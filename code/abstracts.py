from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter
import re
import gc
from typing import Dict, List, Tuple, Optional, Set
import pickle
import os
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import logging
from functools import partial
import time
from scipy import stats
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""

    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_ngrams_for_year(texts: List[str], ngram_range: Tuple[int, int], whitelist: Set[str]) -> Tuple[
    Dict[str, int], Set[str]]:
    """Extract n-grams from texts for a specific year"""
    if not texts or all(not text.strip() for text in texts):
        return {}, set()

    # Use CountVectorizer for n-gram extraction
    if whitelist:
        # If we have a whitelist, restrict to those n-grams
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words='english',
            min_df=1,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
            vocabulary=whitelist
        )
    else:
        # First batch or no whitelist yet - extract all n-grams
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words='english',
            min_df=1,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )

    try:
        # Fit and transform the texts
        count_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Get counts and document presence
        ngram_counts = {}
        ngram_doc_presence = set()

        for i, ngram in enumerate(feature_names):
            count = count_matrix[:, i].sum()
            if count > 0:
                ngram_counts[ngram] = count
                # Check if this n-gram appears in any document
                if (count_matrix[:, i] > 0).any():
                    ngram_doc_presence.add(ngram)

        return ngram_counts, ngram_doc_presence

    except ValueError as e:
        logger.error(f"Error extracting n-grams: {e}")
        return {}, set()


def process_year_worker(args: Tuple) -> Dict:
    """Worker function to process texts for a specific year"""
    year, raw_texts, ngram_range, whitelist = args

    # Preprocess texts
    texts = [preprocess_text(text) for text in raw_texts]
    texts = [text for text in texts if text.strip()]

    if not texts:
        return {
            'year': year,
            'year_doc_count': 0,
            'year_word_count': 0,
            'ngram_counts': {},
            'doc_presence': set(),
            'total_docs': 0
        }

    # Extract n-grams
    ngram_counts, doc_presence = extract_ngrams_for_year(texts, ngram_range, whitelist)

    # Count total words for this year
    total_words = sum(len(text.split()) for text in texts)

    return {
        'year': year,
        'year_doc_count': len(texts),
        'year_word_count': total_words,
        'ngram_counts': ngram_counts,
        'doc_presence': doc_presence,
        'total_docs': len(texts)
    }

class ProgressiveNgramAnalyzer:
    def __init__(self, database_url: str, batch_size: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 target_ngrams: int = 5000,
                 confidence_level: float = 0.95,
                 min_year_frequency: int = 2,
                 n_processes: Optional[int] = None,
                 chunk_size: int = 100):
        """
        Initialize the progressive N-gram analyzer with beta-binomial filtering

        Args:
            database_url: Database connection string
            batch_size: Number of records to fetch from DB at once
            ngram_range: Range of n-grams to extract (min_n, max_n)
            target_ngrams: Target number of top n-grams to keep
            confidence_level: Confidence level for beta-binomial estimates
            min_year_frequency: Minimum frequency per year to be included
            n_processes: Number of processes to use for n-gram extraction
            chunk_size: Size of text chunks for parallel processing
        """
        self.database_url = database_url
        self.batch_size = batch_size
        self.ngram_range = ngram_range
        self.target_ngrams = target_ngrams
        self.confidence_level = confidence_level
        self.min_year_frequency = min_year_frequency
        self.n_processes = n_processes or mp.cpu_count()
        self.chunk_size = chunk_size
        self.engine = create_engine(database_url)

        # Progressive filtering state
        self.ngram_whitelist = set()  # Current whitelist of promising n-grams
        self.ngram_doc_counts = defaultdict(int)  # How many documents contain each n-gram
        self.total_docs_processed = 0
        self.estimated_total_docs = None

        # Storage for results
        self.year_ngram_counts = defaultdict(lambda: defaultdict(int))
        self.year_doc_counts = defaultdict(int)
        self.year_word_counts = defaultdict(int)
        self.global_ngram_counts = defaultdict(int)
        self.total_docs = 0

        # Frequency tracking for progressive filtering
        self.current_top_frequencies = []  # Track current top N frequencies

        logger.info(f"Initialized progressive analyzer with {self.n_processes} processes")

    def get_total_records(self) -> int:
        """Get total number of records to process"""
        count_query = """
                      SELECT COUNT(*) as total
                      FROM articles a
                               JOIN abstracts ab ON a.article_id = ab.article_id
                      WHERE ab.abstract IS NOT NULL \
                        AND ab.abstract != '' \
                      """

        with self.engine.connect() as conn:
            result = pd.read_sql_query(count_query, conn)
            return result['total'].iloc[0]

    def get_year_distribution(self) -> Dict[int, int]:
        """Get distribution of documents per year for uniform sampling"""
        dist_query = """
                     SELECT a.publication_year, COUNT(*) as count
                     FROM articles a
                         JOIN abstracts ab \
                     ON a.article_id = ab.article_id
                     WHERE ab.abstract IS NOT NULL AND ab.abstract != ''
                     GROUP BY a.publication_year
                     ORDER BY a.publication_year \
                     """

        with self.engine.connect() as conn:
            result = pd.read_sql_query(dist_query, conn)
            return dict(zip(result['publication_year'], result['count']))

    def create_stratified_batches(self, total_records: int) -> List[Tuple[int, int]]:
        """Create batches that sample uniformly across years"""
        year_distribution = self.get_year_distribution()
        years = list(year_distribution.keys())

        batches = []
        total_years = len(years)
        docs_per_batch = self.batch_size

        # Calculate how many batches we'll need
        num_batches = math.ceil(total_records / docs_per_batch)

        for batch_idx in range(num_batches):
            # For uniform sampling, we'll use TABLESAMPLE or ORDER BY RANDOM()
            # But for simplicity, we'll use offset-based sampling with year ordering
            offset = batch_idx * docs_per_batch
            batches.append((offset, docs_per_batch))

        return batches

    def estimate_global_frequency_bounds(self, ngram: str, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Estimate lower and upper bounds for global frequency using beta-binomial model

        Args:
            ngram: The n-gram to estimate
            alpha: Significance level (1 - confidence_level)

        Returns:
            (lower_bound, upper_bound) for estimated global frequency
        """
        if self.estimated_total_docs is None or self.total_docs_processed == 0:
            return 0.0, float('inf')

        # Beta-binomial parameters
        docs_with_ngram = self.ngram_doc_counts.get(ngram, 0)
        docs_without_ngram = self.total_docs_processed - docs_with_ngram

        # Beta distribution parameters
        alpha_param = 1 + docs_with_ngram
        beta_param = 1 + docs_without_ngram

        # Calculate confidence interval for the probability
        prob_lower = stats.beta.ppf(alpha / 2, alpha_param, beta_param)
        prob_upper = stats.beta.ppf(1 - alpha / 2, alpha_param, beta_param)

        # Convert to estimated global frequency
        freq_lower = prob_lower * self.estimated_total_docs
        freq_upper = prob_upper * self.estimated_total_docs

        return freq_lower, freq_upper

    def should_keep_ngram(self, ngram: str) -> bool:
        """
        Decide whether to keep an n-gram based on its estimated potential
        """
        if len(self.current_top_frequencies) < self.target_ngrams:
            return True  # Keep everything until we have enough candidates

        # Get the current minimum frequency in our top N
        min_top_freq = min(self.current_top_frequencies)

        # Estimate upper bound for this n-gram's global frequency
        _, freq_upper = self.estimate_global_frequency_bounds(ngram)

        # Keep if upper bound suggests it could make it to top N
        return freq_upper >= min_top_freq

    def update_progressive_filter(self):
        """Update the whitelist based on current frequency estimates"""
        if self.total_docs_processed == 0:
            return

        # Update current top frequencies
        if self.global_ngram_counts:
            all_frequencies = list(self.global_ngram_counts.values())
            all_frequencies.sort(reverse=True)
            self.current_top_frequencies = all_frequencies[:self.target_ngrams]

        # Filter whitelist
        new_whitelist = set()
        for ngram in self.ngram_whitelist:
            if self.should_keep_ngram(ngram):
                new_whitelist.add(ngram)

        removed_count = len(self.ngram_whitelist) - len(new_whitelist)
        if removed_count > 0:
            logger.info(f"Progressive filtering: removed {removed_count} n-grams, "
                        f"keeping {len(new_whitelist)} candidates")

        self.ngram_whitelist = new_whitelist

        # Clean up counts for removed n-grams to save memory
        ngrams_to_remove = []
        for ngram in self.global_ngram_counts:
            if ngram not in self.ngram_whitelist:
                ngrams_to_remove.append(ngram)

        for ngram in ngrams_to_remove:
            del self.global_ngram_counts[ngram]
            if ngram in self.ngram_doc_counts:
                del self.ngram_doc_counts[ngram]

            # Remove from year counts
            for year_counts in self.year_ngram_counts.values():
                if ngram in year_counts:
                    del year_counts[ngram]

    def process_years_parallel(self, df: pd.DataFrame) -> List[Dict]:
        """Process all years in a batch using parallel processing"""
        if df.empty:
            return []

        # Prepare arguments for each year
        year_args = []
        for year, year_group in df.groupby('publication_year'):
            raw_texts = year_group['abstract'].tolist()
            year_args.append((year, raw_texts, self.ngram_range, self.ngram_whitelist))

        # Process years in parallel
        if len(year_args) == 1:
            # Single year, process directly
            return [process_year_worker(year_args[0])]

        with Pool(processes=min(self.n_processes, len(year_args))) as pool:
            year_results = pool.map(process_year_worker, year_args)

        return year_results

    def process_batch(self, offset: int, batch_size: int) -> bool:
        """Process a single batch with progressive filtering"""
        # Sample uniformly across years using random sampling
        query = f"""
        SELECT 
            a.publication_year, ab.abstract
        FROM articles a
        JOIN abstracts ab ON a.article_id = ab.article_id
        WHERE ab.abstract IS NOT NULL AND ab.abstract != ''
        LIMIT {batch_size} OFFSET {offset}
        """

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)

            if df.empty:
                return False

            logger.info(f"Processing batch at offset {offset}, {len(df)} records...")

            # Group by year for processing
            for year, year_group in df.groupby('publication_year'):
                # Preprocess texts
                raw_texts = year_group['abstract'].tolist()
                texts = [preprocess_text(text) for text in raw_texts]
                texts = [text for text in texts if text.strip()]

                if not texts:
                    continue

                # Extract n-grams with current whitelist
                ngram_counts, doc_presence = self.extract_ngrams_parallel(texts)

                # Update whitelist with new n-grams (for first few batches)
                if len(self.ngram_whitelist) < self.target_ngrams * 2:  # Allow some buffer
                    self.ngram_whitelist.update(ngram_counts.keys())

                # Count total words for this year
                total_words = sum(len(text.split()) for text in texts)

                # Update counters
                self.year_doc_counts[year] += len(texts)
                self.year_word_counts[year] += total_words

                # Update n-gram counts and document presence
                for ngram, count in ngram_counts.items():
                    self.year_ngram_counts[year][ngram] += count
                    self.global_ngram_counts[ngram] += count

                for ngram in doc_presence:
                    self.ngram_doc_counts[ngram] += 1

                self.total_docs_processed += len(texts)

            self.total_docs += len(df)

            # Apply progressive filtering every few batches
            if self.total_docs % (self.batch_size * 5) == 0:  # Every 5 batches
                self.update_progressive_filter()
                gc.collect()  # Force garbage collection

            del df
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            return False

    def process_all_batches(self):
        """Process all batches with progressive filtering"""
        logger.info("Getting total record count...")
        total_records = self.get_total_records()
        self.estimated_total_docs = total_records
        logger.info(f"Total records to process: {total_records}")

        # Create stratified batches
        batches = self.create_stratified_batches(total_records)
        logger.info(f"Created {len(batches)} batches for uniform sampling")

        start_time = time.time()

        for batch_idx, (offset, batch_size) in enumerate(batches):
            batch_start = time.time()

            if not self.process_batch(offset, batch_size):
                break

            batch_end = time.time()

            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                elapsed = batch_end - start_time
                progress = (batch_idx + 1) / len(batches)
                estimated_total = elapsed / progress if progress > 0 else 0
                remaining = estimated_total - elapsed

                logger.info(f"Batch {batch_idx + 1}/{len(batches)} ({progress:.1%}). "
                            f"Whitelist size: {len(self.ngram_whitelist)}. "
                            f"Est. remaining: {remaining / 60:.1f} min")

        # Final filtering
        self.update_progressive_filter()

        end_time = time.time()
        logger.info(f"Progressive processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Final whitelist size: {len(self.ngram_whitelist)}")
        logger.info(f"Processed {self.total_docs} documents")

    def calculate_frequencies(self) -> Dict[str, Dict]:
        """Calculate frequency metrics for final n-grams"""
        logger.info("Calculating final frequency metrics...")

        frequency_data = {}

        for ngram in self.ngram_whitelist:
            ngram_data = {
                'absolute_frequency': {},
                'relative_frequency': {},
                'normalized_frequency': {},
                'total_frequency': self.global_ngram_counts.get(ngram, 0),
                'document_frequency': self.ngram_doc_counts.get(ngram, 0),
                'years_present': []
            }

            for year, ngram_counts in self.year_ngram_counts.items():
                if ngram in ngram_counts:
                    count = ngram_counts[ngram]
                    total_ngrams_in_year = sum(ngram_counts.values())
                    total_words_in_year = self.year_word_counts[year]

                    ngram_data['absolute_frequency'][year] = count

                    if total_ngrams_in_year > 0:
                        ngram_data['relative_frequency'][year] = count / total_ngrams_in_year

                    if total_words_in_year > 0:
                        ngram_data['normalized_frequency'][year] = (count / total_words_in_year) * 1000

                    ngram_data['years_present'].append(year)

            frequency_data[ngram] = ngram_data

        return frequency_data

    def get_top_ngrams_by_metric(self, frequency_data: Dict,
                                 metric: str = 'total_frequency',
                                 top_k: int = 50) -> List[Tuple[str, float]]:
        """Get top n-grams by specified metric"""
        if metric == 'total_frequency':
            sorted_ngrams = sorted(
                [(ngram, data['total_frequency']) for ngram, data in frequency_data.items()],
                key=lambda x: x[1], reverse=True
            )
        elif metric == 'document_frequency':
            sorted_ngrams = sorted(
                [(ngram, data['document_frequency']) for ngram, data in frequency_data.items()],
                key=lambda x: x[1], reverse=True
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted_ngrams[:top_k]

    def export_to_csv(self, frequency_data: Dict, filename_prefix: str = "progressive_ngram_analysis"):
        """Export results to CSV files"""
        logger.info("Exporting results to CSV...")

        # Top n-grams
        top_ngrams = self.get_top_ngrams_by_metric(frequency_data, 'total_frequency', 100)
        top_df = pd.DataFrame(top_ngrams, columns=['ngram', 'total_frequency'])
        top_df.to_csv(f"{filename_prefix}_top_ngrams.csv", index=False)

        # Detailed data
        detailed_data = []
        for ngram, data in frequency_data.items():
            for year in data['years_present']:
                detailed_data.append({
                    'ngram': ngram,
                    'year': year,
                    'absolute_frequency': data['absolute_frequency'][year],
                    'relative_frequency': data['relative_frequency'].get(year, 0),
                    'normalized_frequency': data['normalized_frequency'].get(year, 0)
                })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(f"{filename_prefix}_detailed.csv", index=False)

        logger.info(f"Exported results with prefix: {filename_prefix}")

    def run_analysis(self, save_results: bool = True, export_csv: bool = True) -> Dict:
        """Run the complete progressive analysis"""
        logger.info("Starting progressive N-gram frequency analysis...")

        # Process with progressive filtering
        self.process_all_batches()

        # Calculate final frequencies
        frequency_data = self.calculate_frequencies()

        if export_csv:
            self.export_to_csv(frequency_data)

        return frequency_data


# Usage example
def main():
    analyzer = ProgressiveNgramAnalyzer(
        database_url="sqlite:///articles.db",
        batch_size=5000,
        ngram_range=(1, 3),
        target_ngrams=5000,  # Target number of top n-grams
        confidence_level=0.95,  # For beta-binomial estimates
        n_processes=None,
        chunk_size=100
    )

    start_time = time.time()
    frequency_data = analyzer.run_analysis()
    end_time = time.time()

    logger.info(f"Total analysis time: {end_time - start_time:.2f} seconds")

    # Display results
    top_ngrams = analyzer.get_top_ngrams_by_metric(frequency_data, 'total_frequency', 20)
    print("\nTop 20 most frequent n-grams:")
    for i, (ngram, freq) in enumerate(top_ngrams, 1):
        print(f"{i:2d}. {ngram}: {freq:,} occurrences")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()