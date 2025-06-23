from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter
import re
import gc
from typing import Dict, List, Tuple, Optional, Set
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import logging
import time
from scipy import stats
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


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


def extract_ngrams_for_articles(args: Tuple) -> Dict:
    """Extract n-grams from a batch of articles (new worker function)"""
    articles_data, ngram_range, blacklist = args

    if not articles_data:
        return {
            'year_results': {},
            'total_processed': 0
        }

    # Group articles by year
    year_groups = defaultdict(list)
    year_article_ids = defaultdict(list)

    for article_id, year, raw_text in articles_data:
        processed_text = preprocess_text(raw_text)
        if processed_text.strip():
            year_groups[year].append(processed_text)
            year_article_ids[year].append(article_id)

    year_results = {}
    total_processed = 0

    # Process each year in this batch
    for year, texts in year_groups.items():
        article_ids = year_article_ids[year]

        if not texts:
            continue

        # Extract n-grams for this year
        ngram_counts, doc_presence, ngram_articles = extract_ngrams_for_year(
            texts, article_ids, ngram_range, blacklist
        )

        # Count total words for this year
        total_words = sum(len(text.split()) for text in texts)

        year_results[year] = {
            'year_doc_count': len(texts),
            'year_word_count': total_words,
            'ngram_counts': ngram_counts,
            'doc_presence': doc_presence,
            'ngram_articles': ngram_articles,
            'total_docs': len(texts)
        }

        total_processed += len(texts)

    return {
        'year_results': year_results,
        'total_processed': total_processed
    }


def extract_ngrams_for_year(texts: List[str], article_ids: List[int], ngram_range: Tuple[int, int],
                            blacklist: Set[str]) -> Tuple[Dict[str, int], Set[str], Dict[str, set]]:
    """Extract n-grams from texts for a specific year, excluding blacklisted ones"""
    if not texts or all(not text.strip() for text in texts):
        return {}, set(), {}

    # Use CountVectorizer for n-gram extraction
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words='english',
        min_df=1,
        tokenizer=LemmaTokenizer(),
        strip_accents='unicode',
        lowercase=True
    )

    try:
        # Fit and transform the texts
        count_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Get counts and document presence, excluding blacklisted n-grams
        ngram_counts = {}
        ngram_doc_presence = set()
        ngram_articles = defaultdict(set)

        for i, ngram in enumerate(feature_names):
            if ngram in blacklist:
                continue  # Skip blacklisted n-grams

            count = count_matrix[:, i].sum()
            if count > 0:
                ngram_counts[ngram] = count
                # Check if this n-gram appears in any document
                if count_matrix[:, i].nnz > 0:
                    ngram_doc_presence.add(ngram)

                    # Track which articles contain this ngram
                    doc_indices = count_matrix[:, i].nonzero()[0]
                    for doc_idx in doc_indices:
                        ngram_articles[ngram].add(article_ids[doc_idx])

        return ngram_counts, ngram_doc_presence, dict(ngram_articles)

    except ValueError as e:
        logger.error(f"Error extracting n-grams: {e}")
        return {}, set(), {}


def chunk_articles(articles_data: List[Tuple], chunk_size: int) -> List[List[Tuple]]:
    """Split articles into chunks for parallel processing"""
    chunks = []
    for i in range(0, len(articles_data), chunk_size):
        chunks.append(articles_data[i:i + chunk_size])
    return chunks


class TemporalVariationNgramAnalyzer:
    def __init__(self, database_url: str, batch_size: int = 100000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 min_fold_change: float = 3.0,
                 confidence_level: float = 0.95,
                 min_total_frequency: int = 10,
                 min_years_present: int = 3,
                 n_processes: Optional[int] = None,
                 articles_per_chunk: int = 1000):
        """
        Initialize the temporal variation N-gram analyzer with improved parallelization

        Args:
            database_url: Database connection string
            batch_size: Number of records to fetch from DB at once
            ngram_range: Range of n-grams to extract (min_n, max_n)
            min_fold_change: Minimum fold change required (e.g., 2.0 for 2x increase)
            confidence_level: Confidence level for statistical tests
            min_total_frequency: Minimum total frequency across all years
            min_years_present: Minimum number of years n-gram must appear in
            n_processes: Number of processes to use for n-gram extraction
            articles_per_chunk: Number of articles per chunk for parallel processing
        """
        self.database_url = database_url
        self.batch_size = batch_size
        self.ngram_range = ngram_range
        self.min_fold_change = min_fold_change
        self.confidence_level = confidence_level
        self.min_total_frequency = min_total_frequency
        self.min_years_present = min_years_present
        self.n_processes = n_processes or mp.cpu_count()
        self.articles_per_chunk = articles_per_chunk
        self.engine = create_engine(database_url)

        # Progressive filtering state
        self.ngram_blacklist = set()  # N-grams to exclude
        self.total_docs_processed = 0
        self.estimated_total_docs = None

        # Temporal variation tracking
        self.year_ngram_normalized_freq = defaultdict(lambda: defaultdict(float))
        self.year_total_words = defaultdict(int)
        self.ngram_year_counts = defaultdict(lambda: defaultdict(int))
        self.observed_years = set()

        # Storage for results
        self.year_ngram_counts = defaultdict(lambda: defaultdict(int))
        self.year_doc_counts = defaultdict(int)
        self.year_word_counts = defaultdict(int)
        self.global_ngram_counts = defaultdict(int)
        self.total_docs = 0

        # Article tracking
        self.ngram_articles = defaultdict(set)

        # Cursor-based pagination state
        self.last_random_rank = None

        logger.info(f"Initialized temporal variation analyzer with {self.n_processes} processes, "
                    f"targeting {min_fold_change}x fold changes, "
                    f"{articles_per_chunk} articles per chunk")

    def get_total_records(self) -> int:
        """Get total number of records to process"""
        count_query = """
                      SELECT COUNT(*) as total
                      FROM articles a
                               JOIN abstracts ab ON a.article_id = ab.article_id
                      """

        with self.engine.connect() as conn:
            result = pd.read_sql_query(count_query, conn)
            return result['total'].iloc[0]

    def calculate_normalized_frequency(self, count: int, total_words: int, per_k_words: int = 1000) -> float:
        """Calculate normalized frequency per K words"""
        if total_words == 0:
            return 0.0
        return (count / total_words) * per_k_words

    def estimate_year_frequency_bounds(self, ngram: str, year: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Estimate confidence bounds for normalized frequency in a specific year"""
        if year not in self.year_total_words or self.year_total_words[year] == 0:
            return 0.0, 0.0

        observed_count = self.ngram_year_counts[ngram].get(year, 0)
        total_words_year = self.year_total_words[year]

        if observed_count == 0:
            alpha_param = 1
            beta_param = 1
        else:
            alpha_param = observed_count + 1
            beta_param = total_words_year - observed_count + 1

        rate_lower = stats.beta.ppf(alpha / 2, alpha_param, beta_param)
        rate_upper = stats.beta.ppf(1 - alpha / 2, alpha_param, beta_param)

        freq_lower = rate_lower * 1000
        freq_upper = rate_upper * 1000

        return freq_lower, freq_upper

    def has_sufficient_statistical_power(self, ngram: str) -> bool:
        """Check if an n-gram has sufficient frequency for reliable detection"""
        if ngram not in self.global_ngram_counts:
            return True

        if self.total_docs_processed == 0 or self.estimated_total_docs is None:
            return True

        observed_count = self.global_ngram_counts[ngram]
        total_docs_observed = self.total_docs_processed

        alpha_param = observed_count + 1
        beta_param = total_docs_observed - observed_count + 1

        alpha = 1 - self.confidence_level
        rate_upper = stats.beta.ppf(1 - alpha / 2, alpha_param, beta_param)

        estimated_total_count_upper = rate_upper * self.estimated_total_docs

        return estimated_total_count_upper > 1500.0

    def can_achieve_fold_change(self, ngram: str) -> bool:
        """Check if an n-gram can potentially achieve the required fold change"""
        if len(self.observed_years) < 2:
            return True

        if ngram not in self.ngram_year_counts:
            return True

        year_bounds = {}
        for year in self.observed_years:
            if year in self.year_total_words and self.year_total_words[year] > 0:
                lower, upper = self.estimate_year_frequency_bounds(ngram, year)
                year_bounds[year] = (lower, upper)

        if len(year_bounds) < 2:
            return True

        max_upper = max(upper for lower, upper in year_bounds.values())
        min_lower = min(lower for lower, upper in year_bounds.values() if lower > 0)

        if min_lower == 0:
            return max_upper > 0

        max_possible_fold_change = max_upper / min_lower
        return max_possible_fold_change >= self.min_fold_change

    def update_blacklist(self):
        """Add n-grams to blacklist that fail filtering criteria and purge their data"""
        if len(self.observed_years) < 3:
            return

        new_blacklisted = set()
        filter_counts = {
            'low_frequency': 0,
            'insufficient_years': 0,
            'no_statistical_power': 0,
            'no_fold_change': 0
        }

        for ngram in self.global_ngram_counts:
            if ngram in self.ngram_blacklist:
                continue  # Already blacklisted

            total_freq = self.global_ngram_counts.get(ngram, 0)
            years_present = len([year for year in self.observed_years
                                 if self.ngram_year_counts[ngram].get(year, 0) > 0])

            # Check filtering criteria
            should_blacklist = False
            reason = None

            if years_present < self.min_years_present:
                should_blacklist = True
                reason = 'insufficient_years'
            elif not self.has_sufficient_statistical_power(ngram):
                should_blacklist = True
                reason = 'no_statistical_power'
            elif not self.can_achieve_fold_change(ngram):
                should_blacklist = True
                reason = 'no_fold_change'

            if should_blacklist:
                new_blacklisted.add(ngram)
                filter_counts[reason] += 1

        # Add to blacklist
        self.ngram_blacklist.update(new_blacklisted)

        if new_blacklisted:
            total_blacklisted = len(new_blacklisted)
            logger.info(f"Blacklisted {total_blacklisted} n-grams: "
                        f"{filter_counts['low_frequency']} low frequency, "
                        f"{filter_counts['insufficient_years']} insufficient years, "
                        f"{filter_counts['no_statistical_power']} insufficient statistical power, "
                        f"{filter_counts['no_fold_change']} no fold change potential. "
                        f"Total blacklisted: {len(self.ngram_blacklist)}")

        # Clean up data for blacklisted n-grams to save memory
        for ngram in new_blacklisted:
            if ngram in self.global_ngram_counts:
                del self.global_ngram_counts[ngram]
            if ngram in self.ngram_year_counts:
                del self.ngram_year_counts[ngram]
            if ngram in self.year_ngram_normalized_freq:
                del self.year_ngram_normalized_freq[ngram]
            if ngram in self.ngram_articles:  # Purge article tracking
                del self.ngram_articles[ngram]

            for year_counts in self.year_ngram_counts.values():
                if ngram in year_counts:
                    del year_counts[ngram]

    def process_articles_parallel(self, articles_data: List[Tuple]) -> List[Dict]:
        """Process articles using improved parallel processing"""
        if not articles_data:
            return []

        # Split articles into chunks for parallel processing
        chunks = chunk_articles(articles_data, self.articles_per_chunk)

        # Prepare arguments for worker processes
        chunk_args = [(chunk, self.ngram_range, self.ngram_blacklist) for chunk in chunks]

        logger.info(f"Processing {len(articles_data)} articles in {len(chunks)} chunks "
                    f"using {self.n_processes} processes")

        # Use fixed number of processes regardless of data size
        with Pool(processes=self.n_processes) as pool:
            chunk_results = pool.map(extract_ngrams_for_articles, chunk_args)

        return chunk_results

    def process_batch(self, batch_size: int) -> bool:
        """Process a single batch using cursor-based pagination"""
        if self.last_random_rank is None:
            query = """
                    SELECT a.article_id,
                           a.publication_year,
                           a.title,
                           ab.abstract,
                           ao.random_rank
                    FROM articles a
                             JOIN abstracts ab ON a.article_id = ab.article_id
                             JOIN articles_order ao ON a.article_id = ao.article_id
                    ORDER BY ao.random_rank LIMIT :batch_size
                    """
            params = {'batch_size': batch_size}
        else:
            query = """
                    SELECT a.article_id,
                           a.publication_year,
                           a.title,
                           ab.abstract,
                           ao.random_rank
                    FROM articles a
                             JOIN abstracts ab ON a.article_id = ab.article_id
                             JOIN articles_order ao ON a.article_id = ao.article_id
                    WHERE ao.random_rank > :last_rank
                    ORDER BY ao.random_rank LIMIT :batch_size
                    """

            params = {'last_rank': int(self.last_random_rank), 'batch_size': batch_size}

        try:
            logger.info(f"Querying abstracts with cursor at rank {self.last_random_rank}")

            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            # Add this in the process_batch method when df.empty is True:
            if df.empty:
                logger.info("No more records found - batch processing complete")
                return False

            logger.info(f"Retrieved {len(df)} records from database")

            # Update cursor position - use the last random_rank from this batch
            if len(df) > 0:
                self.last_random_rank = df['random_rank'].iloc[-1]
                logger.info(f"Updated cursor to random_rank: {self.last_random_rank}")
            else:
                logger.info("Empty batch received")
                return False

            logger.info(f"Processing batch with cursor at rank {self.last_random_rank}, {len(df)} records...")

            df.fillna("", inplace=True)
            # Prepare articles data as list of tuples
            articles_data = [(row['article_id'], row['publication_year'], row['title'] + '. ' + row['abstract'])
                             for _, row in df.iterrows()]

            # Process articles in parallel
            chunk_results = self.process_articles_parallel(articles_data)

            logger.info(f"Aggregating batch results from {len(chunk_results)} chunks...")

            # Aggregate results from all chunks
            total_processed = 0
            for chunk_result in chunk_results:
                total_processed += chunk_result['total_processed']

                # Process each year's results from this chunk
                for year, year_result in chunk_result['year_results'].items():
                    self.observed_years.add(year)

                    # Update year totals
                    self.year_doc_counts[year] += year_result['year_doc_count']
                    self.year_word_counts[year] += year_result['year_word_count']
                    self.year_total_words[year] += year_result['year_word_count']

                    # Update n-gram counts and temporal data
                    for ngram, count in year_result['ngram_counts'].items():
                        self.year_ngram_counts[year][ngram] += count
                        self.global_ngram_counts[ngram] += count
                        self.ngram_year_counts[ngram][year] += count

                        # Calculate normalized frequency
                        if self.year_total_words[year] > 0:
                            normalized_freq = self.calculate_normalized_frequency(
                                self.ngram_year_counts[ngram][year],
                                self.year_total_words[year]
                            )
                            self.year_ngram_normalized_freq[ngram][year] = normalized_freq

                    # Update article tracking
                    for ngram, article_set in year_result['ngram_articles'].items():
                        self.ngram_articles[ngram].update(article_set)

            self.total_docs_processed += total_processed
            self.total_docs += len(df)

            # Apply blacklist filtering every batch
            if self.total_docs % self.batch_size == 0:
                logger.info(f"Applying blacklist...")
                self.update_blacklist()
                gc.collect()

            del df, articles_data
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"Error processing batch with cursor at rank {self.last_random_rank}: {e}")
            return False

    def process_all_batches(self):
        """Process all batches using cursor-based pagination"""
        logger.info("Getting total record count...")
        total_records = self.get_total_records()
        self.estimated_total_docs = total_records
        logger.info(f"Total records to process: {total_records}")

        start_time = time.time()
        batch_count = 0
        total_processed_docs = 0

        while True:
            batch_start = time.time()

            if not self.process_batch(self.batch_size):
                break

            batch_end = time.time()
            batch_count += 1
            total_processed_docs = self.total_docs  # Use actual count from batches

            # Progress reporting
            if batch_count % 1 == 0:  # Report every batch for debugging, change back to 10 later
                elapsed = batch_end - start_time
                avg_time_per_batch = elapsed / batch_count
                remaining_records = max(0, total_records - total_processed_docs)
                estimated_remaining_batches = remaining_records / self.batch_size if self.batch_size > 0 else 0
                estimated_remaining = estimated_remaining_batches * avg_time_per_batch

                active_ngrams = len(self.global_ngram_counts)
                logger.info(f"Batch {batch_count}: {total_processed_docs:,}/{total_records:,} records processed. "
                            f"Active n-grams: {active_ngrams}, Blacklisted: {len(self.ngram_blacklist)}. "
                            f"Years observed: {len(self.observed_years)}. "
                            f"Cursor at rank: {self.last_random_rank}. "
                            f"Est. remaining: {estimated_remaining / 60:.1f} min")

        # Final blacklist update
        self.update_blacklist()

        end_time = time.time()
        logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Final active n-grams: {len(self.global_ngram_counts)}, "
                    f"Total blacklisted: {len(self.ngram_blacklist)}")

    def calculate_final_statistics(self) -> Dict[str, Dict]:
        """Calculate final temporal variation statistics"""
        logger.info("Calculating final temporal variation statistics...")

        results = {}

        for ngram in self.global_ngram_counts:
            if ngram in self.ngram_blacklist:
                continue

            if ngram not in self.year_ngram_normalized_freq:
                continue

            frequencies = self.year_ngram_normalized_freq[ngram]
            years_present = list(frequencies.keys())
            freq_values = [frequencies[year] for year in years_present if frequencies[year] > 0]

            if len(freq_values) < 2:
                continue

            # Calculate document frequency and estimated total count
            total_count = self.global_ngram_counts[ngram]
            document_frequency = len(years_present)

            # Calculate lower bound estimate for total count
            if self.total_docs_processed > 0 and self.estimated_total_docs:
                alpha_param = total_count + 1
                beta_param = max(1.0, self.total_docs_processed - total_count + 1)
                alpha = 1 - self.confidence_level
                rate_lower = stats.beta.ppf(alpha / 2, alpha_param, beta_param)
                estimated_total_count_lower = rate_lower * self.estimated_total_docs
            else:
                estimated_total_count_lower = total_count

            # Calculate fold change
            max_freq = max(freq_values)
            min_freq = min(freq_values)
            fold_change = max_freq / min_freq if min_freq > 0 else float('inf')

            # Find peak and trough years
            max_year = max(years_present, key=lambda y: frequencies[y])
            min_year = min(years_present, key=lambda y: frequencies[y] if frequencies[y] > 0 else float('inf'))

            # Calculate confidence intervals for the fold change
            max_lower, max_upper = self.estimate_year_frequency_bounds(ngram, max_year)
            min_lower, min_upper = self.estimate_year_frequency_bounds(ngram, min_year)

            fold_change_lower = max_lower / min_upper if min_upper > 0 else 0
            fold_change_upper = max_upper / min_lower if min_lower > 0 else float('inf')

            results[ngram] = {
                'fold_change': fold_change,
                'fold_change_ci_lower': fold_change_lower,
                'fold_change_ci_upper': fold_change_upper,
                'max_frequency': max_freq,
                'min_frequency': min_freq,
                'max_year': max_year,
                'min_year': min_year,
                'years_present': len(years_present),
                'total_frequency': total_count,
                'document_frequency': document_frequency,
                'estimated_total_count_lower': estimated_total_count_lower,
                'yearly_frequencies': dict(frequencies),
                'statistically_significant': (fold_change_lower >= self.min_fold_change and
                                              estimated_total_count_lower > 1000.0),
                'sufficient_power': estimated_total_count_lower > 1000.0
            }

        return results

    def export_temporal_results(self, results: Dict, filename_prefix: str = "temporal_ngrams"):
        """Export temporal variation results"""
        logger.info("Exporting temporal variation results...")

        # Summary of significant n-grams
        significant_data = []
        for ngram, data in results.items():
            if data['statistically_significant']:
                significant_data.append({
                    'ngram': ngram,
                    'fold_change': data['fold_change'],
                    'fold_change_ci_lower': data['fold_change_ci_lower'],
                    'fold_change_ci_upper': data['fold_change_ci_upper'],
                    'max_frequency': data['max_frequency'],
                    'min_frequency': data['min_frequency'],
                    'max_year': data['max_year'],
                    'min_year': data['min_year'],
                    'years_present': data['years_present'],
                    'total_frequency': data['total_frequency'],
                    'document_frequency': data['document_frequency'],
                    'estimated_total_count_lower': data['estimated_total_count_lower']
                })

        significant_df = pd.DataFrame(significant_data)
        significant_df = significant_df.sort_values('fold_change', ascending=False)
        significant_df.to_csv(f"{filename_prefix}_significant.csv", index=False)

        # Time series data
        time_series_data = []
        for ngram, data in results.items():
            for year, freq in data['yearly_frequencies'].items():
                time_series_data.append({
                    'ngram': ngram,
                    'year': year,
                    'normalized_frequency': freq,
                    'fold_change': data['fold_change'],
                    'estimated_total_count_lower': data['estimated_total_count_lower'],
                    'significant': data['statistically_significant']
                })

        time_series_df = pd.DataFrame(time_series_data)
        time_series_df.to_csv(f"{filename_prefix}_timeseries.csv", index=False)

        logger.info(f"Exported {len(significant_data)} significant temporal n-grams")
        return len(significant_data)

    def export_article_mapping(self, filename: str = "ngram_articles.npz"):
        """Export n-gram to article mapping for non-blacklisted n-grams"""
        logger.info("Exporting n-gram to article mapping...")

        # Convert to numpy arrays with ngrams as keys
        arrays_dict = {}
        for ngram, article_set in self.ngram_articles.items():
            if ngram not in self.ngram_blacklist:
                arrays_dict[ngram] = np.array(list(article_set))

        if 'file' in arrays_dict:
            del arrays_dict['file']

        # Save as NPZ with ngrams as keys
        np.savez_compressed(f"output/{filename}", **arrays_dict)

        logger.info(f"Exported article mapping for {len(arrays_dict)} n-grams to {filename}")
        return len(arrays_dict)

    def run_analysis(self, export_results: bool = True) -> Dict:
        """Run the complete temporal variation analysis"""
        logger.info(f"Starting temporal variation analysis (min {self.min_fold_change}x fold change)...")

        # Process with cursor-based pagination
        self.process_all_batches()

        # Calculate final statistics
        results = self.calculate_final_statistics()

        if export_results:
            self.export_temporal_results(results)
            self.export_article_mapping()

        return results


# Usage example
def main():
    analyzer = TemporalVariationNgramAnalyzer(
        database_url="sqlite:///articles.db",
        batch_size=100000,
        ngram_range=(1, 2),
        min_fold_change=3.0,
        confidence_level=0.95,
        min_total_frequency=20,
        min_years_present=1,
        n_processes=7,
        articles_per_chunk=2000
    )

    start_time = time.time()
    results = analyzer.run_analysis()
    end_time = time.time()

    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")

    # Display top temporally varying n-grams
    significant_ngrams = [(ngram, data['fold_change'])
                          for ngram, data in results.items()
                          if data['statistically_significant']]

    significant_ngrams.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop temporally varying n-grams (≥{analyzer.min_fold_change}x fold change):")
    for i, (ngram, fold_change) in enumerate(significant_ngrams[:20], 1):
        data = results[ngram]
        print(f"{i:2d}. {ngram}: {fold_change:.1f}x change "
              f"({data['min_year']}: {data['min_frequency']:.2f} → "
              f"{data['max_year']}: {data['max_frequency']:.2f})")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
