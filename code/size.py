import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from scipy import stats
import argparse


class NgramAnalyzer:
    def __init__(self, base_filename: str, db_path: str = "articles.db", ngram_file: str = "output/ngram_articles.npz"):
        self.base_filename = base_filename
        self.new_links_df = pd.read_csv(f"{base_filename}_new_links.csv")
        self.db_path = db_path
        self.ngram_file = ngram_file

        # Load all n-gram data
        self.ngram_data = np.load(ngram_file)
        self.ngrams = list(self.ngram_data.keys())
        print(f"Found {len(self.ngrams)} n-grams: {self.ngrams}")

        # Load collaboration sizes once for all articles
        self.collaboration_sizes = self.get_all_collaboration_sizes()

    def get_all_collaboration_sizes(self):
        """Get collaboration sizes for all articles once and cache them."""
        print("Loading collaboration sizes for all articles...")
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT aa.article_id, COUNT(aa.author_id) as collaboration_size
                FROM articles_authors aa
                GROUP BY aa.article_id \
                """

        result = pd.read_sql_query(query, conn)
        conn.close()

        print(f"Loaded collaboration sizes for {len(result)} articles")
        # Return as dictionary for fast lookup
        return dict(zip(result['article_id'], result['collaboration_size']))

    def get_total_article_counts(self):
        """Get total article counts by year from database."""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT publication_year, COUNT(*) as total_articles
                FROM articles
                GROUP BY publication_year \
                """

        yearly_totals = pd.read_sql_query(query, conn)
        conn.close()

        return dict(zip(yearly_totals['publication_year'], yearly_totals['total_articles']))

    def calculate_binomial_ci(self, successes, total, alpha=0.05):
        """Calculate 95% credible interval for probability using Beta-Binomial model."""
        if total == 0:
            return 0, 0, 0

        # Beta posterior with non-informative prior Beta(1,1)
        a = successes + 1
        b = total - successes + 1

        # Point estimate
        prob = successes / total

        # Credible interval
        lower = stats.beta.ppf(alpha / 2, a, b)
        upper = stats.beta.ppf(1 - alpha / 2, a, b)

        return prob, lower, upper

    def analyze_ngram(self, ngram):
        """Analyze a single n-gram for new link probability and collaboration size."""
        ngram_articles = set(self.ngram_data[ngram])
        yearly_totals = self.get_total_article_counts()

        # Calculate average collaboration size for ngram articles using cached data
        collab_sizes = [self.collaboration_sizes.get(article_id, 0) for article_id in ngram_articles
                        if article_id in self.collaboration_sizes]

        if collab_sizes:
            avg_collaboration_size = np.mean(collab_sizes)
        else:
            avg_collaboration_size = 0

        # Calculate new link probability across all years
        all_with_new_links = set(
            self.new_links_df[self.new_links_df['distance_before_link'] > 0]['article_id'].unique())
        ngram_with_new_links = all_with_new_links.intersection(ngram_articles)

        # Get total counts
        total_articles = sum(yearly_totals.values())
        total_with_new_links = len(all_with_new_links)
        ngram_total_articles = len(ngram_articles)
        ngram_total_with_new_links = len(ngram_with_new_links)

        # Calculate probabilities
        overall_prob, overall_lower, overall_upper = self.calculate_binomial_ci(
            total_with_new_links, total_articles)

        ngram_prob, ngram_lower, ngram_upper = self.calculate_binomial_ci(
            ngram_total_with_new_links, ngram_total_articles)

        return {
            'ngram': ngram,
            'new_link_probability': ngram_prob,
            'new_link_ci_lower': ngram_lower,
            'new_link_ci_upper': ngram_upper,
            'avg_collaboration_size': avg_collaboration_size,
            'total_articles': ngram_total_articles,
            'articles_with_new_links': ngram_total_with_new_links,
            'overall_new_link_prob': overall_prob  # For comparison
        }

    def analyze_all_ngrams(self):
        """Analyze all n-grams and return results as DataFrame."""
        results = []

        print("Analyzing n-grams...")
        for i, ngram in enumerate(self.ngrams):
            print(f"Processing {i + 1}/{len(self.ngrams)}: {ngram}")
            try:
                result = self.analyze_ngram(ngram)
                print(result)
                results.append(result)
            except Exception as e:
                print(f"Error processing {ngram}: {e}")
                continue

        df = pd.DataFrame(results)

        # Sort by new link probability (descending)
        df = df.sort_values('new_link_probability', ascending=False)

        return df

    def save_results(self, df):
        """Save results to CSV and print summary."""
        output_file = f"{self.base_filename}_ngram_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Number of n-grams analyzed: {len(df)}")
        print(f"Overall new link probability: {df['overall_new_link_prob'].iloc[0]:.4f}")
        print(f"\nTop 10 n-grams by new link probability:")
        print(df[['ngram', 'new_link_probability', 'avg_collaboration_size', 'total_articles']].head(10).to_string(
            index=False))

        print(f"\nBottom 10 n-grams by new link probability:")
        print(df[['ngram', 'new_link_probability', 'avg_collaboration_size', 'total_articles']].tail(10).to_string(
            index=False))

        # Statistics on collaboration size
        print(f"\nCollaboration size statistics:")
        print(f"Mean collaboration size across n-grams: {df['avg_collaboration_size'].mean():.2f}")
        print(f"Median collaboration size across n-grams: {df['avg_collaboration_size'].median():.2f}")
        print(f"Std collaboration size across n-grams: {df['avg_collaboration_size'].std():.2f}")

        # Correlation analysis
        valid_data = df[(df['total_articles'] >= 10) & (df['avg_collaboration_size'] > 0)]
        if len(valid_data) > 1:
            correlation = valid_data['new_link_probability'].corr(valid_data['avg_collaboration_size'])
            print(f"\nCorrelation between new link probability and collaboration size: {correlation:.3f}")
            print(f"(Based on {len(valid_data)} n-grams with ≥10 articles)")

    def create_summary_dataframe(self, df):
        """Create the requested simple DataFrame with two columns."""
        summary_df = df[['ngram', 'new_link_probability', 'avg_collaboration_size']].copy()
        summary_df.columns = ['ngram', 'new_link_probability', 'average_collaboration_size']
        return summary_df


def main():
    parser = argparse.ArgumentParser(description='Analyze all n-grams for new link probability and collaboration size')
    parser.add_argument('base_filename', nargs='?', default='coauthorship_analysis',
                        help='Base filename of the analysis output files')
    parser.add_argument('--db', default='articles.db',
                        help='Path to SQLite database (default: articles.db)')
    parser.add_argument('--ngrams', default='output/ngram_articles.npz',
                        help='Path to n-gram articles file (default: output/ngram_articles.npz)')
    args = parser.parse_args()

    # Check files exist
    required_files = [f"{args.base_filename}_new_links.csv", args.db, args.ngrams]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"Missing files: {missing}")
        return

    analyzer = NgramAnalyzer(args.base_filename, args.db, args.ngrams)

    # Analyze all n-grams
    results_df = analyzer.analyze_all_ngrams()

    # Save detailed results
    analyzer.save_results(results_df)

    # Create and save simple summary DataFrame
    summary_df = analyzer.create_summary_dataframe(results_df)
    summary_file = f"{args.base_filename}_ngram_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSimple summary saved to: {summary_file}")
    print("\nSample of summary DataFrame:")
    print(summary_df.head().to_string(index=False))


if __name__ == "__main__":
    main()