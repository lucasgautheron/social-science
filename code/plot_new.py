import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sqlite3
from pathlib import Path
from scipy import stats

ngram = "preregistered"

# Load ngram articles
data = np.load("output/ngram_articles.npz")
ngram_articles = set(data[ngram])


class NewLinksAnalyzer:
    def __init__(self, base_filename: str, db_path: str = "articles.db"):
        self.base_filename = base_filename
        self.new_links_df = pd.read_csv(f"{base_filename}_new_links.csv")
        self.ngram_articles = ngram_articles
        self.db_path = db_path

    def get_yearly_article_counts(self):
        """Get total article counts and ngram article counts by year from database."""
        conn = sqlite3.connect(self.db_path)

        # Get all years from new_links data
        years = sorted(self.new_links_df['publication_year'].unique())

        yearly_counts = {}
        for year in years:
            # Total articles for this year
            year = int(year)
            total_query = """
                          SELECT COUNT(*) as total_articles
                          FROM articles
                          WHERE publication_year = :year
                          """
            total_result = pd.read_sql_query(total_query, conn, params={"year": year})
            total_articles = total_result['total_articles'].iloc[0]

            print(total_articles)

            # Ngram articles for this year
            ngram_query = """
                          SELECT article_id
                          FROM articles
                          WHERE publication_year = :year \
                          """
            year_articles = pd.read_sql_query(ngram_query, conn, params={"year": year})
            ngram_articles_year = len(set(year_articles['article_id']).intersection(self.ngram_articles))

            yearly_counts[year] = {
                'total_articles': total_articles,
                'ngram_articles': ngram_articles_year
            }

        conn.close()
        return yearly_counts

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

        assert lower >= 0
        assert upper <= 1

        return prob, lower, upper

    def get_yearly_probabilities(self):
        """Calculate probability of new links for all papers and ngram papers by year."""
        # Get true article counts from database
        yearly_counts = self.get_yearly_article_counts()

        results = {}

        for year in sorted(self.new_links_df['publication_year'].unique()):
            year_data = self.new_links_df[self.new_links_df['publication_year'] == year]

            # Articles with new links
            all_with_new_links = set(year_data[year_data['distance_before_link'] > 0]['article_id'].unique())
            ngram_with_new_links = all_with_new_links.intersection(self.ngram_articles)

            # True denominators from database
            total_articles = yearly_counts[year]['total_articles']
            ngram_articles_count = yearly_counts[year]['ngram_articles']

            # Calculate probabilities and credible intervals
            all_prob, all_lower, all_upper = self.calculate_binomial_ci(
                len(all_with_new_links), total_articles)

            ngram_prob, ngram_lower, ngram_upper = self.calculate_binomial_ci(
                len(ngram_with_new_links), ngram_articles_count)

            results[year] = {
                'all_prob': all_prob,
                'all_lower': all_lower,
                'all_upper': all_upper,
                'all_count': total_articles,
                'all_with_links': len(all_with_new_links),
                'ngram_prob': ngram_prob,
                'ngram_lower': ngram_lower,
                'ngram_upper': ngram_upper,
                'ngram_count': ngram_articles_count,
                'ngram_with_links': len(ngram_with_new_links)
            }

        return results

    def plot_probabilities(self):
        """Create grid of probability plots, one per year."""
        probabilities = self.get_yearly_probabilities()
        years = sorted(probabilities.keys())

        if not years:
            print("No data to plot")
            return

        # Calculate grid size (2 columns)
        n_years = len(years)
        cols = 2
        rows = (n_years + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(8, 1.25 * rows))
        if n_years == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, year in enumerate(years):
            ax = axes[i] if n_years > 1 else axes[0]
            data = probabilities[year]

            # Bar positions
            x_pos = [0, 1]
            probabilities_vals = [data['all_prob'], data['ngram_prob']]
            lower_errs = [max(0, data['all_prob'] - data['all_lower']),
                          max(0, data['ngram_prob'] - data['ngram_lower'])]
            upper_errs = [max(0, data['all_upper'] - data['all_prob']),
                          max(0, data['ngram_upper'] - data['ngram_prob'])]

            # Create bars with error bars
            bars = ax.bar(x_pos, probabilities_vals,
                          color=['skyblue', 'lightgreen'],
                          alpha=0.7, width=0.6)

            print(lower_errs)
            print(upper_errs)
            # # Add error bars for credible intervals
            ax.errorbar(x_pos, probabilities_vals,
                        yerr=[lower_errs, upper_errs],
                        fmt='none', color='black', capsize=5, capthick=2)

            # Add count labels on bars
            for j, (bar, prob, count, with_links) in enumerate(zip(bars, probabilities_vals,
                                                                   [data['all_count'], data['ngram_count']],
                                                                   [data['all_with_links'], data['ngram_with_links']])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{prob:.3f}',
                        ha='center', va='bottom', fontsize=8)

            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['All Publications', f'{ngram}'])
            # ax.set_ylabel('Probability of New Links')
            ax.set_title(f'{year}', fontsize=10)
            ax.set_ylim(0, max(1.0, max(data['all_upper'], data['ngram_upper']) * 1.1))

            # Add grid for easier reading
            ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for i in range(n_years, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.base_filename}_new_links_probability.png", dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self):
        """Print summary statistics."""
        probabilities = self.get_yearly_probabilities()

        print(f"\nSummary: Probability of New Links by Year")
        print("=" * 60)
        print(f"{'Year':<6} {'All Pubs':<15} {'CI':<20} {ngram + ' Pubs':<15} {'CI':<20}")
        print("-" * 60)

        for year in sorted(probabilities.keys()):
            data = probabilities[year]
            all_ci = f"[{data['all_lower']:.3f}, {data['all_upper']:.3f}]"
            ngram_ci = f"[{data['ngram_lower']:.3f}, {data['ngram_upper']:.3f}]"

            print(f"{year:<6} {data['all_prob']:.3f}<11> {all_ci:<20} "
                  f"{data['ngram_prob']:.3f}<11> {ngram_ci:<20}")


def main():
    parser = argparse.ArgumentParser(description='Analyze probability of new links in publications')
    parser.add_argument('base_filename', nargs='?', default='coauthorship_analysis',
                        help='Base filename of the analysis output files')
    parser.add_argument('--db', default='articles.db',
                        help='Path to SQLite database (default: articles.db)')
    args = parser.parse_args()

    # Check files exist
    required_files = [f"{args.base_filename}_new_links.csv", args.db]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"Missing files: {missing}")
        return

    analyzer = NewLinksAnalyzer(args.base_filename, args.db)
    analyzer.plot_probabilities()
    analyzer.print_summary()


if __name__ == "__main__":
    main()
