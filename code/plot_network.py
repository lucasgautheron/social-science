import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ast
from pathlib import Path
from scipy import stats


class DistributionAnalyzer:
    def __init__(self, base_filename: str, articles, ngram):
        self.base_filename = base_filename
        self.new_links_df = pd.read_csv(f"{base_filename}_new_links.csv")
        self.yearly_dist_df = pd.read_csv(f"{base_filename}_yearly_distributions.csv")
        self.ngram_articles = set(articles)  # Convert to set for faster lookup
        self.ngram = ngram

    def calculate_binomial_ci(self, counts_dict, total_count, alpha=0.05):
        """Calculate 95% credible intervals for CDF using Beta-Binomial conjugate prior."""
        if total_count == 0:
            return [], [], []

        distances = sorted(counts_dict.keys())
        lower_bounds = []
        upper_bounds = []
        cdf_values = []

        cumulative_count = 0
        for d in distances:
            cumulative_count += counts_dict[d]

            # For CDF, we want the credible interval for P(X <= d)
            # This follows Beta(cumulative_count + 1, total_count - cumulative_count + 1)
            # Using non-informative prior Beta(1, 1)
            a = cumulative_count + 1
            b = total_count - cumulative_count + 1

            # Calculate credible interval
            lower = stats.beta.ppf(alpha / 2, a, b)
            upper = stats.beta.ppf(1 - alpha / 2, a, b)
            cdf_val = cumulative_count / total_count

            lower_bounds.append(lower)
            upper_bounds.append(upper)
            cdf_values.append(cdf_val)

        return cdf_values, lower_bounds, upper_bounds

    def get_distributions_by_year(self):
        """Get normalized distributions and CDFs with credible intervals for new links, random pairs, and ngram articles by year."""
        results = {}

        # New links distributions
        for year in self.new_links_df['publication_year'].unique():
            year_data = self.new_links_df[
                (self.new_links_df['publication_year'] == year) &
                (self.new_links_df['distance_before_link'] > 0)
                ]

            # ngram articles subset
            ngram_year_data = year_data[year_data['article_id'].isin(self.ngram_articles)]

            if len(year_data) > 0:
                # New links distribution (all)
                new_counts = year_data['distance_before_link'].value_counts().to_dict()
                new_total = sum(new_counts.values())

                # Calculate CDF with credible intervals
                new_distances = sorted(new_counts.keys())
                new_cdf, new_lower, new_upper = self.calculate_binomial_ci(new_counts, new_total)

                # ngram links distribution
                ngram_distances = []
                ngram_cdf = []
                ngram_lower = []
                ngram_upper = []
                if len(ngram_year_data) > 0:
                    ngram_counts = ngram_year_data['distance_before_link'].value_counts().to_dict()
                    ngram_total = sum(ngram_counts.values())

                    ngram_distances = sorted(ngram_counts.keys())
                    ngram_cdf, ngram_lower, ngram_upper = self.calculate_binomial_ci(ngram_counts, ngram_total)

                # Random pairs distribution from CSV
                yearly_info = self.yearly_dist_df[self.yearly_dist_df['year'] == year]
                random_distances = []
                random_cdf = []
                random_lower = []
                random_upper = []
                n_random_pairs = 0

                if len(yearly_info) > 0:
                    distance_counts_str = yearly_info['distance_counts'].iloc[0]
                    n_random_pairs = yearly_info['connected_pairs'].iloc[0]

                    # Parse the distance_counts string
                    try:
                        distance_counts = ast.literal_eval(distance_counts_str)
                        random_total = sum(distance_counts.values())
                        if random_total > 0:
                            random_distances = sorted([int(d) for d in distance_counts.keys()])
                            # Convert keys to int for proper sorting
                            int_distance_counts = {int(d): c for d, c in distance_counts.items()}
                            random_cdf, random_lower, random_upper = self.calculate_binomial_ci(int_distance_counts,
                                                                                                random_total)
                    except:
                        print(f"Could not parse distance_counts for year {year}")

                results[year] = {
                    'new_distances': new_distances,
                    'new_cdf': new_cdf,
                    'new_lower': new_lower,
                    'new_upper': new_upper,
                    'ngram_distances': ngram_distances,
                    'ngram_cdf': ngram_cdf,
                    'ngram_lower': ngram_lower,
                    'ngram_upper': ngram_upper,
                    'random_distances': random_distances,
                    'random_cdf': random_cdf,
                    'random_lower': random_lower,
                    'random_upper': random_upper,
                    'n_new_links': len(year_data),
                    'n_ngram_links': len(ngram_year_data),
                    'n_random_pairs': int(n_random_pairs)
                }

        return results

    def plot_distributions(self):
        """Create grid of CDF plots with credible intervals, one per year."""
        distributions = self.get_distributions_by_year()
        years = sorted(distributions.keys())

        if not years:
            print("No data to plot")
            return

        # Calculate grid size (2 columns)
        n_years = len(years)
        cols = 2
        rows = (n_years + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6, 1.5 * rows))
        if n_years == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, year in enumerate(years):
            ax = axes[i] if n_years > 1 else axes[0]
            data = distributions[year]

            # Plot CDFs as lines with shaded credible intervals
            if data['new_distances'] and data['new_cdf']:
                ax.plot(data['new_distances'], data['new_cdf'], 'o-', color='blue',
                        label='New links', alpha=0.8, markersize=4)
                ax.fill_between(data['new_distances'], data['new_lower'], data['new_upper'],
                                color='blue', alpha=0.2)

            if data['ngram_distances'] and data['ngram_cdf']:
                ax.plot(data['ngram_distances'], data['ngram_cdf'], '^-', color='green',
                        label=self.ngram, alpha=0.8, markersize=4)
                ax.fill_between(data['ngram_distances'], data['ngram_lower'], data['ngram_upper'],
                                color='green', alpha=0.2)

            if data['random_distances'] and data['random_cdf']:
                ax.plot(data['random_distances'], data['random_cdf'], 's-', color='red',
                        label='Random baseline', alpha=0.8, markersize=4)
                ax.fill_between(data['random_distances'], data['random_lower'], data['random_upper'],
                                color='red', alpha=0.2)

            # Set log scale for x-axis only (CDF goes from 0 to 1)
            ax.set_xscale('log')

            ax.set_title(
                f'{year}',
                fontsize=9)
            # ax.set_xlabel('Distance')
            # ax.set_ylabel('Cumulative Probability')
            ax.set_ylim(0.8, 1.05)

            if i == len(years) - 1:  # Only show legend on last plot
                ax.legend()

        # Hide unused subplots
        for i in range(n_years, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.base_filename}_cdf_by_year.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze co-authorship distance distributions')
    parser.add_argument("--ngram")
    parser.add_argument('base_filename', nargs='?', default='coauthorship_analysis')
    args = parser.parse_args()
    ngram = args.ngram

    data = np.load("output/ngram_articles.npz")
    articles = data[ngram]

    print(articles)

    # Check files exist
    required_files = [f"{args.base_filename}_new_links.csv",
                      f"{args.base_filename}_yearly_distributions.csv"]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"Missing files: {missing}")
        return

    analyzer = DistributionAnalyzer(args.base_filename, articles, ngram)
    analyzer.plot_distributions()

    print("ok")


if __name__ == "__main__":
    main()
