import argparse
import json
import logging
import pickle
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoauthorshipNetworkAnalyzer:
    def __init__(self, database_url: str, sample_size: int = 10000, batch_size: int = 1000):
        """
        Initialize the coauthorship network analyzer.

        Args:
            database_url: SQLAlchemy database connection string
            sample_size: Number of author pairs to sample for distance distribution
            batch_size: Number of articles to process in each batch
        """
        self.engine = create_engine(database_url)
        self.sample_size = sample_size
        self.batch_size = batch_size

        # Network to track coauthorship relationships
        self.coauthorship_graph = nx.Graph()

        # Track existing edges to identify new links
        self.existing_edges = set()

        # Store results
        self.new_links_data = []
        self.yearly_distributions = {}

        # Performance optimization: cache shortest paths
        self.path_cache = {}

        # Track processed years for distribution sampling
        self.processed_years = set()

    def get_date_range(self) -> Tuple[str, str]:
        """Get the min and max publication dates for pagination."""
        query = """
                SELECT MIN(a.publication_date) as min_date, \
                       MAX(a.publication_date) as max_date
                FROM articles a
                         JOIN articles_authors aa ON a.article_id = aa.article_id
                WHERE a.publication_date IS NOT NULL \
                """

        result = pd.read_sql(query, self.engine)
        return result['min_date'].iloc[0], result['max_date'].iloc[0]

    def load_articles_batch(self, start_date: str, batch_size: int) -> Tuple[pd.DataFrame, str]:
        """
        Load a batch of articles with their authors, sorted by publication date.
        Uses date-based pagination instead of OFFSET for better performance.

        Returns:
            Tuple of (DataFrame, next_start_date)
        """
        query = """
                WITH ordered_articles AS (SELECT DISTINCT a.article_id, \
                                                          a.publication_date, \
                                                          a.publication_year \
                                          FROM articles a \
                                                   JOIN articles_authors aa ON a.article_id = aa.article_id \
                                          WHERE a.publication_date IS NOT NULL \
                                            AND a.publication_date >= :start_date \
                                          ORDER BY a.publication_date, a.article_id
                    LIMIT :batch_size
                    )
                SELECT oa.article_id, \
                       oa.publication_date, \
                       oa.publication_year, \
                       aa.author_id
                FROM ordered_articles oa
                         JOIN articles_authors aa ON oa.article_id = aa.article_id
                ORDER BY oa.publication_date, oa.article_id, aa.author_id \
                """

        df = pd.read_sql(query, self.engine, params={'start_date': start_date, 'batch_size': batch_size})

        # Determine next start date
        next_start_date = None
        if len(df) > 0:
            # Get the last date in this batch
            unique_articles = df.groupby(['article_id', 'publication_date']).first().reset_index()
            if len(unique_articles) == batch_size:
                # If we got a full batch, next start date is the day after the last article
                last_date = unique_articles['publication_date'].iloc[-1]
                # For next batch, start from the same date to catch any articles on the same day
                next_start_date = last_date

        return df, next_start_date

    def calculate_shortest_path_distance(self, author1: int, author2: int) -> int:
        """
        Calculate shortest path distance between two authors.
        Returns -1 if no path exists.
        """
        # Check cache first
        cache_key = tuple(sorted([author1, author2]))
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        try:
            if author1 == author2:
                distance = 0
            elif self.coauthorship_graph.has_edge(author1, author2):
                distance = 1
            elif author1 in self.coauthorship_graph and author2 in self.coauthorship_graph:
                distance = nx.shortest_path_length(self.coauthorship_graph, author1, author2)
            else:
                distance = -1  # No path exists
        except nx.NetworkXNoPath:
            distance = -1

        # Cache the result
        self.path_cache[cache_key] = distance
        return distance

    def sample_author_pairs_for_distribution(self, year: int) -> List[Tuple[int, int, int]]:
        """
        Sample random author pairs to estimate distance distribution for a given year.
        Returns list of (author1, author2, distance) tuples.
        """
        if len(self.coauthorship_graph.nodes()) < 2:
            return []

        nodes = list(self.coauthorship_graph.nodes())
        sample_pairs = []

        # Sample random pairs
        for _ in range(min(self.sample_size, len(nodes) * (len(nodes) - 1) // 2)):
            author1, author2 = random.sample(nodes, 2)
            distance = self.calculate_shortest_path_distance(author1, author2)
            sample_pairs.append((author1, author2, distance))

        return sample_pairs

    def clear_path_cache(self):
        """Clear the path cache to prevent memory issues."""
        self.path_cache.clear()

    def process_batch(self, batch_df: pd.DataFrame) -> int:
        """
        Process a batch of articles and their authors.

        Args:
            batch_df: DataFrame containing article_id, publication_date, publication_year, author_id

        Returns:
            Number of articles processed in this batch
        """
        # Group by article to get co-author lists
        article_groups = batch_df.groupby(['article_id', 'publication_date', 'publication_year'])

        articles_processed_in_batch = 0

        for (article_id, pub_date, pub_year), group in article_groups:
            coauthors = group['author_id'].tolist()

            # Skip single-author papers
            if len(coauthors) < 2:
                continue

            if len(coauthors) > 16:
                continue

            # Sample yearly distribution if we haven't done this year yet
            if pub_year not in self.processed_years and len(self.coauthorship_graph.nodes()) > 1:
                logger.info(f"Sampling distance distribution for year {pub_year}")
                yearly_sample = self.sample_author_pairs_for_distribution(pub_year)

                # Calculate distribution statistics
                distances = [d for _, _, d in yearly_sample if d > 0]  # Exclude disconnected pairs
                if distances:
                    self.yearly_distributions[pub_year] = {
                        'mean_distance': np.mean(distances),
                        'median_distance': np.median(distances),
                        'std_distance': np.std(distances),
                        'distance_counts': dict(zip(*np.unique(distances, return_counts=True))),
                        'total_pairs_sampled': len(yearly_sample),
                        'connected_pairs': len(distances),
                        'network_size': len(self.coauthorship_graph.nodes()),
                        'network_edges': len(self.coauthorship_graph.edges())
                    }
                    self.processed_years.add(pub_year)

            # Process all author pairs in this article
            for i in range(len(coauthors)):
                for j in range(i + 1, len(coauthors)):
                    author1, author2 = coauthors[i], coauthors[j]
                    edge = tuple(sorted([author1, author2]))

                    # Check if this is a new collaboration
                    if edge not in self.existing_edges:
                        # Calculate distance before adding the edge
                        distance = self.calculate_shortest_path_distance(author1, author2)

                        # Record the new link
                        self.new_links_data.append({
                            'author1': author1,
                            'author2': author2,
                            'article_id': article_id,
                            'publication_date': pub_date,
                            'publication_year': pub_year,
                            'distance_before_link': distance,
                            # 'network_size_before': len(self.coauthorship_graph.nodes()),
                            # 'network_edges_before': len(self.coauthorship_graph.edges())
                        })

                        # Mark this edge as processed
                        self.existing_edges.add(edge)

                        # Add authors to graph if not present
                        if author1 not in self.coauthorship_graph:
                            self.coauthorship_graph.add_node(author1)
                        if author2 not in self.coauthorship_graph:
                            self.coauthorship_graph.add_node(author2)

                    # Add or strengthen the edge
                    if self.coauthorship_graph.has_edge(author1, author2):
                        # Increase collaboration count
                        self.coauthorship_graph[author1][author2]['weight'] += 1
                        self.coauthorship_graph[author1][author2]['articles'].append(article_id)
                    else:
                        # Add new edge
                        self.coauthorship_graph.add_edge(
                            author1, author2,
                            weight=1,
                            articles=[article_id],
                            first_collaboration_date=pub_date,
                            first_collaboration_year=pub_year
                        )

            articles_processed_in_batch += 1

        return articles_processed_in_batch

    def process_articles_chronologically(self):
        """
        Main processing function that analyzes the network evolution in batches.
        Uses date-based pagination instead of OFFSET for better performance.
        """
        # Get date range for processing
        min_date, max_date = self.get_date_range()
        logger.info(f"Processing articles from {min_date} to {max_date}")

        processed_articles = 0
        current_start_date = min_date

        logger.info("Starting chronological batch processing...")

        while current_start_date is not None:
            # Load batch of articles with their authors
            logger.info(f"Loading batch starting from date {current_start_date}")
            batch_df, next_start_date = self.load_articles_batch(current_start_date, self.batch_size)

            if len(batch_df) == 0:
                logger.info("No more articles to process")
                break

            # Process the batch
            articles_in_batch = len(batch_df.groupby('article_id'))
            logger.info(f"Processing batch with {articles_in_batch} articles, "
                        f"{len(batch_df)} total author-article records")

            batch_processed = self.process_batch(batch_df)
            processed_articles += batch_processed

            print(len(self.existing_edges), len(self.coauthorship_graph.edges()))
            logger.info(f"Processed {batch_processed} articles in batch, "
                        f"Total processed: {processed_articles}, "
                        f"Network size: {len(self.coauthorship_graph.nodes())} nodes, "
                        f"{len(self.coauthorship_graph.edges())} edges, "
                        f"New links found: {len(self.new_links_data)}")

            # Clear path cache periodically to manage memory
            if processed_articles % (self.batch_size * 5) == 0:
                logger.info("Clearing path cache to manage memory")
                self.clear_path_cache()

            # Move to next batch
            if next_start_date == current_start_date:
                # If we're on the same date, we need to handle articles from the same day
                # Find the next date after current batch
                last_date_in_batch = batch_df['publication_date'].max()
                query = """
                        SELECT MIN(publication_date) as next_date
                        FROM articles
                        WHERE publication_date > :last_date \
                        """
                result = pd.read_sql(query, self.engine, params={'last_date': last_date_in_batch})
                current_start_date = result['next_date'].iloc[0] if not pd.isna(result['next_date'].iloc[0]) else None
            else:
                current_start_date = next_start_date

            # Break if we've processed fewer articles than batch size and no next date
            if articles_in_batch < self.batch_size and current_start_date is None:
                break

        # Sample distributions for any remaining years not yet processed
        remaining_years = set()
        for link_data in self.new_links_data:
            year = link_data['publication_year']
            if year not in self.processed_years:
                remaining_years.add(year)

        for year in remaining_years:
            if len(self.coauthorship_graph.nodes()) > 1:
                logger.info(f"Sampling distance distribution for remaining year {year}")
                yearly_sample = self.sample_author_pairs_for_distribution(year)
                distances = [d for _, _, d in yearly_sample if d > 0]
                if distances:
                    self.yearly_distributions[year] = {
                        'mean_distance': np.mean(distances),
                        'median_distance': np.median(distances),
                        'std_distance': np.std(distances),
                        'distance_counts': dict(zip(*np.unique(distances, return_counts=True))),
                        'total_pairs_sampled': len(yearly_sample),
                        'connected_pairs': len(distances),
                        'network_size': len(self.coauthorship_graph.nodes()),
                        'network_edges': len(self.coauthorship_graph.edges())
                    }

        logger.info(f"Completed processing. Total articles processed: {processed_articles}, "
                    f"Total new links: {len(self.new_links_data)}, "
                    f"Years with distributions: {len(self.yearly_distributions)}")

    def get_new_links_dataframe(self) -> pd.DataFrame:
        """Convert new links data to pandas DataFrame."""
        return pd.DataFrame(self.new_links_data)

    def get_yearly_distributions_dataframe(self) -> pd.DataFrame:
        """Convert yearly distributions to pandas DataFrame."""
        rows = []
        for year, stats in self.yearly_distributions.items():
            row = {'year': year}
            row.update(stats)
            rows.append(row)
        return pd.DataFrame(rows)

    def save_results(self, base_filename: str):
        """Save results to files."""
        Path(base_filename).expanduser().parent.mkdir(parents=True, exist_ok=True)
        # Save new links data
        new_links_df = self.get_new_links_dataframe()
        new_links_df.to_csv(f"{base_filename}_new_links.csv", index=False)
        logger.info(f"Saved new links data to {base_filename}_new_links.csv")

        # Save yearly distributions
        yearly_dist_df = self.get_yearly_distributions_dataframe()
        yearly_dist_df.to_csv(f"{base_filename}_yearly_distributions.csv", index=False)
        logger.info(f"Saved yearly distributions to {base_filename}_yearly_distributions.csv")

        # Save network
        with open(f"{base_filename}_network.gpickle", "wb") as handle:
            pickle.dump(self.coauthorship_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved network to {base_filename}_network.gpickle")

        # Save detailed yearly distributions as JSON
        with open(f"{base_filename}_yearly_distributions_detailed.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_compatible = {}
            for year, data in self.yearly_distributions.items():
                json_compatible[str(year)] = {}
                for key, value in data.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_compatible[str(year)][key] = value.item()
                    elif isinstance(value, dict):
                        json_compatible[str(year)][key] = {
                            str(k): int(v) if isinstance(v, (np.integer, np.floating)) else v
                            for k, v in value.items()}
                    else:
                        json_compatible[str(year)][key] = value

            json.dump(json_compatible, f, indent=2)
        logger.info(f"Saved detailed yearly distributions to {base_filename}_yearly_distributions_detailed.json")

    def get_network_summary(self) -> Dict:
        """Get summary statistics of the final network."""
        if len(self.coauthorship_graph.nodes()) == 0:
            return {"error": "Empty network"}

        # Basic network statistics
        summary = {
            'total_nodes': len(self.coauthorship_graph.nodes()),
            'total_edges': len(self.coauthorship_graph.edges()),
            'density': nx.density(self.coauthorship_graph),
            'is_connected': nx.is_connected(self.coauthorship_graph),
        }

        if summary['is_connected']:
            summary['diameter'] = nx.diameter(self.coauthorship_graph)
            summary['average_shortest_path_length'] = nx.average_shortest_path_length(self.coauthorship_graph)
        else:
            # For disconnected graphs, analyze the largest component
            largest_cc = max(nx.connected_components(self.coauthorship_graph), key=len)
            largest_cc_subgraph = self.coauthorship_graph.subgraph(largest_cc)
            summary['largest_component_size'] = len(largest_cc)
            summary['largest_component_diameter'] = nx.diameter(largest_cc_subgraph)
            summary['largest_component_avg_path_length'] = nx.average_shortest_path_length(largest_cc_subgraph)
            summary['number_of_components'] = nx.number_connected_components(self.coauthorship_graph)

        # Degree statistics
        degrees = [d for n, d in self.coauthorship_graph.degree()]
        summary['avg_degree'] = np.mean(degrees)
        summary['median_degree'] = np.median(degrees)
        summary['max_degree'] = max(degrees)
        summary['min_degree'] = min(degrees)

        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze coauthorship network evolution.")
    parser.add_argument("--db-path", default="articles.db", help="Read-only source corpus.")
    parser.add_argument("--output-prefix", default="output/coauthorship")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyzer = CoauthorshipNetworkAnalyzer(
        f"sqlite:///{args.db_path}",
        args.sample_size,
        args.batch_size,
    )
    analyzer.process_articles_chronologically()
    analyzer.save_results(args.output_prefix)
    network_summary = analyzer.get_network_summary()
    logger.info("Network Summary:")
    for key, value in network_summary.items():
        logger.info(f"  {key}: {value}")

    # Print sample of results
    new_links_df = analyzer.get_new_links_dataframe()
    if len(new_links_df) > 0:
        logger.info("\nSample of new links data:")
        logger.info(new_links_df.head(10).to_string())

    yearly_dist_df = analyzer.get_yearly_distributions_dataframe()
    if len(yearly_dist_df) > 0:
        logger.info("\nSample of yearly distributions:")
        logger.info(yearly_dist_df.head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
