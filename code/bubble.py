import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
from collections import defaultdict, Counter
from functools import partial


class TopicAuthorRetriever:
    def __init__(self, database_url, topics_csv_path):
        """
        Initialize the retriever with database connection and topic classifications.

        Args:
            database_url: SQLAlchemy database URL
            topics_csv_path: Path to the CSV file with article_id -> topic mappings
        """
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

        # Load topic classifications
        print("Loading topic classifications...")
        topics_df = pd.read_csv(topics_csv_path)
        self.topics = topics_df.set_index("article_id")["topic_reduced"].to_dict()
        print(f"Loaded {len(self.topics)} article-topic mappings")

        # Get unique topics
        self.unique_topics = sorted(set(self.topics.values()))
        print(f"Found {len(self.unique_topics)} unique topics: {self.unique_topics}")

    def get_authors_by_topic(self, target_topic, include_stats=True):
        """
        Retrieve all authors who have published articles in the specified topic.

        Args:
            target_topic: The topic to search for
            include_stats: Whether to include publication statistics per author

        Returns:
            pandas.DataFrame: Authors with their information and optionally statistics
        """
        if target_topic not in self.unique_topics:
            print(f"Topic '{target_topic}' not found in available topics.")
            print(f"Available topics: {self.unique_topics}")
            return pd.DataFrame()

        # Get article IDs for the target topic
        target_article_ids = [
            article_id for article_id, topic in self.topics.items()
            if topic == target_topic
        ]

        if not target_article_ids:
            print(f"No articles found for topic '{target_topic}'")
            return pd.DataFrame()

        print(f"Found {len(target_article_ids)} articles for topic '{target_topic}'")

        session = self.Session()
        try:
            # Create a temporary table approach or use IN clause
            # For large datasets, we'll use IN clause with chunking if needed
            if len(target_article_ids) > 1000:
                # Process in chunks for very large datasets
                all_authors = []
                chunk_size = 1000

                for i in range(0, len(target_article_ids), chunk_size):
                    chunk = target_article_ids[i:i + chunk_size]
                    chunk_authors = self._get_authors_chunk(session, chunk, include_stats)
                    all_authors.extend(chunk_authors)

                # Combine results and aggregate if needed
                if include_stats:
                    authors_df = pd.DataFrame(all_authors)
                    # Group by author and sum statistics
                    authors_df = authors_df.groupby(['author_id', 'name', 'orcid', 'gender']).agg({
                        'article_count': 'sum',
                        'first_publication_year': 'min',
                        'last_publication_year': 'max'
                    }).reset_index()
                else:
                    # Remove duplicates
                    authors_df = pd.DataFrame(all_authors).drop_duplicates(subset=['author_id'])
            else:
                # Process all at once for smaller datasets
                all_authors = self._get_authors_chunk(session, target_article_ids, include_stats)
                authors_df = pd.DataFrame(all_authors)

            return authors_df.sort_values('author_id') if not authors_df.empty else authors_df

        finally:
            session.close()

    def _get_authors_chunk(self, session, article_ids, include_stats):
        """Helper method to get authors for a chunk of article IDs."""
        # Convert article_ids to string for SQL IN clause
        article_ids_str = ','.join(map(str, article_ids))

        if include_stats:
            query = text("""
                SELECT 
                    a.author_id,
                    a.name,
                    a.orcid,
                    a.gender,
                    COUNT(DISTINCT aa.article_id) as article_count,
                    MIN(art.publication_year) as first_publication_year,
                    MAX(art.publication_year) as last_publication_year
                FROM authors a
                INNER JOIN articles_authors aa ON a.author_id = aa.author_id
                INNER JOIN articles art ON aa.article_id = art.article_id
                WHERE aa.article_id IN ({})
                GROUP BY a.author_id, a.name, a.orcid, a.gender
                ORDER BY article_count DESC, a.author_id
            """.format(article_ids_str))
        else:
            query = text("""
                SELECT DISTINCT
                    a.author_id,
                    a.name,
                    a.orcid,
                    a.gender
                FROM authors a
                INNER JOIN articles_authors aa ON a.author_id = aa.author_id
                WHERE aa.article_id IN ({})
                ORDER BY a.author_id
            """.format(article_ids_str))

        result = session.execute(query)
        return [dict(row._mapping) for row in result]

    def get_topic_statistics(self):
        """Get statistics about articles per topic."""
        topic_stats = pd.Series(self.topics).value_counts().sort_index()
        return topic_stats.to_dict()

    def get_author_details_by_topic(self, target_topic, min_articles=1):
        """
        Get detailed author information including their articles in the topic.

        Args:
            target_topic: The topic to search for
            min_articles: Minimum number of articles an author must have in this topic

        Returns:
            dict: Dictionary with author details and their articles
        """
        if target_topic not in self.unique_topics:
            print(f"Topic '{target_topic}' not found in available topics.")
            return {}

        # Get article IDs for the target topic
        target_article_ids = [
            article_id for article_id, topic in self.topics.items()
            if topic == target_topic
        ]

        if not target_article_ids:
            return {}

        session = self.Session()
        try:
            article_ids_str = ','.join(map(str, target_article_ids))

            query = text("""
                SELECT 
                    a.author_id,
                    a.name,
                    a.orcid,
                    a.gender,
                    art.article_id,
                    art.title,
                    art.publication_year,
                    aa.position
                FROM authors a
                INNER JOIN articles_authors aa ON a.author_id = aa.author_id
                INNER JOIN articles art ON aa.article_id = art.article_id
                WHERE aa.article_id IN ({})
                ORDER BY a.author_id, art.publication_year DESC
            """.format(article_ids_str))

            result = session.execute(query)
            rows = [dict(row._mapping) for row in result]

            # Group by author
            authors_details = {}
            for row in rows:
                author_id = row['author_id']
                if author_id not in authors_details:
                    authors_details[author_id] = {
                        'author_info': {
                            'author_id': author_id,
                            'name': row['name'],
                            'orcid': row['orcid'],
                            'gender': row['gender']
                        },
                        'articles': []
                    }

                authors_details[author_id]['articles'].append({
                    'article_id': row['article_id'],
                    'title': row['title'],
                    'publication_year': row['publication_year'],
                    'author_position': row['position']
                })

            # Filter by minimum articles
            filtered_authors = {
                author_id: details for author_id, details in authors_details.items()
                if len(details['articles']) >= min_articles
            }

            return filtered_authors

        finally:
            session.close()

    def get_authors_topic_distribution_before_target(self, target_topic):
        """
        Get the topic distribution of every author before their first paper in the given topic.

        Args:
            target_topic: The target topic to analyze

        Returns:
            dict: Dictionary with author information and their topic distribution before entering target topic
        """
        if target_topic not in self.unique_topics:
            print(f"Topic '{target_topic}' not found in available topics.")
            return {}

        print(f"Analyzing topic distribution before first publication in topic '{target_topic}'...")

        # Get all authors who have published in the target topic
        authors_in_target = self.get_authors_by_topic(target_topic, include_stats=False)

        if authors_in_target.empty:
            print(f"No authors found for topic '{target_topic}'")
            return {}

        author_ids = authors_in_target['author_id'].tolist()
        print(f"Found {len(author_ids)} authors who published in topic '{target_topic}'")

        session = self.Session()
        try:
            results = {}

            # Process authors in chunks to avoid memory issues
            chunk_size = 50000
            for i in range(0, len(author_ids), chunk_size):
                chunk_author_ids = author_ids[i:i + chunk_size]
                chunk_results = self._analyze_authors_chunk(session, chunk_author_ids, target_topic)
                results.update(chunk_results)

                print(f"Processed {min(i + chunk_size, len(author_ids))}/{len(author_ids)} authors")

            return results

        finally:
            session.close()

    def _analyze_authors_chunk(self, session, author_ids, target_topic):
        """Analyze a chunk of authors for their topic distribution before target topic."""
        author_ids_str = ','.join(map(str, author_ids))

        # Get all articles for these authors with publication years
        query = text("""
            SELECT 
                a.author_id,
                a.name,
                a.orcid,
                a.gender,
                art.article_id,
                art.publication_year
            FROM authors a
            INNER JOIN articles_authors aa ON a.author_id = aa.author_id
            INNER JOIN articles art ON aa.article_id = art.article_id
            WHERE a.author_id IN ({})
            ORDER BY a.author_id, art.publication_year ASC
        """.format(author_ids_str))

        result = session.execute(query)
        rows = [dict(row._mapping) for row in result]

        # Group articles by author
        authors_articles = defaultdict(list)
        author_info = {}

        for row in rows:
            author_id = row['author_id']
            authors_articles[author_id].append({
                'article_id': row['article_id'],
                'publication_year': row['publication_year']
            })
            if author_id not in author_info:
                author_info[author_id] = {
                    'name': row['name'],
                    'orcid': row['orcid'],
                    'gender': row['gender']
                }

        # Analyze each author's topic distribution before target topic
        results = {}
        for author_id, articles in authors_articles.items():
            # Find first publication in target topic
            target_articles = [
                art for art in articles
                if art['article_id'] in self.topics and self.topics[art['article_id']] == target_topic
            ]

            if not target_articles:
                continue  # Author has no articles in target topic (shouldn't happen)

            first_target_year = min(art['publication_year'] for art in target_articles)

            # Get articles published before first target topic article
            prior_articles = [
                art for art in articles
                if art['publication_year'] < first_target_year and art['article_id'] in self.topics
            ]

            # Count topic distribution for prior articles
            prior_topics = [self.topics[art['article_id']] for art in prior_articles]
            topic_counts = Counter(prior_topics)

            assert target_topic not in prior_topics

            # Calculate topic distribution (percentages)
            total_prior_articles = len(prior_articles)
            topic_distribution = {}
            if total_prior_articles > 0:
                for topic, count in topic_counts.items():
                    topic_distribution[topic] = {
                        'count': count,
                        'percentage': (count / total_prior_articles) * 100
                    }

            results[author_id] = {
                'author_info': author_info[author_id],
                'first_target_topic_year': first_target_year,
                'total_prior_articles': total_prior_articles,
                'prior_topic_distribution': topic_distribution,
                'total_articles': len(articles)
            }

        return results

    def get_aggregated_topic_distribution_before_target(self, target_topic):
        """
        Get aggregated statistics about topic distributions before entering target topic.

        Args:
            target_topic: The target topic to analyze

        Returns:
            dict: Aggregated statistics about prior topic distributions
        """
        author_distributions = self.get_authors_topic_distribution_before_target(target_topic)

        if not author_distributions:
            return {}

        # Aggregate statistics
        all_prior_topics = Counter()
        authors_with_prior_work = 0
        total_authors = len(author_distributions)

        for author_data in author_distributions.values():
            if author_data['total_prior_articles'] > 0:
                authors_with_prior_work += 1
                for topic, data in author_data['prior_topic_distribution'].items():
                    all_prior_topics[topic] += data['count']

        # Calculate overall percentages
        total_prior_articles = sum(all_prior_topics.values())
        aggregated_distribution = {}

        if total_prior_articles > 0:
            for topic, count in all_prior_topics.items():
                aggregated_distribution[topic] = {
                    'total_articles': count,
                    'percentage_of_prior_work': (count / total_prior_articles) * 100,
                    'authors_count': sum(1 for auth in author_distributions.values()
                                         if topic in auth['prior_topic_distribution'])
                }

        return {
            'target_topic': target_topic,
            'total_authors_in_target': total_authors,
            'authors_with_prior_work': authors_with_prior_work,
            'authors_without_prior_work': total_authors - authors_with_prior_work,
            'total_prior_articles': total_prior_articles,
            'prior_topic_distribution': aggregated_distribution
        }


def topic_vector(d: dict, N_topics: int, normalize: bool = False):
    v = np.zeros(N_topics)

    print(d)

    for t in d:
        v[t] = d[t]['count']

    return v / np.sum(v) if normalize else v


# Example usage and utility functions
def main():
    """Example usage of the TopicAuthorRetriever."""
    topic_labels = pd.read_csv("data/topic_list.csv").set_index("Topic")["Name"].to_dict()

    # Configuration
    database_url = "sqlite:///articles.db"  # Adjust as needed
    topics_csv_path = "data/all_article_topic_classifications_nn.csv"

    # Initialize retriever
    retriever = TopicAuthorRetriever(database_url, topics_csv_path)

    # Show available topics
    print("\n=== Topic Statistics ===")
    topic_stats = retriever.get_topic_statistics()

    N_topics = len(topic_stats)

    # Example: Get authors for a specific topic
    target_topic = 266

    print(f"\n=== Authors for topic: '{target_topic}' ===")

    # Get basic author list
    authors_df = retriever.get_authors_by_topic(target_topic, include_stats=True)

    if not authors_df.empty:
        print(f"\nFound {len(authors_df)} unique authors")

    # Get detailed individual author analysis (sample)
    individual_analysis = retriever.get_authors_topic_distribution_before_target(target_topic)

    print("ok")
    # Show a few examples of authors with prior work
    authors_with_prior = {k: v for k, v in individual_analysis.items()
                          if v['total_prior_articles'] > 0}

    for topic in range(len(topic_labels)):
        print(topic_labels[topic])
        most_prolific = max(
            authors_with_prior,
            key=lambda author: (
                authors_with_prior[author]['prior_topic_distribution'].get(topic, {'count': 0}).get('count', 0)
            )
        )
        print(most_prolific)

    N_authors = len(authors_with_prior)
    prior_topic_matrix = np.zeros((N_authors, N_topics))

    i = 0
    print("Constructing matrix")
    for author, data in authors_with_prior.items():
        prior_topic_matrix[i] = topic_vector(data['prior_topic_distribution'], N_topics, normalize=False)
        i += 1


    np.save(f"output/author_topic_{target_topic}.npy", prior_topic_matrix)

    comb = prior_topic_matrix.T @ prior_topic_matrix + 1
    comb /= N_authors

    for i in range(N_topics):
        if comb[i].sum() > 0:
            v = comb[i]
            indices = np.argpartition(v, -5)[-5:]

            print(f"{topic_labels[i]}:")
            for idx in indices:
                print(f"{topic_labels[idx]} ({v[idx]})")


def search_authors_by_topic(database_url, topics_csv_path, search_topic):
    """
    Convenience function to quickly search for authors by topic.

    Args:
        database_url: Database connection string
        topics_csv_path: Path to topics CSV file
        search_topic: Topic to search for

    Returns:
        pandas.DataFrame: Authors in the specified topic
    """
    retriever = TopicAuthorRetriever(database_url, topics_csv_path)
    return retriever.get_authors_by_topic(search_topic, include_stats=True)


if __name__ == "__main__":
    main()
