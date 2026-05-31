import pandas as pd
import numpy as np
import sqlite3
import pickle
import logging
from sqlalchemy import create_engine, text
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go
from plotly.offline import plot
import os
from typing import List, Tuple, Optional
import gc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BERTopicAnalyzer:
    def __init__(self,
                 embeddings_db_path: str = "article_embeddings.db",
                 source_db_url: str = "sqlite:///articles.db",
                 output_dir: str = "bertopic_results",
                 n_samples: Optional[int] = None,
                 random_seed: int = 42):

        self.embeddings_db_path = embeddings_db_path
        self.source_db_url = source_db_url
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.random_seed = random_seed

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize variables
        self.embeddings = None
        self.documents = None
        self.article_ids = None
        self.topic_model = None
        self.topics = None
        self.probabilities = None

        np.random.seed(random_seed)

    def load_embeddings_and_documents(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load embeddings and corresponding documents from databases"""
        logger.info("Loading embeddings and documents...")

        # Get available embeddings
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        if self.n_samples:
            cursor.execute("SELECT article_id FROM embeddings ORDER BY RANDOM() LIMIT ?", (self.n_samples,))
        else:
            cursor.execute("SELECT article_id FROM embeddings")

        available_article_ids = [str(row[0]) for row in cursor.fetchall()]  # Ensure strings
        conn.close()

        logger.info(f"Found {len(available_article_ids)} available embeddings")

        if self.n_samples and len(available_article_ids) > self.n_samples:
            # Random sample without replacement
            np.random.shuffle(available_article_ids)
            selected_article_ids = available_article_ids[:self.n_samples]
        else:
            selected_article_ids = available_article_ids

        logger.info(f"Processing {len(selected_article_ids)} articles")

        # Load embeddings in batches to manage memory
        batch_size = 10000
        embeddings_list = []
        documents_list = []
        valid_article_ids = []

        for i in tqdm(range(0, len(selected_article_ids), batch_size), desc="Loading batches"):
            batch_ids = selected_article_ids[i:i + batch_size]

            # Load embeddings for this batch
            batch_embeddings = self._load_embeddings_batch(batch_ids)

            # Load documents for this batch
            batch_documents = self._load_documents_batch(batch_ids)

            # Filter out any missing documents or embeddings
            valid_batch_ids = []
            valid_batch_embeddings = []
            valid_batch_documents = []

            for article_id in batch_ids:
                if article_id in batch_embeddings and article_id in batch_documents:
                    valid_batch_ids.append(article_id)
                    valid_batch_embeddings.append(batch_embeddings[article_id])
                    valid_batch_documents.append(batch_documents[article_id])

            if valid_batch_embeddings:
                embeddings_list.extend(valid_batch_embeddings)
                documents_list.extend(valid_batch_documents)
                valid_article_ids.extend(valid_batch_ids)

            # Clean up batch data
            del batch_embeddings, batch_documents
            gc.collect()

        if not embeddings_list:
            raise ValueError("No valid embeddings and documents found")

        embeddings = np.array(embeddings_list)
        logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

        return embeddings, documents_list, valid_article_ids

    def _load_embeddings_batch(self, article_ids: List[str]) -> dict:
        """Load a batch of embeddings from the database"""
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        # Create placeholders for IN query - ensure article_ids are strings
        article_ids_str = [str(aid) for aid in article_ids]
        placeholders = ','.join([':id' + str(i) for i in range(len(article_ids_str))])
        params = {f'id{i}': aid for i, aid in enumerate(article_ids_str)}

        cursor.execute(f"SELECT article_id, embedding FROM embeddings WHERE article_id IN ({placeholders})", params)
        results = cursor.fetchall()
        conn.close()

        embeddings_dict = {}
        for article_id, embedding_blob in results:
            embeddings_dict[str(article_id)] = pickle.loads(embedding_blob)

        return embeddings_dict

    def _load_documents_batch(self, article_ids: List[str]) -> dict:
        """Load a batch of documents from the source database"""
        engine = create_engine(self.source_db_url)

        # Convert article_ids to strings and create query
        article_ids_str = "', '".join(str(aid) for aid in article_ids)

        query = f"""
        SELECT a.article_id, a.title, ab.abstract
        FROM articles a
        JOIN abstracts ab ON a.article_id = ab.article_id
        WHERE a.article_id IN ('{article_ids_str}')
        """

        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)

        documents_dict = {}
        for _, row in df.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else ""
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
            combined_text = f"{title} . {abstract}".strip()
            documents_dict[str(row['article_id'])] = combined_text

        return documents_dict

    def create_bertopic_model(self, min_cluster_size: int = 50, n_neighbors: int = 15,
                              min_dist: float = 0.0, metric: str = 'cosine',
                              umap_components: int = 10) -> BERTopic:
        """Create and configure BERTopic model with UMAP for large-scale analysis"""

        logger.info("Configuring BERTopic model...")
        logger.info(f"Dimensionality reduction: UMAP({umap_components})")

        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=umap_components,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_seed,
            low_memory=True,  # Use less memory
            verbose=True
        )

        # Use default HDBSCAN (let BERTopic handle it)
        hdbscan_model = None

        # Vectorizer for topic representation
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            max_features=10000  # Limit vocabulary size
        )

        # Memory-efficient topic representation
        representation_model = KeyBERTInspired()

        # Create BERTopic model with UMAP only
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,  # Use default HDBSCAN
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            nr_topics="auto",  # Let the model decide
            calculate_probabilities=True,
            verbose=True
        )

        return topic_model

    def fit_bertopic_model(self, min_cluster_size: int = 50, representation_subset_size: int = 10000,
                           umap_components: int = 10):
        """Fit BERTopic model with UMAP and subset representation training"""
        logger.info("Fitting BERTopic model...")

        # Load data
        self.embeddings, self.documents, self.article_ids = self.load_embeddings_and_documents()

        if len(self.documents) > representation_subset_size:
            logger.info(f"Using memory-efficient approach:")
            logger.info(f"  - UMAP + HDBSCAN on full dataset ({len(self.documents)} documents)")
            logger.info(f"  - Representation training on subset ({representation_subset_size} documents)")

            # Step 1: Create model without representation model for clustering
            clustering_model = self.create_bertopic_model(
                min_cluster_size=min_cluster_size,
                umap_components=umap_components
            )
            # Remove representation model to prevent it from training on full dataset
            clustering_model.representation_model = None

            # Step 2: Fit clustering on FULL dataset (UMAP + HDBSCAN)
            logger.info("Performing dimensionality reduction and clustering on full dataset...")
            self.topics, self.probabilities = clustering_model.fit_transform(
                self.documents, self.embeddings
            )

            # Step 3: Update topic representations using subset of documents
            logger.info("Updating topic representations using subset...")
            subset_indices = np.random.choice(
                len(self.documents),
                size=min(representation_subset_size, len(self.documents)),
                replace=False
            )

            # Get subset data
            subset_docs = [self.documents[i] for i in subset_indices]

            # Update topic representations using the public API
            clustering_model.update_topics(subset_docs, n_gram_range=(1, 2))

            # Assign the final model
            self.topic_model = clustering_model

        else:
            # If dataset is small enough, use standard approach
            logger.info("Dataset small enough - using standard training on all documents")
            self.topic_model = self.create_bertopic_model(
                min_cluster_size=min_cluster_size,
                umap_components=umap_components
            )
            self.topics, self.probabilities = self.topic_model.fit_transform(
                self.documents, self.embeddings
            )

        logger.info(f"BERTopic training completed. Found {len(set(self.topics))} topics.")

        # Reduce outliers to improve topic coherence (using full dataset)
        logger.info("Reducing outliers...")
        self.topics = self.topic_model.reduce_outliers(
            self.documents,
            self.topics,
            strategy="embeddings",
            embeddings=self.embeddings,
            threshold=0.1
        )

        logger.info(f"After outlier reduction: {len(set(self.topics))} topics.")

    def save_topic_classifications(self):
        """Save article topic classifications to CSV"""
        logger.info("Saving topic classifications...")

        # Create DataFrame with article classifications
        classifications_df = pd.DataFrame({
            'article_id': self.article_ids,
            'topic': self.topics,
            'probability': [prob.max() for prob in self.probabilities] if self.probabilities is not None else None
        })

        # Add topic labels
        topic_labels = {topic: self.topic_model.get_topic_info(topic)['Name'].iloc[0]
                        for topic in set(self.topics) if topic != -1}
        topic_labels[-1] = "Outlier"  # Outlier topic

        classifications_df['topic_label'] = classifications_df['topic'].map(topic_labels)

        # Save to CSV
        classifications_path = os.path.join(self.output_dir, "article_topic_classifications.csv")
        classifications_df.to_csv(classifications_path, index=False)
        logger.info(f"Saved topic classifications to {classifications_path}")

    def save_topic_list(self):
        """Save topic information to CSV"""
        logger.info("Saving topic list...")

        # Get topic information
        topic_info = self.topic_model.get_topic_info()

        # Save to CSV
        topic_list_path = os.path.join(self.output_dir, "topic_list.csv")
        topic_info.to_csv(topic_list_path, index=False)
        logger.info(f"Saved topic list to {topic_list_path}")

        # Also save detailed topic words
        detailed_topics = []
        for topic in topic_info['Topic']:
            if topic != -1:  # Skip outlier topic
                topic_words = self.topic_model.get_topic(topic)
                topic_words_str = '; '.join([f"{word}:{score:.3f}" for word, score in topic_words[:20]])
                detailed_topics.append({
                    'Topic': topic,
                    'Count': topic_info[topic_info['Topic'] == topic]['Count'].iloc[0],
                    'Name': topic_info[topic_info['Topic'] == topic]['Name'].iloc[0],
                    'Top_Words': topic_words_str
                })

        detailed_df = pd.DataFrame(detailed_topics)
        detailed_path = os.path.join(self.output_dir, "detailed_topics.csv")
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed topics to {detailed_path}")

    def save_hierarchical_structure(self):
        """Save hierarchical topic structure"""
        logger.info("Creating and saving hierarchical structure...")

        try:
            # Create hierarchical topics
            hierarchical_topics = self.topic_model.hierarchical_topics(self.documents)

            # Save hierarchy data
            hierarchy_path = os.path.join(self.output_dir, "topic_hierarchy.csv")
            hierarchical_topics.to_csv(hierarchy_path, index=False)
            logger.info(f"Saved topic hierarchy to {hierarchy_path}")

            # Create and save hierarchy visualization
            fig = self.topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            hierarchy_viz_path = os.path.join(self.output_dir, "topic_hierarchy.html")
            fig.write_html(hierarchy_viz_path)
            logger.info(f"Saved hierarchy visualization to {hierarchy_viz_path}")

        except Exception as e:
            logger.error(f"Error creating hierarchical structure: {e}")
            logger.info("Continuing without hierarchical analysis...")

    def save_additional_visualizations(self):
        """Save additional BERTopic visualizations"""
        logger.info("Creating additional visualizations...")

        try:
            # Topic word scores
            fig_words = self.topic_model.visualize_barchart(top_k_topics=20)
            fig_words.write_html(os.path.join(self.output_dir, "topic_words.html"))

            # Intertopic distance map
            fig_topics = self.topic_model.visualize_topics()
            fig_topics.write_html(os.path.join(self.output_dir, "intertopic_distance.html"))

            # Topic distribution
            if self.probabilities is not None and len(self.probabilities) > 0:
                fig_dist = self.topic_model.visualize_distribution(self.probabilities[0])
                fig_dist.write_html(os.path.join(self.output_dir, "topic_distribution_sample.html"))

            logger.info("Saved additional visualizations")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

    def run_complete_analysis(self, min_cluster_size: int = 50, representation_subset_size: int = 10000,
                              umap_components: int = 10):
        """Run the complete BERTopic analysis pipeline with UMAP"""
        logger.info("Starting complete BERTopic analysis...")

        # Fit the model with UMAP and representation subset
        self.fit_bertopic_model(
            min_cluster_size=min_cluster_size,
            representation_subset_size=representation_subset_size,
            umap_components=umap_components
        )

        # Save all results
        self.save_topic_classifications()
        self.save_topic_list()
        self.save_hierarchical_structure()
        self.save_additional_visualizations()

        # Save model for future use
        model_path = os.path.join(self.output_dir, "bertopic_model")
        self.topic_model.save(model_path, serialization="pickle")
        logger.info(f"Saved BERTopic model to {model_path}")

        logger.info("Complete BERTopic analysis finished!")

        # Print summary
        self._print_analysis_summary()

    def _print_analysis_summary(self):
        """Print analysis summary"""
        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        n_outliers = sum(1 for topic in self.topics if topic == -1)

        print("\n" + "=" * 50)
        print("BERTOPIC ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total documents analyzed: {len(self.documents):,}")
        print(f"Number of topics found: {n_topics}")
        print(f"Number of outliers: {n_outliers:,} ({n_outliers / len(self.topics) * 100:.1f}%)")
        print(f"Results saved to: {self.output_dir}")
        print("\nTop 10 topics by size:")

        topic_info = self.topic_model.get_topic_info().head(11)  # +1 for outlier topic
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic in top list
                print(f"  Topic {row['Topic']:2d}: {row['Count']:5,} docs - {row['Name']}")

        print("=" * 50)


def main():
    """Main function to run BERTopic analysis"""

    # Configuration
    EMBEDDINGS_DB = "article_embeddings.db"
    SOURCE_DB = "sqlite:///articles.db"
    OUTPUT_DIR = "bertopic_results"
    N_SAMPLES = 100000  # Set to None to use all available embeddings
    MIN_CLUSTER_SIZE = 50  # Minimum cluster size for HDBSCAN
    REPRESENTATION_SUBSET_SIZE = 10000  # Use smaller subset for representation training
    UMAP_COMPONENTS = 10  # UMAP final dimensions

    # Create analyzer
    analyzer = BERTopicAnalyzer(
        embeddings_db_path=EMBEDDINGS_DB,
        source_db_url=SOURCE_DB,
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        random_seed=42
    )

    # Run complete analysis with UMAP only
    try:
        analyzer.run_complete_analysis(
            min_cluster_size=MIN_CLUSTER_SIZE,
            representation_subset_size=REPRESENTATION_SUBSET_SIZE,
            umap_components=UMAP_COMPONENTS
        )
        print("\n✓ BERTopic analysis completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
