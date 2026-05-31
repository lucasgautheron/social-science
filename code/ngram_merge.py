import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import networkx as nx
import argparse
from sqlalchemy import create_engine
from scipy.stats import beta

# Add argument parser for the ratio parameter
parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=float, default=2.0, help="Minimum peak-to-pre-peak ratio threshold")
args = parser.parse_args()

# Database connection - update this to match your setup
database_url = "sqlite:///articles.db"  # or your PostgreSQL URL
engine = create_engine(database_url)

topics = pd.read_csv("data/all_article_topic_classifications_nn.csv")
topics = topics.set_index("article_id")["topic_reduced"].to_dict()

topic_labels = pd.read_csv("data/topic_list.csv").set_index("Topic")["Name"].to_dict()


def get_papers_per_year(database_url="sqlite:///articles.db"):
    """
    Returns the total number of papers per year in the database.
    """
    engine = create_engine(database_url)

    query = """
            SELECT publication_year,
                   COUNT(DISTINCT article_id) as total
            FROM articles
            WHERE publication_year IS NOT NULL
            GROUP BY publication_year
            ORDER BY publication_year
            """

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    return df


def get_all_article_dates():
    """
    Retrieve all article publication years in a single query.
    Returns a dictionary mapping article_id to publication_year.
    """
    query = """
            SELECT article_id, publication_year
            FROM articles
            WHERE publication_year IS NOT NULL
            """

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    return dict(zip(df['article_id'], df['publication_year']))


def calculate_peak_ratio(ngram_articles, article_dates, total_papers_per_year):
    """
    Calculate the peak-to-pre-peak ratio for a given n-gram.
    """
    try:
        # Get publication years for this ngram's articles
        years = [article_dates.get(article_id) for article_id in ngram_articles
                 if article_id in article_dates]

        if len(years) == 0:
            return 0

        # Count articles per year for this ngram
        year_counts = pd.Series(years).value_counts().to_dict()

        # Create per_year dataframe
        per_year = total_papers_per_year.copy()
        per_year['n'] = per_year.index.map(year_counts).fillna(0)
        per_year["fraction"] = (1 + per_year["n"]) / (2 + per_year["total"])

        # Calculate peak ratio
        fraction = per_year["fraction"].values
        peak = fraction.argmax()

        if peak <= 1:  # Need at least 2 pre-peak points
            return 0

        pre_peak = fraction[:peak]
        if len(pre_peak) == 0:
            return 0

        peak_ratio = fraction[peak] / pre_peak.min()
        return peak_ratio

    except Exception as e:
        print(f"Error calculating peak ratio: {e}")
        return 0


# Load your data
data = np.load("output/ngram_articles.npz")

# Retrieve all article dates and total papers per year once
print("Loading article dates and yearly totals...")
article_dates = get_all_article_dates()
total_papers_per_year = get_papers_per_year().set_index("publication_year")

# Filter n-grams based on peak ratio
print("Filtering n-grams based on peak ratio...")
filtered_data = {}
total_ngrams = 0
filtered_count = 0
ngrams_topics = {}

for ngram in data.keys():
    total_ngrams += 1
    if len(data[ngram]) < 1000:  # Skip n-grams with too few articles
        continue

    peak_ratio = calculate_peak_ratio(data[ngram], article_dates, total_papers_per_year)

    if peak_ratio >= args.ratio:
        filtered_data[ngram] = data[ngram]
        filtered_count += 1
        print(f"Included: {ngram} (peak ratio: {peak_ratio:.2f})")
    else:
        print(f"Filtered: {ngram} (peak ratio: {peak_ratio:.2f} < {args.ratio})")

    ngrams_topics[ngram] = [topics[article_id] for article_id in data[ngram] if article_id in topics]
    print(len(ngrams_topics[ngram]))

print(f"\nFiltering summary:")
print(f"Total n-grams: {total_ngrams}")
print(f"N-grams meeting ratio threshold: {filtered_count}")

# Use filtered data for the rest of the analysis
data = filtered_data
ngrams = list(data.keys())


def create_sparse_bow_matrix(data):
    """
    Create a sparse bag-of-words matrix from n-gram to article mappings.

    Returns:
    - bow_matrix: scipy sparse matrix (articles x n-grams)
    - article_ids: list of unique article IDs
    - ngram_list: list of n-grams corresponding to matrix columns
    """

    ngram_list = list(data.keys())

    # Get all unique article IDs
    all_article_ids = set()
    for ngram in ngram_list:
        all_article_ids.update(data[ngram])

    article_ids = sorted(list(all_article_ids))

    # Create mappings for efficient indexing
    article_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(ngram_list)}

    # Prepare data for sparse matrix construction
    row_indices = []
    col_indices = []

    print(ngram_list)

    # For each n-gram, add entries for all articles containing it
    for ngram in ngram_list:
        ngram_idx = ngram_to_idx[ngram]
        for article_id in data[ngram]:
            article_idx = article_to_idx[article_id]
            row_indices.append(article_idx)
            col_indices.append(ngram_idx)

    # Create sparse matrix (binary: 1 if ngram appears in article, 0 otherwise)
    bow_matrix = csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(len(article_ids), len(ngram_list)),
    )

    return bow_matrix, article_ids, ngram_list


def create_topic_bow_matrix(article_ids, topics):
    """
    Create a sparse bag-of-words matrix for topics using the same approach as your ngram matrix.

    Returns:
    - topic_bow_matrix: scipy sparse matrix (articles x topics)
    - unique_topics: list of unique topics
    """
    # Get unique topics from articles that are in our dataset
    filtered_topics = {aid: topics[aid] for aid in article_ids if aid in topics}
    unique_topics = sorted(list(set(filtered_topics.values())))

    # Create mappings
    article_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}
    topic_to_idx = {topic: idx for idx, topic in enumerate(unique_topics)}

    # Prepare data for sparse matrix construction
    row_indices = []
    col_indices = []

    # For each article, add entry for its topic
    for article_id, topic in filtered_topics.items():
        if article_id in article_to_idx:  # Make sure article is in our dataset
            article_idx = article_to_idx[article_id]
            topic_idx = topic_to_idx[topic]
            row_indices.append(article_idx)
            col_indices.append(topic_idx)

    # Create sparse matrix (binary: 1 if article has this topic, 0 otherwise)
    topic_bow_matrix = csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(len(article_ids), len(unique_topics)),
    )

    return topic_bow_matrix, unique_topics


# Create the sparse matrix
bow_matrix, article_ids, ngram_list = create_sparse_bow_matrix(data)
ngram_to_idx = {ngram: i for i, ngram in enumerate(ngram_list)}

# Add this after your existing ngram NPMI calculation
print("Creating topic bow matrix...")
topic_bow_matrix, unique_topics = create_topic_bow_matrix(article_ids, topics)

print(f"Topic matrix shape: {topic_bow_matrix.shape}")
print(f"Number of unique topics: {len(unique_topics)}")

# Calculate topic-ngram NPMI using your efficient method
print("Calculating topic-ngram NPMI matrix...")

# Use the same K value as your existing code
# K = bow_matrix.shape[0]  # or use your 9.5e6 if that's your preference
K = len(topics)

# K = bow_matrix.shape[0]
K = len(topics)
N = bow_matrix.shape[1]
N_topics = topic_bow_matrix.shape[1]
N_ngrams = bow_matrix.shape[1]

coocc = (bow_matrix.T @ bow_matrix).todense()
marginal = np.array(bow_matrix.sum(axis=0))[0, :]

cond_prob = coocc / marginal[:, np.newaxis]

coocc = np.log(coocc / K)
baseline = np.log(marginal * marginal / K / K)
npmi = baseline / coocc - 1
npmi = np.array(npmi)

print(npmi)

np.fill_diagonal(npmi, -1)
np.fill_diagonal(cond_prob, 0)

maximum_npmi_idx = npmi.argmax(axis=0)
maximum_cond_prob_idx = cond_prob.argmax(axis=0)

# Calculate co-occurrence matrix: topics x ngrams
# This is equivalent to: topic_bow_matrix.T @ bow_matrix
topic_ngram_coocc = (topic_bow_matrix.T @ bow_matrix).todense()

# Calculate marginals using your approach
topic_marginal = np.array(topic_bow_matrix.sum(axis=0))[0, :]  # marginal counts for topics
ngram_marginal = marginal  # reuse your existing ngram marginal

# Calculate NPMI using your efficient log-based approach
topic_ngram_coocc_log = np.log(topic_ngram_coocc / K)
topic_ngram_baseline = np.log(np.outer(topic_marginal, ngram_marginal) / K / K)
topic_ngram_npmi = topic_ngram_baseline / topic_ngram_coocc_log - 1
topic_ngram_npmi = np.array(topic_ngram_npmi)

# Handle numerical issues (same as your original code)
topic_ngram_npmi = np.nan_to_num(topic_ngram_npmi, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Topic-ngram NPMI matrix shape: {topic_ngram_npmi.shape}")
print(f"NPMI range: [{topic_ngram_npmi.min():.4f}, {topic_ngram_npmi.max():.4f}]")

# Save the results
np.savez_compressed("output/topic_ngram_npmi.npz",
                    npmi=topic_ngram_npmi,
                    cooccurrence=topic_ngram_coocc,
                    topics=unique_topics,
                    ngrams=ngram_list,
                    topic_marginal=topic_marginal)

# Analysis: Top ngrams per topic
print(f"\nTop 5 n-grams per topic (by NPMI):")
for topic_idx, topic in enumerate(unique_topics):
    if topic_marginal[topic_idx] > 100:  # Only show topics with reasonable support
        top_ngram_indices = np.argsort(topic_ngram_npmi[topic_idx])[-5:][::-1]
        print(f"\n{topic} ({int(topic_marginal[topic_idx])} articles):")
        for idx in top_ngram_indices:
            ngram = ngram_list[idx]
            npmi_score = topic_ngram_npmi[topic_idx, idx]
            if npmi_score > 0:  # Only show positive associations
                print(f"  {ngram}: {npmi_score:.4f}")

# Optional: Create topic-enhanced ngram graph
print("\nCreating topic-enhanced ngram graph...")
G_enhanced = nx.Graph()

for i in range(N_ngrams):
    ngram = ngram_list[i]

    # Find the topic with highest NPMI for this ngram
    best_topic_idx = np.argmax(topic_ngram_npmi[:, i])
    best_topic = topic_labels[unique_topics[best_topic_idx]]
    best_npmi = topic_ngram_npmi[best_topic_idx, i]

    # Add node with topic information
    G_enhanced.add_node(ngram,
        topic=best_topic,
        topic_npmi=best_npmi,
        original_topic=max(set(ngrams_topics[ngram]), key=ngrams_topics[ngram].count) if ngrams_topics[
            ngram] else "unknown"
    )

    # Add edges based on your existing ngram-ngram NPMI
    indices = np.argwhere(npmi[i] > 0).flatten()
    for idx in indices:
        connected_ngram = ngram_list[idx]
        if cond_prob[i, idx] > 0.1:
            G_enhanced.add_edge(ngram, connected_ngram, weight=cond_prob[i, idx])

nx.write_gexf(G_enhanced, "output/ngrams_with_topics.gexf")

print("Topic-ngram analysis complete!")
print(f"- Saved topic-ngram NPMI matrix to output/topic_ngram_npmi.npz")
print(f"- Saved enhanced ngram graph to output/ngrams_with_topics.gexf")
#
# G = nx.Graph()
# for i in range(N):
#     ngram = ngram_list[i]
#
#     G.add_node(ngram_list[i], topic=max(set(ngrams_topics[ngram]), key=ngrams_topics[ngram].count))
#
#     indices = np.argwhere(npmi[i] > 0).flatten()
#
#     for idx in indices:
#         connected_ngram = ngram_list[idx]
#         if cond_prob[i, idx] > 0.1:
#             G.add_edge(ngram, connected_ngram, weight=cond_prob[i, idx])
#
# nx.write_gexf(G, "output/ngrams.gexf")
# die()
#
# print(cond_prob)
#
# mapping = dict()
# population = dict()
#
# G = nx.DiGraph()
# for i in range(N):
#     ngram = ngram_list[i]
#     indices = np.argwhere(marginal > marginal[i]).flatten()
#
#     G.add_node(ngram)
#
#     if len(indices) > 0:
#         replacement = max(indices, key=lambda idx: npmi[i, idx])
#
#         if npmi[i, replacement] > 0.2:
#             G.add_edge(ngram, ngram_list[replacement], weight=1)
#             mapping[ngram] = ngram_list[replacement]
#             continue
#
#     mapping[ngram] = ngram
#
#
# def follow_path(node):
#     neighbors = list(G.neighbors(node))
#     if len(neighbors) == 0:
#         return node
#     else:
#         return follow_path(neighbors[0])
#
#
# for keyword in mapping.keys():
#     target = follow_path(keyword)
#     population[target] = population.get(target, 0) + marginal[ngram_to_idx[keyword]]
#
# bubbles = sorted(population.keys(), key=lambda x: population[x], reverse=True)
#
# for bubble in bubbles:
#     keywords = [x for x in mapping if follow_path(x) == bubble]
#     print(f"{bubble} ({population[bubble]}, {len(keywords)})")
#
# # print(mapping["coronavirus"])
# # print(mapping["covid"])
# # print(mapping)
#
# nx.write_gexf(G, "output/ngrams.gexf")
