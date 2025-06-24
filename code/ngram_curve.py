from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import datetime
from scipy.stats import beta
from cmdstanpy import CmdStanModel

parser = argparse.ArgumentParser()
parser.add_argument("--ngram")
args = parser.parse_args()

data = np.load("output/ngram_articles.npz")
articles = data[args.ngram]

# Database connection - update this to match your setup
database_url = "sqlite:///articles.db"  # or your PostgreSQL URL
engine = create_engine(database_url)


def get_papers_per_year(database_url="sqlite:///articles.db"):
    """
    Returns the total number of papers per year in the database.

    Args:
        database_url (str): Database connection URL

    Returns:
        pandas.DataFrame: DataFrame with publication_year and total columns
    """
    engine = create_engine(database_url)

    query = """
            SELECT publication_year,
                   COUNT(DISTINCT article_id) as total
            FROM articles
            WHERE publication_year IS NOT NULL
            GROUP BY publication_year
            ORDER BY publication_year \
            """

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    return df


def process_batch(articles):
    whitelist = ", ".join(map(str, list(articles)))

    # Query to count keyword articles and total articles per year
    query = f"""
            SELECT a.publication_date,
                   a.publication_year,
                   a.article_id
            FROM articles a
            WHERE a.article_id IN ({whitelist}) \
            """

    # Execute query and get results
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    return df


step = 1000
df = pd.concat([
    process_batch(articles[start:start + step])
    for start in range(0, len(articles), step)
])


def str2dt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


counts = df.groupby(["publication_year"]).agg(n=("article_id", pd.Series.nunique))

per_year = get_papers_per_year().set_index("publication_year")
print(per_year)
per_year = per_year.merge(counts, how="left", left_index=True, right_index=True)
per_year.fillna(0, inplace=True)
per_year["fraction"] = (1 + per_year["n"]) / (2 + per_year["total"])
per_year["low"] = per_year.apply(
    lambda row: beta.ppf(0.05, row["n"] + 1, row["total"] - row["n"] + 1),
    axis=1,
)
per_year["high"] = per_year.apply(
    lambda row: beta.ppf(0.95, row["n"] + 1, row["total"] - row["n"] + 1),
    axis=1,
)

plt.plot(per_year["fraction"])
plt.errorbar(per_year.index, per_year["fraction"], yerr=(per_year["fraction"] - per_year["low"],
                                                         per_year["high"] - per_year["fraction"]))
plt.show()
print(counts)
