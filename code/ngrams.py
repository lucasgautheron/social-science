from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import datetime

data = np.load("ngram_articles.npz")

# Database connection - update this to match your setup
database_url = "sqlite:///articles.db"  # or your PostgreSQL URL
engine = create_engine(database_url)


def process_batch(articles):
    whitelist = ", ".join(map(str, list(articles)))

    # Query to count keyword articles and total articles per year
    query = f"""
            SELECT a.publication_date,
                   a.article_id
            FROM articles a
            WHERE a.article_id IN ({whitelist}) \
            """

    # Execute query and get results
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    return df


dfs = []

for ngram in data.keys():
    if len(data[ngram]) < 1000:
        continue

    print(ngram)
    step = 10000
    dfs.append(pd.concat([
        process_batch(data[ngram][start:start + step])
        for start in range(0, len(data[ngram]), step)
    ]).assign(ngram=ngram))

dfs = pd.concat(dfs)
