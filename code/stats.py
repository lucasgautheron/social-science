from sqlalchemy import create_engine, text
import pandas as pd

# Database connection - update this to match your setup
database_url = "sqlite:///articles.db"  # or your PostgreSQL URL
engine = create_engine(database_url)

# Query to count keyword articles and total articles per year
query = """
SELECT 
    a.publication_year,
    COUNT(*) as total_articles,
    SUM(CASE WHEN ab.abstract LIKE '% ukraine %' THEN 1 ELSE 0 END) as keyword_articles
FROM articles a
JOIN abstracts ab ON a.article_id = ab.article_id
GROUP BY a.publication_year
ORDER BY a.publication_year
"""

# Execute query and get results
with engine.connect() as conn:
    df = pd.read_sql_query(query, conn)

# Calculate fraction
df['keyword_fraction'] = df['keyword_articles'] / df['total_articles']
df['keyword_percentage'] = df['keyword_fraction'] * 100

# Display results
print("Articles containing 'keyword' in abstract by year:")
print(df[['publication_year', 'keyword_articles', 'total_articles', 'keyword_percentage']].to_string(index=False, float_format='%.2f'))

# Calculate overall stats
total_keyword = df['keyword_articles'].sum()
total_all = df['total_articles'].sum()
overall_fraction = total_keyword / total_all

print(f"\nOverall Statistics:")
print(f"Total articles with 'keyword': {total_keyword:,}")
print(f"Total articles: {total_all:,}")
print(f"Overall fraction: {overall_fraction:.4f} ({overall_fraction*100:.2f}%)")
print(f"Year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
print(f"Average percentage per year: {df['keyword_percentage'].mean():.2f}%")