import requests
import time
from urllib import parse
from functools import reduce
import os
import gzip
import numpy as np
import pandas as pd
import json
import datetime
import re
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    ForeignKey,
    Table,
    Index,
    BigInteger,
    Date,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import JSON

from genderComputer import GenderComputer

gc = GenderComputer()

Base = declarative_base()

ENABLE_REFERENCES = False

# Association tables for many-to-many relationships
articles_authors_table = Table(
    "articles_authors",
    Base.metadata,
    Column(
        "article_id", BigInteger, ForeignKey("articles.article_id"), primary_key=True
    ),
    Column("author_id", BigInteger, ForeignKey("authors.author_id"), primary_key=True),
    Column("position", String),
)

articles_affiliations_table = Table(
    "articles_affiliations",
    Base.metadata,
    Column(
        "article_id", BigInteger, ForeignKey("articles.article_id"), primary_key=True
    ),
    Column("author_id", BigInteger, ForeignKey("authors.author_id"), primary_key=True),
    Column(
        "institution_id",
        BigInteger,
        ForeignKey("institutions.institution_id"),
        primary_key=True,
    ),
    Column("type", String),
)

articles_topics_table = Table(
    "articles_topics",
    Base.metadata,
    Column(
        "article_id", BigInteger, ForeignKey("articles.article_id"), primary_key=True
    ),
    Column("topic_id", BigInteger, ForeignKey("topics.topic_id"), primary_key=True),
    Column("score", Float),
)

articles_concepts_table = Table(
    "articles_concepts",
    Base.metadata,
    Column(
        "article_id", BigInteger, ForeignKey("articles.article_id"), primary_key=True
    ),
    Column(
        "concept_id", BigInteger, ForeignKey("concepts.concept_id"), primary_key=True
    ),
    Column(
        "score", Float
    ),
)


# Define database models
class Article(Base):
    __tablename__ = "articles"

    article_id = Column(BigInteger, primary_key=True)
    title = Column(Text)
    publication_year = Column(Integer)
    publication_date = Column(Date)
    domain = Column(Integer)
    field = Column(Integer)
    subfield = Column(Integer)
    domains = Column(JSON)  # Store array as JSON
    fields = Column(JSON)  # Store array as JSON
    subfields = Column(JSON)  # Store array as JSON
    url = Column(Text)
    language = Column(String)
    source = Column(BigInteger)

    # Relationships
    authors = relationship(
        "Author", secondary=articles_authors_table, back_populates="articles"
    )
    institutions = relationship(
        "Institution", secondary=articles_affiliations_table, back_populates="articles"
    )
    topics = relationship(
        "Topic", secondary=articles_topics_table, back_populates="articles"
    )
    concepts = relationship(
        "Concept", secondary=articles_concepts_table, back_populates="articles"
    )
    abstract = relationship("Abstract", uselist=False, back_populates="article")

    # Add indexes for common queries
    __table_args__ = (
        Index("idx_publication_year", "publication_year"),
        Index("idx_domain", "domain"),
        Index("idx_field", "field"),
    )


class Abstract(Base):
    __tablename__ = "abstracts"

    article_id = Column(BigInteger, ForeignKey("articles.article_id"), primary_key=True)
    abstract = Column(Text)

    # Relationship
    article = relationship("Article", back_populates="abstract")


class Author(Base):
    __tablename__ = "authors"

    author_id = Column(BigInteger, primary_key=True)
    name = Column(String)
    orcid = Column(String)
    gender = Column(String)

    # Relationships
    articles = relationship(
        "Article", secondary=articles_authors_table, back_populates="authors"
    )
    institutions = relationship(
        "Institution", secondary=articles_affiliations_table, back_populates="authors"
    )

    # Add indexes
    __table_args__ = (
        Index("idx_author_name", "name"),
        Index("idx_author_gender", "gender"),
    )


class Institution(Base):
    __tablename__ = "institutions"

    institution_id = Column(BigInteger, primary_key=True)
    name = Column(String)
    country_code = Column(String)
    lineage = Column(JSON)  # Store array as JSON

    # Relationships
    articles = relationship(
        "Article", secondary=articles_affiliations_table, back_populates="institutions"
    )
    authors = relationship(
        "Author", secondary=articles_affiliations_table, back_populates="institutions"
    )

    # Add indexes
    __table_args__ = (
        Index("idx_institution_name", "name"),
        Index("idx_country_code", "country_code"),
    )


class Topic(Base):
    __tablename__ = "topics"

    topic_id = Column(BigInteger, primary_key=True)
    name = Column(String)

    # Relationships
    articles = relationship(
        "Article", secondary=articles_topics_table, back_populates="topics"
    )


class Concept(Base):
    __tablename__ = "concepts"

    concept_id = Column(BigInteger, primary_key=True)
    name = Column(String)
    level = Column(Integer)

    # Relationships
    articles = relationship(
        "Article", secondary=articles_concepts_table, back_populates="concepts"
    )


class Reference(Base):
    __tablename__ = "references"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cites = Column(BigInteger, ForeignKey("articles.article_id"))
    cited = Column(BigInteger)  # May not exist in our database

    # Add indexes for citation queries
    __table_args__ = (
        Index("idx_cites", "cites"),
        Index("idx_cited", "cited"),
    )


early_date = 2016


def now():
    dt = datetime.now() - datetime(2000, 1, 1)
    s = (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1e6
    return s


def read_json_from_gzip(path):
    with gzip.open(path, "rt") as fp:
        data = fp.read()
    data = json.loads(data)
    return data


def url_to_id(url):
    return int(url.replace("https://openalex.org/", "")[1:])


def clean_domain(url):
    return int(url.replace("https://openalex.org/domains/", ""))


def clean_field(url):
    return int(url.replace("https://openalex.org/fields/", ""))


def clean_subfield(url):
    return int(url.replace("https://openalex.org/subfields/", ""))


def clean_source(url):
    return url.replace("https://openalex.org/S", "")


def get_arxiv(url):
    if url is None:
        return None
    pattern = r"https?://arxiv\.org/abs/([a-z0-9\.]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


github_repo_regex = (
    r"(?:https?:\/\/)?(?:www\.)?github\.com\/([^\/\s.,(){}]+\/[^\/\s.,(){}]+)"
)


def extract_repositories(words):
    repositories = []
    for word in words:
        match = re.search(github_repo_regex, word)
        if match:
            repositories.append(match.group(1))
    return repositories


class OptimizedSQLCompiler:
    def __init__(self, database_url, batch_size=10000):
        """
        Initialize the optimized SQL compiler with database connection.

        Args:
            database_url: SQLAlchemy database URL
            batch_size: Number of records to insert in each batch (increased default)
        """
        # Optimize connection pool for bulk operations
        self.engine = create_engine(
            database_url,
            echo=False,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.Session = sessionmaker(bind=self.engine)
        self.batch_size = batch_size
        self.database_type = self._detect_database_type(database_url)

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Temporary storage for batch processing - larger batches
        self.temp_data = {
            "articles": [],
            "abstracts": [],
            "authors": [],
            "institutions": [],
            "topics": [],
            "concepts": [],
            "references": [],
            "articles_authors": [],
            "articles_affiliations": [],
            "articles_topics": [],
            "articles_concepts": [],
            "files": []
        }

        # Track processed IDs within current batch to avoid duplicates
        self.batch_processed = {
            "articles": set(),
            "authors": set(),
            "institutions": set(),
            "topics": set(),
            "concepts": set(),
        }

        self.repositories = []
        self.n_urls = 0

    def _detect_database_type(self, database_url):
        """Detect database type from URL for optimized SQL queries."""
        if database_url.startswith('postgresql'):
            return 'postgresql'
        elif database_url.startswith('mysql'):
            return 'mysql'
        elif database_url.startswith('sqlite'):
            return 'sqlite'
        else:
            return 'unknown'

    def _get_upsert_sql(self, table_name, columns, conflict_columns=None):
        """Generate database-specific upsert SQL."""
        # Quote table name to handle reserved keywords
        quoted_table = f'"{table_name}"'

        if self.database_type == 'postgresql':
            cols = ', '.join(f'"{col}"' for col in columns)
            values = ', '.join(f':{col}' for col in columns)
            if conflict_columns:
                conflict_cols = ', '.join(f'"{col}"' for col in conflict_columns)
                return f"""
                    INSERT INTO {quoted_table} ({cols}) 
                    VALUES ({values})
                    ON CONFLICT ({conflict_cols}) DO NOTHING
                """
            else:
                return f"""
                    INSERT INTO {quoted_table} ({cols}) 
                    VALUES ({values})
                    ON CONFLICT DO NOTHING
                """
        elif self.database_type == 'mysql':
            cols = ', '.join(f'`{col}`' for col in columns)
            values = ', '.join(f':{col}' for col in columns)
            return f"""
                INSERT IGNORE INTO {quoted_table} ({cols}) 
                VALUES ({values})
            """
        else:  # SQLite
            cols = ', '.join(f'"{col}"' for col in columns)
            values = ', '.join(f':{col}' for col in columns)
            return f"""
                INSERT OR IGNORE INTO {quoted_table} ({cols}) 
                VALUES ({values})
            """

    def flush_batch_data(self):
        """Insert accumulated data using optimized bulk operations with raw SQL."""
        if not any(self.temp_data.values()):
            return

        session = self.Session()
        try:
            # Use raw SQL for maximum performance

            # Bulk insert articles
            if self.temp_data["articles"]:
                article_sql = self._get_upsert_sql(
                    'articles',
                    ['article_id', 'title', 'publication_year', 'publication_date',
                     'domain', 'field', 'subfield', 'domains', 'fields', 'subfields',
                     'url', 'language', 'source'],
                    ['article_id']
                )
                session.execute(text(article_sql), self.temp_data["articles"])
                print(f"Inserted {len(self.temp_data['articles'])} articles")

            # Bulk insert abstracts
            if self.temp_data["abstracts"]:
                abstract_sql = self._get_upsert_sql(
                    'abstracts',
                    ['article_id', 'abstract'],
                    ['article_id']
                )
                session.execute(text(abstract_sql), self.temp_data["abstracts"])
                print(f"Inserted {len(self.temp_data['abstracts'])} abstracts")

            # Bulk insert authors
            if self.temp_data["authors"]:
                author_sql = self._get_upsert_sql(
                    'authors',
                    ['author_id', 'name', 'orcid', 'gender'],
                    ['author_id']
                )
                session.execute(text(author_sql), self.temp_data["authors"])
                print(f"Inserted {len(self.temp_data['authors'])} authors")

            # Bulk insert institutions
            if self.temp_data["institutions"]:
                institution_sql = self._get_upsert_sql(
                    'institutions',
                    ['institution_id', 'name', 'country_code', 'lineage'],
                    ['institution_id']
                )
                session.execute(text(institution_sql), self.temp_data["institutions"])
                print(f"Inserted {len(self.temp_data['institutions'])} institutions")

            # Bulk insert topics
            if self.temp_data["topics"]:
                topic_sql = self._get_upsert_sql(
                    'topics',
                    ['topic_id', 'name'],
                    ['topic_id']
                )
                session.execute(text(topic_sql), self.temp_data["topics"])
                print(f"Inserted {len(self.temp_data['topics'])} topics")

            # Bulk insert concepts
            if self.temp_data["concepts"]:
                concept_sql = self._get_upsert_sql(
                    'concepts',
                    ['concept_id', 'name', 'level'],
                    ['concept_id']
                )
                session.execute(text(concept_sql), self.temp_data["concepts"])
                print(f"Inserted {len(self.temp_data['concepts'])} concepts")

            # Bulk insert references
            if self.temp_data["references"] and ENABLE_REFERENCES:
                # For SQLite, we need to handle the reserved keyword "references" differently
                if self.database_type == 'sqlite':
                    # Use executemany with proper SQLite syntax
                    session.execute(
                        text('INSERT OR IGNORE INTO "references" ("cites", "cited") VALUES (:cites, :cited)'),
                        self.temp_data["references"]
                    )
                else:
                    reference_sql = self._get_upsert_sql(
                        'references',
                        ['cites', 'cited']
                    )
                    session.execute(text(reference_sql), self.temp_data["references"])
                print(f"Inserted {len(self.temp_data['references'])} references")

            session.commit()

            # Bulk insert relationship tables
            if self.temp_data["articles_authors"]:
                if self.database_type == 'sqlite':
                    session.execute(
                        text(
                            'INSERT OR IGNORE INTO "articles_authors" ("article_id", "author_id", "position") VALUES (:article_id, :author_id, :position)'),
                        self.temp_data["articles_authors"]
                    )
                else:
                    articles_authors_sql = self._get_upsert_sql(
                        'articles_authors',
                        ['article_id', 'author_id', 'position'],
                        ['article_id', 'author_id']
                    )
                    session.execute(text(articles_authors_sql), self.temp_data["articles_authors"])
                print(f"Inserted {len(self.temp_data['articles_authors'])} article-author relationships")

            if self.temp_data["articles_affiliations"]:
                if self.database_type == 'sqlite':
                    session.execute(
                        text(
                            'INSERT OR IGNORE INTO "articles_affiliations" ("article_id", "author_id", "institution_id", "type") VALUES (:article_id, :author_id, :institution_id, :type)'),
                        self.temp_data["articles_affiliations"]
                    )
                else:
                    articles_affiliations_sql = self._get_upsert_sql(
                        'articles_affiliations',
                        ['article_id', 'author_id', 'institution_id', 'type'],
                        ['article_id', 'author_id', 'institution_id']
                    )
                    session.execute(text(articles_affiliations_sql), self.temp_data["articles_affiliations"])
                print(f"Inserted {len(self.temp_data['articles_affiliations'])} article-affiliation relationships")

            if self.temp_data["articles_topics"]:
                if self.database_type == 'sqlite':
                    session.execute(
                        text(
                            'INSERT OR IGNORE INTO "articles_topics" ("article_id", "topic_id", "score") VALUES (:article_id, :topic_id, :score)'),
                        self.temp_data["articles_topics"]
                    )
                else:
                    articles_topics_sql = self._get_upsert_sql(
                        'articles_topics',
                        ['article_id', 'topic_id', 'score'],
                        ['article_id', 'topic_id']
                    )
                    session.execute(text(articles_topics_sql), self.temp_data["articles_topics"])
                print(f"Inserted {len(self.temp_data['articles_topics'])} article-topic relationships")

            if self.temp_data["articles_concepts"]:
                if self.database_type == 'sqlite':
                    session.execute(
                        text(
                            'INSERT OR IGNORE INTO "articles_concepts" ("article_id", "concept_id", "score") VALUES (:article_id, :concept_id, :score)'),
                        self.temp_data["articles_concepts"]
                    )
                else:
                    articles_concepts_sql = self._get_upsert_sql(
                        'articles_concepts',
                        ['article_id', 'concept_id', "score"],
                        ['article_id', 'concept_id', "score"]
                    )
                    session.execute(text(articles_concepts_sql), self.temp_data["articles_concepts"])
                print(f"Inserted {len(self.temp_data['articles_concepts'])} article-concept relationships")

            session.commit()

        except Exception as e:
            session.rollback()
            print(f"Error inserting batch: {e}")
            raise
        finally:
            session.close()

        self.load_processed()
        self.processed += self.temp_data["files"]
        open("processed", "w+").write("\n".join(self.processed))
        self.load_processed()

        # Clear temporary data and batch tracking
        for key in self.temp_data:
            self.temp_data[key].clear()
        for key in self.batch_processed:
            self.batch_processed[key].clear()

    def add_article(self, data):
        article_id = url_to_id(data["id"])

        # Skip if already processed in current batch
        if article_id in self.batch_processed["articles"]:
            return

        source = data.get("primary_location", 0)
        if source:
            source = source.get("source", 0)
            if source:
                if "id" in source and source["id"] is not None:
                    source = int(clean_source(source["id"]))
                else:
                    source = 0

        if source is None:
            source = 0

        if not (data["publication_year"] >= early_date):
            return

        if data.get("language", "") != "en":
            return

        article = {
            "article_id": article_id,
            "title": data["title"],
            "publication_year": data["publication_year"],
            "publication_date": datetime.strptime(data["publication_date"], "%Y-%m-%d"),
            "domain": clean_domain(data["primary_topic"]["domain"]["id"]),
            "field": clean_field(data["primary_topic"]["field"]["id"]),
            "subfield": clean_subfield(data["primary_topic"]["subfield"]["id"]),
            "domains": json.dumps([]),  # Convert to JSON string
            "fields": json.dumps([]),
            "subfields": json.dumps([]),
            "url": None,
            "language": data.get("language", ""),
            "source": int(source),
        }

        domains = []
        fields = []
        subfields = []

        for location in data["locations"]:
            if location["is_oa"] and article["url"] is None:
                article["url"] = location["landing_page_url"]
                self.n_urls += 1
                continue

        # Process abstract
        if data["abstract_inverted_index"] is not None:
            if len(data["abstract_inverted_index"]) > 1:
                abstract_words = [""] * int(
                    reduce(
                        lambda x, y: np.max(np.maximum(x, np.max(y))),
                        data["abstract_inverted_index"].values(),
                    )
                    + 1
                )

                for word in data["abstract_inverted_index"]:
                    for pos in data["abstract_inverted_index"][word]:
                        abstract_words[pos] = word

                self.temp_data["abstracts"].append({
                    "article_id": article_id,
                    "abstract": " ".join(abstract_words),
                })

        # Process topics
        for topic in data["topics"]:
            domains.append(clean_domain(topic["domain"]["id"]))
            fields.append(clean_field(topic["field"]["id"]))
            subfields.append(clean_subfield(topic["subfield"]["id"]))

        # Update article with JSON arrays
        article["domains"] = json.dumps(domains)
        article["fields"] = json.dumps(fields)
        article["subfields"] = json.dumps(subfields)

        self.temp_data["articles"].append(article)
        self.batch_processed["articles"].add(article_id)

        # Process references
        for reference in data["referenced_works"]:
            reference_id = url_to_id(reference)
            self.temp_data["references"].append({
                "cites": article_id,
                "cited": reference_id
            })

        # Process authors and affiliations
        for author_data in data["authorships"]:
            author_id = url_to_id(author_data["author"]["id"])

            if author_id not in self.batch_processed["authors"]:
                author_name = author_data["author"]["display_name"]
                try:
                    gender = gc.resolveGender(author_name, None)
                except:
                    gender = None
                if gender == "female":
                    gender = "f"
                elif gender == "male":
                    gender = "m"

                self.temp_data["authors"].append({
                    "author_id": author_id,
                    "name": author_name,
                    "orcid": author_data["author"]["orcid"],
                    "gender": gender,
                })
                self.batch_processed["authors"].add(author_id)

            self.temp_data["articles_authors"].append({
                "author_id": author_id,
                "article_id": article_id,
                "position": author_data["author_position"],
            })

            # Process institutions
            for institution in author_data["institutions"]:
                institution_id = url_to_id(institution["id"])

                if institution_id not in self.batch_processed["institutions"]:
                    self.temp_data["institutions"].append({
                        "institution_id": institution_id,
                        "name": institution["display_name"],
                        "country_code": institution["country_code"],
                        "lineage": json.dumps([
                            url_to_id(ancestor)
                            for ancestor in institution["lineage"]
                            if ancestor != institution["id"]
                        ]),
                    })
                    self.batch_processed["institutions"].add(institution_id)

                self.temp_data["articles_affiliations"].append({
                    "article_id": article_id,
                    "author_id": author_id,
                    "institution_id": institution_id,
                    "type": institution["type"],
                })

        # Process topics
        for topic in data["topics"]:
            topic_id = url_to_id(topic["id"])

            if topic_id not in self.batch_processed["topics"]:
                self.temp_data["topics"].append({
                    "topic_id": topic_id,
                    "name": topic["display_name"],
                })
                self.batch_processed["topics"].add(topic_id)

            self.temp_data["articles_topics"].append({
                "article_id": article_id,
                "topic_id": topic_id,
                "score": topic["score"],
            })

        # Check if we need to flush batch
        if len(self.temp_data["articles"]) >= self.batch_size:
            self.flush_batch_data()

            stats = compiler.get_stats()
            print("Current database statistics:")
            for table, count in stats.items():
                print(f"{table}: {count}")

    def add_concept(self, data):
        concept_id = url_to_id(data["id"])

        if concept_id not in self.batch_processed["concepts"]:
            self.temp_data["concepts"].append({
                "concept_id": concept_id,
                "name": data["display_name"],
                "level": data["level"],
            })
            self.batch_processed["concepts"].add(concept_id)

    def load_processed(self):
        if os.path.exists("processed"):
            self.processed = open("processed", "r").read().split("\n")
        else:
            self.processed = []

    def compile_works(self, raw_data_location, pattern=None):
        extension = "gz"

        self.load_processed()

        for root, dirnames, filenames in os.walk(raw_data_location):
            for filename in filenames:
                path = os.path.join(root, filename)

                if not filename.endswith(extension):
                    continue

                if not os.path.exists(path):
                    continue

                if pattern and (not re.match(pattern, path)):
                    continue

                if path in self.processed:
                    print(f"Skipping: {path}")
                    continue

                print(f"Processing: {path}")
                data = read_json_from_gzip(path)

                for r in data:
                    self.add_article(r)

                self.temp_data["files"].append(path)

                # print(f"Processed articles in batch: {len(self.temp_data['articles'])}")

        # Flush remaining data
        if self.temp_data["articles"]:
            self.flush_batch_data()

    def compile_concepts(self, raw_data_location):
        extension = "gz"

        for root, dirnames, filenames in os.walk(raw_data_location):
            for filename in filenames:
                path = os.path.join(root, filename)

                if not filename.endswith(extension):
                    continue

                if not os.path.exists(path):
                    continue

                print(f"Processing concepts: {path}")
                data = read_json_from_gzip(path)
                for r in data["results"]:
                    self.add_concept(r)

        # Flush concepts
        if self.temp_data["concepts"]:
            self.flush_batch_data()

    def get_stats(self):
        """Get database statistics."""
        session = self.Session()
        try:
            stats = {
                "articles": session.query(Article).count(),
                "authors": session.query(Author).count(),
                "institutions": session.query(Institution).count(),
                "topics": session.query(Topic).count(),
                "concepts": session.query(Concept).count(),
                "references": session.query(Reference).count(),
            }
            return stats
        finally:
            session.close()


# Example usage:
if __name__ == "__main__":
    # For PostgreSQL (recommended for best performance)
    # database_url = "postgresql://postgres@localhost/socialscience"

    # For SQLite (simpler setup)
    database_url = "sqlite:///articles.db"

    # Use the optimized compiler with larger batch size
    compiler = OptimizedSQLCompiler(database_url, batch_size=10000)
    compiler.compile_works("output")

    # Print final statistics
    stats = compiler.get_stats()
    print("Final database statistics:")
    for table, count in stats.items():
        print(f"{table}: {count}")
