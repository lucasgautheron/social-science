import os
import gzip
import json
import datetime
import re
from datetime import datetime, date as datetime_date
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

gc = None


def resolve_gender(author_name):
    global gc
    if gc is None:
        from genderComputer import GenderComputer

        gc = GenderComputer()
    return gc.resolveGender(author_name, None)

Base = declarative_base()

ENABLE_REFERENCES = False
DEFAULT_PROGRESS_INTERVAL = 1000
DEFAULT_PARQUET_BATCH_SIZE = 1000
DEFAULT_PARQUET_READER = "pyarrow"

WORKS_PARQUET_COLUMNS = [
    "id",
    "doi",
    "title",
    "display_name",
    "ids",
    "indexed_in",
    "publication_date",
    "publication_year",
    "language",
    "type",
    "authorships",
    "authors_count",
    "corresponding_author_ids",
    "corresponding_institution_ids",
    "primary_topic",
    "topics",
    "keywords",
    "concepts",
    "locations",
    "locations_count",
    "primary_location",
    "best_oa_location",
    "sustainable_development_goals",
    "awards",
    "funders",
    "institutions",
    "countries_distinct_count",
    "institutions_distinct_count",
    "open_access",
    "is_paratext",
    "is_retracted",
    "is_xpac",
    "biblio",
    "referenced_works",
    "referenced_works_count",
    "related_works",
    "abstract_inverted_index",
    "cited_by_count",
    "counts_by_year",
    "apc_list",
    "apc_paid",
    "fwci",
    "citation_normalized_percentile",
    "cited_by_percentile_year",
    "mesh",
    "has_content",
    "has_fulltext",
    "created_date",
    "updated_date",
]

WORKS_PARQUET_READ_COLUMNS = [
    "id",
    "title",
    "display_name",
    "publication_date",
    "publication_year",
    "language",
    "authorships",
    "primary_topic",
    "topics",
    "concepts",
    "primary_location",
    "best_oa_location",
    "abstract_inverted_index",
]


def get_works_parquet_read_columns():
    columns = list(WORKS_PARQUET_READ_COLUMNS)
    if ENABLE_REFERENCES:
        columns.append("referenced_works")
    return columns

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


def is_missing(value):
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float) and value != value:
        return True
    return False


def normalize_nested(value):
    if is_missing(value):
        return None
    if hasattr(value, "as_py"):
        return normalize_nested(value.as_py())
    if isinstance(value, dict):
        return {key: normalize_nested(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_nested(item) for item in value)
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        try:
            return normalize_nested(value.tolist())
        except (AttributeError, TypeError, ValueError):
            return value
    return value


def parse_json_like(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def ensure_list(value):
    value = normalize_nested(parse_json_like(value))
    if is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return []


def ensure_dict(value):
    value = normalize_nested(parse_json_like(value))
    if isinstance(value, dict):
        return value
    return {}


def extract_openalex_numeric_id(value, prefix=None):
    if is_missing(value):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return int(value)

    identifier = str(value).strip().rstrip("/")
    if not identifier:
        return None

    identifier = identifier.rsplit("/", 1)[-1]
    if prefix and identifier.startswith(prefix):
        identifier = identifier[len(prefix):]
    elif identifier and identifier[0].isalpha():
        identifier = identifier[1:]

    return int(identifier)


def url_to_id(url):
    return extract_openalex_numeric_id(url)


def safe_url_to_id(url):
    try:
        return url_to_id(url)
    except (AttributeError, TypeError, ValueError):
        return None


def clean_domain(url):
    return extract_openalex_numeric_id(url)


def clean_field(url):
    return extract_openalex_numeric_id(url)


def clean_subfield(url):
    return extract_openalex_numeric_id(url)


def clean_source(url):
    return extract_openalex_numeric_id(url, "S")


def parse_publication_date(value):
    if is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, datetime_date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    return value


def iter_abstract_entries(inverted_index):
    inverted_index = normalize_nested(parse_json_like(inverted_index))
    if is_missing(inverted_index):
        return

    if isinstance(inverted_index, dict):
        for word, positions in inverted_index.items():
            yield word, ensure_list(positions)
        return

    for item in ensure_list(inverted_index):
        item = normalize_nested(item)
        if isinstance(item, tuple) and len(item) == 2:
            yield item[0], ensure_list(item[1])
        elif isinstance(item, dict):
            if "key" in item and "value" in item:
                yield item["key"], ensure_list(item["value"])
            elif "word" in item and "positions" in item:
                yield item["word"], ensure_list(item["positions"])
            elif len(item) == 1:
                word, positions = next(iter(item.items()))
                yield word, ensure_list(positions)


def reconstruct_abstract(inverted_index):
    entries = []
    for word, positions in iter_abstract_entries(inverted_index):
        clean_positions = []
        for position in positions:
            if is_missing(position):
                continue
            clean_positions.append(int(position))
        if clean_positions:
            entries.append((str(word), clean_positions))

    if not entries:
        return None

    max_position = max(max(positions) for _, positions in entries)
    abstract_words = [""] * (max_position + 1)
    for word, positions in entries:
        for position in positions:
            abstract_words[position] = word

    return " ".join(abstract_words)


def iter_json_work_records(path):
    data = read_json_from_gzip(path)
    if isinstance(data, dict):
        data = data.get("results", [])
    for record in data:
        yield normalize_nested(record)


def iter_pandas_parquet_work_records(path):
    import pandas as pd

    try:
        frame = pd.read_parquet(path, columns=get_works_parquet_read_columns())
    except (KeyError, ValueError):
        frame = pd.read_parquet(path)

    for record in frame.to_dict(orient="records"):
        yield normalize_nested(record)


def iter_parquet_work_records(
    path,
    batch_size=DEFAULT_PARQUET_BATCH_SIZE,
    reader=DEFAULT_PARQUET_READER,
):
    reader = reader.lower()

    if reader == "pandas":
        yield from iter_pandas_parquet_work_records(path)
        return

    if reader != "pyarrow":
        raise ValueError("parquet reader must be 'pyarrow' or 'pandas'")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        yield from iter_pandas_parquet_work_records(path)
        return

    if pq is not None:
        parquet_file = pq.ParquetFile(path)
        read_columns = get_works_parquet_read_columns()
        columns = [
            column
            for column in read_columns
            if column in parquet_file.schema_arrow.names
        ]
        if not columns:
            columns = None

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for record in batch.to_pylist():
                yield record
        return


def detect_work_file_format(filename):
    if filename.endswith(".parquet"):
        return "parquet"
    if filename.endswith(".gz"):
        return "json_gzip"
    return None


def iter_work_records(
    path,
    batch_size=DEFAULT_PARQUET_BATCH_SIZE,
    parquet_reader=DEFAULT_PARQUET_READER,
):
    input_format = detect_work_file_format(path)
    if input_format == "parquet":
        yield from iter_parquet_work_records(
            path,
            batch_size=batch_size,
            reader=parquet_reader,
        )
    elif input_format == "json_gzip":
        yield from iter_json_work_records(path)


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
        self.database_type = self._detect_database_type(database_url)
        engine_options = {"echo": False}
        if self.database_type != "sqlite":
            # Optimize connection pool for bulk operations on server databases.
            engine_options.update(
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        self.engine = create_engine(database_url, **engine_options)
        self.Session = sessionmaker(bind=self.engine)
        self.batch_size = batch_size

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
        self.insert_stats = {
            key: {"inserted": 0, "attempted": 0}
            for key in self.temp_data
            if key != "files"
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

    def _record_insert_stats(self, key, label, attempted, result):
        inserted = getattr(result, "rowcount", None)
        if inserted is None or inserted < 0:
            inserted = attempted

        self.insert_stats[key]["inserted"] += inserted
        self.insert_stats[key]["attempted"] += attempted

        if inserted == attempted:
            print(f"Inserted {inserted} {label}")
        else:
            print(f"Inserted {inserted}/{attempted} {label}")

    def print_insert_stats(self):
        print("Insertion statistics for this run:")
        for key, stats in self.insert_stats.items():
            if stats["attempted"] == 0:
                continue
            print(
                f"{key}: {stats['inserted']} inserted "
                f"({stats['attempted']} attempted)"
            )

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
                result = session.execute(text(article_sql), self.temp_data["articles"])
                self._record_insert_stats(
                    "articles",
                    "articles",
                    len(self.temp_data["articles"]),
                    result,
                )

            # Bulk insert abstracts
            if self.temp_data["abstracts"]:
                abstract_sql = self._get_upsert_sql(
                    'abstracts',
                    ['article_id', 'abstract'],
                    ['article_id']
                )
                result = session.execute(text(abstract_sql), self.temp_data["abstracts"])
                self._record_insert_stats(
                    "abstracts",
                    "abstracts",
                    len(self.temp_data["abstracts"]),
                    result,
                )

            # Bulk insert authors
            if self.temp_data["authors"]:
                author_sql = self._get_upsert_sql(
                    'authors',
                    ['author_id', 'name', 'orcid', 'gender'],
                    ['author_id']
                )
                result = session.execute(text(author_sql), self.temp_data["authors"])
                self._record_insert_stats(
                    "authors",
                    "authors",
                    len(self.temp_data["authors"]),
                    result,
                )

            # Bulk insert institutions
            if self.temp_data["institutions"]:
                institution_sql = self._get_upsert_sql(
                    'institutions',
                    ['institution_id', 'name', 'country_code', 'lineage'],
                    ['institution_id']
                )
                result = session.execute(text(institution_sql), self.temp_data["institutions"])
                self._record_insert_stats(
                    "institutions",
                    "institutions",
                    len(self.temp_data["institutions"]),
                    result,
                )

            # Bulk insert topics
            if self.temp_data["topics"]:
                topic_sql = self._get_upsert_sql(
                    'topics',
                    ['topic_id', 'name'],
                    ['topic_id']
                )
                result = session.execute(text(topic_sql), self.temp_data["topics"])
                self._record_insert_stats(
                    "topics",
                    "topics",
                    len(self.temp_data["topics"]),
                    result,
                )

            # Bulk insert concepts
            if self.temp_data["concepts"]:
                concept_sql = self._get_upsert_sql(
                    'concepts',
                    ['concept_id', 'name', 'level'],
                    ['concept_id']
                )
                result = session.execute(text(concept_sql), self.temp_data["concepts"])
                self._record_insert_stats(
                    "concepts",
                    "concepts",
                    len(self.temp_data["concepts"]),
                    result,
                )

            # Bulk insert references
            if self.temp_data["references"] and ENABLE_REFERENCES:
                # For SQLite, we need to handle the reserved keyword "references" differently
                if self.database_type == 'sqlite':
                    # Use executemany with proper SQLite syntax
                    result = session.execute(
                        text('INSERT OR IGNORE INTO "references" ("cites", "cited") VALUES (:cites, :cited)'),
                        self.temp_data["references"]
                    )
                else:
                    reference_sql = self._get_upsert_sql(
                        'references',
                        ['cites', 'cited']
                    )
                    result = session.execute(text(reference_sql), self.temp_data["references"])
                self._record_insert_stats(
                    "references",
                    "references",
                    len(self.temp_data["references"]),
                    result,
                )

            session.commit()

            # Bulk insert relationship tables
            if self.temp_data["articles_authors"]:
                if self.database_type == 'sqlite':
                    result = session.execute(
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
                    result = session.execute(text(articles_authors_sql), self.temp_data["articles_authors"])
                self._record_insert_stats(
                    "articles_authors",
                    "article-author relationships",
                    len(self.temp_data["articles_authors"]),
                    result,
                )

            if self.temp_data["articles_affiliations"]:
                if self.database_type == 'sqlite':
                    result = session.execute(
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
                    result = session.execute(text(articles_affiliations_sql), self.temp_data["articles_affiliations"])
                self._record_insert_stats(
                    "articles_affiliations",
                    "article-affiliation relationships",
                    len(self.temp_data["articles_affiliations"]),
                    result,
                )

            if self.temp_data["articles_topics"]:
                if self.database_type == 'sqlite':
                    result = session.execute(
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
                    result = session.execute(text(articles_topics_sql), self.temp_data["articles_topics"])
                self._record_insert_stats(
                    "articles_topics",
                    "article-topic relationships",
                    len(self.temp_data["articles_topics"]),
                    result,
                )

            if self.temp_data["articles_concepts"]:
                if self.database_type == 'sqlite':
                    result = session.execute(
                        text(
                            'INSERT OR IGNORE INTO "articles_concepts" ("article_id", "concept_id", "score") VALUES (:article_id, :concept_id, :score)'),
                        self.temp_data["articles_concepts"]
                    )
                else:
                    articles_concepts_sql = self._get_upsert_sql(
                        'articles_concepts',
                        ['article_id', 'concept_id', "score"],
                        ['article_id', 'concept_id']
                    )
                    result = session.execute(text(articles_concepts_sql), self.temp_data["articles_concepts"])
                self._record_insert_stats(
                    "articles_concepts",
                    "article-concept relationships",
                    len(self.temp_data["articles_concepts"]),
                    result,
                )

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
        article_id = safe_url_to_id(data.get("id"))
        if article_id is None:
            print("Skipping work with missing or invalid id")
            return

        # Skip if already processed in current batch
        if article_id in self.batch_processed["articles"]:
            return

        primary_location = ensure_dict(data.get("primary_location"))
        source = ensure_dict(primary_location.get("source"))
        source_id = safe_url_to_id(source.get("id")) or 0

        minimum_publication_year = getattr(self, "minimum_publication_year", early_date)
        publication_year = data.get("publication_year")
        if is_missing(publication_year):
            return
        publication_year = int(publication_year)

        if minimum_publication_year is not None and publication_year < minimum_publication_year:
            return

        allowed_languages = getattr(self, "allowed_languages", {"en"})
        language = data.get("language", "")
        if (
            allowed_languages is not None
            and language not in allowed_languages
        ):
            return

        primary_topic = ensure_dict(data.get("primary_topic"))
        primary_domain = ensure_dict(primary_topic.get("domain"))
        primary_field = ensure_dict(primary_topic.get("field"))
        primary_subfield = ensure_dict(primary_topic.get("subfield"))

        article = {
            "article_id": article_id,
            "title": data.get("title") or data.get("display_name"),
            "publication_year": publication_year,
            "publication_date": parse_publication_date(data.get("publication_date")),
            "domain": safe_url_to_id(primary_domain.get("id")),
            "field": safe_url_to_id(primary_field.get("id")),
            "subfield": safe_url_to_id(primary_subfield.get("id")),
            "domains": json.dumps([]),  # Convert to JSON string
            "fields": json.dumps([]),
            "subfields": json.dumps([]),
            "url": None,
            "language": language,
            "source": source_id or 0,
        }

        domains = []
        fields = []
        subfields = []
        topics = [ensure_dict(topic) for topic in ensure_list(data.get("topics"))]

        for location in [
            primary_location,
            ensure_dict(data.get("best_oa_location")),
        ]:
            if location.get("is_oa") and article["url"] is None:
                article["url"] = location.get("landing_page_url")
                self.n_urls += 1

        for location in ensure_list(data.get("locations")):
            location = ensure_dict(location)
            if location.get("is_oa") and article["url"] is None:
                article["url"] = location.get("landing_page_url")
                self.n_urls += 1
                break

        # Process abstract
        abstract = reconstruct_abstract(data.get("abstract_inverted_index"))
        if abstract:
            self.temp_data["abstracts"].append({
                "article_id": article_id,
                "abstract": abstract,
            })

        # Process topics
        for topic in topics:
            domain_id = ensure_dict(topic.get("domain")).get("id")
            field_id = ensure_dict(topic.get("field")).get("id")
            subfield_id = ensure_dict(topic.get("subfield")).get("id")
            domain_id = safe_url_to_id(domain_id)
            field_id = safe_url_to_id(field_id)
            subfield_id = safe_url_to_id(subfield_id)
            if domain_id is not None:
                domains.append(domain_id)
            if field_id is not None:
                fields.append(field_id)
            if subfield_id is not None:
                subfields.append(subfield_id)

        # Update article with JSON arrays
        article["domains"] = json.dumps(domains)
        article["fields"] = json.dumps(fields)
        article["subfields"] = json.dumps(subfields)

        self.temp_data["articles"].append(article)
        self.batch_processed["articles"].add(article_id)

        # Process references only when they will actually be inserted.
        if ENABLE_REFERENCES:
            for reference in ensure_list(data.get("referenced_works")):
                reference_id = safe_url_to_id(reference)
                if reference_id is None:
                    continue
                self.temp_data["references"].append({
                    "cites": article_id,
                    "cited": reference_id
                })

        # Process authors and affiliations
        for author_data in ensure_list(data.get("authorships")):
            author_data = ensure_dict(author_data)
            author = ensure_dict(author_data.get("author"))
            author_openalex_id = author.get("id")
            if author_openalex_id is None:
                print(f"Skipping authorship with missing author id for article {article_id}")
                continue

            author_id = safe_url_to_id(author_openalex_id)
            if author_id is None:
                print(f"Skipping authorship with invalid author id for article {article_id}")
                continue

            if author_id not in self.batch_processed["authors"]:
                author_name = author.get("display_name")
                try:
                    gender = resolve_gender(author_name)
                except:
                    gender = None
                if gender == "female":
                    gender = "f"
                elif gender == "male":
                    gender = "m"

                self.temp_data["authors"].append({
                    "author_id": author_id,
                    "name": author_name,
                    "orcid": author.get("orcid"),
                    "gender": gender,
                })
                self.batch_processed["authors"].add(author_id)

            self.temp_data["articles_authors"].append({
                "author_id": author_id,
                "article_id": article_id,
                "position": author_data.get("author_position"),
            })

            # Process institutions
            for institution in ensure_list(author_data.get("institutions")):
                institution = ensure_dict(institution)
                institution_openalex_id = institution.get("id")
                institution_id = safe_url_to_id(institution_openalex_id)
                if institution_id is None:
                    print(f"Skipping institution with missing id for article {article_id}")
                    continue

                if institution_id not in self.batch_processed["institutions"]:
                    self.temp_data["institutions"].append({
                        "institution_id": institution_id,
                        "name": institution.get("display_name"),
                        "country_code": institution.get("country_code"),
                        "lineage": json.dumps([
                            safe_url_to_id(ancestor)
                            for ancestor in ensure_list(institution.get("lineage"))
                            if ancestor != institution_openalex_id
                            and safe_url_to_id(ancestor) is not None
                        ]),
                    })
                    self.batch_processed["institutions"].add(institution_id)

                self.temp_data["articles_affiliations"].append({
                    "article_id": article_id,
                    "author_id": author_id,
                    "institution_id": institution_id,
                    "type": institution.get("type"),
                })

        # Process topics
        for topic in topics:
            topic_id = safe_url_to_id(topic.get("id"))
            if topic_id is None:
                print(f"Skipping topic with missing id for article {article_id}")
                continue

            if topic_id not in self.batch_processed["topics"]:
                self.temp_data["topics"].append({
                    "topic_id": topic_id,
                    "name": topic.get("display_name"),
                })
                self.batch_processed["topics"].add(topic_id)

            self.temp_data["articles_topics"].append({
                "article_id": article_id,
                "topic_id": topic_id,
                "score": topic.get("score"),
            })

        # Process concepts embedded in works parquet/JSON records.
        for concept in ensure_list(data.get("concepts")):
            concept = ensure_dict(concept)
            concept_id = safe_url_to_id(concept.get("id"))
            if concept_id is None:
                print(f"Skipping concept with missing id for article {article_id}")
                continue

            if concept_id not in self.batch_processed["concepts"]:
                self.temp_data["concepts"].append({
                    "concept_id": concept_id,
                    "name": concept.get("display_name"),
                    "level": concept.get("level"),
                })
                self.batch_processed["concepts"].add(concept_id)

            self.temp_data["articles_concepts"].append({
                "article_id": article_id,
                "concept_id": concept_id,
                "score": concept.get("score"),
            })

        # Check if we need to flush batch
        if len(self.temp_data["articles"]) >= self.batch_size:
            self.flush_batch_data()

    def add_concept(self, data):
        data = ensure_dict(data)
        concept_id = safe_url_to_id(data.get("id"))
        if concept_id is None:
            return

        if concept_id not in self.batch_processed["concepts"]:
            self.temp_data["concepts"].append({
                "concept_id": concept_id,
                "name": data.get("display_name"),
                "level": data.get("level"),
            })
            self.batch_processed["concepts"].add(concept_id)

    def load_processed(self):
        if os.path.exists("processed"):
            self.processed = open("processed", "r").read().split("\n")
        else:
            self.processed = []

    def compile_works(
        self,
        raw_data_location,
        pattern=None,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
        parquet_reader=DEFAULT_PARQUET_READER,
        parquet_batch_size=DEFAULT_PARQUET_BATCH_SIZE,
    ):
        self.load_processed()
        total_raw_records = 0
        start_time = now()

        def candidate_paths():
            if os.path.isfile(raw_data_location):
                yield raw_data_location
                return

            for root, dirnames, filenames in os.walk(raw_data_location):
                dirnames.sort()
                for filename in sorted(filenames):
                    yield os.path.join(root, filename)

        for path in candidate_paths():
            input_format = detect_work_file_format(path)
            if input_format is None:
                continue

            if not os.path.exists(path):
                continue

            if pattern and (not re.match(pattern, path)):
                continue

            if path in self.processed:
                print(f"Skipping: {path}")
                continue

            print(f"Processing ({input_format}): {path}")
            file_raw_records = 0
            file_start_time = now()
            for record in iter_work_records(
                path,
                batch_size=parquet_batch_size,
                parquet_reader=parquet_reader,
            ):
                total_raw_records += 1
                file_raw_records += 1
                self.add_article(record)

                if (
                    progress_interval
                    and total_raw_records % progress_interval == 0
                ):
                    elapsed = max(now() - start_time, 1e-9)
                    print(
                        "Progress: "
                        f"{total_raw_records:,} raw rows "
                        f"({total_raw_records / elapsed:.1f} rows/s), "
                        f"{len(self.temp_data['articles']):,} buffered articles, "
                        f"{len(self.temp_data['authors']):,} buffered authors, "
                        f"{len(self.temp_data['references']):,} buffered references"
                    )

            self.temp_data["files"].append(path)
            file_elapsed = max(now() - file_start_time, 1e-9)
            print(
                f"Finished file: {path} "
                f"({file_raw_records:,} raw rows in {file_elapsed:.1f}s, "
                f"{file_raw_records / file_elapsed:.1f} rows/s)"
            )

            # print(f"Processed articles in batch: {len(self.temp_data['articles'])}")

        # Flush remaining data
        if any(self.temp_data.values()):
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

# Example usage:
if __name__ == "__main__":
    # For PostgreSQL (recommended for best performance)
    # database_url = "postgresql://postgres@localhost/socialscience"

    # For SQLite (simpler setup)
    database_url = "sqlite:///articles.db"

    # Use the optimized compiler with larger batch size
    compiler = OptimizedSQLCompiler(database_url, batch_size=10000)
    compiler.compile_works("output")

    compiler.print_insert_stats()
