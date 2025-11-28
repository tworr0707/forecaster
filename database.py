import sqlite3
import os
import wikipediaapi
import pickle
import datetime as dt
import pandas as pd
import threading
import hashlib
import requests
import torch
import gc
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
from tqdm import tqdm
from logger import setup_logger
from config import EMBEDDING_MODEL_PATH, MAX_EMBEDDING_SIZE

logger = setup_logger(__name__)
pd.set_option('display.max_colwidth', None)

class Database:
    """
    Database provides methods to store, retrieve, and manage articles, embeddings,
    forecast data, and ensemble results using a SQLite backend.
    
    Attributes:
        db_path (str): Filesystem path to the SQLite database file.
        embedding_model_path (str): Transformer model path for generating embeddings.
    """

    _conn: sqlite3.Connection | None = None
    _lock = threading.RLock()

    def __init__(
        self,
        db_path: str | None = None,
        embedding_model_path: str = EMBEDDING_MODEL_PATH,
    ) -> None:
        """
        Initialise the Database, set up tables, and fetch top new articles.
        
        Parameters:
            db_path (str | None): Path to the SQLite database file; defaults to project directory.
            embedding_model_path (str): Model identifier for sentence-transformers.
        """
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "database.db")

        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        self.model: Optional[SentenceTransformer] = None

        self._init_tables_and_pragmas()

    def unload_embedding_model(self) -> None:
        """Release the cached embedding model and clear accelerator memory."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def _connect(self) -> sqlite3.Connection:
        """
        Create and configure a new SQLite connection.
        
        Returns:
            sqlite3.Connection: Connection with WAL mode and foreign keys enabled.
        
        Raises:
            sqlite3.OperationalError: If the database cannot be opened.
        """
        try:
            if Database._conn is None:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30,
                    isolation_level="IMMEDIATE",
                    check_same_thread=False,
                )
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                Database._conn = conn
            return Database._conn
        except sqlite3.OperationalError as e:
            logger.error("SQLite error: %s", e)
            raise

    def _init_tables_and_pragmas(self) -> None:
        """
        Create database tables and enforce pragmas for WAL mode and foreign keys.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    title TEXT PRIMARY KEY,
                    text  TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    date_time      DATETIME NOT NULL,
                    query          TEXT     NOT NULL,
                    model          TEXT     NOT NULL,
                    chunk          TEXT,
                    expected_value REAL     NOT NULL,
                    entropy        REAL     NOT NULL,
                    PRIMARY KEY (date_time, query, chunk, model)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ensembles (
                    date_time      DATETIME NOT NULL,
                    query          TEXT     NOT NULL,
                    expected_value REAL     NOT NULL,
                    entropy        REAL     NOT NULL,
                    PRIMARY KEY (date_time, query)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    title     TEXT PRIMARY KEY
                              REFERENCES articles(title)
                              ON DELETE CASCADE
                              ON UPDATE CASCADE,
                    embedding BLOB,
                    model     TEXT    NOT NULL
                )
            """
            )
            conn.commit()

    def store_article(self, article_title: str, article_text: str) -> None:
        """
        Encode and store an article text and its embedding.
        
        Parameters:
            article_title (str): Title of the article.
            article_text (str): Full text of the article.
        
        Raises:
            ValueError: If encoding or database operations fail.
        """
        
        article_title = article_title.strip().replace('_', ' ')

        try:
            if self.model is None:
                logger.info("Loading embedding model…")
                self.model = SentenceTransformer(self.embedding_model_path)
            embedding_vector = self.model.encode(article_text[:MAX_EMBEDDING_SIZE])
            embedding_blob = pickle.dumps(embedding_vector)
        except Exception as e:
            logger.error("Error encoding article: %s", e)
            raise ValueError(f"Error encoding article: {e}") from e
        try:
            with Database._lock, self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO articles (title, text)
                    VALUES (?, ?)""",
                    (article_title, article_text),
                )
                cursor.execute(
                    """INSERT OR REPLACE INTO article_embeddings (title, embedding, model)
                    VALUES (?, ?, ?)""",
                    (article_title, embedding_blob, self.embedding_model_path),
                )
                conn.commit()
            logger.info("Stored article and embedding: %s", article_title)
        except sqlite3.Error as e:
            logger.error("Database error storing article: %s", e)
            raise ValueError(f"Database error storing article: {e}") from e
        except Exception as e:
            logger.error("Error storing article: %s", e)
            raise ValueError(f"Error storing article: {e}") from e

    def load_article(self, article_title: str) -> str | None:
        """
        Load the text of an article from the database, fetching from Wikipedia if necessary.
        
        Parameters:
            article_title (str): Title of the article to load.
        
        Returns:
            Optional[str]: Article text or None if not found.
        """
        with Database._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM articles WHERE title = ?", (article_title,))
            row = cursor.fetchone()

        if not row:
            logger.warning('Article "%s" not found in the database.', article_title)
            return None
        return row[0]

    def get_article(self, article_title: str) -> None:
        """
        Fetch an article from Wikipedia and store it in the database if it exists.
        
        Parameters:
            article_title (str): Title of the Wikipedia article.
        
        Raises:
            ValueError: If fetching or storing the article fails.
        """
        wiki = wikipediaapi.Wikipedia("Forecaster", "en")
        try:
            page = wiki.page(article_title)
            if not page.exists():
                logger.warning('Wikipedia page does not exist: "%s"', article_title)
                return
        except Exception as e:
            logger.error("Error fetching article: %s", e)
            raise ValueError(f"Error fetching article: {e}") from e
        
        try:
            self.store_article(article_title, page.text)
        except Exception as e:
            logger.error("Error storing article: %s", e)
            raise ValueError(f"Error storing article: {e}") from e

    def article_exists(self, article_title: str) -> bool:
        """
        Check if an article exists in the database.
        
        Parameters:
            article_title (str): Title to check.
        
        Returns:
            bool: True if the article exists, False otherwise.
        """
        with Database._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM articles WHERE title = ? COLLATE NOCASE",
                (article_title,),
            )
            return cursor.fetchone() is not None

    def list_articles(self) -> List[str]:
        """
        List all article titles stored in the database in alphabetical order.
        
        Returns:
            List[str]: Sorted list of article titles.
        """
        with Database._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT title FROM articles ORDER BY title")
            return [row[0] for row in cursor.fetchall()]

    def delete_article(self, article_title: str) -> None:
        """
        Delete an article and its embedding from the database.
        
        Parameters:
            article_title (str): Title of the article to delete.
        """
        with Database._lock, self._connect() as conn:
            conn.execute("DELETE FROM articles WHERE title = ?", (article_title,))
            conn.commit()
        logger.info("Deleted article '%s'", article_title)

    def store_forecast(
        self,
        date_time: dt.datetime,
        query: str,
        model: str,
        chunk: str,
        expected_value: float,
        entropy: float,
    ) -> None:
        """
        Store a single model forecast result in the database.
        
        Parameters:
            date_time (datetime): Timestamp of the forecast.
            query (str): Forecast query identifier.
            model (str): Model name.
            chunk (str): Text chunk used.
            expected_value (float): Forecasted value.
            entropy (float): Forecast entropy.
        """
        with Database._lock, self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO forecasts
                        (date_time, query, model, chunk, expected_value, entropy)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date_time, query, chunk, model)
                    DO UPDATE SET
                        expected_value = excluded.expected_value,
                        entropy        = excluded.entropy
                    """,
                    (date_time, query, model, chunk, expected_value, entropy),
                )
                conn.commit()
                logger.info(
                    "Stored forecast (%s) – %.0f%%, entropy %.3f",
                    model,
                    expected_value,
                    entropy,
                )
            except sqlite3.IntegrityError as e:
                logger.error("Integrity error storing forecast: %s", e)
            except Exception as e:
                logger.error("Error storing forecast: %s", e)

    def store_ensemble(
        self,
        date_time: dt.datetime,
        query: str,
        expected_value: float,
        entropy: float,
    ) -> None:
        """
        Store an ensemble forecast result in the database.
        
        Parameters:
            date_time (datetime): Timestamp of the ensemble.
            query (str): Forecast query identifier.
            expected_value (float): Combined forecast value.
            entropy (float): Combined entropy.
        """
        with Database._lock, self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO ensembles
                        (date_time, query, expected_value, entropy)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(date_time, query)
                    DO UPDATE SET
                        expected_value = excluded.expected_value,
                        entropy        = excluded.entropy
                    """,
                    (date_time, query, expected_value, entropy),
                )
                conn.commit()
                logger.info(
                    "Stored ensemble – %.0f%%, entropy %.3f", expected_value, entropy
                )
            except sqlite3.IntegrityError as e:
                logger.error("Integrity error storing ensemble: %s", e)

    def delete_table(self, table_name: str) -> None:
        """
        Drop a table from the database if it exists.
        
        Parameters:
            table_name (str): Name of the table to drop.
        """
        allowed = {"articles", "article_embeddings", "forecasts", "ensembles"}
        if table_name not in allowed:
            raise ValueError(f"Unsafe table name: {table_name!r}")
        with Database._lock, self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
        logger.info("Dropped table '%s'", table_name)

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load an entire table into a pandas DataFrame.
        
        Parameters:
            table_name (str): Name of the table to load.
        
        Returns:
            pd.DataFrame: Contents of the table.
        """
        allowed = {"articles", "article_embeddings", "forecasts", "ensembles"}
        if table_name not in allowed:
            raise ValueError(f"Unsafe table name: {table_name!r}")
        try:
            with Database._lock, self._connect() as conn:
                return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            logger.error("Error loading table '%s': %s", table_name, e)
            return pd.DataFrame()

    def get_article_embedding(self, article_title: str):
        """
        Retrieve a stored embedding for a given article.
        
        Parameters:
            article_title (str): Title of the article.
        
        Returns:
            Optional[np.ndarray]: Embedding array or None if not found or unloadable.
        """
        with Database._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT embedding FROM article_embeddings WHERE title = ?",
                (article_title,),
            )
            row = cursor.fetchone()
        if not row:
            logger.warning("No embedding found for article '%s'", article_title)
            return None
        try:
            return pickle.loads(row[0])
        except pickle.UnpicklingError as e:
            logger.error("UnpicklingError loading embedding for '%s': %s", article_title, e)
            return None
        except Exception as e:
            logger.error("Error loading embedding for '%s': %s", article_title, e)
            return None

    def clean_duplicates(self) -> None:
        """
        Remove duplicate articles by title or normalized text, keeping the earliest entry.
        """
        logger.info("Cleaning up duplicate articles by title or text.")
        with Database._lock, self._connect() as conn:
            cursor = conn.cursor()
            # Remove duplicate titles (case‑insensitive, normalizing underscores): keep the row with the smallest rowid for each title
            cursor.execute(
                """
                DELETE FROM articles
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM articles
                    GROUP BY LOWER(REPLACE(title, '_', ' '))
                );
                """
            )
            # Remove duplicate texts: keep the row with the smallest rowid for each normalised text
            rows = cursor.execute(
                "SELECT rowid, text FROM articles ORDER BY rowid DESC"
            ).fetchall()
            seen_hashes = set()
            for rowid, text in rows:
                normalized = " ".join(text.split())
                digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    cursor.execute("DELETE FROM articles WHERE rowid = ?", (rowid,))
                else:
                    seen_hashes.add(digest)
            # Remove embeddings for articles that no longer exist
            cursor.execute(
                """
                DELETE FROM article_embeddings
                WHERE title NOT IN (SELECT title FROM articles)
                """
            )
            conn.commit()
        logger.info("Cleaned up duplicate articles by title or text.")

    def get_top_new_articles(self, n: int = 10) -> List[str]:
        """
        Retrieve titles of the top n most viewed Wikipedia articles on the previous day.

        Parameters:
            n (int): Number of top articles to retrieve (default 10).

        Returns:
            List[str]: List of article titles.
        """
        endpoint = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access"
        yesterday = dt.datetime.now() - dt.timedelta(days=2)
        date_path = yesterday.strftime("%Y/%m/%d")
        url = f"{endpoint}/{date_path}"
        # Network request
        try:
            headers = {"User-Agent": "Forecaster"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Network error fetching top viewed articles: %s", e)
            return []
        # JSON decoding
        try:
            data = response.json()
        except ValueError as e:
            logger.warning("Error decoding JSON from Pageviews API: %s", e)
            return []
        # Extract titles
        try:
            items = data.get("items", [])
            if not items:
                return []
            articles = items[0].get("articles", [])
            titles = [entry.get("article") for entry in articles if entry.get("article")]
            result = titles[:n]
            logger.info("Retrieved %d new articles", len(result))
            return result
        except Exception as e:
            logger.warning("Unexpected data format in pageviews response: %s", e)
            return []


    def update_embeddings(self) -> None:
        """
        Recalculate and store embeddings for *all* articles using the model
        specified by ``self.embedding_model_path``.

        If an article already has an embedding, it will be replaced. Any errors
        during encoding or database operations are logged and the update
        continues with remaining articles.

        Raises
        ------
        ValueError
            If a fatal database error prevents updates from being committed.
        """
        logger.info("Starting full‑database embedding refresh with model '%s'.",
                    self.embedding_model_path)

        # Determine which articles need re-embedding
        try:
            with Database._lock, self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT title, model FROM article_embeddings")
                existing = dict(cursor.fetchall())
            all_titles = self.list_articles()
            titles_to_update = [t for t in all_titles if existing.get(t) != self.embedding_model_path]
            if not titles_to_update:
                logger.warning("No articles found – nothing to update.")
                return
        except Exception as e:
            logger.error("Failed to list articles for embedding refresh: %s", e)
            raise ValueError("Unable to list articles for embedding refresh.") from e

        try:
            if self.model is None:
                logger.info("Loading embedding model…")
                self.model = SentenceTransformer(self.embedding_model_path)
            model = self.model
        except Exception as e:
            logger.error(
                "Failed to load sentence-transformer model '%s': %s",
                self.embedding_model_path,
                e,
            )
            raise ValueError("Unable to load embedding model.") from e

        updated = 0
        for title in tqdm(titles_to_update, desc="Updating embeddings"):
            try:
                text = self.load_article(title)
                if not text:
                    logger.warning("Skipping embedding refresh – no text for '%s'.", title)
                    continue

                embedding_blob = pickle.dumps(model.encode(text[:MAX_EMBEDDING_SIZE]))

                with Database._lock, self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO article_embeddings (title, embedding, model)
                        VALUES (?, ?, ?)
                        """,
                        (title, embedding_blob, self.embedding_model_path),
                    )
                    conn.commit()
                updated += 1
            except Exception as e:
                logger.error("Error updating embedding for '%s': %s", title, e)
                continue  # continue updating remaining articles

        self.unload_embedding_model()

        logger.info("Embedding refresh complete – updated %d / %d articles.",
                    updated, len(titles_to_update))
