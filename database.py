"""Aurora-backed persistence layer for articles, embeddings, forecasts, and logic.

The Database class now targets Aurora PostgreSQL Serverless v2 with pgvector.
If no Aurora connection details are present (or USE_DB_STUB=true), an in-memory
stub backend is used so tests/dev can run without AWS.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import register_vector

from logger import setup_logger
from config import (
    AURORA_CONNECTION_STRING,
    AURORA_DB_HOST,
    AURORA_DB_NAME,
    AURORA_DB_PASSWORD,
    AURORA_DB_PORT,
    AURORA_DB_USER,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
    EMBEDDING_DIM,
    USE_DB_STUB,
)

logger = setup_logger(__name__)


@dataclass
class _StubStorage:
    articles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    forecasts: List[Dict[str, Any]] = field(default_factory=list)
    ensembles: List[Dict[str, Any]] = field(default_factory=list)
    logic_runs: List[Dict[str, Any]] = field(default_factory=list)


class Database:
    """Aurora/Postgres database wrapper with optional in-memory stub."""

    _lock = threading.RLock()

    def __init__(
        self,
        connection_string: Optional[str] = None,
        use_stub: Optional[bool] = None,
    ) -> None:
        self.connection_string = connection_string or self._build_connection_string()
        self.use_stub = use_stub if use_stub is not None else (USE_DB_STUB or not self.connection_string)
        self._engine: Optional[Engine] = None
        self._stub = _StubStorage()

        if self.use_stub:
            logger.warning("Database running in in-memory stub mode; no data will persist. Set Aurora env vars to enable Postgres.")
        else:
            self._engine = self._create_engine(self.connection_string)
            register_vector(self._engine)
            self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _build_connection_string(self) -> Optional[str]:
        if AURORA_CONNECTION_STRING:
            return AURORA_CONNECTION_STRING
        if all([AURORA_DB_USER, AURORA_DB_PASSWORD, AURORA_DB_HOST, AURORA_DB_NAME]):
            return (
                f"postgresql+psycopg2://{AURORA_DB_USER}:{AURORA_DB_PASSWORD}"
                f"@{AURORA_DB_HOST}:{AURORA_DB_PORT}/{AURORA_DB_NAME}"
            )
        return None

    def _create_engine(self, conn_str: str) -> Engine:
        logger.info("Creating Aurora engine with pool_size=%s, max_overflow=%s", DB_POOL_SIZE, DB_MAX_OVERFLOW)
        return create_engine(
            conn_str,
            poolclass=QueuePool,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_pre_ping=True,
        )

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        assert self._engine is not None
        ddl_articles = f"""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS articles (
                id              bigserial PRIMARY KEY,
                title           text UNIQUE NOT NULL,
                text            text NOT NULL,
                embedding       vector({EMBEDDING_DIM}),
                embedding_model text,
                created_at      timestamptz DEFAULT now(),
                updated_at      timestamptz DEFAULT now()
            );
        """
        ddl_forecasts = """
            CREATE TABLE IF NOT EXISTS forecasts (
                date_time      timestamptz NOT NULL,
                query          text        NOT NULL,
                model          text        NOT NULL,
                chunk          text,
                expected_value double precision NOT NULL,
                entropy        double precision NOT NULL,
                PRIMARY KEY (date_time, query, chunk, model)
            );
        """
        ddl_ensembles = """
            CREATE TABLE IF NOT EXISTS ensembles (
                date_time      timestamptz NOT NULL,
                query          text        NOT NULL,
                expected_value double precision NOT NULL,
                entropy        double precision NOT NULL,
                PRIMARY KEY (date_time, query)
            );
        """
        ddl_logic = """
            CREATE TABLE IF NOT EXISTS logic_runs (
                id           bigserial PRIMARY KEY,
                date_time    timestamptz NOT NULL,
                query        text        NOT NULL,
                chunk_index  int         NOT NULL,
                context_hash text        NOT NULL,
                model        text        NOT NULL,
                logic_text   text        NOT NULL
            );
        """
        with self._engine.begin() as conn:
            conn.execute(text(ddl_articles))
            conn.execute(text(ddl_forecasts))
            conn.execute(text(ddl_ensembles))
            conn.execute(text(ddl_logic))
        logger.info("Aurora schema ensured (articles/forecasts/ensembles/logic_runs).")

    # ------------------------------------------------------------------
    # Article APIs
    # ------------------------------------------------------------------
    def store_article(
        self,
        title: str,
        text_value: str,
        embedding: Optional[np.ndarray] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        title = title.strip().replace("_", " ")
        emb_list: Optional[List[float]] = None
        if embedding is not None:
            emb_arr = np.asarray(embedding, dtype=float)
            if emb_arr.shape[-1] != EMBEDDING_DIM:
                raise ValueError(f"Embedding dimension {emb_arr.shape[-1]} does not match configured EMBEDDING_DIM={EMBEDDING_DIM}.")
            emb_list = emb_arr.tolist()

        if self.use_stub:
            with Database._lock:
                self._stub.articles[title] = {
                    "text": text_value,
                    "embedding": emb_list,
                    "embedding_model": embedding_model,
                    "updated_at": dt.datetime.utcnow(),
                }
            return

        assert self._engine is not None
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO articles (title, text, embedding, embedding_model)
                    VALUES (:title, :text_value, :embedding, :embedding_model)
                    ON CONFLICT (title) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = now()
                    """
                ),
                {
                    "title": title,
                    "text_value": text_value,
                    "embedding": emb_list,
                    "embedding_model": embedding_model,
                },
            )

    def load_article(self, title: str) -> Optional[str]:
        if self.use_stub:
            return self._stub.articles.get(title, {}).get("text")

        assert self._engine is not None
        with self._engine.begin() as conn:
            row = conn.execute(text("SELECT text FROM articles WHERE title = :title"), {"title": title}).fetchone()
            return row[0] if row else None

    def article_exists(self, title: str) -> bool:
        if self.use_stub:
            return title in self._stub.articles

        assert self._engine is not None
        with self._engine.begin() as conn:
            row = conn.execute(text("SELECT 1 FROM articles WHERE title = :title"), {"title": title}).fetchone()
            return row is not None

    def list_articles(self) -> List[str]:
        if self.use_stub:
            return sorted(self._stub.articles.keys())

        assert self._engine is not None
        with self._engine.begin() as conn:
            rows = conn.execute(text("SELECT title FROM articles ORDER BY title"))
            return [r[0] for r in rows.fetchall()]

    def get_article_embedding(self, title: str) -> Optional[np.ndarray]:
        if self.use_stub:
            emb = self._stub.articles.get(title, {}).get("embedding")
            return np.array(emb) if emb is not None else None

        assert self._engine is not None
        with self._engine.begin() as conn:
            row = conn.execute(text("SELECT embedding FROM articles WHERE title = :title"), {"title": title}).fetchone()
            if not row or row[0] is None:
                return None
            return np.array(row[0], dtype=float)

    def search_articles_by_embedding(self, embedding: np.ndarray, top_n: int = 4) -> List[Tuple[str, str]]:
        emb_arr = np.asarray(embedding, dtype=float)
        if emb_arr.shape[-1] != EMBEDDING_DIM:
            raise ValueError(f"Query embedding dimension {emb_arr.shape[-1]} does not match EMBEDDING_DIM={EMBEDDING_DIM}.")

        if self.use_stub:
            results: List[Tuple[str, float]] = []
            for title, payload in self._stub.articles.items():
                doc_emb = payload.get("embedding")
                if doc_emb is None:
                    continue
                sim = self._cosine_similarity(emb_arr, np.array(doc_emb))
                results.append((title, sim))
            ranked = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
            return [(t, self._stub.articles[t]["text"]) for t, _ in ranked]

        assert self._engine is not None
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT title, text
                    FROM articles
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> :query_vec
                    LIMIT :top_n
                    """
                ),
                {"query_vec": emb_arr.tolist(), "top_n": top_n},
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------------
    # Forecast / ensemble / logic persistence
    # ------------------------------------------------------------------
    def store_forecast(
        self,
        date_time: dt.datetime,
        query: str,
        model: str,
        chunk: Optional[str],
        expected_value: float,
        entropy: float,
    ) -> None:
        record = {
            "date_time": date_time,
            "query": query,
            "model": model,
            "chunk": chunk,
            "expected_value": expected_value,
            "entropy": entropy,
        }
        if self.use_stub:
            with Database._lock:
                self._stub.forecasts.append(record)
            return

        assert self._engine is not None
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO forecasts (date_time, query, model, chunk, expected_value, entropy)
                    VALUES (:date_time, :query, :model, :chunk, :expected_value, :entropy)
                    ON CONFLICT (date_time, query, chunk, model) DO UPDATE SET
                        expected_value = EXCLUDED.expected_value,
                        entropy        = EXCLUDED.entropy
                    """
                ),
                record,
            )

    def store_ensemble(
        self,
        date_time: dt.datetime,
        query: str,
        expected_value: float,
        entropy: float,
    ) -> None:
        record = {
            "date_time": date_time,
            "query": query,
            "expected_value": expected_value,
            "entropy": entropy,
        }
        if self.use_stub:
            with Database._lock:
                self._stub.ensembles.append(record)
            return

        assert self._engine is not None
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ensembles (date_time, query, expected_value, entropy)
                    VALUES (:date_time, :query, :expected_value, :entropy)
                    ON CONFLICT (date_time, query) DO UPDATE SET
                        expected_value = EXCLUDED.expected_value,
                        entropy        = EXCLUDED.entropy
                    """
                ),
                record,
            )

    def store_logic(
        self,
        date_time: dt.datetime,
        query: str,
        chunk_index: int,
        context_hash: str,
        model: str,
        logic_text: str,
    ) -> None:
        record = {
            "date_time": date_time,
            "query": query,
            "chunk_index": chunk_index,
            "context_hash": context_hash,
            "model": model,
            "logic_text": logic_text,
        }
        if self.use_stub:
            with Database._lock:
                self._stub.logic_runs.append(record)
            return

        assert self._engine is not None
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO logic_runs (date_time, query, chunk_index, context_hash, model, logic_text)
                    VALUES (:date_time, :query, :chunk_index, :context_hash, :model, :logic_text)
                    """
                ),
                record,
            )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def load_table(self, table_name: str) -> pd.DataFrame:
        allowed = {"articles", "forecasts", "ensembles", "logic_runs"}
        if table_name not in allowed:
            raise ValueError(f"Unsafe table name: {table_name!r}")

        if self.use_stub:
            if table_name == "articles":
                rows = []
                for title, payload in self._stub.articles.items():
                    rows.append({"title": title, **payload})
                return pd.DataFrame(rows)
            return pd.DataFrame(getattr(self._stub, table_name, []))

        assert self._engine is not None
        with self._engine.begin() as conn:
            return pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)

    def clean_duplicates(self) -> None:
        """No-op in Aurora; uniqueness enforced via constraints."""
        if self.use_stub:
            return
        # With UNIQUE(title) duplicates cannot exist; nothing to do.

    def get_top_new_articles(self, n: int = 10) -> List[str]:
        endpoint = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access"
        yesterday = dt.datetime.now() - dt.timedelta(days=2)
        date_path = yesterday.strftime("%Y/%m/%d")
        url = f"{endpoint}/{date_path}"
        try:
            headers = {"User-Agent": "Forecaster"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            if not items:
                return []
            articles = items[0].get("articles", [])
            titles = [entry.get("article") for entry in articles if entry.get("article")]
            return titles[:n]
        except Exception as e:
            logger.warning("Failed to fetch top new articles: %s", e)
            return []

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / denom) if denom else 0.0
