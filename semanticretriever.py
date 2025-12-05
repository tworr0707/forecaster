from typing import Optional, List, Tuple
import numpy as np
import traceback

from database import Database
from bedrock_embeddings import EmbeddingClient
from config import EMBEDDING_MODEL_ID
from logger import setup_logger

logger = setup_logger(__name__)


class SemanticRetriever:
    """Semantic retrieval backed by Bedrock embeddings + Aurora pgvector."""

    def __init__(
        self,
        embedding_model_id: str = EMBEDDING_MODEL_ID,
        get_new_articles: bool = False,
        db: Optional[Database] = None,
        embed_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self.db = db or Database()
        self.embedding_client = embed_client or EmbeddingClient(model_id=embedding_model_id)
        self.get_new_articles = get_new_articles

        if self.get_new_articles:
            # Admin path: fetch + embed new top-viewed articles
            try:
                titles = self.db.get_top_new_articles()
                logger.info("Fetched %d new article candidates", len(titles))
                self.refresh_articles(titles)
            except Exception as e:
                logger.warning("Skipping auto article refresh: %s", e)

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------
    def convert_query_to_embedding(self, query: str) -> np.ndarray:
        embeddings = self.embedding_client.embed_texts([query])
        if not embeddings:
            raise ValueError("Embedding client returned no vector for query.")
        return np.array(embeddings[0], dtype=float)

    def get_top_n_articles(self, query: str, top_n: int = 4) -> List[str]:
        query_emb = self.convert_query_to_embedding(query)
        rows: List[Tuple[str, str]] = self.db.search_articles_by_embedding(query_emb, top_n=top_n)
        texts = [text for _, text in rows]
        logger.info("Retrieved %d context articles from Aurora", len(texts))
        return texts

    def get_context(self, query: str, top_n: int = 4) -> str:
        texts = self.get_top_n_articles(query, top_n=top_n)
        if not texts:
            raise ValueError("No context articles retrieved; check embeddings or database content.")
        return "\n\n".join(texts)

    # ------------------------------------------------------------------
    # Admin refresh helpers
    # ------------------------------------------------------------------
    def refresh_articles(self, titles: List[str]) -> None:
        """Fetch text for given titles, embed, and store/update in DB.

        - New titles are fetched and inserted.
        - Existing titles are re-fetched and re-embedded to refresh content.
        """
        if not titles:
            return

        texts: List[str] = []
        kept_titles: List[str] = []

        for title in titles:
            try:
                # Attempt to fetch and store (handles both new and existing)
                txt = self.db.fetch_and_store_article(title)
                if not txt:
                    continue
                kept_titles.append(title)
                texts.append(txt)
            except Exception as e:
                logger.error("Failed to fetch/store article '%s': %s\n%s", title, e, traceback.format_exc())

        if not texts:
            logger.warning("No texts fetched for refresh; skipping embedding.")
            return

        try:
            embeddings = self.embedding_client.embed_texts(texts)
        except Exception as e:
            logger.error("Embedding failed during refresh: %s", e, exc_info=True)
            return

        if len(embeddings) != len(kept_titles):
            logger.warning("Mismatch in embeddings (%d) vs titles (%d); aborting store.", len(embeddings), len(kept_titles))
            return

        for title, emb, txt in zip(kept_titles, embeddings, texts):
            try:
                self.db.store_article(title, txt, embedding=np.array(emb, dtype=float), embedding_model=self.embedding_client.model_id)
            except Exception as e:
                logger.error("Failed to store embedded article '%s': %s", title, e, exc_info=True)
