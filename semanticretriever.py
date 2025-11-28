from typing import Optional, List, Tuple, Set, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch
from tqdm import tqdm
import traceback

from database import Database
from config import EMBEDDING_MODEL_PATH, CROSS_ENCODER_MODEL_PATH
from logger import setup_logger

logger = setup_logger(__name__)


class SemanticRetriever:
    """Semantic retrieval backed by dense embeddings and optional cross-encoder re-ranking."""

    def __init__(
        self,
        embedding_model_path: str = EMBEDDING_MODEL_PATH,
        get_new_articles: bool = False,
        cross_encoder_model_path: str = CROSS_ENCODER_MODEL_PATH,
    ) -> None:
        self.db = Database()
        self.embedding_model_path = embedding_model_path
        self.cross_encoder_model_path = cross_encoder_model_path

        if get_new_articles:
            self._ensure_latest_articles()

        logger.info("SemanticRetriever initialised.")

    # ------------------------------------------------------------------
    # Article maintenance helpers
    # ------------------------------------------------------------------
    def _ensure_latest_articles(self) -> None:
        try:
            new_arts: List[str] = self.db.get_top_new_articles()
            for art in tqdm(new_arts, desc="Fetching embeddings for new articles"):
                try:
                    self.db.get_article(art)
                except Exception as e:
                    logger.error("Failed to get article '%s': %s\n%s", art, e, traceback.format_exc())

            try:
                self.db.clean_duplicates()
            except Exception as e:
                logger.error("Failed to clean duplicates: %s\n%s", e, traceback.format_exc())
        except Exception as e:
            logger.error("Failed to ensure latest articles: %s\n%s", e, traceback.format_exc())
            # Surface a clear signal to callers
            raise RuntimeError("Retriever failed to refresh articles; check DB/network logs.") from e

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def convert_query_to_embedding(self, query: str) -> np.ndarray:
        """Encode the query using the configured sentence-transformer model."""
        model: Optional[SentenceTransformer] = None
        try:
            logger.info("Converting query to embedding: %s", query)
            logger.info("Loading embedding model…")
            model = SentenceTransformer(self.embedding_model_path)
            return model.encode(query)
        except Exception as e:
            logger.error("Error converting query '%s' to embedding: %s\n%s", query, e, traceback.format_exc())
            raise ValueError(f"Error converting query to embedding for query {query}: {e}") from e
        finally:
            logger.info("Unloading embedding model…")
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def get_articles_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Return every stored article embedding, generating missing ones on demand."""
        logger.info("Fetching stored embeddings for all articles…")
        try:
            titles: List[str] = self.db.list_articles()
            pairs: List[Tuple[str, np.ndarray]] = []
            for title in titles:
                emb: Optional[np.ndarray] = None
                try:
                    emb = self.db.get_article_embedding(title)
                except Exception as e:
                    logger.error("Error getting embedding for '%s': %s\n%s", title, e, traceback.format_exc())
                    continue
                if emb is None:
                    logger.info("Embedding missing for article '%s', generating now.", title)
                    try:
                        self.db.get_article(title)
                    except Exception as e:
                        logger.error("Error fetching article '%s': %s\n%s", title, e, traceback.format_exc())
                        continue

                    text: Optional[str] = None
                    try:
                        text = self.db.load_article(title)
                    except Exception as e:
                        logger.error("Error loading article '%s': %s\n%s", title, e, traceback.format_exc())
                        continue
                    if text:
                        try:
                            emb = self.db.get_article_embedding(title)
                        except Exception as e:
                            logger.error("Error retrieving embedding for '%s': %s\n%s", title, e, traceback.format_exc())
                            continue
                        if emb is None:
                            logger.warning("Failed to generate embedding for article '%s'", title)
                            continue
                    else:
                        logger.warning("Article text missing for '%s'; cannot generate embedding.", title)
                        continue
                pairs.append((title, emb))
            return pairs
        except Exception as e:
            logger.error("Error retrieving article embeddings: %s\n%s", e, traceback.format_exc())
            raise RuntimeError("Retriever failed to fetch embeddings; check DB/network logs.") from e

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------
    def get_context(self, query: str, top_n: int = 4) -> str:
        """Concatenate the top-N article texts for the supplied query."""
        logger.info("Retrieving context for query: %s", query)
        try:
            top_texts = self.get_top_n_articles(query, top_n=top_n)
            logger.info("Retrieved %d articles for query.", len(top_texts))
            return "\n\n".join(top_texts)
        except Exception as e:
            logger.error("Failed to get context for query '%s': %s\n%s", query, e, traceback.format_exc())
            raise ValueError(f"Failed to get context for query {query}: {e}") from e
        finally:
            self.db.unload_embedding_model()

    def _rerank_with_cross_encoder(
        self,
        query: str,
        titles: List[str],
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        """Re-rank candidates with the configured cross-encoder."""
        model: Optional[CrossEncoder] = None
        try:
            logger.info("Loading cross-encoder model '%s'…", self.cross_encoder_model_path)
            model = CrossEncoder(self.cross_encoder_model_path)
            pairs: List[Tuple[str, str]] = []
            for t in titles:
                txt = ""
                try:
                    txt = self.db.load_article(t) or ""
                except Exception as e:
                    logger.warning("Could not load article '%s' for re-ranking: %s\n%s", t, e, traceback.format_exc())
                pairs.append((query, txt))
            scores = model.predict(pairs, batch_size=batch_size).tolist()
            scores_arr = np.array(scores)
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            norm_scores = (exp_scores / exp_scores.sum()).tolist()
            ranked = sorted(zip(titles, norm_scores), key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error("Cross-encoder error during re-ranking: %s\n%s", e, traceback.format_exc())
            ranked = [(t, 0.0) for t in titles]
        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return ranked

    def get_top_n_articles(
        self,
        query: str,
        top_n: int = 4,
        use_cross_encoder: bool = False,
        pool_size: int = 100,
    ) -> List[str]:
        """Return the texts of the top-N most similar articles."""
        try:
            query_emb = self.convert_query_to_embedding(query)
            pairs = self.get_articles_embeddings()
            similarities: List[Tuple[str, float]] = []
            for title, emb in pairs:
                try:
                    sim = self.calc_norm(query_emb, emb)
                except Exception as e:
                    logger.warning("Skipping article '%s' due to similarity error: %s\n%s", title, e, traceback.format_exc())
                    continue
                similarities.append((title, sim))

            ranked = sorted(similarities, key=lambda x: x[1], reverse=True)

            if use_cross_encoder and ranked:
                cand_titles = [t for t, _ in ranked[:pool_size]]
                ranked = self._rerank_with_cross_encoder(query, cand_titles)

            unique_texts: List[str] = []
            seen: Set[str] = set()
            for title, _ in ranked:
                try:
                    self.db.get_article(title)
                    text = self.db.load_article(title)
                except Exception as e:
                    logger.error("Error loading article '%s': %s\n%s", title, e, traceback.format_exc())
                    continue
                if text and text not in seen:
                    unique_texts.append(text)
                    seen.add(text)
                    if len(unique_texts) >= top_n:
                        break

            return unique_texts
        except Exception as e:
            logger.error("Error getting top %d articles: %s\n%s", top_n, e, traceback.format_exc())
            raise ValueError(f"Error getting top {top_n} articles: {e}") from e

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    def calc_norm(self, vec_1: np.ndarray, vec_2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        vec1 = np.array(vec_1)
        vec2 = np.array(vec_2)
        norm_val = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2) / norm_val) if norm_val else 0.0
