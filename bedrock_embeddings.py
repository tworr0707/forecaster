"""Bedrock embedding client (Cohere via bedrock-runtime) with stub fallback."""

from __future__ import annotations

import hashlib
import json
import os
from typing import List

import boto3

from config import (
    AWS_REGION,
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    USE_DB_STUB,
    BEDROCK_STUB,
    BEDROCK_STUB_FALLBACK,
)
from logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingClient:
    def __init__(self, model_id: str = EMBEDDING_MODEL_ID, use_stub: bool | None = None):
        self.model_id = model_id
        self.stub_mode = use_stub if use_stub is not None else USE_DB_STUB or BEDROCK_STUB
        if not self.stub_mode:
            self.client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        else:
            self.client = None
            logger.warning("EmbeddingClient running in stub mode; embeddings are deterministic hashes, not Bedrock results.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.stub_mode:
            return [self._hash_to_unit_vector(t) for t in texts]

        max_batch = EMBED_BATCH_SIZE
        results: List[List[float]] = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i : i + max_batch]
            payload = {
                "texts": batch,
                "input_type": "search_document",
            }
            try:
                body = json.dumps(payload)
                response = self.client.invoke_model(modelId=self.model_id, body=body)
                output = json.loads(response["body"].read())
                embeddings = output.get("embeddings") or output.get("embedding") or []
                if not embeddings:
                    raise ValueError("No embeddings returned from Bedrock")
                results.extend(embeddings)
            except Exception as e:
                logger.error("Bedrock embedding call failed: %s", e, exc_info=True)
                if BEDROCK_STUB_FALLBACK:
                    logger.warning("Falling back to stub embeddings after Bedrock error...")
                    results.extend([self._hash_to_unit_vector(t) for t in batch])
                    continue
                raise
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hash_to_unit_vector(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding for offline/test mode."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat digest to fill EMBEDDING_DIM then normalise
        raw = bytearray()
        while len(raw) < EMBEDDING_DIM:
            raw.extend(digest)
        raw = raw[:EMBEDDING_DIM]
        vec = [((b / 255.0) * 2 - 1) for b in raw]
        # L2 normalise
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
