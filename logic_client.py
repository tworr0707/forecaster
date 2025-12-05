"""Bedrock logic-generation client (Claude) with <thinking> stripping and stub mode."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

import boto3

from config import (
    AWS_REGION,
    LOGIC_MODEL_ID,
    USE_DB_STUB,
    BEDROCK_LOGIC_STUB,
    BEDROCK_LOGIC_STUB_FALLBACK,
)
from logger import setup_logger

logger = setup_logger(__name__)

SYSTEM_PROMPT = (
    "You are an analytic reasoning assistant for forecasting. "
    "Provide concise but rigorous reasoning that helps a forecaster, "
    "without giving probabilities or final answers."
)


class LogicClient:
    def __init__(self, model_id: str = LOGIC_MODEL_ID, use_stub: Optional[bool] = None) -> None:
        self.model_id = model_id
        self.stub_mode = use_stub if use_stub is not None else USE_DB_STUB or BEDROCK_LOGIC_STUB
        if not self.stub_mode:
            self.client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        else:
            self.client = None
            logger.warning("LogicClient running in stub mode; returning deterministic placeholder logic.")

    def generate_logic(self, query: str, context_chunk: str) -> str:
        if self.stub_mode:
            return self._stub_logic(query, context_chunk)

        user_prompt = f"Knowledge (chunk):\n{context_chunk}\n\nQuery: {query}\n\nExplain relevant causal/forecasting reasoning only."
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
        }

        try:
            body = json.dumps(payload)
            response = self.client.invoke_model(modelId=self.model_id, body=body)
            data = json.loads(response["body"].read())
            msg = data.get("output_text")
            if not msg:
                # Anthropic format returns messages
                outputs = data.get("content") or data.get("messages") or []
                if outputs and isinstance(outputs, list):
                    # handle anthropic content blocks
                    if "content" in outputs[0]:
                        blocks = outputs[0].get("content", [])
                        texts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
                        msg = "\n".join(filter(None, texts))
                    elif "text" in outputs[0]:
                        msg = outputs[0].get("text")
            if not msg:
                raise ValueError("No logic text returned from Bedrock response")
            return self._strip_thinking(msg)
        except Exception as e:
            logger.error("Logic generation failed: %s", e, exc_info=True)
            if BEDROCK_LOGIC_STUB_FALLBACK:
                logger.warning("Falling back to stub logic after Bedrock error.")
                return self._stub_logic(query, context_chunk)
            raise

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _stub_logic(query: str, context_chunk: str) -> str:
        digest = hashlib.sha256((query + context_chunk[:200]).encode("utf-8")).hexdigest()
        return f"Stub reasoning (offline): key factors hashed {digest[:12]}."
