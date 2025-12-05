"""Bedrock logic-generation client (Claude) with <thinking> stripping and stub mode."""

from __future__ import annotations

import hashlib
import json
import os
import datetime as dt
from typing import Optional

import boto3

from config import (
    AWS_REGION,
    LOGIC_MODEL_ID,
    USE_DB_STUB,
    BEDROCK_LOGIC_STUB,
    BEDROCK_LOGIC_STUB_FALLBACK,
    SATS_PATH,
)
from logger import setup_logger

logger = setup_logger(__name__)


def _load_sats_text() -> str:
    sats_path = SATS_PATH or os.path.join(os.path.dirname(__file__), "sats.txt")
    try:
        if os.path.exists(sats_path):
            with open(sats_path) as f:
                raw = f.read()
                return raw.replace('{', '{{').replace('}', '}}')
    except Exception as e:
        logger.warning("Error reading sats.txt: %s", e)
    return "Standard Analytical Techniques (SATs) guidelines not available."


SATS_TEXT = _load_sats_text()

SYSTEM_PROMPT = (
    "You are an expert AI superforecaster, trained to predict the future using a large body of knowledge and data. "
    f"The current date is {dt.datetime.now().strftime('%Y-%m-%d')}. "
    'Your responses should be exceptionally detailed and elaborate. Use as many tokens as possible to fully explore every nuance of your analysis. '
    'Do not hold back on any detail—explain every point thoroughly, provide multiple examples and perspectives, and ensure that your response is exhaustive. '
    'Brevity is not a concern; your goal is to generate a maximally comprehensive and in-depth explanation. '
    f'Approach each query methodically, ensuring your response is rigorous, objective, and unabridged. Specifically think through the SATs Guidelines:\n{SATS_TEXT}\n'
    'Critically evaluate assumptions, consider multiple perspectives, and clearly identify any gaps in the available information. '
    'Your analysis should be articulate, well-organised, and demonstrate a high standard of intellectual inquiry.'
    "Focus solely on the underlying logic required to answer the query. IMPORTANT: Do NOT attempt to answer the query or suggest a likelihood."
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

        
        user_prompt = f"Knowledge:\n{context_chunk}\n\nQuery: {query}\n\n"
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": None,
            "temperature": 0.3,
            "top_p": 0.9,
            "system": SYSTEM_PROMPT,
            "messages": [
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
            return msg
        except Exception as e:
            logger.error("Logic generation failed: %s", e, exc_info=True)
            if BEDROCK_LOGIC_STUB_FALLBACK:
                logger.warning("Falling back to stub logic after Bedrock error.")
                return self._stub_logic(query, context_chunk)
            raise

    @staticmethod
    def _stub_logic(query: str, context_chunk: str) -> str:
        context_chunk = context_chunk or ""
        digest = hashlib.sha256((query + context_chunk[:200]).encode("utf-8")).hexdigest()
        return f"Stub reasoning (offline): key factors hashed {digest[:12]}."
