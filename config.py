"""Unified configuration for embeddings, retriever, and vLLM/Run:ai.

Edit this file per deployment (e.g., S3 model paths, parallelism, loader
settings). Keep secrets in environment variables, not here.
"""

import os

# ---------------------------------------------------------------------------
# AWS / Aurora / Bedrock (all secrets via env vars)
# ---------------------------------------------------------------------------
# AWS region for Bedrock/S3/RDS calls
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")

# Aurora connection: prefer full URI; otherwise supply parts via env.
# Example URI: postgresql+psycopg2://user:pass@host:5432/dbname
AURORA_CONNECTION_STRING = os.getenv("AURORA_CONNECTION_STRING")
AURORA_DB_USER = os.getenv("AURORA_DB_USER", "CHANGE_ME_USER")  # set to Aurora user
AURORA_DB_PASSWORD = os.getenv("AURORA_DB_PASSWORD")  # store in Secrets Manager/SSM
AURORA_SECRET_ARN = os.getenv("AURORA_SECRET_ARN")    # optional: fetched at runtime if set
AURORA_DB_HOST = os.getenv("AURORA_DB_HOST", "aurora-cluster.endpoint")
AURORA_DB_PORT = os.getenv("AURORA_DB_PORT", "5432")
AURORA_DB_NAME = os.getenv("AURORA_DB_NAME", "forecaster")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))            # SQLAlchemy core pool
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "5"))      # extra conns above pool
USE_DB_STUB = os.getenv("USE_DB_STUB", "false").lower() == "true"  # in-memory dev mode

# Bedrock model ids (non-secret; set per environment)
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "cohere.embed-english-v3")
LOGIC_MODEL_ID = os.getenv("LOGIC_MODEL_ID", "anthropic.claude-3-7-sonnet-20241022")

# Bedrock stub toggles for offline/dev
BEDROCK_STUB = os.getenv("BEDROCK_STUB", "false").lower() == "true"
BEDROCK_STUB_FALLBACK = os.getenv("BEDROCK_STUB_FALLBACK", "false").lower() == "true"
BEDROCK_LOGIC_STUB = os.getenv("BEDROCK_LOGIC_STUB", "false").lower() == "true"
BEDROCK_LOGIC_STUB_FALLBACK = os.getenv("BEDROCK_LOGIC_STUB_FALLBACK", "false").lower() == "true"

# Embedding settings: dimension must match Bedrock embedding model
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))   # per Bedrock batch call

# Logic: toggle and max concurrent chunk calls to Bedrock logic model
USE_LOGIC = os.getenv("USE_LOGIC", "false")
LOGIC_MAX_WORKERS = int(os.getenv("LOGIC_MAX_WORKERS", "4"))

# Artefacts / plots S3 bucket and key prefix
PLOTS_BUCKET = os.getenv("PLOTS_BUCKET", "forecaster-plots-dev")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")                 # used in S3 key prefix

# Embedding / retriever (legacy local ST models; used only in fallback paths)
EMBEDDING_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
MAX_EMBEDDING_SIZE = 20_000  # chars to include when embedding article text locally
CROSS_ENCODER_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L12-v2"

# Legacy HF forecasting (kept for reference)
FORECAST_MODEL_PATHS = {
    "llama-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4-8b": "mlx-community/phi-4-8bit",
    "phi4": "microsoft/phi-4",
    "llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-8B": "meta-llama/llama-3.1-8B-Instruct",
    "llama-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-405B": "meta-llama/Llama-3.1-405B-Instruct",
    "nemotron": "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "mistral": "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    "qwen-32B": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "gemma3": "mlx-community/gemma-3-27b-it-qat-4bit",
    "llama-1B-base": "meta-llama/Llama-3.2-1B",
    "gpt-oss": "lmstudio-community/gpt-oss-20b-MLX-8bit",
}
DEFAULT_MODEL = "llama-3B"
DEFAULT_LOGIC_MODEL_PATH = "mlx-community/QwQ-32B-4bit"

# vLLM / Run:ai configuration (primary path)
FORECAST_MODEL_PATHS_VLLM = {
    "llama-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4": "microsoft/phi-4",
    "llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-405B": "meta-llama/Llama-3.1-405B-Instruct",
    "nemotron": "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
}

DEFAULT_MODEL_VLLM = "llama-3B"

VLLM_CONFIG = {
    "tensor_parallel_size": None,          # GPUs to shard across (TP); None = all visible
    "pipeline_parallel_size": 1,           # >1 only for multi-node or very large models
    "distributed_executor_backend": "mp",  # "mp" or "ray"
    "gpu_memory_utilization": 0.90,
    "swap_space_gb": None,                 # Enable for KV offload if needed
    "max_model_len": None,                 # Let vLLM infer if None
    "load_format": None,                   # set to "runai_streamer" only in Run:ai/S3 environments
    "model_loader_extra_config": {
        "concurrency": 16,                 # Tune for S3 bandwidth; 16–64 typical
        "distributed": False,              # True for multi-node streaming
        # "memory_limit": 5_368_709_120,   # Optional cap in bytes
    },
}

# vLLM logprob extraction settings
# NUMBER_LOGPROB_TOP_K: how many top logprobs to request per step; missing tokens get a tiny floor prob
NUMBER_LOGPROB_TOP_K = 100_000

def validate_config():
    """Basic validation to fail fast with actionable errors."""
    tp = VLLM_CONFIG.get("tensor_parallel_size") or 0
    if tp < 0:
        raise ValueError("tensor_parallel_size must be >=0 or None")
    pp = VLLM_CONFIG.get("pipeline_parallel_size") or 0
    if pp < 1:
        raise ValueError("pipeline_parallel_size must be >=1")
    gmu = VLLM_CONFIG.get("gpu_memory_utilization", 0.9)
    if not (0.0 < gmu <= 0.99):
        raise ValueError("gpu_memory_utilization must be in (0,1].")
    mlc = VLLM_CONFIG.get("model_loader_extra_config", {})
    if mlc and not isinstance(mlc, dict):
        raise ValueError("model_loader_extra_config must be a dict.")
    if isinstance(mlc, dict):
        conc = mlc.get("concurrency", 0)
        if conc and conc < 1:
            raise ValueError("model_loader_extra_config.concurrency must be >=1.")
    if NUMBER_LOGPROB_TOP_K < 101:
        raise ValueError("NUMBER_LOGPROB_TOP_K must be at least 101 to cover numbers 0–100 with minimal flooring.")

    if VLLM_CONFIG.get("load_format") == "runai_streamer":
        try:
            import importlib
            if importlib.util.find_spec("runai_model_streamer") is None:
                raise ImportError
        except ImportError:
            raise ImportError("runai_model_streamer is required for load_format='runai_streamer'. Install vllm[runai] and runai-model-streamer.")
        # basic URI check for S3 targets
        for name, uri in FORECAST_MODEL_PATHS_VLLM.items():
            if uri.startswith("s3://"):
                continue
            if "://" in uri and not uri.startswith("http"):
                raise ValueError(f"Model path for {name} should be S3/HF/local; got {uri}")

    if EMBEDDING_DIM <= 0:
        raise ValueError("EMBEDDING_DIM must be positive.")
    if LOGIC_MAX_WORKERS < 1:
        raise ValueError("LOGIC_MAX_WORKERS must be >=1.")
