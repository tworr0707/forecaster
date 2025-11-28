"""Unified configuration for embeddings, retriever, and vLLM/Run:ai.

Edit this file per deployment (e.g., S3 model paths, parallelism, loader
settings). Keep secrets in environment variables, not here.
"""

# Embedding / retriever
EMBEDDING_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
MAX_EMBEDDING_SIZE = 20_000
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
    "load_format": "runai_streamer",
    "model_loader_extra_config": {
        "concurrency": 16,                 # Tune for S3 bandwidth; 16–64 typical
        "distributed": False,              # True for multi-node streaming
        # "memory_limit": 5_368_709_120,   # Optional cap in bytes
    },
}

# Probability-cache bound for next_token_probs (per Agent forecast call)
MAX_LOGPROB_CACHE = 128  # set >=101 to cover all number tokens 0–100


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
    if MAX_LOGPROB_CACHE < 101:
        raise ValueError("MAX_LOGPROB_CACHE must be >=101 to cover numbers 0-100.")

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
