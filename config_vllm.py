"""Deployment-centric vLLM configuration.

Edit this file for AWS deployment (e.g., S3 model paths, parallelism,
Run:ai loader settings). Constructor args can still override these values at
runtime, but config should be the single source of truth for infra defaults.
"""

# Map friendly model keys to their storage locations. Swap to S3 URIs when
# deploying (e.g., "s3://your-bucket/Meta-Llama-3-70B"). Defaults remain HF
# hub IDs so local testing keeps working.
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
    # Number of GPUs to shard the model across (tensor parallel). If None, use all visible GPUs.
    "tensor_parallel_size": None,
    # Pipeline parallel partitions (use >1 for multi-node or very large models).
    "pipeline_parallel_size": 1,
    # Backend for distributed executor: "mp" (default) or "ray".
    "distributed_executor_backend": "mp",
    # Fraction of each GPU's memory vLLM may use (0-1).
    "gpu_memory_utilization": 0.90,
    # Host swap space in GB to offload KV cache (helps with large prompts/models).
    "swap_space_gb": None,
    # Optional hard cap on context length; if None, let vLLM infer from model.
    "max_model_len": None,
    # Loader format and extra tuning for Run:ai streaming from S3.
    "load_format": "runai_streamer",
    "model_loader_extra_config": {
        # Increase for faster S3 reads; tune per bandwidth/CPU. 16–64 common.
        "concurrency": 16,
        # Set True when using multi-GPU and high-throughput storage.
        "distributed": False,
        # Optional: cap host RAM used for streaming buffer (bytes).
        # "memory_limit": 5_368_709_120,
    },
}

