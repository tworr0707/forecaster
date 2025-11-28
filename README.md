# Forecaster (vLLM edition)

Lightweight forecasting ensemble that runs locally or on multi‑GPU AWS instances using vLLM. The torch subpackage defaults to vLLM with Run:ai S3 streaming for large models.

## Repository layout
- `torch/agents_vllm.py` – vLLM-based agent (forecasts + optional logic model).
- `torch/ensemble_torch.py` – orchestrates multiple agents and stores results.
- `torch/config_vllm.py` – single source of truth for model paths and vLLM/Run:ai settings.

## Quick start (local single box)
```bash
cd torch
pip install -r requirements.txt
python forecaster_torch.py
```
Defaults load smaller HF models for local testing.

## Configure models and parallelism
Edit `torch/config_vllm.py`:
- Set `FORECAST_MODEL_PATHS_VLLM` to your model URIs (HF or S3).
- Tune `VLLM_CONFIG` (`tensor_parallel_size`, `pipeline_parallel_size`, `distributed_executor_backend`, `gpu_memory_utilization`, `load_format`, `model_loader_extra_config`).

## Deploying on AWS EC2 with vLLM + Run:ai S3 streaming
This repo is configured to run large models on multi-GPU AWS instances using vLLM and the Run:ai streamer to pull weights directly from S3.

### Prereqs
- Multi-GPU EC2 (e.g., p4d/p5) with CUDA/NCCL working.
- IAM role on the instance with read access to your S3 model bucket.
- Python 3.10+.

Install deps:
```bash
pip install -r requirements.txt
```
`vllm[runai]` is pinned to a stable 0.10.x release for Run:ai support.

### Configure model paths and loader
Edit `torch/config_vllm.py`:
- Set `FORECAST_MODEL_PATHS_VLLM` entries to your S3 URIs (e.g., `s3://your-bucket/Llama-3-70B`).
- Keep `VLLM_CONFIG.load_format` as `runai_streamer`.
- Tune `model_loader_extra_config` (`concurrency`, `distributed`, optional `memory_limit`).
- Adjust `tensor_parallel_size`, `pipeline_parallel_size`, and `distributed_executor_backend` to match your GPU topology.

### Running in-process (Python)
```bash
cd torch
python forecaster_torch.py
```
The ensemble uses `agents_vllm.Agent`, which loads via Run:ai from S3 per `config_vllm.py`.

### Running vLLM server (optional)
```bash
vllm serve \\
  --model s3://your-bucket/Llama-3-70B \\
  --load-format runai_streamer \\
  --tensor-parallel-size 8 \\
  --pipeline-parallel-size 1 \\
  --distributed-executor-backend mp \\
  --gpu-memory-utilization 0.90 \\
  --model-loader-extra-config '{\"concurrency\":32}'
```
Then point clients to the OpenAI-compatible endpoint (client wiring not included here).

### AWS + S3 notes
- Prefer IAM instance roles instead of static credentials. If needed, export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (avoid committing them).
- Keep S3 bucket in the same region as the instance.
- For very large models, enable `swap_space_gb` in `config_vllm.py` to allow KV offload.

### Performance tuning checklist
- `tensor_parallel_size`: typically number of GPUs on the node.
- `pipeline_parallel_size`: >1 only if the model still doesn’t fit and you span nodes.
- `gpu_memory_utilization`: 0.90–0.95 on dedicated boxes.
- `model_loader_extra_config.concurrency`: 16–64 depending on network/storage.
- Pin GPUs with `CUDA_VISIBLE_DEVICES` if needed.

### Logging & troubleshooting
- `agents_vllm` logs TP/PP/backend and loader settings at startup.
- If loads hang, check S3 IAM permissions/VPC endpoints; enable `NCCL_DEBUG=INFO` for multi-GPU issues.

## Notes
- Requirements pin `vllm[runai]` to 0.10.x for Run:ai streamer compatibility.
- Sensitive files like `runpod.json` are removed/ignored; keep credentials out of the repo.
*** End Patch
