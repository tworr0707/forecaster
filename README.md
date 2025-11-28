# Forecaster (vLLM edition)

Lightweight forecasting ensemble that runs locally or on multi-GPU AWS instances using vLLM. The `torch` package defaults to vLLM with Run:ai S3 streaming for large models.

---
## 0) Security / hygiene before going public
- No secrets tracked. Set `HF_TOKEN`, `AWS_*`, etc. via environment only.
- Past HF token existed in git history—scrub history before open-sourcing (e.g., `git filter-repo --replace-text <file-with-token>`).
- `.gitignore` excludes logs, DBs, caches, secrets.

---
## 1) Repository layout
- `torch/agents_vllm.py` – vLLM-based agent (forecast + optional logic model).
- `torch/ensemble_torch.py` – orchestrates agents and writes metrics to DB.
- `torch/config_vllm.py` – model URI map and vLLM/Run:ai defaults (TP/PP/backend, loader config).
- `torch/logger_torch.py` – root logger; set `LOG_FILE` env to redirect.
- `tests/test_agent_stub.py` – smoke tests with fake LLM/tokenizer.

---
## 2) Prereqs
- Python 3.10+.
- NVIDIA GPUs with CUDA; for AWS: p4d/p5 preferred. Install matching driver/CUDA runtime.
- Network access to S3 bucket that stores models.

---
## 3) Install (local dev)
```bash
cd torch
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Dependencies are pinned: `vllm[runai]` 0.10.x, `torch` 2.3–<2.5, `transformers` 4.42–<4.46, `sentence-transformers` 2.7–<2.8.

---
## 4) Configure models and parallelism
Edit `torch/config_vllm.py`:
- Set `FORECAST_MODEL_PATHS_VLLM` to HF IDs for local dev or S3 URIs for prod (e.g., `s3://your-bucket/Llama-3-70B`).
- Tune `VLLM_CONFIG`:
  - `tensor_parallel_size`: GPUs per node (often = GPU count).
  - `pipeline_parallel_size`: >1 only if spanning nodes or model still doesn’t fit.
  - `distributed_executor_backend`: `mp` (default) or `ray`.
  - `gpu_memory_utilization`: 0.90–0.95 for dedicated boxes.
  - `load_format`: keep `runai_streamer` for S3 streaming.
  - `model_loader_extra_config`: `concurrency` (16–64 typical), `distributed` (True for multi-node pulls), optional `memory_limit`.
  - `swap_space_gb`: enable KV offload if context/model is very large.

Env helpers:
```bash
export HF_TOKEN=your_token              # for gated HF models
export CUDA_VISIBLE_DEVICES=0,1,2,3     # select GPUs
export LOG_FILE=/var/log/forecaster.log # optional
```

---
## 5) Run locally (single node, in-process)
```bash
cd torch
python forecaster_torch.py
```
Uses vLLM agents with models defined in `config_vllm.py`. Forecasts and ensemble metrics stored via `database_torch.py` (SQLite by default).

---
## 6) AWS EC2 with vLLM + Run:ai S3 streaming (step-by-step)
### 6.1 Launch + IAM
- Instance: p4d/p5 (8x A100/H100) or bigger if needed.
- Attach an IAM role with at least:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::your-bucket/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::your-bucket"]
    }
  ]
}
```
- If in a private subnet, create an S3 Gateway VPC Endpoint to avoid public egress.

### 6.2 Prep the instance
```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y git tmux python3-pip
python3 -m pip install --upgrade pip
git clone https://github.com/tworr0707/forecaster.git
cd forecaster/torch
python3 -m pip install -r requirements.txt
```

### 6.3 Set environment for this session
```bash
export HF_TOKEN=your_hf_token               # only if model is gated on HF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust to GPUs you want to use
export LOG_FILE=/var/log/forecaster.log     # optional log path
```

### 6.4 Point models to S3 and tune parallelism
Edit `torch/config_vllm.py`:
- Change `FORECAST_MODEL_PATHS_VLLM` entries to your S3 URIs, e.g.:
  ` "llama-70B": "s3://your-bucket/llama-3-70b" `
- Set `tensor_parallel_size` = GPUs per node (e.g., 8 on p4d).
- Keep `pipeline_parallel_size` = 1 for single-node; raise for multi-node.
- Keep `load_format = "runai_streamer"`; raise `model_loader_extra_config["concurrency"]` (start 32–64).
- Consider `swap_space_gb` if prompts are very long.

### 6.5 Option A: In-process run (simplest)
```bash
cd torch
python forecaster_torch.py
```
This loads the model via Run:ai streamer directly in the Python process and writes forecasts to the local SQLite DB.

### 6.6 Option B: Run vLLM server (API mode)
Start server:
```bash
vllm serve \
  --model s3://your-bucket/llama-3-70b \
  --load-format runai_streamer \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 \
  --model-loader-extra-config '{"concurrency":48}'
```
Then point your client (not provided here) to the OpenAI-compatible endpoint on `http://<host>:8000/v1`.

### 6.7 Verify the load
- Check logs (`LOG_FILE` or console) for: TP/PP settings, load_format=runai_streamer, and S3 download progress.
- Run a tiny probe:
```bash
curl http://localhost:8000/health
```
(server mode) or observe forecast logs for token generation (in-process).

### 6.8 Troubleshooting
- Stuck at download: verify IAM role + bucket policy + S3 endpoint; try lowering/raising `concurrency`.
- OOM during load: lower `tensor_parallel_size` or raise `swap_space_gb`.
- NCCL init errors: set `NCCL_DEBUG=INFO` and ensure GPUs are visible; try `distributed_executor_backend=mp` (default) if Ray causes device mismatch.
- Slow first token: ensure bucket is in same region; consider warmup by generating a short prompt after start.

---
## 7) Logging and paths
- Default log file: `runpod_forecast_logs.log` in CWD; override with `LOG_FILE`.
- `.gitignore` excludes DBs/logs; avoid committing artifacts.

---
## 8) Testing
Run lightweight smoke tests (no real model downloads):
```bash
cd torch
pytest
```
Tests stub the LLM/tokenizer and ensure cache eviction + ensemble path work.

---
## 9) Performance tuning checklist
- Start with TP = number of GPUs on the node; raise `concurrency` for faster S3 streaming.
- Use `pipeline_parallel_size` >1 only for multi-node or ultra-large models.
- Keep `gpu_memory_utilization` at 0.90–0.95; add `swap_space_gb` if contexts are very long.
- Pin GPUs with `CUDA_VISIBLE_DEVICES` to isolate workloads.

---
## 10) Operational cautions
- Build script `build_pod.sh` installs system packages and logs into HF; run only in trusted, root-capable environments. It populates HF cache; for Run:ai streaming you still need models in S3.
- Ensure `HF_TOKEN` and any AWS creds are passed via environment/instance roles, never committed.
