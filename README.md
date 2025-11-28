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
## 6) AWS EC2 with vLLM + Run:ai S3 streaming
1) Spin up p4d/p5 (or similar) with IAM role granting S3 read.
2) VPC: add S3 gateway endpoint if private subnet.
3) Configure `torch/config_vllm.py` model URIs to S3.
4) Install deps on the instance as in section 3.
5) Run:
```bash
cd torch
python forecaster_torch.py
```
or run vLLM server instead of in-process:
```bash
vllm serve \
  --model s3://your-bucket/Llama-3-70B \
  --load-format runai_streamer \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.90 \
  --model-loader-extra-config '{"concurrency":32}'
```
Point clients to the OpenAI-compatible endpoint (client wiring not included here).

IAM policy minimum (attach to instance role):
- `s3:GetObject` on `arn:aws:s3:::your-bucket/*`
- `s3:ListBucket` on `arn:aws:s3:::your-bucket`

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

*** End Patch
