# Forecaster (vLLM edition)

Forecaster is a forecasting engine that uses large language models (LLMs) to estimate the probability (0–100%) of future events. It:
- Uses **vLLM** for high-throughput inference on single or multi‑GPU machines.
- Supports **Run:ai S3 streaming** so large models can be stored in S3 and streamed into vLLM.
- Uses a **retrieval component** to pull contextual information (e.g. wiki articles) and stores everything in a local SQLite database.
- Produces plots summarising ensemble forecasts over time.

This README explains what the project does and gives step‑by‑step instructions for local and AWS usage.

---
## 0. Security & hygiene
- No secrets are tracked in the repo. Set `HF_TOKEN`, `AWS_*`, etc. via environment variables only.
- `.gitignore` excludes logs, databases, caches, and secrets.

---
## 1. Repository layout (root)
- `forecaster.py` – main entrypoint; runs a set of queries through the ensemble and generates plots.
- `ensemble.py` – orchestrates vLLM agents, aggregates results, and writes metrics to SQLite via `database.py`.
- `agents_vllm.py` – main vLLM-based agent (forecasts + optional “logic” / chain‑of‑thought model).
- `agents.py` – **legacy** Hugging Face transformers path (deprecated; kept for reference only).
- `semanticretriever.py` – fetches articles and embeddings to provide contextual knowledge.
- `database.py` – SQLite-backed storage for articles, embeddings, and forecast results.
- `config.py` – unified configuration for embedding models, HF defaults, vLLM/Run:ai settings, and cache limits.
- `logger.py` – configures root logging (file + console); honours `LOG_FILE` and `LOG_DIR`.
- `build_pod.sh` – optional setup helper script for a GPU VM/container (installs system & Python deps and downloads HF models).
- `tests/test_agent_stub.py` – smoke tests using fake LLM/tokeniser to validate core glue code.

---
## 2. Requirements
- **Python**: 3.10+.
- **GPUs**: NVIDIA GPUs with CUDA. On AWS, p4d/p5 (8× A100/H100) work well.
- **CUDA / drivers**: Ensure your GPU driver and CUDA runtime match the torch/vLLM wheels.
  - Tested combo: `torch` 2.3.x + CUDA 12.1 + `vllm[runai]` 0.10.x.
- **Network**:
  - Access to the **Hugging Face Hub** (for HF IDs) and/or
  - Access to an **S3 bucket** containing your model weights (for Run:ai streaming).

---
## 3. Installation (local dev, from repo root)

From the repo root:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies:
- `vllm[runai]` (pinned to a 0.10.x release).
- `torch` 2.3–<2.5.
- `transformers` 4.42–<4.46.
- `sentence-transformers` 2.7–<2.8.
- `runai-model-streamer` – required when `VLLM_CONFIG["load_format"] == "runai_streamer"`.

If you hit CUDA or wheel errors, explicitly install the torch build that matches your CUDA driver (see PyTorch “Previous Versions” matrix).

---
## 4. Configuration

All configuration lives in `config.py`. This file controls:

### 4.1 Embedding / retriever
- `EMBEDDING_MODEL_PATH` – sentence-transformers / Qwen embedding model used by `semanticretriever.py`.
- `MAX_EMBEDDING_SIZE` – max characters of article text to embed.
- `CROSS_ENCODER_MODEL_PATH` – cross‑encoder model for reranking (if used).

### 4.2 Forecasting (legacy HF path)
- `FORECAST_MODEL_PATHS` – model name → HF ID/path mapping for the legacy `agents.py` path.
- `DEFAULT_MODEL`, `DEFAULT_LOGIC_MODEL_PATH` – defaults for the legacy path (kept for reference).

### 4.3 Forecasting (vLLM + Run:ai, primary path)
- `FORECAST_MODEL_PATHS_VLLM` – model name → path mapping used by `agents_vllm.py`:
  - For local dev, use HF IDs like `meta-llama/Llama-3.2-3B-Instruct`.
  - For production with Run:ai, use S3 URIs like `s3://your-bucket/Llama-3-70B`.
- `DEFAULT_MODEL_VLLM` – default vLLM model key (e.g. `"llama-3B"`).
- `VLLM_CONFIG`:
  - `tensor_parallel_size`: GPUs per node for tensor parallelism (often = GPU count).
  - `pipeline_parallel_size`: >1 for multi‑node or very large models; keep 1 for single node.
  - `distributed_executor_backend`: `"mp"` (default) or `"ray"`.
  - `gpu_memory_utilization`: fraction of each GPU’s memory vLLM may use (0.0–1.0; recommended 0.90–0.95).
  - `swap_space_gb`: host swap (in GB) to offload KV cache (for long contexts).
  - `max_model_len`: optional hard cap on context length; `None` lets vLLM infer from the model.
  - `load_format`: `"runai_streamer"` to enable Run:ai S3 streaming; `None` to use default vLLM loader.
  - `model_loader_extra_config`: dict passed to Run:ai loader, e.g.:
    - `concurrency`: number of concurrent threads reading from S3 (16–64 typical).
    - `distributed`: `True` for multi‑node streaming setups.
    - Optional `memory_limit` cap in bytes.
- `MAX_LOGPROB_CACHE` – bound on the per‑forecast logprob cache used when computing probabilities for numbers 0–100:
  - Keep **≥101** to cover all numbers 0–100.
  - Lower only if you’re very RAM‑constrained.

### 4.4 Env variables

Typical environment configuration:
```bash
export HF_TOKEN=your_hf_token              # if you use gated HF models
export CUDA_VISIBLE_DEVICES=0,1,2,3        # select GPUs
export LOG_DIR=/var/log/forecaster         # optional log directory
export LOG_FILE=/var/log/forecaster/app.log# optional log path
```

The code validates `config.py` on startup via `validate_config()` in `forecaster.py`. If there’s a misconfiguration (invalid TP/PP, bad concurrency, missing Run:ai module when required), the app will exit with a clear error message.

---
## 5. Running locally (single node, in‑process)

The simplest way to run the ensemble locally is:
```bash
python forecaster.py
```

What this does:
- Validates `config.py` (including Run:ai module presence and basic ranges).
- Checks that CUDA is available and that `tensor_parallel_size` does not exceed `torch.cuda.device_count()`.
- Logs environment info (detected CUDA version, GPU count, vLLM version).
- Uses `agents_vllm.Agent` instances and `ensemble.Ensemble` to:
  - For each hard‑coded query in `forecaster.py`, retrieve context (via `semanticretriever.py`),
  - Run one or more vLLM agents,
  - Persist per‑agent and ensemble metrics into `database.db` (via `database.py`),
  - Generate plots via `analysis.py` (saved into a plots directory).

You can modify the list of queries in `forecaster.py` or extend it to read queries from a file or CLI arguments.

---
## 6. Running on AWS EC2 with vLLM + Run:ai S3 streaming

This section assumes:
- You want to run large models (e.g. 70B+) stored in S3.
- You’re using EC2 p4d/p5 instances.

### 6.1 Launch + IAM

1. Launch a p4d/p5 instance (or larger) with an IAM role that can read your model bucket.
2. Attach a role with at least:

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

3. If your instance is in a private subnet, create an **S3 VPC Gateway Endpoint** to avoid public egress for weight downloads.

### 6.2 Prep the instance

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y git tmux python3-pip
python3 -m pip install --upgrade pip
git clone https://github.com/tworr0707/forecaster.git
cd forecaster
python3 -m pip install -r requirements.txt
```

### 6.3 Set environment

```bash
export HF_TOKEN=your_hf_token               # only if model is gated on HF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust to GPUs you want to use
export LOG_DIR=/var/log/forecaster
export LOG_FILE=/var/log/forecaster/app.log
```

### 6.4 Point models to S3 and tune parallelism

Edit `config.py`:
- Set `FORECAST_MODEL_PATHS_VLLM` to your S3 URIs, for example:
  ```python
  FORECAST_MODEL_PATHS_VLLM = {
      "llama-70B": "s3://your-bucket/llama-3-70b",
      # ...
  }
  ```
- Set `tensor_parallel_size` equal to the number of GPUs you want to use on the node (e.g. 8 on p4d).
- Keep `pipeline_parallel_size = 1` for single‑node; only increase when using multiple nodes.
- Keep `load_format = "runai_streamer"` when using Run:ai S3 streaming.
- Raise `model_loader_extra_config["concurrency"]` to 32–64 if you have good network bandwidth to S3.
- Consider `swap_space_gb` if you have very long prompts and are hitting OOMs.

### 6.5 Option A: In‑process run (simplest)

Run the same script as locally:
```bash
python forecaster.py
```

This will:
- Initialise vLLM with Run:ai loader pointing at your S3 models.
- Run the hard‑coded set of queries.
- Persist forecasts and plots as described above.

### 6.6 Option B: vLLM server (API mode)

If you prefer to run vLLM as a standalone server and call it from a separate client:

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

Then point an OpenAI‑compatible client at `http://<host>:8000/v1`. (The current repo does not contain a server‑mode client, but you can adapt `agents_vllm.py` to use HTTP instead of in‑process vLLM.)

### 6.7 Verifying the load

- Check logs (`LOG_FILE` or console) for:
  - Effective TP/PP settings,
  - `load_format=runai_streamer`,
  - S3 download progress.
- Optionally curl a health endpoint if you’re in server mode:

```bash
curl http://localhost:8000/health
```

- For in‑process runs, watch the logs for the first forecast run and monitor GPUs with `nvidia-smi`.

### 6.8 Troubleshooting

- **Stuck at download**:
  - Verify IAM role + bucket policy.
  - Ensure S3 VPC endpoint is configured (for private subnets).
  - Try lowering or raising `model_loader_extra_config["concurrency"]`.
- **OOM during load or generation**:
  - Lower `tensor_parallel_size` or GPU memory utilisation.
  - Increase `swap_space_gb` to offload KV cache.
- **NCCL init errors**:
  - Set `NCCL_DEBUG=INFO`.
  - Confirm GPUs are visible and no other processes are using them.
  - Stick to `distributed_executor_backend="mp"` if `ray` causes device mismatches.
- **Slow first token**:
  - Ensure S3 bucket is in the same region as the EC2 instance.
  - Consider a small warm‑up prompt after startup to cache weights/paths.

---
## 7. Logging & paths

- Default log file: `runpod_forecast_logs.log` in `LOG_DIR` (or CWD if unset).
- Logs use a **rotating file handler** (size‑limited with backups) to avoid filling disk.
- `LOG_DIR` and `LOG_FILE` can be overridden via environment variables.
- `.gitignore` excludes:
  - `database.db` and other DB files,
  - plots, logs, coverage artifacts,
  - notebook checkpoints and IDE metadata.

For production, consider:
- Shipping logs to a central log stack (CloudWatch, ELK, etc.).
- Adding structured logging if you want to correlate forecasts with upstream systems.

---
## 8. Testing

There is a small smoke test suite:

```bash
pytest
```

Currently, `tests/test_agent_stub.py`:
- Uses a fake LLM/tokeniser to exercise:
  - `assure_agent` integer token logic,
  - logprob caching and eviction,
  - the ensemble path with stubbed probabilities.

Recommended additional tests before a production deployment:
- A test that runs `forecaster.main()` with stubbed agents (no real GPU/model) to ensure the full pipeline doesn’t regress.
- Negative tests for `validate_config()` (invalid TP/PP, bad concurrency, missing Run:ai module).
- Tests that verify retriever failure falls back to `context=None` without crashing.
- Tests that verify `LOG_FILE`/`LOG_DIR` overrides work and that logs are written where expected.

---
## 9. Performance tuning checklist

- Start with `tensor_parallel_size` equal to the number of GPUs on the node.
- Use `pipeline_parallel_size > 1` only when:
  - The model does not fit in a single node, or
  - You are intentionally spanning multiple nodes.
- Set `gpu_memory_utilization` to 0.90–0.95 on dedicated GPU boxes; lower if sharing.
- Increase `model_loader_extra_config["concurrency"]` (e.g., 32–64) for higher S3 throughput, but watch for saturation.
- Use `swap_space_gb` when you need very long prompts and can tolerate some host memory usage for KV cache.
- Pin GPUs with `CUDA_VISIBLE_DEVICES` to isolate workloads or to run side‑by‑side experiments.
- Keep `MAX_LOGPROB_CACHE >= 101`; only lower if absolutely necessary, and monitor memory.

---
## 10. Operational cautions

- **build_pod.sh**
  - Installs system packages and Python dependencies and logs into Hugging Face.
  - Intended for **disposable VM/container** usage only.
  - Requires explicit opt‑in (`CONFIRM_BUILDPOD`) and root privileges.
  - It populates the local HF cache; for Run:ai S3 streaming you still need models uploaded to S3.
- **Secrets**
  - Always provide `HF_TOKEN` and any AWS credentials via IAM roles or environment variables.
  - Never commit tokens or credentials to the repo.
- **Legacy HF path**
  - `agents.py` and the associated HF config entries are **deprecated**.
  - Prefer `agents_vllm.py` and the vLLM/Run:ai path for any new deployments.

With the above configuration and operational practices, Forecaster can be run on a single high‑end GPU workstation or on multi‑GPU AWS nodes with S3‑backed models. Adjust the config and environment to match your hardware, models, and reliability requirements. 
