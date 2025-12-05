# Forecaster

GPU forecaster that retrieves Wikipedia context, generates logic with Bedrock Claude, and forecasts probabilities (0–100) with vLLM. Data lives in Aurora Postgres + pgvector; plots go to S3. A stub mode exists for offline dev.

## Project Layout
- `agents.py` – vLLM agent (forecast distribution), logprob extraction.
- `ensemble.py` – orchestrates retrieval, logic (Bedrock), agents, persistence, plotting.
- `semanticretriever.py` – Bedrock embeddings + pgvector search; optional article refresh.
- `database.py` – Aurora/pgvector (shared engine); stub store for dev.
- `logic_client.py` / `bedrock_embeddings.py` – Bedrock clients with stub fallbacks.
- `config.py` – all tunables (models, DB, S3, logic toggle).
- `infra/` – CDK stack for VPC + Aurora Serverless v2 + GPU EC2 + endpoints.
- `build_pod.sh` – optional EC2 bootstrap (venv + deps; HF download opt‑in).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
For dev/stub runs (no AWS): set `USE_DB_STUB=true`, `BEDROCK_STUB=true`, `BEDROCK_LOGIC_STUB=true` in env or code.

## Configuration
Edit `config.py`:
- Aurora: `AURORA_CONNECTION_STRING` or host/user/password/DB; optional `AURORA_SECRET_ARN` for Secrets Manager.
- Bedrock: `EMBEDDING_MODEL_ID`, `LOGIC_MODEL_ID`, `USE_LOGIC` ("true"/"false" string).
- Models: `FORECAST_MODEL_PATHS_VLLM`, `DEFAULT_MODEL_VLLM`; `load_format=None` by default.
- S3: `PLOTS_BUCKET`, `ENVIRONMENT`.
- Logprob: `NUMBER_LOGPROB_TOP_K` ≥ 101.

## Run Locally
```bash
python forecaster.py
```
Hard-coded queries in `forecaster.py`; writes forecasts/ensembles to Aurora (or stub) and saves plots (S3 if configured, else local).

## AWS Deploy (CDK)
```bash
cd infra
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
cdk synth -c config=stack-config.yaml
cdk deploy ForecasterStack -c config=stack-config.yaml
```
Stack creates VPC (public/app/db), Aurora Serverless v2 with pgvector, GPU EC2 with IAM for Bedrock+S3+Secrets, VPC endpoints, and a plots bucket. Outputs include Aurora endpoint/secret ARN and EC2 instance ID.

## Notes
- Shared stub/engine prevents pool explosion; keep `DB_POOL_SIZE` modest.
- Bedrock IAM in CDK is scoped to provided model ARNs (cohere embed multilingual, Claude 3.7 Sonnet).
- OOMs per agent are logged and skipped; batch continues.
- Multi-token numbers are batched and handle leading-space tokens to avoid probability bias.
