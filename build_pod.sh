#!/usr/bin/env bash
set -euo pipefail

# Set HF_TOKEN in the environment before running (for gated models).
: "${HF_TOKEN:?Environment variable HF_TOKEN must be set}"
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "This script installs system packages; run as root or with sudo." >&2
  exit 1
fi

echo "Updating package list..."
apt-get update -y
apt-get upgrade -y
apt-get install -y python3-pip tmux

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

echo "Installing Hugging Face CLI..."
python3 -m pip install "huggingface_hub[cli]"

echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

model_list=(
    "meta-llama/Llama-3.3-70B-Instruct"
)

for model in "${model_list[@]}"; do
    echo "Downloading model $model to HF cache at $HF_HOME ..."
    huggingface-cli download "$model"
    echo "Done! $model downloaded."
done

echo "Note: This script populates the local Hugging Face cache. For Run:ai S3 streaming, upload models to S3 and set the S3 URIs in torch/config_vllm.py."
