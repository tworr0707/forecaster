#!/usr/bin/env bash
set -euo pipefail

# Safety: require explicit opt-in to avoid accidental local execution
: "${CONFIRM_BUILDPOD:?Set CONFIRM_BUILDPOD=yes to run this script}"

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "Run as root or with sudo." >&2
  exit 1
fi

# Tunables
VENV_PATH=${VENV_PATH:-/opt/forecaster/venv}
USE_HF_DOWNLOAD=${USE_HF_DOWNLOAD:-false}   # set true to pull HF weights locally
HF_HOME=${HF_HOME:-/opt/hf-cache}
PIP_REQS=${PIP_REQS:-requirements.txt}

export DEBIAN_FRONTEND=noninteractive

echo "[system] apt-get update/install"
apt-get update -y
apt-get install -y \
  python3-pip python3-venv git tmux curl unzip \
  build-essential libpq-dev libssl-dev pkg-config \
  awscli

echo "[python] creating venv at $VENV_PATH"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip

echo "[python] installing requirements from $PIP_REQS"
pip install -r "$PIP_REQS"

if [ "$USE_HF_DOWNLOAD" = "true" ]; then
  : "${HF_TOKEN:?Set HF_TOKEN to download gated HF models}"
  export HF_HOME
  pip install "huggingface_hub[cli]"
  echo "[hf] logging in"
  huggingface-cli login --token "$HF_TOKEN" --timeout 60
  model_list=(
    "meta-llama/Llama-3.3-70B-Instruct"
  )
  for model in "${model_list[@]}"; do
    echo "[hf] downloading $model to $HF_HOME"
    huggingface-cli download "$model" --timeout 300 || { echo "Download failed for $model"; exit 1; }
  done
  echo "[hf] download complete"
else
  echo "[hf] Skipping HF model download (USE_HF_DOWNLOAD=false)"
fi

echo "[done] Environment ready. Activate venv with: source $VENV_PATH/bin/activate"
