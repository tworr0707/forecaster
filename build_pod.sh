#!/usr/bin/env bash

export HF_TOKEN="<REMOVED>"
export HF_HOME=/workspace/.cache/huggingface   

: "${HF_TOKEN:?Environment variable HF_TOKEN must be set}"

echo "Updating package list..."
apt-get update -y
apt upgrade -y
apt install python3-pip
apt install tmux -y

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing Hugging Face CLI..."
pip install "huggingface_hub[cli]"

echo "Checking HF_TOKEN..."
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable not set"
  exit 1
fi

echo "Logging in to Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

model_list=(
    #"meta-llama/Llama-3.2-3B-Instruct"
    #"meta-llama/Llama-3.2-1B-Instruct"
    #"microsoft/phi-4"
    #"meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    #"Qwen/QwQ-32B"
    #"nvidia/Llama-3_3-Nemotron-Super-49B-v1"
)

for model in "${model_list[@]}"; do
    echo "Downloading model $model..."
    huggingface-cli download "$model"
    echo "Done! $model downloaded."
done