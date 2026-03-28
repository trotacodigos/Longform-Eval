#!/bin/bash
# Sequential model download script
# Runs overnight; logs each model's result

LOG_FILE="download_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

download() {
    local model_id=$1
    log "Starting: $model_id"
    huggingface-cli download "$model_id"
    if [ $? -eq 0 ]; then
        log "Done: $model_id"
    else
        log "FAILED: $model_id"
    fi
}

# ── 소형 (7~32B) ──────────────────────────────────────────
#download "tencent/Hunyuan-MT-7B"
download "google/gemma-3-27b-it"
download "Qwen/Qwen3.5-27B"
download "LGAI-EXAONE/EXAONE-4.0-32B"
download "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"

# ── 중형 (70~72B) ─────────────────────────────────────────
download "Unbabel/Tower-Plus-72B" #"Unbabel/TowerInstruct-72B-v0.2"
download "meta-llama/Llama-3.3-70B-Instruct"

log "All small/medium models downloaded."