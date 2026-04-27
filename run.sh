#!/usr/bin/env bash
set -euo pipefail

# MPS-first end-to-end launcher.
# You can override these environment variables before calling this script.
SAVE_DIR="${SAVE_DIR:-outputs}"
PROMPT="${PROMPT:-Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.}"
SCENE_TYPE="${SCENE_TYPE:-outdoor}"
SEED="${SEED:-16806}"
NUM_LAYERS="${NUM_LAYERS:-3}"
QUALITY="${QUALITY:-high}"
LORA_PATH="${LORA_PATH:-checkpoints/pano_lora_720x1440_v1.safetensors}"
DEPTH_MODEL="${DEPTH_MODEL:-DepthAnythingv2}"
MPS_RASTERIZER="${MPS_RASTERIZER:-cpp}"

python inference_mps.py \
  --prompt "$PROMPT" \
  --save_dir "$SAVE_DIR" \
  --scene_type "$SCENE_TYPE" \
  --seed "$SEED" \
  --num_layers "$NUM_LAYERS" \
  --quality "$QUALITY" \
  --lora_path "$LORA_PATH" \
  --depth_model "$DEPTH_MODEL" \
  --mps_rasterizer "$MPS_RASTERIZER"

