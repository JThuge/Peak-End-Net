#!/bin/bash
# ============================================================
# Peak-End Net Training Script
# ============================================================
# Usage:
#   bash run.sh                  # Single GPU
#   bash run.sh --multi-gpu 4    # Multi-GPU (4 GPUs)
# ============================================================

set -e

# ============================================================
# Data paths (UPDATE THESE to match your setup)
# ============================================================
TRAIN_CSV="./data/train.csv"
VAL_CSV="./data/val.csv"
VIDEO_PATHS_JSON="./data/video_paths.json"
EXTRACTED_FRAMES_DIR=""          # Optional: pre-extracted frames directory

# ============================================================
# Output
# ============================================================
OUTPUT_DIR="./output_peak_end_net"

# ============================================================
# Hyperparameters
# ============================================================
EPOCHS=30
LR=1e-3
BATCH_SIZE_TRAIN=16
BATCH_SIZE_VAL=8
MAX_FRAMES=12
NUM_WORKERS=4

# Model architecture
PRETRAINED_CLIP_NAME="ViT-L/14"
FRAME_CONTEXT_WINDOW=3
END_WINDOW=3
PEAK_TEMPERATURE=0.5
CONTRAST_TEMPERATURE=0.3
END_DECAY_RATE=2.0
RHYTHM_DIM=64
END_RATIO=0.25

# Loss weights
FRAME_CONSISTENCY_WEIGHT=0.5
PEAK_CONSTRAINT_WEIGHT=0.3

# ============================================================
# Build command
# ============================================================
CMD="train.py"
ARGS="
    --train_csv ${TRAIN_CSV}
    --val_csv ${VAL_CSV}
    --video_paths_json ${VIDEO_PATHS_JSON}
    --output_dir ${OUTPUT_DIR}
    --epochs ${EPOCHS}
    --lr ${LR}
    --batch_size_train ${BATCH_SIZE_TRAIN}
    --batch_size_val ${BATCH_SIZE_VAL}
    --max_frames ${MAX_FRAMES}
    --num_workers ${NUM_WORKERS}
    --pretrained_clip_name ${PRETRAINED_CLIP_NAME}
    --frame_context_window ${FRAME_CONTEXT_WINDOW}
    --end_window ${END_WINDOW}
    --peak_temperature ${PEAK_TEMPERATURE}
    --contrast_temperature ${CONTRAST_TEMPERATURE}
    --end_decay_rate ${END_DECAY_RATE}
    --rhythm_dim ${RHYTHM_DIM}
    --end_ratio ${END_RATIO}
    --frame_consistency_weight ${FRAME_CONSISTENCY_WEIGHT}
    --peak_constraint_weight ${PEAK_CONSTRAINT_WEIGHT}
"

if [ -n "${EXTRACTED_FRAMES_DIR}" ]; then
    ARGS="${ARGS} --extracted_frames_dir ${EXTRACTED_FRAMES_DIR}"
fi

# ============================================================
# Launch
# ============================================================
MULTI_GPU=0
if [ "$1" = "--multi-gpu" ] && [ -n "$2" ]; then
    MULTI_GPU=$2
fi

if [ "${MULTI_GPU}" -gt 1 ] 2>/dev/null; then
    echo "============================================================"
    echo "Launching Peak-End Net training with ${MULTI_GPU} GPUs (DDP)"
    echo "============================================================"
    torchrun --nproc_per_node=${MULTI_GPU} ${CMD} ${ARGS}
else
    echo "============================================================"
    echo "Launching Peak-End Net training (single GPU)"
    echo "============================================================"
    python ${CMD} ${ARGS}
fi
