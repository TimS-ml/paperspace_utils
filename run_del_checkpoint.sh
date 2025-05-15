#!/bin/bash

BASE_DIR="/notebooks/x-transformers/checkpoints"

LATEST_DIR=$(find "$BASE_DIR" -type d -name "*_*_*" | sort -r | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo "Found dir: $LATEST_DIR"
    python del_checkpoint.py --model_dir "$LATEST_DIR"
else
    echo "Dir not found"
    exit 1
fi
