#!/bin/bash

BASE_DIR="/notebooks/x-transformers/checkpoints"

TOP_DIRS=$(find "$BASE_DIR" -type d -name "*_*_*" | sort -r | head -3)

if [ -z "$TOP_DIRS" ]; then
    echo "No dirs found"
    exit 1
fi

for DIR in $TOP_DIRS; do
    echo "Processing: $DIR"
    python del_checkpoint.py --model_dir "$DIR"
done
