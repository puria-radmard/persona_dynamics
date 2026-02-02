#!/bin/bash
# Run activation capture for OLMo-3 base model checkpoints
# Using transcripts from the instruct model

TRANSCRIPT_DIR="outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main"
BASE_MODEL="allenai/OLMo-3-1125-32B"
TOKENIZER="allenai/OLMo-3.1-32B-Instruct"
NUM_GPUS=2
BATCH_SIZE=16  # Adjust based on VRAM - start with 16, try 32 if no OOM

# List of checkpoints to process
CHECKPOINTS=(
    "stage1-step337000"
    "stage1-step499000"
    "stage1-step656000"
    "stage2-ingredient1-step0"
    "stage2-ingredient1-step7000"
    "stage2-ingredient1-step14000"
    "stage2-ingredient1-step21000"
)

# Run each checkpoint
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo "========================================"
    echo "Processing checkpoint: $CHECKPOINT"
    echo "========================================"
    
    python -m capture_activations \
        --model-name "$BASE_MODEL" \
        --tokenizer-name "$TOKENIZER" \
        --checkpoint "$CHECKPOINT" \
        --transcript-dir "$TRANSCRIPT_DIR" \
        --minimum-rating 2 \
        --num-gpus "$NUM_GPUS" \
        --batch-size "$BATCH_SIZE" \
        --resume
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "FAILED: $CHECKPOINT"
        exit 1
    fi
    
    echo "Completed: $CHECKPOINT"
    echo ""
done

echo "========================================"
echo "All checkpoints completed!"
echo "========================================"
