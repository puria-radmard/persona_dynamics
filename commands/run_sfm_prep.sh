#!/bin/bash
# Run activation preparation capture for all SFM base models
# Captures activations at last token position (prefill only)

TOKENIZER="geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo"
NUM_GPUS=1
BATCH_SIZE=64
CHECKPOINT="main"

# List of base models to process
MODELS=(
    "geodesic-research/sfm_unfiltered_cpt_misalignment_upsampled_dpo"
    "geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_dpo"
    "geodesic-research/sfm_baseline_unfiltered_dpo"
    "geodesic-research/sfm_baseline_filtered_dpo"
    "geodesic-research/sfm_filtered_e2e_alignment_upsampled_dpo"
    "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo"
    "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo"
    "geodesic-research/sfm_filtered_cpt_alignment_upsampled_dpo"
)

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Processing model: $MODEL"
    echo "========================================"
    
    python -m activation_preparation.capture_activations \
        --model-name "$MODEL" \
        --tokenizer-name "$TOKENIZER" \
        --checkpoint "$CHECKPOINT" \
        --num-gpus "$NUM_GPUS" \
        --batch-size "$BATCH_SIZE" \
        --resume
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "FAILED: $MODEL"
        exit 1
    fi
    
    echo "Completed: $MODEL"
    echo ""
done

echo "========================================"
echo "All models completed!"
echo "========================================"
