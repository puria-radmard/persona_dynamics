#!/bin/bash
# Run activation capture for SFM base models
# Using transcripts from the Olmo instruct model

TRANSCRIPT_DIR="outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main"
TOKENIZER="geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo"
NUM_GPUS=1
BATCH_SIZE=128
CHECKPOINT="main"

# List of base models to process (in reverse order)
MODELS=(
    # "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base"
    "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base"
    "geodesic-research/sfm_filtered_e2e_alignment_upsampled_base"
    "geodesic-research/sfm_baseline_filtered_base"
    "geodesic-research/sfm_baseline_unfiltered_base"

    "geodesic-research/sfm_unfiltered_cpt_misalignment_upsampled_base"
    "geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_base"
    "geodesic-research/sfm_filtered_cpt_alignment_upsampled_base"
)

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Processing model: $MODEL"
    echo "========================================"
    
    python -m capture_activations \
        --model-name "$MODEL" \
        --tokenizer-name "$TOKENIZER" \
        --checkpoint "$CHECKPOINT" \
        --transcript-dir "$TRANSCRIPT_DIR" \
        --minimum-rating 2 \
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