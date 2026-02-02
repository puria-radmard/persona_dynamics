# Chain with && so failures stop the pipeline
# Add --resume for robustness if interrupted

# source /workspace/setup_env.sh

# python -m generate_transcripts \
#     --run-name replicate \
#     --model-name google/gemma-2-27b-it \
#     --batch-size 128 \
#     --num-gpus 2

python -m capture_activations \
    --transcript-dir outputs/transcripts/replicate/google_gemma-2-27b-it/main \
    --model-name google/gemma-2-27b-it \
    --num-gpus 2 \
    --resume

# Assuming filtering done already!

# python -m plot_histogram --activations-dir outputs/transcripts/replicate/google_gemma-2-27b-it/main/activations/google_gemma-2-27b-it/main/ --filtering-dir outputs/transcripts/replicate/google_gemma-2-27b-it/main/filtering/ --minimum-rating 2