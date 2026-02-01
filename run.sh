# Chain with && so failures stop the pipeline
# Add --resume for robustness if interrupted

source /workspace/setup_env.sh

# python -m generate_transcripts \
#     --run-name upgrade \
#     --model-name allenai/Olmo-3.1-32B-Instruct \
#     --batch-size 128 \
#     --num-gpus 2 \
#     --resume \

python -m capture_activations \
    --transcript-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main \
    --model-name allenai/Olmo-3.1-32B-Instruct \
    --num-gpus 2 \
    --resume \

# Assuming filtering done already!

# python -m plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/allenai_Olmo-3.1-32B-Instruct/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2