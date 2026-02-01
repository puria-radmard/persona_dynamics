# Chain with && so failures stop the pipeline
# Add --resume for robustness if interrupted

python -m generate_transcripts \
    --run-name wdef \
    --model-name allenai/Olmo-3-7B-Instruct \
    --batch-size 64 \
    --num-gpus 2 \
    --max-model-len 4096 \
    --resume \
&& python -m capture_activations \
    --transcript-dir outputs/transcripts/wdef/allenai_Olmo-3-7B-Instruct/main \
    --model-name allenai/Olmo-3-7B-Instruct \
    --num-gpus 2 \
    --resume \
&& python -m plot_histogram \
    --activations-dir outputs/transcripts/wdef/allenai_Olmo-3-7B-Instruct/main/activations/allenai_Olmo-3-7B-Instruct/main