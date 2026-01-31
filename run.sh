python -m generate_transcripts --run-name test --model-name allenai/Olmo-3-7B-Instruct --batch-size 32 --num-gpus 2 --max-model-len 4096

python -m capture_activations --model-name allenai/Olmo-3-7B-Instruct --batch-size 4 --transcript-dir outputs/transcripts/test_20260130_221255/allenai_Olmo-3-7B-Instruct/main --num-gpus 2 

python -m plot_histogram --activations-dir outputs/transcripts/test_20260130_221255/allenai_Olmo-3-7B-Instruct/main/activations/allenai_Olmo-3-7B-Instruct/main


python -m generate_transcripts --model-name allenai/Olmo-3-7B-Instruct --run-name test --timestamp 20260130_221255 --resume --max-model-len 4096 --num-gpus 2