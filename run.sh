export CUDA_VISIBLE_DEVICES=1

# python -m generate_transcripts --run-name test --model-name Qwen/Qwen3-4B-Instruct-2507 --batch-size 4
python -m capture_activations --model-name Qwen/Qwen3-4B-Instruct-2507 --batch-size 4 --transcript-dir outputs/transcripts/test_20260130_105234/Qwen_Qwen3-4B-Instruct-2507/main

