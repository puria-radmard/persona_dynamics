# Persona Dynamics Research

Investigating how persona representations change over the course of post-training in LLMs, building on the "Assistant Axis" paper by Lu et al. (2026).

## Project Structure

```
├── data/
│   └── roles/
│       └── instructions/
│           ├── advocate.json
│           ├── analyst.json
│           └── ...
├── src/
│   ├── vllm_utils.py          # vLLM utilities for generation
│   ├── hf_utils.py            # HuggingFace utilities for activation capture
│   ├── generate_transcripts.py # Script 1: Generate role-play transcripts
│   └── capture_activations.py  # Script 2: Extract activations from transcripts
└── outputs/
    ├── transcripts/           # Generated transcripts
    └── activations/           # Extracted activations
```

## Setup

```bash
pip install vllm transformers torch tqdm
```

## Usage

### Step 1: Generate Transcripts

Generate role-play transcripts using vLLM:

```bash
python src/generate_transcripts.py \
    --model-name allenai/OLMo-2-7B-Instruct \
    --checkpoint main \
    --run-name persona_v1 \
    --num-rollouts 3
```

**Arguments:**
- `--model-name`: HuggingFace model ID (default: `allenai/OLMo-2-7B-Instruct`)
- `--checkpoint`: Git revision for model checkpoint (default: `main`)
- `--roles`: Specific roles to process (default: all)
- `--num-rollouts`: Number of rollouts per (system_prompt, question) pair
- `--run-name`: Name for this run (required)
- `--resume`: Resume from existing run (requires `--timestamp`)
- `--temperature`: Sampling temperature (default: 0.8)
- `--max-tokens`: Maximum generation length (default: 512)

**Output format** (`outputs/transcripts/<run>/<model>/<checkpoint>/<role>.jsonl`):
```json
{"system_prompt_idx": 0, "question_idx": 5, "rollout_idx": 0, "response": "..."}
```

### Step 2: Capture Activations

Extract activations from transcripts using HuggingFace:

```bash
python src/capture_activations.py \
    --model-name allenai/OLMo-2-7B-Instruct \
    --checkpoint main \
    --transcript-dir outputs/transcripts/persona_v1_20250129/allenai_OLMo-2-7B-Instruct/main \
    --output-dir outputs/activations/persona_v1/main
```

**Arguments:**
- `--model-name`: HuggingFace model ID
- `--checkpoint`: Git revision for model checkpoint
- `--transcript-dir`: Directory containing transcript JSONL files
- `--role-data-dir`: Directory with role instruction JSONs (default: `data/roles/instructions`)
- `--output-dir`: Where to save activations
- `--layers`: Specific layer indices to capture (default: middle layer)
- `--all-layers`: Capture all layers
- `--aggregation`: How to aggregate over response tokens: `mean`, `last`, or `none`

**Output format:**
- `<role>_activations.pt`: PyTorch tensor of shape `(num_samples, hidden_dim)` per layer
- `<role>_metadata.json`: List of `{system_prompt_idx, question_idx, rollout_idx}` dicts

## Research Pipeline

### Part 1: Replication with OLMo-2 7B

1. Generate transcripts with final checkpoint
2. Capture activations in final checkpoint and base model
3. Run PCA to identify persona space and Assistant Axis

### Part 2: Persona Dynamics over RLVR

Available checkpoints for `allenai/OLMo-2-7B-Instruct`:
- Post-SFT: `allenai/OLMo-2-7B-Instruct-SFT`
- Post-DPO: `allenai/OLMo-2-7B-Instruct-DPO`
- RLVR checkpoints: Use revisions on `allenai/OLMo-2-7B-Instruct`

For each checkpoint:
1. Capture activations using the same transcripts (generated from final checkpoint)
2. Project into persona space
3. Track movement along Assistant Axis and other PCs

### Part 3: Alignment Pretraining Effects

Repeat the above with synthetic data-upsampled models.

## Data Format

### Role Instruction Files (`data/roles/instructions/<role>.json`)

```json
{
  "instruction": [
    {"pos": "You are an advocate who passionately champions..."},
    {"pos": "Please be an advocate who dedicates..."}
  ],
  "questions": [
    "What are your thoughts on recent changes to environmental regulations?",
    "How should society address income inequality?"
  ]
}
```

## Notes

- The paper uses 275 roles and 240 questions
- Filtering is applied based on whether the role was successfully expressed
- We start from post-SFT checkpoint to ensure instruction following capability
- Transcripts from final checkpoint are reused across intermediate checkpoints to control for text variation
