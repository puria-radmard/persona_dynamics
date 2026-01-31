"""Capture activations from transcripts using HuggingFace."""

import argparse
import json
import os
import torch
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
import multiprocessing as mp

# Load .env file if present (for HF_TOKEN, etc.)
from dotenv import load_dotenv
_script_dir = Path(__file__).parent.resolve()
load_dotenv(_script_dir / ".env")
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture activations from role-play transcripts"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/OLMo-2-7B-Instruct",
        help="HuggingFace model ID for activation capture (can differ from transcript model)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint/revision",
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="Directory containing transcript JSONL files (activations saved in subdir)",
    )
    parser.add_argument(
        "--role-data-dir",
        type=str,
        default="data/roles/instructions",
        help="Directory containing role instruction JSONs",
    )
    parser.add_argument(
        "--roles",
        type=str,
        nargs="+",
        default=None,
        help="Specific roles to process (default: all in transcript dir)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to capture (default: middle layer only)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Capture all layers (overrides --layers)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "last", "none"],
        default="mean",
        help="How to aggregate over response tokens (none = save all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (1 recommended for memory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type for model",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for data parallelism (splits roles across GPUs)",
    )
    
    return parser.parse_args()


def load_transcripts(transcript_path: Path) -> list[dict]:
    """Load transcripts from JSONL file."""
    transcripts = []
    with open(transcript_path) as f:
        for line in f:
            transcripts.append(json.loads(line))
    return transcripts


def load_role_data(role_data_dir: Path, role: str) -> dict:
    """Load role instructions and questions."""
    role_path = role_data_dir / f"{role}.json"
    with open(role_path) as f:
        return json.load(f)


def get_roles_to_process(
    transcript_dir: Path,
    roles: Optional[list[str]],
) -> list[str]:
    """Get list of roles to process."""
    available = [f.stem for f in transcript_dir.glob("*.jsonl")]
    
    if roles is None:
        return available
    
    # Validate requested roles exist
    for role in roles:
        if role not in available:
            raise FileNotFoundError(f"No transcript found for role: {role}")
    
    return roles


def process_role(
    role: str,
    transcripts: list[dict],
    role_data: dict,
    model,
    tokenizer,
    layer_indices: list[int],
    aggregation: str,
    device: str,
) -> dict:
    """
    Process all transcripts for a role and extract activations.
    
    Returns:
        Dict with:
            - 'activations': dict mapping layer_idx to tensor of shape:
                - (num_samples, hidden_dim) if aggregation != 'none'
                - list of (num_tokens, hidden_dim) tensors if aggregation == 'none'
            - 'metadata': list of dicts with indices for each sample
    """
    from hf_utils import extract_response_activations
    
    instructions = role_data["instruction"]
    questions = role_data["questions"]
    
    # Storage for activations per layer
    layer_activations = {layer_idx: [] for layer_idx in layer_indices}
    metadata = []
    
    for transcript in tqdm(transcripts, desc=f"  {role}", leave=False):
        sp_idx = transcript["system_prompt_idx"]
        q_idx = transcript["question_idx"]
        r_idx = transcript["rollout_idx"]
        response = transcript["response"]
        
        system_prompt = instructions[sp_idx]["pos"]
        question = questions[q_idx]
        
        # Extract activations
        try:
            activations = extract_response_activations(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_message=question,
                assistant_response=response,
                layer_indices=layer_indices,
                device=device,
            )
        except Exception as e:
            logger.warning(f"  Failed to extract activations for {role} "
                          f"(sp={sp_idx}, q={q_idx}, r={r_idx}): {e}")
            continue
        
        # Aggregate and store
        for layer_idx, acts in activations.items():
            # acts shape: (num_response_tokens, hidden_dim)
            if aggregation == "mean":
                aggregated = acts.mean(dim=0)  # (hidden_dim,)
            elif aggregation == "last":
                aggregated = acts[-1]  # (hidden_dim,)
            else:  # none
                aggregated = acts  # (num_tokens, hidden_dim)
            
            layer_activations[layer_idx].append(aggregated)
        
        metadata.append({
            "system_prompt_idx": sp_idx,
            "question_idx": q_idx,
            "rollout_idx": r_idx,
        })
    
    # Stack activations if aggregated
    result_activations = {}
    for layer_idx, acts_list in layer_activations.items():
        if aggregation != "none":
            # Stack into (num_samples, hidden_dim)
            result_activations[layer_idx] = torch.stack(acts_list, dim=0)
        else:
            # Keep as list of variable-length tensors
            result_activations[layer_idx] = acts_list
    
    return {
        "activations": result_activations,
        "metadata": metadata,
    }


def save_activations(
    output_dir: Path,
    role: str,
    data: dict,
    aggregation: str,
):
    """Save activations and metadata for a role."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activations
    activations_file = output_dir / f"{role}_activations.pt"
    torch.save(data["activations"], activations_file)
    
    # Save metadata
    metadata_file = output_dir / f"{role}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(data["metadata"], f)
    
    logger.info(f"  Saved {len(data['metadata'])} samples to {output_dir}")


def worker_fn(
    gpu_id: int,
    roles: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    transcript_dir: Path,
    role_data_dir: Path,
):
    """Worker function that runs on a single GPU."""
    # Load .env in worker process too
    from dotenv import load_dotenv
    _script_dir = Path(__file__).parent.resolve()
    load_dotenv(_script_dir / ".env")
    load_dotenv()
    
    # Set CUDA device before importing model utilities
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from hf_utils import (
        load_hf_model,
        extract_response_activations,
        get_num_layers,
    )
    
    worker_logger = logging.getLogger(f"worker_{gpu_id}")
    worker_logger.info(f"GPU {gpu_id}: Processing {len(roles)} roles")
    
    # Get dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load model
    worker_logger.info(f"GPU {gpu_id}: Loading model")
    model, tokenizer = load_hf_model(
        args.model_name,
        revision=args.checkpoint,
        device="cuda",  # Always cuda:0 since we set CUDA_VISIBLE_DEVICES
        torch_dtype=torch_dtype,
    )
    
    # Determine layers
    num_layers = get_num_layers(model)
    if args.all_layers:
        layer_indices = list(range(num_layers))
    elif args.layers is not None:
        layer_indices = args.layers
    else:
        layer_indices = [num_layers // 2]
    
    worker_logger.info(f"GPU {gpu_id}: Capturing layers {layer_indices}")
    
    # Process assigned roles
    for role in tqdm(roles, desc=f"GPU {gpu_id}", position=gpu_id):
        transcript_path = transcript_dir / f"{role}.jsonl"
        transcripts = load_transcripts(transcript_path)
        role_data = load_role_data(role_data_dir, role)
        
        data = process_role(
            role=role,
            transcripts=transcripts,
            role_data=role_data,
            model=model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            aggregation=args.aggregation,
            device="cuda",
        )
        
        save_activations(output_dir, role, data, args.aggregation)
    
    worker_logger.info(f"GPU {gpu_id}: Done!")


def main():
    args = parse_args()
    
    transcript_dir = Path(args.transcript_dir)
    role_data_dir = Path(args.role_data_dir)
    
    # Construct output dir: transcript_dir/activations/<model_name>/<checkpoint>
    model_name_safe = args.model_name.replace("/", "_")
    checkpoint_safe = args.checkpoint if args.checkpoint else "main"
    output_dir = transcript_dir / "activations" / model_name_safe / checkpoint_safe
    
    # Validate directories
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
    if not role_data_dir.exists():
        raise FileNotFoundError(f"Role data directory not found: {role_data_dir}")
    
    # Get roles to process
    roles = get_roles_to_process(transcript_dir, args.roles)
    logger.info(f"Processing {len(roles)} roles")
    logger.info(f"Output path: {output_dir}")
    
    # Create output dir and save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "layers": args.layers,
        "all_layers": args.all_layers,
        "aggregation": args.aggregation,
        "transcript_dir": str(transcript_dir),
        "role_data_dir": str(role_data_dir),
        "roles": args.roles,
        "num_gpus": args.num_gpus,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    if args.num_gpus == 1:
        # Single GPU mode
        from hf_utils import (
            load_hf_model,
            extract_response_activations,
            get_num_layers,
        )
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[args.dtype]
        
        logger.info(f"Loading model: {args.model_name} (checkpoint: {args.checkpoint})")
        model, tokenizer = load_hf_model(
            args.model_name,
            revision=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
        )
        
        num_layers = get_num_layers(model)
        logger.info(f"Model has {num_layers} layers")
        
        if args.all_layers:
            layer_indices = list(range(num_layers))
        elif args.layers is not None:
            layer_indices = args.layers
            for idx in layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(f"Invalid layer index {idx}. Must be in [0, {num_layers})")
        else:
            layer_indices = [num_layers // 2]
        
        logger.info(f"Capturing layers: {layer_indices}")
        
        for role in tqdm(roles, desc="Roles"):
            logger.info(f"Processing role: {role}")
            
            transcript_path = transcript_dir / f"{role}.jsonl"
            transcripts = load_transcripts(transcript_path)
            role_data = load_role_data(role_data_dir, role)
            
            logger.info(f"  Loaded {len(transcripts)} transcripts")
            
            data = process_role(
                role=role,
                transcripts=transcripts,
                role_data=role_data,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
                aggregation=args.aggregation,
                device=args.device,
            )
            
            save_activations(output_dir, role, data, args.aggregation)
        
        logger.info("Done!")
    
    else:
        # Multi-GPU mode
        logger.info(f"Using {args.num_gpus} GPUs for data parallelism")
        
        # Split roles across GPUs
        role_chunks = [[] for _ in range(args.num_gpus)]
        for i, role in enumerate(roles):
            role_chunks[i % args.num_gpus].append(role)
        
        for gpu_id, chunk in enumerate(role_chunks):
            logger.info(f"GPU {gpu_id}: {len(chunk)} roles")
        
        # Spawn workers
        mp.set_start_method("spawn", force=True)
        
        processes = []
        for gpu_id, chunk_roles in enumerate(role_chunks):
            if chunk_roles:
                p = mp.Process(
                    target=worker_fn,
                    args=(gpu_id, chunk_roles, args, output_dir, transcript_dir, role_data_dir),
                )
                p.start()
                processes.append(p)
        
        for p in processes:
            p.join()
        
        logger.info("All workers complete!")


if __name__ == "__main__":
    main()
