"""Generate role-play transcripts using vLLM."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
import multiprocessing as mp

# Load .env file if present (for HF_TOKEN, etc.)
from dotenv import load_dotenv
# Look for .env in script directory and current directory
_script_dir = Path(__file__).parent.resolve()
load_dotenv(_script_dir / ".env")  # script directory
load_dotenv()  # also try current directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate role-play transcripts for persona analysis"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/OLMo-2-7B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint/revision (e.g., 'step-1000' or 'main')",
    )
    parser.add_argument(
        "--roles",
        type=str,
        nargs="+",
        default=None,
        help="Roles to run (default: all in data/roles/instructions)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per (system_prompt, question) pair",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for this run (used in output path)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run directory",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp of run to resume (required if --resume)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/roles/instructions",
        help="Directory containing role instruction JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/transcripts",
        help="Base directory for output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (reduce if OOM)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for data parallelism (splits roles across GPUs)",
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.resume and args.timestamp is None:
        parser.error("--timestamp is required when --resume is set")
    
    return args


def get_role_files(data_dir: str, roles: Optional[list[str]]) -> dict[str, Path]:
    """
    Get paths to role instruction files.
    
    Args:
        data_dir: Directory containing role JSONs
        roles: Specific roles to load (None = all)
        
    Returns:
        Dict mapping role name to file path
    """
    data_path = Path(data_dir)
    
    if roles is None:
        # Load all JSON files
        role_files = {f.stem: f for f in data_path.glob("*.json")}
    else:
        role_files = {}
        for role in roles:
            role_path = data_path / f"{role}.json"
            if not role_path.exists():
                raise FileNotFoundError(f"Role file not found: {role_path}")
            role_files[role] = role_path
    
    if not role_files:
        raise ValueError(f"No role files found in {data_dir}")
    
    logger.info(f"Found {len(role_files)} role files")
    return role_files


def load_role_data(role_path: Path) -> dict:
    """Load role instructions and questions from JSON."""
    with open(role_path) as f:
        return json.load(f)


def get_output_path(
    output_dir: str,
    run_name: str,
    timestamp: str,
    model_name: str,
    checkpoint: Optional[str],
) -> Path:
    """
    Construct output directory path.
    
    Structure: output_dir/run_name_timestamp/model_name/checkpoint/
    """
    # Sanitize model name for filesystem
    model_name_safe = model_name.replace("/", "_")
    checkpoint_safe = checkpoint if checkpoint else "main"
    
    return Path(output_dir) / f"{run_name}_{timestamp}" / model_name_safe / checkpoint_safe


def load_completed_items(output_path: Path, role: str) -> set[tuple[int, int, int]]:
    """
    Load set of already-completed (system_prompt_idx, question_idx, rollout_idx) tuples.
    """
    completed = set()
    role_file = output_path / f"{role}.jsonl"
    
    if role_file.exists():
        with open(role_file) as f:
            for line in f:
                item = json.loads(line)
                key = (
                    item["system_prompt_idx"],
                    item["question_idx"],
                    item["rollout_idx"],
                )
                completed.add(key)
    
    return completed


def generate_for_role(
    role: str,
    role_data: dict,
    llm,
    tokenizer,
    output_path: Path,
    num_rollouts: int,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    resume: bool,
) -> int:
    """
    Generate transcripts for a single role.
    
    Returns:
        Number of new items generated
    """
    # Import here to support deferred loading for multi-GPU
    from vllm_utils import format_chat_prompt, generate_responses
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{role}.jsonl"
    
    # Get completed items if resuming
    completed = load_completed_items(output_path, role) if resume else set()
    
    if completed:
        logger.info(f"  Resuming: {len(completed)} items already completed")
    
    # Build list of items to generate
    items_to_generate = []
    
    instructions = role_data["instruction"]
    questions = role_data["questions"]
    
    for sp_idx, instruction in enumerate(instructions):
        system_prompt = instruction["pos"]
        for q_idx, question in enumerate(questions):
            for r_idx in range(num_rollouts):
                key = (sp_idx, q_idx, r_idx)
                if key not in completed:
                    items_to_generate.append({
                        "system_prompt": system_prompt,
                        "system_prompt_idx": sp_idx,
                        "question": question,
                        "question_idx": q_idx,
                        "rollout_idx": r_idx,
                    })
    
    if not items_to_generate:
        logger.info(f"  All items already completed for {role}")
        return 0
    
    logger.info(f"  Generating {len(items_to_generate)} items for {role}")
    
    # Process in batches
    num_generated = 0
    num_batches = (len(items_to_generate) + batch_size - 1) // batch_size
    
    for batch_start in tqdm(
        range(0, len(items_to_generate), batch_size),
        desc=f"  {role}",
        total=num_batches,
        leave=False,
    ):
        batch = items_to_generate[batch_start:batch_start + batch_size]
        
        # Format prompts
        prompts = [
            format_chat_prompt(tokenizer, item["system_prompt"], item["question"])
            for item in batch
        ]
        
        # Generate
        responses = generate_responses(
            llm,
            prompts,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Save results - only indices and response, not full text
        with open(output_file, "a") as f:
            for item, response in zip(batch, responses):
                result = {
                    "system_prompt_idx": item["system_prompt_idx"],
                    "question_idx": item["question_idx"],
                    "rollout_idx": item["rollout_idx"],
                    "response": response,
                }
                f.write(json.dumps(result) + "\n")
        
        num_generated += len(batch)
    
    return num_generated


def save_run_config(output_path: Path, args: argparse.Namespace, timestamp: str):
    """Save complete run configuration for reproducibility."""
    config = {
        # Model settings
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "max_model_len": args.max_model_len,
        # Data settings
        "data_dir": args.data_dir,
        "roles": args.roles,
        # Generation settings
        "num_rollouts": args.num_rollouts,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        # Parallelism
        "num_gpus": args.num_gpus,
        # Run metadata
        "run_name": args.run_name,
        "timestamp": timestamp,
        "output_dir": args.output_dir,
    }
    
    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def worker_fn(gpu_id: int, roles: list[str], args: argparse.Namespace, timestamp: str, output_path: Path):
    """Worker function that runs on a single GPU."""
    # Load .env in worker process too (spawn doesn't inherit everything)
    from dotenv import load_dotenv
    _script_dir = Path(__file__).parent.resolve()
    load_dotenv(_script_dir / ".env")
    load_dotenv()
    
    # IMPORTANT: Set CUDA device before importing vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Now import vLLM (must be after setting CUDA_VISIBLE_DEVICES)
    from vllm_utils import load_model
    
    worker_logger = logging.getLogger(f"worker_{gpu_id}")
    worker_logger.info(f"GPU {gpu_id}: Processing {len(roles)} roles")
    
    # Load model
    worker_logger.info(f"GPU {gpu_id}: Loading model")
    llm, tokenizer = load_model(
        args.model_name,
        revision=args.checkpoint,
        max_model_len=args.max_model_len,
    )
    
    # Process assigned roles
    total_generated = 0
    data_dir = Path(args.data_dir)
    
    for role in tqdm(roles, desc=f"GPU {gpu_id}", position=gpu_id):
        role_path = data_dir / f"{role}.json"
        role_data = load_role_data(role_path)
        
        num_generated = generate_for_role(
            role=role,
            role_data=role_data,
            llm=llm,
            tokenizer=tokenizer,
            output_path=output_path,
            num_rollouts=args.num_rollouts,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resume=args.resume,
        )
        
        total_generated += num_generated
    
    worker_logger.info(f"GPU {gpu_id}: Done! Generated {total_generated} items")
    return total_generated


def main():
    args = parse_args()
    
    # Setup timestamp
    if args.resume:
        timestamp = args.timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Run: {args.run_name}_{timestamp}")
    
    # Get role files
    role_files = get_role_files(args.data_dir, args.roles)
    all_roles = sorted(role_files.keys())
    
    # Setup output path
    output_path = get_output_path(
        args.output_dir,
        args.run_name,
        timestamp,
        args.model_name,
        args.checkpoint,
    )
    logger.info(f"Output path: {output_path}")
    
    # Save config (on first run or if not exists)
    output_path.mkdir(parents=True, exist_ok=True)
    if not (output_path / "config.json").exists():
        save_run_config(output_path, args, timestamp)
    
    if args.num_gpus == 1:
        # Single GPU: run directly (import here to allow multi-GPU to set CUDA_VISIBLE_DEVICES first)
        from vllm_utils import load_model
        
        logger.info(f"Loading model: {args.model_name} (checkpoint: {args.checkpoint})")
        llm, tokenizer = load_model(
            args.model_name,
            revision=args.checkpoint,
            max_model_len=args.max_model_len,
        )
        
        total_generated = 0
        for role, role_path in tqdm(sorted(role_files.items()), desc="Roles"):
            logger.info(f"Processing role: {role}")
            role_data = load_role_data(role_path)
            
            num_generated = generate_for_role(
                role=role,
                role_data=role_data,
                llm=llm,
                tokenizer=tokenizer,
                output_path=output_path,
                num_rollouts=args.num_rollouts,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                resume=args.resume,
            )
            total_generated += num_generated
        
        logger.info(f"Done! Generated {total_generated} total items")
    
    else:
        # Multi-GPU: split roles across workers
        logger.info(f"Using {args.num_gpus} GPUs for data parallelism")
        
        # Split roles across GPUs
        role_chunks = [[] for _ in range(args.num_gpus)]
        for i, role in enumerate(all_roles):
            role_chunks[i % args.num_gpus].append(role)
        
        for gpu_id, chunk in enumerate(role_chunks):
            logger.info(f"GPU {gpu_id}: {len(chunk)} roles")
        
        # Spawn workers
        mp.set_start_method("spawn", force=True)
        
        processes = []
        for gpu_id, roles in enumerate(role_chunks):
            if roles:  # Skip empty chunks
                p = mp.Process(
                    target=worker_fn,
                    args=(gpu_id, roles, args, timestamp, output_path),
                )
                p.start()
                processes.append(p)
        
        # Wait for all workers
        for p in processes:
            p.join()
        
        logger.info("All workers complete!")
    
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()