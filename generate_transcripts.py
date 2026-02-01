"""Generate role-play transcripts using vLLM.

Uses shared extraction questions (240) across all roles, matching the paper methodology.

Generates:
1. For each role: responses with role's system prompts -> <role>.jsonl
2. One default file: responses with default system prompts -> default.jsonl

With 2 GPUs:
- GPU 0: processes roles (first half)
- GPU 1: processes roles (second half) + default
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
import multiprocessing as mp

from dotenv import load_dotenv
_script_dir = Path(__file__).parent.resolve()
load_dotenv(_script_dir / ".env")
load_dotenv()

from config import get_model_display_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate role-play transcripts for persona analysis"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        # default="allenai/OLMo-2-7B-Instruct",
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
        help="Specific roles to run (default: all in role-data-dir except 'default')",
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
        "--questions-file",
        type=str,
        default="data/extraction_questions.jsonl",
        help="Path to shared extraction questions JSONL",
    )
    parser.add_argument(
        "--role-data-dir",
        type=str,
        default="data/roles/instructions",
        help="Directory containing role instruction JSONs (for system prompts)",
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
        help="Batch size for vLLM generation",
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
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096 * 2,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs for parallel processing",
    )
    parser.add_argument(
        "--skip-default",
        action="store_true",
        help="Skip generating the default.jsonl file",
    )
    
    return parser.parse_args()


def load_shared_questions(questions_file: str) -> list[str]:
    """Load shared extraction questions from JSONL file."""
    questions_path = Path(questions_file)
    
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}\n"
            "Expected format: JSONL with 'question' and 'id' fields"
        )
    
    questions = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    
    logger.info(f"Loaded {len(questions)} shared extraction questions")
    return questions


def load_role_files(role_data_dir: str, roles: Optional[list[str]]) -> dict[str, Path]:
    """Get paths to role instruction files, excluding 'default'."""
    data_path = Path(role_data_dir)
    
    if roles is None:
        role_files = {
            f.stem: f for f in data_path.glob("*.json")
            if f.stem != "default"
        }
    else:
        role_files = {}
        for role in roles:
            if role == "default":
                continue
            role_path = data_path / f"{role}.json"
            if not role_path.exists():
                raise FileNotFoundError(f"Role file not found: {role_path}")
            role_files[role] = role_path
    
    return role_files


def load_role_system_prompts(role_path: Path) -> list[str]:
    """Load system prompts from a role JSON file."""
    with open(role_path) as f:
        role_data = json.load(f)
    return [instr["pos"] for instr in role_data["instruction"]]


def load_default_prompts(role_data_dir: str, model_name: str) -> list[str]:
    """Load and format default system prompts."""
    default_path = Path(role_data_dir) / "default.json"
    
    if not default_path.exists():
        raise FileNotFoundError(
            f"default.json not found at {default_path}. "
            "This is required for generating default baseline transcripts."
        )
    
    with open(default_path) as f:
        default_data = json.load(f)
    
    display_name = get_model_display_name(model_name)
    
    prompts = []
    for instruction in default_data["instruction"]:
        prompt = instruction["pos"]
        if "{model_name}" in prompt:
            prompt = prompt.format(model_name=display_name)
        prompts.append(prompt)
    
    logger.info(f"Loaded {len(prompts)} default system prompts (model: {display_name})")
    return prompts


def get_output_path(
    output_dir: str,
    run_name: str,
    model_name: str,
    checkpoint: Optional[str],
) -> Path:
    """Construct output directory path."""
    model_name_safe = model_name.replace("/", "_")
    checkpoint_safe = checkpoint if checkpoint else "main"
    return Path(output_dir) / f"{run_name}" / model_name_safe / checkpoint_safe


def load_completed_items(output_path: Path, filename: str) -> set[tuple[int, int, int]]:
    """Load set of completed (system_prompt_idx, question_idx, rollout_idx) tuples."""
    completed = set()
    filepath = output_path / f"{filename}.jsonl"
    
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                key = (item["system_prompt_idx"], item["question_idx"], item["rollout_idx"])
                completed.add(key)
    
    return completed


def generate_for_file(
    output_filename: str,
    system_prompts: list[str],
    questions: list[str],
    llm,
    tokenizer,
    output_path: Path,
    num_rollouts: int,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    resume: bool,
    worker_logger,
) -> int:
    """
    Generate transcripts and save to a JSONL file.
    
    Returns number of new items generated.
    """
    from vllm_utils import format_chat_prompt, generate_responses
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{output_filename}.jsonl"
    
    # Check completed items if resuming
    completed = load_completed_items(output_path, output_filename) if resume else set()
    
    if completed:
        worker_logger.info(f"  Resuming {output_filename}: {len(completed)} already done")
    
    # Build items to generate
    items_to_generate = []
    for sp_idx, system_prompt in enumerate(system_prompts):
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
        worker_logger.info(f"  {output_filename}: all items complete")
        return 0
    
    worker_logger.info(f"  {output_filename}: generating {len(items_to_generate)} items")
    
    # Process in batches
    num_generated = 0
    num_batches = (len(items_to_generate) + batch_size - 1) // batch_size
    
    for batch_start in tqdm(
        range(0, len(items_to_generate), batch_size),
        desc=f"  {output_filename}",
        total=num_batches,
        leave=False,
    ):
        batch = items_to_generate[batch_start:batch_start + batch_size]
        
        prompts = [
            format_chat_prompt(tokenizer, item["system_prompt"], item["question"])
            for item in batch
        ]
        
        responses = generate_responses(
            llm, prompts, temperature=temperature, max_tokens=max_tokens
        )
        
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


def save_config(
    output_path: Path,
    args: argparse.Namespace,
    default_prompts: list[str],
    questions: list[str],
):
    """Save run configuration."""
    config = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "max_model_len": args.max_model_len,
        "questions_file": args.questions_file,
        "role_data_dir": args.role_data_dir,
        "roles": args.roles,
        "num_rollouts": args.num_rollouts,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_gpus": args.num_gpus,
        "run_name": args.run_name,
        "default_prompts": default_prompts,
        "num_questions": len(questions),
    }
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def worker_fn(
    gpu_id: int,
    role_items: list[tuple[str, list[str]]],  # List of (role_name, system_prompts)
    generate_default: bool,
    default_prompts: list[str],
    questions: list[str],
    args: argparse.Namespace,
    output_path: Path,
):
    """Worker function for a single GPU."""
    from dotenv import load_dotenv
    _script_dir = Path(__file__).parent.resolve()
    load_dotenv(_script_dir / ".env")
    load_dotenv()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_CACHE_ROOT"] = f"/tmp/vllm_cache_gpu_{gpu_id}"
    
    from vllm_utils import load_model
    
    worker_logger = logging.getLogger(f"worker_{gpu_id}")
    worker_logger.info(f"GPU {gpu_id}: {len(role_items)} roles" + 
                       (" + default" if generate_default else ""))
    
    # Load model
    worker_logger.info(f"GPU {gpu_id}: Loading model...")
    llm, tokenizer = load_model(
        args.model_name,
        revision=args.checkpoint,
        max_model_len=args.max_model_len,
    )
    worker_logger.info(f"GPU {gpu_id}: Model loaded")
    
    total_generated = 0
    
    # Process roles
    for role_name, system_prompts in tqdm(role_items, desc=f"GPU {gpu_id} roles", position=gpu_id):
        num = generate_for_file(
            output_filename=role_name,
            system_prompts=system_prompts,
            questions=questions,
            llm=llm,
            tokenizer=tokenizer,
            output_path=output_path,
            num_rollouts=args.num_rollouts,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resume=args.resume,
            worker_logger=worker_logger,
        )
        total_generated += num
    
    # Process default if assigned
    if generate_default:
        worker_logger.info(f"GPU {gpu_id}: Processing default")
        num = generate_for_file(
            output_filename="default",
            system_prompts=default_prompts,
            questions=questions,
            llm=llm,
            tokenizer=tokenizer,
            output_path=output_path,
            num_rollouts=args.num_rollouts,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resume=args.resume,
            worker_logger=worker_logger,
        )
        total_generated += num
    
    worker_logger.info(f"GPU {gpu_id}: Complete! Generated {total_generated} items")


def main():
    args = parse_args()
    
    logger.info(f"Run: {args.run_name}")
    
    # Load shared extraction questions
    questions = load_shared_questions(args.questions_file)
    
    # Load role files (excluding default)
    role_files = load_role_files(args.role_data_dir, args.roles)
    logger.info(f"Found {len(role_files)} role files")
    
    # Load default system prompts
    default_prompts = load_default_prompts(args.role_data_dir, args.model_name)
    
    # Load system prompts for each role
    roles_data = []
    for role_name, role_path in sorted(role_files.items()):
        system_prompts = load_role_system_prompts(role_path)
        roles_data.append((role_name, system_prompts))
    
    logger.info(f"Loaded system prompts for {len(roles_data)} roles")
    
    # Setup output path
    output_path = get_output_path(
        args.output_dir, args.run_name, args.model_name, args.checkpoint
    )
    logger.info(f"Output: {output_path}")
    
    # Save config
    output_path.mkdir(parents=True, exist_ok=True)
    if not (output_path / "config.json").exists():
        save_config(output_path, args, default_prompts, questions)
    
    # Calculate expected outputs
    n_roles = len(roles_data)
    n_prompts_per_role = 5  # Typically 5 system prompts per role
    n_questions = len(questions)
    n_rollouts = args.num_rollouts
    
    per_role = n_prompts_per_role * n_questions * n_rollouts
    total_role = n_roles * per_role
    default_total = len(default_prompts) * n_questions * n_rollouts
    
    logger.info(f"Expected outputs:")
    logger.info(f"  Per role: {n_prompts_per_role} prompts × {n_questions} questions × {n_rollouts} rollouts = {per_role}")
    logger.info(f"  All roles: {n_roles} × {per_role} = {total_role}")
    logger.info(f"  Default: {len(default_prompts)} prompts × {n_questions} questions × {n_rollouts} rollouts = {default_total}")
    logger.info(f"  Total: {total_role + default_total}")
    
    # --- Execution ---
    
    if args.num_gpus == 1:
        from vllm_utils import load_model
        
        logger.info(f"Loading model: {args.model_name}")
        llm, tokenizer = load_model(
            args.model_name, revision=args.checkpoint, max_model_len=args.max_model_len
        )
        
        total = 0
        
        # Process all roles
        for role_name, system_prompts in tqdm(roles_data, desc="Roles"):
            logger.info(f"Processing: {role_name}")
            total += generate_for_file(
                output_filename=role_name,
                system_prompts=system_prompts,
                questions=questions,
                llm=llm,
                tokenizer=tokenizer,
                output_path=output_path,
                num_rollouts=args.num_rollouts,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                resume=args.resume,
                worker_logger=logger,
            )
        
        # Process default
        if not args.skip_default:
            logger.info("Processing: default")
            total += generate_for_file(
                output_filename="default",
                system_prompts=default_prompts,
                questions=questions,
                llm=llm,
                tokenizer=tokenizer,
                output_path=output_path,
                num_rollouts=args.num_rollouts,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                resume=args.resume,
                worker_logger=logger,
            )
        
        logger.info(f"Done! Generated {total} items")
    
    else:
        # Multi-GPU: distribute roles across GPUs, assign default to last GPU
        logger.info(f"Using {args.num_gpus} GPUs")
        
        # Round-robin split of roles
        chunks = [[] for _ in range(args.num_gpus)]
        for i, item in enumerate(roles_data):
            chunks[i % args.num_gpus].append(item)
        
        # Assign default to the GPU with fewest roles (usually last)
        default_gpu = min(range(args.num_gpus), key=lambda i: len(chunks[i]))
        
        for i, chunk in enumerate(chunks):
            has_default = (i == default_gpu) and not args.skip_default
            logger.info(f"  GPU {i}: {len(chunk)} roles" + (" + default" if has_default else ""))
        
        mp.set_start_method("spawn", force=True)
        
        processes = []
        for gpu_id, chunk in enumerate(chunks):
            generate_default = (gpu_id == default_gpu) and not args.skip_default
            if chunk or generate_default:
                p = mp.Process(
                    target=worker_fn,
                    args=(gpu_id, chunk, generate_default, default_prompts, questions, args, output_path),
                )
                p.start()
                processes.append((gpu_id, p))
        
        # Wait and check exit codes
        failed = []
        for gpu_id, p in processes:
            p.join()
            if p.exitcode != 0:
                failed.append((gpu_id, p.exitcode))
        
        if failed:
            for gpu_id, code in failed:
                logger.error(f"GPU {gpu_id} FAILED (exit code {code})")
            sys.exit(1)
        
        logger.info("All workers complete!")
    
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()