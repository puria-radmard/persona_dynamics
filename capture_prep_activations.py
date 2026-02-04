"""Capture activations at generation start position (prefill only).

Unlike the transcript-based approach, this captures activations at the last token
position (just after the assistant header) without any actual response text.
This measures the model's internal state when "about to generate" as a persona.

For each role:
- Uses role-specific questions from data/roles/instructions/{role}.json
- 5 system prompts × 40 questions = 200 samples per role

For default:
- Uses shared questions from data/extraction_questions.jsonl  
- 5 system prompts × 240 questions = 1200 samples

Output: outputs/activation_preparation/<model>/<checkpoint>/
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
import multiprocessing as mp

from dotenv import load_dotenv
_script_dir = Path(__file__).parent.resolve()
load_dotenv(_script_dir.parent / ".env")
load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(_script_dir.parent))
from config import get_model_display_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture activations at generation start position"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="HuggingFace tokenizer ID (default: same as model-name)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint/revision",
    )
    parser.add_argument(
        "--role-data-dir",
        type=str,
        default="data/roles/instructions",
        help="Directory containing role instruction JSONs",
    )
    parser.add_argument(
        "--shared-questions-file",
        type=str,
        default="data/extraction_questions.jsonl",
        help="Path to shared extraction questions JSONL (for default)",
    )
    parser.add_argument(
        "--roles",
        type=str,
        nargs="+",
        default=None,
        help="Specific roles to process (default: all in role-data-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/activation_preparation",
        help="Base directory for output",
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
        help="Capture all layers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing",
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
        "--resume",
        action="store_true",
        help="Skip roles that already have activation files",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel processing",
    )
    
    return parser.parse_args()


def load_role_data(role_path: Path) -> tuple[list[str], list[str]]:
    """
    Load system prompts and questions from a role JSON file.
    
    Returns:
        (system_prompts, questions)
    """
    with open(role_path) as f:
        data = json.load(f)
    
    system_prompts = [instr["pos"] for instr in data["instruction"]]
    questions = data.get("questions", [])
    
    return system_prompts, questions


def load_default_prompts(role_data_dir: Path, model_name: str) -> list[str]:
    """Load and format default system prompts."""
    default_path = role_data_dir / "default.json"
    
    with open(default_path) as f:
        data = json.load(f)
    
    display_name = get_model_display_name(model_name)
    
    prompts = []
    for instruction in data["instruction"]:
        prompt = instruction["pos"]
        if "{model_name}" in prompt:
            prompt = prompt.format(model_name=display_name)
        prompts.append(prompt)
    
    return prompts


def load_shared_questions(questions_file: str) -> list[str]:
    """Load shared extraction questions from JSONL file."""
    questions = []
    with open(questions_file) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    return questions


def get_roles_to_process(role_data_dir: Path, roles: Optional[list[str]]) -> list[str]:
    """Get list of roles to process (excluding 'default')."""
    if roles is not None:
        return [r for r in roles if r != "default"]
    
    available = []
    for f in role_data_dir.glob("*.json"):
        if f.stem != "default":
            available.append(f.stem)
    
    return sorted(available)


def get_completed_roles(output_dir: Path) -> set[str]:
    """Get set of roles that already have activation files."""
    completed = set()
    for f in output_dir.glob("*_activations.pt"):
        name = f.stem.replace("_activations", "")
        completed.add(name)
    return completed


def get_output_path(output_dir: str, model_name: str, checkpoint: Optional[str]) -> Path:
    """Construct output directory path."""
    model_safe = model_name.replace("/", "_")
    checkpoint_safe = checkpoint if checkpoint else "main"
    return Path(output_dir) / model_safe / checkpoint_safe


def process_role(
    role_name: str,
    system_prompts: list[str],
    questions: list[str],
    model,
    tokenizer,
    layer_indices: list[int],
    batch_size: int,
    device: str,
) -> dict:
    """
    Process a single role and extract activations.
    
    Returns:
        Dict with 'activations' and 'metadata'
    """
    from hf_utils import extract_last_token_activations_batched
    
    # Build all (system_prompt, question) pairs
    samples = []
    for sp_idx, system_prompt in enumerate(system_prompts):
        for q_idx, question in enumerate(questions):
            samples.append({
                "system_prompt": system_prompt,
                "system_prompt_idx": sp_idx,
                "question": question,
                "question_idx": q_idx,
            })
    
    layer_activations = {layer_idx: [] for layer_idx in layer_indices}
    metadata = []
    
    # Process in batches
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    for batch_start in tqdm(range(0, len(samples), batch_size),
                            desc=f"  {role_name}",
                            total=num_batches,
                            leave=False):
        batch = samples[batch_start:batch_start + batch_size]
        
        batch_system_prompts = [s["system_prompt"] for s in batch]
        batch_questions = [s["question"] for s in batch]
        
        try:
            batch_activations = extract_last_token_activations_batched(
                model=model,
                tokenizer=tokenizer,
                system_prompts=batch_system_prompts,
                user_messages=batch_questions,
                layer_indices=layer_indices,
                device=device,
            )
            
            for i, item_acts in enumerate(batch_activations):
                if item_acts is None:
                    continue
                
                for layer_idx, acts in item_acts.items():
                    layer_activations[layer_idx].append(acts)
                
                metadata.append({
                    "system_prompt_idx": batch[i]["system_prompt_idx"],
                    "question_idx": batch[i]["question_idx"],
                })
                
        except Exception as e:
            logger.warning(f"Batch failed for {role_name}: {e}")
            continue
    
    # Stack activations
    result_activations = {}
    for layer_idx, acts_list in layer_activations.items():
        if acts_list:
            result_activations[layer_idx] = torch.stack(acts_list, dim=0)
    
    return {
        "activations": result_activations,
        "metadata": metadata,
    }


def save_activations(output_dir: Path, name: str, data: dict):
    """Save activations and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    activations_file = output_dir / f"{name}_activations.pt"
    torch.save(data["activations"], activations_file)
    
    metadata_file = output_dir / f"{name}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(data["metadata"], f)
    
    logger.info(f"  Saved {len(data['metadata'])} samples for {name}")


def worker_fn(
    gpu_id: int,
    role_names: list[str],
    process_default: bool,
    args: argparse.Namespace,
    output_dir: Path,
    role_data_dir: Path,
    default_prompts: list[str],
    shared_questions: list[str],
):
    """Worker function for a single GPU."""
    from dotenv import load_dotenv
    _script_dir = Path(__file__).parent.resolve()
    load_dotenv(_script_dir.parent / ".env")
    load_dotenv()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    sys.path.insert(0, str(_script_dir.parent))
    from hf_utils import load_hf_model, get_num_layers
    from transformers import AutoTokenizer
    
    worker_logger = logging.getLogger(f"worker_{gpu_id}")
    worker_logger.info(f"GPU {gpu_id}: Processing {len(role_names)} roles" +
                       (" + default" if process_default else ""))
    
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
        device="cuda",
        torch_dtype=torch_dtype,
    )
    
    # Override tokenizer if specified
    if args.tokenizer_name:
        worker_logger.info(f"GPU {gpu_id}: Loading separate tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Determine layers
    num_layers = get_num_layers(model)
    if args.all_layers:
        layer_indices = list(range(num_layers))
    elif args.layers is not None:
        layer_indices = args.layers
    else:
        layer_indices = [num_layers // 2]
    
    worker_logger.info(f"GPU {gpu_id}: Capturing layers {layer_indices}")
    
    # Process roles
    for role_name in tqdm(role_names, desc=f"GPU {gpu_id} roles", position=gpu_id):
        role_path = role_data_dir / f"{role_name}.json"
        system_prompts, questions = load_role_data(role_path)
        
        if not questions:
            worker_logger.warning(f"  {role_name}: no questions found, skipping")
            continue
        
        worker_logger.info(f"  {role_name}: {len(system_prompts)} prompts × {len(questions)} questions")
        
        data = process_role(
            role_name=role_name,
            system_prompts=system_prompts,
            questions=questions,
            model=model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            batch_size=args.batch_size,
            device="cuda",
        )
        
        if data["metadata"]:
            save_activations(output_dir, role_name, data)
        else:
            worker_logger.warning(f"  No activations captured for {role_name}")
    
    # Process default if assigned
    if process_default:
        worker_logger.info(f"GPU {gpu_id}: Processing default")
        
        data = process_role(
            role_name="default",
            system_prompts=default_prompts,
            questions=shared_questions,
            model=model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            batch_size=args.batch_size,
            device="cuda",
        )
        
        if data["metadata"]:
            save_activations(output_dir, "default", data)
        else:
            worker_logger.warning(f"  No activations captured for default")
    
    worker_logger.info(f"GPU {gpu_id}: Complete!")


def main():
    args = parse_args()
    
    role_data_dir = Path(args.role_data_dir)
    if not role_data_dir.exists():
        raise FileNotFoundError(f"Role data directory not found: {role_data_dir}")
    
    # Get roles to process
    roles_to_process = get_roles_to_process(role_data_dir, args.roles)
    logger.info(f"Found {len(roles_to_process)} roles to process")
    
    # Load default prompts and shared questions
    display_model = args.tokenizer_name if args.tokenizer_name else args.model_name
    default_prompts = load_default_prompts(role_data_dir, display_model)
    shared_questions = load_shared_questions(args.shared_questions_file)
    
    logger.info(f"Default: {len(default_prompts)} prompts × {len(shared_questions)} questions")
    
    # Setup output path
    output_dir = get_output_path(args.output_dir, args.model_name, args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")
    
    # Check for completed if resuming
    if args.resume:
        completed = get_completed_roles(output_dir)
        if completed:
            logger.info(f"Resuming: {len(completed)} already completed")
            roles_to_process = [r for r in roles_to_process if r not in completed]
            logger.info(f"Remaining: {len(roles_to_process)} roles")
    
    process_default = "default" not in get_completed_roles(output_dir) if args.resume else True
    
    # Save config
    config = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "layers": args.layers,
        "all_layers": args.all_layers,
        "role_data_dir": str(role_data_dir),
        "shared_questions_file": args.shared_questions_file,
        "batch_size": args.batch_size,
        "num_gpus": args.num_gpus,
        "capture_method": "last_token_prefill",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # --- Execution ---
    
    if args.num_gpus == 1:
        from hf_utils import load_hf_model, get_num_layers
        from transformers import AutoTokenizer
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[args.dtype]
        
        logger.info(f"Loading model: {args.model_name}")
        model, tokenizer = load_hf_model(
            args.model_name,
            revision=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
        )
        
        if args.tokenizer_name:
            logger.info(f"Loading separate tokenizer: {args.tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        num_layers = get_num_layers(model)
        logger.info(f"Model has {num_layers} layers")
        
        if args.all_layers:
            layer_indices = list(range(num_layers))
        elif args.layers is not None:
            layer_indices = args.layers
        else:
            layer_indices = [num_layers // 2]
        
        logger.info(f"Capturing layers: {layer_indices}")
        
        # Process roles
        for role_name in tqdm(roles_to_process, desc="Roles"):
            role_path = role_data_dir / f"{role_name}.json"
            system_prompts, questions = load_role_data(role_path)
            
            if not questions:
                logger.warning(f"  {role_name}: no questions found, skipping")
                continue
            
            logger.info(f"Processing: {role_name} ({len(system_prompts)} × {len(questions)} = {len(system_prompts) * len(questions)} samples)")
            
            data = process_role(
                role_name=role_name,
                system_prompts=system_prompts,
                questions=questions,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
                batch_size=args.batch_size,
                device=args.device,
            )
            
            if data["metadata"]:
                save_activations(output_dir, role_name, data)
            else:
                logger.warning(f"No activations captured for {role_name}")
        
        # Process default
        if process_default:
            logger.info(f"Processing: default ({len(default_prompts)} × {len(shared_questions)} = {len(default_prompts) * len(shared_questions)} samples)")
            
            data = process_role(
                role_name="default",
                system_prompts=default_prompts,
                questions=shared_questions,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
                batch_size=args.batch_size,
                device=args.device,
            )
            
            if data["metadata"]:
                save_activations(output_dir, "default", data)
            else:
                logger.warning(f"No activations captured for default")
        
        logger.info("Done!")
    
    else:
        # Multi-GPU
        logger.info(f"Using {args.num_gpus} GPUs")
        
        # Round-robin split
        chunks = [[] for _ in range(args.num_gpus)]
        for i, role in enumerate(roles_to_process):
            chunks[i % args.num_gpus].append(role)
        
        # Assign default to GPU with fewest roles
        default_gpu = min(range(args.num_gpus), key=lambda i: len(chunks[i]))
        
        for i, chunk in enumerate(chunks):
            has_default = (i == default_gpu) and process_default
            logger.info(f"  GPU {i}: {len(chunk)} roles" + (" + default" if has_default else ""))
        
        mp.set_start_method("spawn", force=True)
        
        processes = []
        for gpu_id, chunk in enumerate(chunks):
            do_default = (gpu_id == default_gpu) and process_default
            if chunk or do_default:
                p = mp.Process(
                    target=worker_fn,
                    args=(gpu_id, chunk, do_default, args, output_dir, role_data_dir,
                          default_prompts, shared_questions),
                )
                p.start()
                processes.append((gpu_id, p))
        
        # Wait and check
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
    
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()