"""Capture activations from transcripts using HuggingFace.

Handles:
- <role>.jsonl - activations from role system prompts (using shared questions)
- default.jsonl - activations from default system prompts (using shared questions)

All transcripts use the same shared extraction questions.
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
load_dotenv(_script_dir / ".env")
load_dotenv()

from config import get_model_display_name

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
        help="HuggingFace model ID for activation capture",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="HuggingFace tokenizer ID (default: same as model-name). "
             "Use this when capturing activations from a base model but "
             "transcripts were generated with an instruct model.",
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
        help="Directory containing transcript JSONL files",
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
        "--transcripts",
        type=str,
        nargs="+",
        default=None,
        help="Specific transcript files to process (default: all in transcript dir)",
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
        help="Number of GPUs for data parallelism",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip transcripts that already have activation files",
    )
    
    # Filtering options
    parser.add_argument(
        "--filtering-dir",
        type=str,
        default=None,
        help="Directory containing filtering results (to skip low-rated samples). "
             "If not specified, checks for 'filtering' subdirectory in transcript-dir.",
    )
    parser.add_argument(
        "--minimum-rating",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Minimum judge rating to include. Requires filtering results to exist. "
             "Roles without filtering data will be SKIPPED (use --resume to fill in later).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for activation extraction. Higher values use more VRAM but are faster. "
             "Start with 8-16 and increase until you hit OOM.",
    )
    
    return parser.parse_args()


def load_shared_questions(questions_file: str) -> list[str]:
    """Load shared extraction questions from JSONL file."""
    questions_path = Path(questions_file)
    
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    questions = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    
    return questions


def load_transcripts(transcript_path: Path) -> list[dict]:
    """Load transcripts from JSONL file."""
    transcripts = []
    with open(transcript_path) as f:
        for line in f:
            transcripts.append(json.loads(line))
    return transcripts


def load_role_system_prompts(role_data_dir: Path, role: str) -> list[str]:
    """Load system prompts from a role JSON file."""
    role_path = role_data_dir / f"{role}.json"
    with open(role_path) as f:
        role_data = json.load(f)
    return [instr["pos"] for instr in role_data["instruction"]]


def load_default_prompts(role_data_dir: Path, model_name: str) -> list[str]:
    """Load and format default system prompts."""
    default_path = role_data_dir / "default.json"
    
    if not default_path.exists():
        raise FileNotFoundError(f"default.json not found at {default_path}")
    
    with open(default_path) as f:
        default_data = json.load(f)
    
    display_name = get_model_display_name(model_name)
    
    prompts = []
    for instruction in default_data["instruction"]:
        prompt = instruction["pos"]
        if "{model_name}" in prompt:
            prompt = prompt.format(model_name=display_name)
        prompts.append(prompt)
    
    return prompts


def get_transcripts_to_process(
    transcript_dir: Path,
    transcripts: Optional[list[str]],
) -> list[str]:
    """Get list of transcript names (without .jsonl) to process."""
    available = [f.stem for f in transcript_dir.glob("*.jsonl")]
    
    if transcripts is None:
        return sorted(available)
    
    # Validate requested exist
    for name in transcripts:
        if name not in available:
            raise FileNotFoundError(f"No transcript found: {name}")
    
    return sorted(transcripts)


def get_completed_transcripts(output_dir: Path) -> set[str]:
    """Get set of transcripts that already have activation files."""
    completed = set()
    for f in output_dir.glob("*_activations.pt"):
        name = f.stem.replace("_activations", "")
        completed.add(name)
    return completed


# --- Filtering utilities (from plot_pca.py) ---

def load_filtering_results(filtering_dir: Path, role_name: str) -> list[dict] | None:
    """Load filtering results for a role. Returns None if file doesn't exist."""
    filtering_path = filtering_dir / f"{role_name}.jsonl"
    if not filtering_path.exists():
        return None
    
    results = []
    with open(filtering_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_passing_indices(
    filtering_results: list[dict],
    minimum_rating: int,
) -> set[tuple[int, int, int]]:
    """Get set of (sp_idx, q_idx, r_idx) tuples that pass the filter."""
    passing = set()
    for r in filtering_results:
        try:
            response = r.get("judge_response", "")
            if response is None:
                continue
            # Extract first digit found
            rating = None
            for char in response.strip():
                if char.isdigit():
                    rating = int(char)
                    break
            
            if rating is not None and rating >= minimum_rating:
                key = (r["system_prompt_idx"], r["question_idx"], r["rollout_idx"])
                passing.add(key)
        except (ValueError, KeyError):
            continue
    return passing


def get_roles_with_filtering(filtering_dir: Path) -> set[str]:
    """Get set of role names that have filtering results."""
    return {f.stem for f in filtering_dir.glob("*.jsonl")}


def filter_transcripts(
    transcripts: list[dict],
    filtering_dir: Path,
    transcript_name: str,
    minimum_rating: int,
) -> tuple[list[dict] | None, str]:
    """
    Filter transcripts based on judge ratings.
    
    Returns:
        (filtered_transcripts, status_message)
        filtered_transcripts is None if role should be skipped (no filtering data)
    
    Note: 'default' transcripts are never filtered (needed for assistant vector).
    """
    # Never filter default - it's needed unfiltered for the assistant vector
    if transcript_name == "default":
        return transcripts, f"all {len(transcripts)} samples (default, no filtering)"
    
    filtering_results = load_filtering_results(filtering_dir, transcript_name)
    
    if filtering_results is None:
        return None, f"no filtering data (skipped, use --resume later)"
    
    passing = get_passing_indices(filtering_results, minimum_rating)
    
    if len(passing) == 0:
        return [], f"no samples with rating >= {minimum_rating}"
    
    filtered = [
        t for t in transcripts
        if (t["system_prompt_idx"], t["question_idx"], t["rollout_idx"]) in passing
    ]
    
    return filtered, f"{len(filtered)}/{len(transcripts)} samples (rating >= {minimum_rating})"


# --- Main processing ---

def process_transcript(
    transcript_name: str,
    transcripts: list[dict],
    system_prompts: list[str],
    questions: list[str],
    model,
    tokenizer,
    layer_indices: list[int],
    aggregation: str,
    device: str,
    batch_size: int = 1,
) -> dict:
    """
    Process all transcripts and extract activations.
    
    Args:
        batch_size: Number of samples to process in parallel. Higher = more VRAM usage.
    
    Returns:
        Dict with 'activations' and 'metadata'
    """
    from hf_utils import extract_response_activations, extract_response_activations_batched
    
    layer_activations = {layer_idx: [] for layer_idx in layer_indices}
    metadata = []
    
    if batch_size == 1:
        # Original single-sample processing
        for transcript in tqdm(transcripts, desc=f"  {transcript_name}", leave=False):
            sp_idx = transcript["system_prompt_idx"]
            q_idx = transcript["question_idx"]
            r_idx = transcript["rollout_idx"]
            response = transcript["response"]
            
            system_prompt = system_prompts[sp_idx]
            question = questions[q_idx]
            
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
                logger.warning(f"  Failed for {transcript_name} "
                              f"(sp={sp_idx}, q={q_idx}, r={r_idx}): {e}")
                continue
            
            for layer_idx, acts in activations.items():
                if aggregation == "mean":
                    aggregated = acts.mean(dim=0)
                elif aggregation == "last":
                    aggregated = acts[-1]
                else:
                    aggregated = acts
                
                layer_activations[layer_idx].append(aggregated)
            
            metadata.append({
                "system_prompt_idx": sp_idx,
                "question_idx": q_idx,
                "rollout_idx": r_idx,
            })
    else:
        # Batched processing
        num_batches = (len(transcripts) + batch_size - 1) // batch_size
        
        for batch_start in tqdm(range(0, len(transcripts), batch_size), 
                                desc=f"  {transcript_name}", 
                                total=num_batches,
                                leave=False):
            batch = transcripts[batch_start:batch_start + batch_size]
            
            # Prepare batch inputs
            batch_system_prompts = []
            batch_user_messages = []
            batch_responses = []
            batch_metadata = []
            
            for transcript in batch:
                sp_idx = transcript["system_prompt_idx"]
                q_idx = transcript["question_idx"]
                r_idx = transcript["rollout_idx"]
                
                batch_system_prompts.append(system_prompts[sp_idx])
                batch_user_messages.append(questions[q_idx])
                batch_responses.append(transcript["response"])
                batch_metadata.append({
                    "system_prompt_idx": sp_idx,
                    "question_idx": q_idx,
                    "rollout_idx": r_idx,
                })
            
            try:
                # Batched extraction
                batch_activations = extract_response_activations_batched(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompts=batch_system_prompts,
                    user_messages=batch_user_messages,
                    assistant_responses=batch_responses,
                    layer_indices=layer_indices,
                    device=device,
                )
                
                # Process each item in batch
                for i, item_activations in enumerate(batch_activations):
                    if item_activations is None:
                        # This item failed
                        continue
                    
                    for layer_idx, acts in item_activations.items():
                        if aggregation == "mean":
                            aggregated = acts.mean(dim=0)
                        elif aggregation == "last":
                            aggregated = acts[-1]
                        else:
                            aggregated = acts
                        
                        layer_activations[layer_idx].append(aggregated)
                    
                    metadata.append(batch_metadata[i])
                    
            except Exception as e:
                logger.warning(f"  Batch failed for {transcript_name}, falling back to single: {e}")
                # Fallback to single-sample processing for this batch
                for i, transcript in enumerate(batch):
                    sp_idx = transcript["system_prompt_idx"]
                    q_idx = transcript["question_idx"]
                    r_idx = transcript["rollout_idx"]
                    
                    try:
                        activations = extract_response_activations(
                            model=model,
                            tokenizer=tokenizer,
                            system_prompt=system_prompts[sp_idx],
                            user_message=questions[q_idx],
                            assistant_response=transcript["response"],
                            layer_indices=layer_indices,
                            device=device,
                        )
                    except Exception as e2:
                        logger.warning(f"    Failed single (sp={sp_idx}, q={q_idx}, r={r_idx}): {e2}")
                        continue
                    
                    for layer_idx, acts in activations.items():
                        if aggregation == "mean":
                            aggregated = acts.mean(dim=0)
                        elif aggregation == "last":
                            aggregated = acts[-1]
                        else:
                            aggregated = acts
                        
                        layer_activations[layer_idx].append(aggregated)
                    
                    metadata.append(batch_metadata[i])
    
    # Stack if aggregated
    result_activations = {}
    for layer_idx, acts_list in layer_activations.items():
        if acts_list:
            if aggregation != "none":
                result_activations[layer_idx] = torch.stack(acts_list, dim=0)
            else:
                result_activations[layer_idx] = acts_list
    
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
    transcript_names: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    transcript_dir: Path,
    role_data_dir: Path,
    questions: list[str],
    default_prompts: list[str],
    filtering_dir: Optional[Path],
):
    """Worker function for a single GPU."""
    from dotenv import load_dotenv
    _script_dir = Path(__file__).parent.resolve()
    load_dotenv(_script_dir / ".env")
    load_dotenv()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from hf_utils import load_hf_model, get_num_layers
    from transformers import AutoTokenizer
    
    worker_logger = logging.getLogger(f"worker_{gpu_id}")
    worker_logger.info(f"GPU {gpu_id}: Processing {len(transcript_names)} transcripts")
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    worker_logger.info(f"GPU {gpu_id}: Loading model")
    model, tokenizer = load_hf_model(
        args.model_name,
        revision=args.checkpoint,
        device="cuda",
        torch_dtype=torch_dtype,
    )
    
    # Override tokenizer if specified (for base model + instruct tokenizer)
    if args.tokenizer_name:
        worker_logger.info(f"GPU {gpu_id}: Loading separate tokenizer from {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = get_num_layers(model)
    if args.all_layers:
        layer_indices = list(range(num_layers))
    elif args.layers is not None:
        layer_indices = args.layers
    else:
        layer_indices = [num_layers // 2]
    
    worker_logger.info(f"GPU {gpu_id}: Capturing layers {layer_indices}")
    
    skipped_no_filtering = []
    skipped_no_passing = []
    processed = []
    
    for transcript_name in tqdm(transcript_names, desc=f"GPU {gpu_id}", position=gpu_id):
        # Load transcripts
        transcript_path = transcript_dir / f"{transcript_name}.jsonl"
        transcripts = load_transcripts(transcript_path)
        
        # Apply filtering if enabled
        if filtering_dir and args.minimum_rating:
            filtered, status = filter_transcripts(
                transcripts=transcripts,
                filtering_dir=filtering_dir,
                transcript_name=transcript_name,
                minimum_rating=args.minimum_rating,
            )
            
            if filtered is None:
                # No filtering data - skip this role
                skipped_no_filtering.append(transcript_name)
                continue
            elif len(filtered) == 0:
                # No samples pass filter
                skipped_no_passing.append(transcript_name)
                continue
            else:
                transcripts = filtered
                worker_logger.info(f"  {transcript_name}: {status}")
        
        # Get system prompts
        if transcript_name == "default":
            system_prompts = default_prompts
        else:
            system_prompts = load_role_system_prompts(role_data_dir, transcript_name)
        
        data = process_transcript(
            transcript_name=transcript_name,
            transcripts=transcripts,
            system_prompts=system_prompts,
            questions=questions,
            model=model,
            tokenizer=tokenizer,
            layer_indices=layer_indices,
            aggregation=args.aggregation,
            device="cuda",
            batch_size=args.batch_size,
        )
        
        if data["metadata"]:  # Only save if we have data
            save_activations(output_dir, transcript_name, data)
            processed.append(transcript_name)
        else:
            worker_logger.warning(f"  No activations captured for {transcript_name}")
    
    # Summary
    worker_logger.info(f"GPU {gpu_id}: Done!")
    worker_logger.info(f"  Processed: {len(processed)}")
    if skipped_no_filtering:
        worker_logger.info(f"  Skipped (no filtering data): {len(skipped_no_filtering)}")
    if skipped_no_passing:
        worker_logger.info(f"  Skipped (no passing samples): {len(skipped_no_passing)}")


def main():
    args = parse_args()
    
    transcript_dir = Path(args.transcript_dir)
    role_data_dir = Path(args.role_data_dir)
    
    # Output dir structure: transcript_dir/activations/<model>/<checkpoint>
    model_name_safe = args.model_name.replace("/", "_")
    checkpoint_safe = args.checkpoint if args.checkpoint else "main"
    output_dir = transcript_dir / "activations" / model_name_safe / checkpoint_safe
    
    # Validate
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
    if not role_data_dir.exists():
        raise FileNotFoundError(f"Role data directory not found: {role_data_dir}")
    
    # Setup filtering
    filtering_dir = None
    if args.filtering_dir:
        filtering_dir = Path(args.filtering_dir)
    elif args.minimum_rating:
        # Check for default filtering location
        default_filtering = transcript_dir / "filtering"
        if default_filtering.exists():
            filtering_dir = default_filtering
            logger.info(f"Using default filtering dir: {filtering_dir}")
        else:
            raise ValueError(
                f"--minimum-rating specified but no filtering dir found. "
                f"Either specify --filtering-dir or run query_judge first."
            )
    
    if filtering_dir:
        if not filtering_dir.exists():
            raise FileNotFoundError(f"Filtering directory not found: {filtering_dir}")
        if args.minimum_rating:
            roles_with_filtering = get_roles_with_filtering(filtering_dir)
            logger.info(f"Filtering enabled: minimum rating >= {args.minimum_rating}")
            logger.info(f"  Roles with filtering data: {len(roles_with_filtering)}")
        else:
            logger.info(f"Filtering dir exists but --minimum-rating not set, no filtering applied")
            filtering_dir = None
    
    # Load shared questions
    questions = load_shared_questions(args.questions_file)
    logger.info(f"Loaded {len(questions)} shared extraction questions")
    
    # Load default prompts (use tokenizer model name for display name if specified)
    display_model = args.tokenizer_name if args.tokenizer_name else args.model_name
    default_prompts = load_default_prompts(role_data_dir, display_model)
    logger.info(f"Loaded {len(default_prompts)} default system prompts")
    
    # Get transcripts to process
    transcripts_to_process = get_transcripts_to_process(transcript_dir, args.transcripts)
    logger.info(f"Found {len(transcripts_to_process)} transcript files")
    
    # Filter if resuming
    if args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = get_completed_transcripts(output_dir)
        if completed:
            logger.info(f"Resuming: {len(completed)} already completed")
            transcripts_to_process = [t for t in transcripts_to_process if t not in completed]
            logger.info(f"Remaining: {len(transcripts_to_process)} to process")
    
    if not transcripts_to_process:
        logger.info("All transcripts already processed!")
        return
    
    # Preview filtering impact
    if filtering_dir and args.minimum_rating:
        roles_with_filtering = get_roles_with_filtering(filtering_dir)
        will_process = [t for t in transcripts_to_process if t in roles_with_filtering or t == "default"]
        will_skip = [t for t in transcripts_to_process if t not in roles_with_filtering and t != "default"]
        logger.info(f"With current filtering data:")
        logger.info(f"  Will process: {len(will_process)} roles (includes 'default' if present)")
        logger.info(f"  Will skip (no filtering data yet): {len(will_skip)} roles")
        if will_skip and len(will_skip) <= 10:
            logger.info(f"    Skipped roles: {', '.join(sorted(will_skip))}")
    
    logger.info(f"Processing {len(transcripts_to_process)} transcripts")
    logger.info(f"Output: {output_dir}")
    
    # Tokenizer info
    if args.tokenizer_name:
        logger.info(f"Model: {args.model_name} (checkpoint: {args.checkpoint})")
        logger.info(f"Tokenizer: {args.tokenizer_name} (separate)")
    else:
        logger.info(f"Model & Tokenizer: {args.model_name} (checkpoint: {args.checkpoint})")
    
    if args.batch_size > 1:
        logger.info(f"Batch size: {args.batch_size} (increase if VRAM available, decrease if OOM)")
    
    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "layers": args.layers,
        "all_layers": args.all_layers,
        "aggregation": args.aggregation,
        "transcript_dir": str(transcript_dir),
        "questions_file": args.questions_file,
        "role_data_dir": str(role_data_dir),
        "num_gpus": args.num_gpus,
        "num_questions": len(questions),
        "filtering_dir": str(filtering_dir) if filtering_dir else None,
        "minimum_rating": args.minimum_rating,
        "batch_size": args.batch_size,
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
        
        logger.info(f"Loading model: {args.model_name} (checkpoint: {args.checkpoint})")
        model, tokenizer = load_hf_model(
            args.model_name,
            revision=args.checkpoint,
            device=args.device,
            torch_dtype=torch_dtype,
        )
        
        # Override tokenizer if specified
        if args.tokenizer_name:
            logger.info(f"Loading separate tokenizer from {args.tokenizer_name}")
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
        
        skipped_no_filtering = []
        skipped_no_passing = []
        processed = []
        
        for transcript_name in tqdm(transcripts_to_process, desc="Transcripts"):
            transcript_path = transcript_dir / f"{transcript_name}.jsonl"
            transcripts = load_transcripts(transcript_path)
            
            # Apply filtering if enabled
            if filtering_dir and args.minimum_rating:
                filtered, status = filter_transcripts(
                    transcripts=transcripts,
                    filtering_dir=filtering_dir,
                    transcript_name=transcript_name,
                    minimum_rating=args.minimum_rating,
                )
                
                if filtered is None:
                    # No filtering data - skip
                    skipped_no_filtering.append(transcript_name)
                    continue
                elif len(filtered) == 0:
                    skipped_no_passing.append(transcript_name)
                    continue
                else:
                    transcripts = filtered
                    logger.info(f"  {transcript_name}: {status}")
            
            # Get system prompts
            if transcript_name == "default":
                system_prompts = default_prompts
            else:
                system_prompts = load_role_system_prompts(role_data_dir, transcript_name)
            
            logger.info(f"Processing: {transcript_name} ({len(transcripts)} samples)")
            
            data = process_transcript(
                transcript_name=transcript_name,
                transcripts=transcripts,
                system_prompts=system_prompts,
                questions=questions,
                model=model,
                tokenizer=tokenizer,
                layer_indices=layer_indices,
                aggregation=args.aggregation,
                device=args.device,
                batch_size=args.batch_size,
            )
            
            if data["metadata"]:
                save_activations(output_dir, transcript_name, data)
                processed.append(transcript_name)
            else:
                logger.warning(f"No activations captured for {transcript_name}")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info(f"  Processed: {len(processed)}")
        if skipped_no_filtering:
            logger.info(f"  Skipped (no filtering data): {len(skipped_no_filtering)}")
            logger.info(f"    Run --resume after more filtering batches complete")
        if skipped_no_passing:
            logger.info(f"  Skipped (no passing samples): {len(skipped_no_passing)}")
        logger.info("=" * 60)
    
    else:
        # Multi-GPU
        logger.info(f"Using {args.num_gpus} GPUs")
        
        # Round-robin split
        chunks = [[] for _ in range(args.num_gpus)]
        for i, name in enumerate(transcripts_to_process):
            chunks[i % args.num_gpus].append(name)
        
        for i, chunk in enumerate(chunks):
            logger.info(f"GPU {i}: {len(chunk)} transcripts")
        
        mp.set_start_method("spawn", force=True)
        
        processes = []
        for gpu_id, chunk in enumerate(chunks):
            if chunk:
                p = mp.Process(
                    target=worker_fn,
                    args=(gpu_id, chunk, args, output_dir, transcript_dir, role_data_dir, 
                          questions, default_prompts, filtering_dir),
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
        logger.info("If roles were skipped due to missing filtering data, run again with --resume")
    
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()