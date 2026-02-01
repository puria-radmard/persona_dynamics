"""Query LLM judge for filtering role-play transcripts.

Uses OpenAI Batch API to evaluate whether responses demonstrate proper role-playing.
Each role's JSON contains an eval_prompt with {question} and {answer} placeholders.

Modes:
- dry: Print stats and example queries without making API calls
- send: Submit batch job to OpenAI
- receive: Poll for completion and download results

Usage:
    # Preview what would be sent
    python -m query_judge \
        --transcript-dir outputs/transcripts/run1/model/main \
        --mode dry

    # Submit batch job
    python -m query_judge \
        --transcript-dir outputs/transcripts/run1/model/main \
        --mode send

    # Check status and download results
    python -m query_judge \
        --transcript-dir outputs/transcripts/run1/model/main \
        --mode receive
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import logging

from dotenv import load_dotenv
_script_dir = Path(__file__).parent.resolve()
load_dotenv(_script_dir / ".env")
load_dotenv()

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query LLM judge for filtering role-play transcripts"
    )
    
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="Directory containing transcript JSONL files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dry", "send", "receive"],
        required=True,
        help="dry: preview queries, send: submit batch, receive: poll and download",
    )
    parser.add_argument(
        "--role-data-dir",
        type=str,
        default="data/roles/instructions",
        help="Directory containing role instruction JSONs (for eval_prompt)",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default="data/extraction_questions.jsonl",
        help="Path to shared extraction questions JSONL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="OpenAI model for judging",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        nargs="+",
        default=None,
        help="Specific transcripts to process (default: all except 'default')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dry mode examples",
    )
    
    return parser.parse_args()


def load_shared_questions(questions_file: str) -> list[str]:
    """Load shared extraction questions from JSONL file."""
    questions = []
    with open(questions_file) as f:
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


def load_eval_prompt(role_data_dir: Path, role_name: str) -> str:
    """Load eval_prompt from role JSON file.
    
    The eval_prompt contains {question} and {answer} placeholders.
    """
    role_path = role_data_dir / f"{role_name}.json"
    with open(role_path) as f:
        role_data = json.load(f)
    
    eval_prompt = role_data.get("eval_prompt")
    if not eval_prompt:
        raise ValueError(f"No eval_prompt found in {role_path}")
    return eval_prompt


def get_transcripts_to_process(
    transcript_dir: Path,
    transcripts: list[str] | None,
) -> list[str]:
    """Get list of transcript names to process (excluding 'default')."""
    available = [f.stem for f in transcript_dir.glob("*.jsonl") if f.stem != "default"]
    
    if transcripts is None:
        return sorted(available)
    
    for name in transcripts:
        if name not in available:
            raise FileNotFoundError(f"No transcript found: {name}")
    
    return sorted(transcripts)


def format_judge_query(
    eval_prompt: str,
    question: str,
    answer: str,
) -> str:
    """Format a judge query by substituting {question} and {answer} into eval_prompt."""
    return eval_prompt.format(question=question, answer=answer)


def build_batch_requests(
    transcript_dir: Path,
    role_data_dir: Path,
    questions: list[str],
    transcript_names: list[str],
    model: str,
) -> list[dict]:
    """
    Build all batch API requests.
    
    Returns list of dicts with:
    - custom_id: unique identifier for matching results
    - method: "POST"
    - url: "/v1/chat/completions"
    - body: the request body
    """
    requests = []
    
    for role_name in transcript_names:
        transcript_path = transcript_dir / f"{role_name}.jsonl"
        transcripts = load_transcripts(transcript_path)
        eval_prompt = load_eval_prompt(role_data_dir, role_name)
        
        for transcript in transcripts:
            sp_idx = transcript["system_prompt_idx"]
            q_idx = transcript["question_idx"]
            r_idx = transcript["rollout_idx"]
            response = transcript["response"]
            question = questions[q_idx]
            
            # Format the judge prompt using eval_prompt
            judge_prompt = format_judge_query(
                eval_prompt=eval_prompt,
                question=question,
                answer=response,
            )
            
            # Create unique ID for this request
            custom_id = f"{role_name}|{sp_idx}|{q_idx}|{r_idx}"
            
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": judge_prompt}
                    ],
                    "max_tokens": 16,  # Just need a single digit
                    "temperature": 0.0,  # Deterministic judging
                }
            }
            requests.append(request)
    
    return requests


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        import tiktoken
        # GPT-4 mini likely uses cl100k_base or similar
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Rough estimate if tiktoken not installed
        return len(text) // 4


def mode_dry(
    requests: list[dict],
    transcript_names: list[str],
    seed: int,
):
    """Dry run: print stats, chunking plan, and example queries."""
    random.seed(seed)
    
    total_requests = len(requests)
    
    # Compute chunking plan (this also counts tokens)
    print("\n" + "=" * 60)
    print("CHUNKING PLAN")
    print("=" * 60)
    
    chunks = chunk_requests_by_tokens(requests, transcript_names)
    
    total_tokens = sum(chunk["token_count"] for chunk in chunks)
    
    for i, chunk in enumerate(chunks, 1):
        chunk_name = get_chunk_filename(chunk)
        start_idx, end_idx = chunk["request_indices"]
        print(f"\nChunk {i}: {chunk_name}")
        print(f"  Roles: {len(chunk['roles'])} ({chunk['start_role']} → {chunk['end_role']})")
        print(f"  Requests: {end_idx - start_idx:,}")
        print(f"  Tokens: {chunk['token_count']:,}")
        print(f"  Est. file size: {chunk['estimated_file_size_mb']:.1f} MB")
    
    print("\n" + "=" * 60)
    print("DRY RUN SUMMARY")
    print("=" * 60)
    print(f"Total requests: {total_requests:,}")
    print(f"Total input tokens: {total_tokens:,}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Estimated cost (gpt-4.1-mini batch @ $0.20/1M input): ${total_tokens * 0.20 / 1_000_000:.2f}")
    print(f"\nScoring scale (from eval_prompt):")
    print(f"  0 = Not role-playing, refuses as AI")
    print(f"  1 = Not role-playing, answers as AI") 
    print(f"  2 = Partial role-playing (AI with role attributes)")
    print(f"  3 = Fully role-playing")
    
    # Show example queries
    print("\n" + "-" * 60)
    print("EXAMPLE QUERIES (3 random)")
    print("-" * 60)
    
    examples = random.sample(requests, min(3, len(requests)))
    for i, req in enumerate(examples, 1):
        content = req['body']['messages'][0]['content']
        print(f"\n[Example {i}] custom_id: {req['custom_id']}")
        print(f"Prompt ({len(content)} chars):\n{content}")


def chunk_requests_by_tokens(
    requests: list[dict],
    transcript_names: list[str],
    max_tokens: int = 35_000_000,
    max_file_size_mb: int = 150,
) -> list[dict]:
    """
    Chunk requests by role, respecting token and file size limits.
    
    Returns list of chunk info dicts:
    {
        "start_role": "addict",
        "end_role": "fool", 
        "roles": ["addict", "advocate", ..., "fool"],
        "request_indices": (0, 45000),  # slice indices into requests list
        "token_count": 34_500_000,
        "estimated_file_size_mb": 140,
    }
    """
    # Group requests by role (they're already sorted by role from build_batch_requests)
    role_boundaries = []  # List of (role_name, start_idx, end_idx, token_count)
    
    current_role = None
    start_idx = 0
    role_tokens = 0
    
    for i, req in tqdm(enumerate(requests), total=len(requests), desc="Computing chunks"):
        role_name = req["custom_id"].split("|")[0]
        
        if role_name != current_role:
            if current_role is not None:
                role_boundaries.append((current_role, start_idx, i, role_tokens))
            current_role = role_name
            start_idx = i
            role_tokens = 0
        
        role_tokens += count_tokens(req["body"]["messages"][0]["content"])
    
    # Don't forget the last role
    if current_role is not None:
        role_boundaries.append((current_role, start_idx, len(requests), role_tokens))
    
    # Now chunk roles together respecting limits
    chunks = []
    chunk_roles = []
    chunk_start_idx = 0
    chunk_tokens = 0
    chunk_size_estimate = 0
    
    for role_name, start_idx, end_idx, role_token_count in role_boundaries:
        # Estimate file size: ~200 bytes overhead per request + content
        role_requests = requests[start_idx:end_idx]
        role_size_estimate = sum(
            len(json.dumps(req)) for req in role_requests
        ) / (1024 * 1024)  # MB
        
        # Check if adding this role would exceed limits
        would_exceed_tokens = (chunk_tokens + role_token_count) > max_tokens
        would_exceed_size = (chunk_size_estimate + role_size_estimate) > max_file_size_mb
        
        if chunk_roles and (would_exceed_tokens or would_exceed_size):
            # Save current chunk and start new one
            chunks.append({
                "start_role": chunk_roles[0],
                "end_role": chunk_roles[-1],
                "roles": chunk_roles,
                "request_indices": (chunk_start_idx, start_idx),
                "token_count": chunk_tokens,
                "estimated_file_size_mb": chunk_size_estimate,
            })
            chunk_roles = []
            chunk_start_idx = start_idx
            chunk_tokens = 0
            chunk_size_estimate = 0
        
        chunk_roles.append(role_name)
        chunk_tokens += role_token_count
        chunk_size_estimate += role_size_estimate
    
    # Don't forget the last chunk
    if chunk_roles:
        chunks.append({
            "start_role": chunk_roles[0],
            "end_role": chunk_roles[-1],
            "roles": chunk_roles,
            "request_indices": (chunk_start_idx, len(requests)),
            "token_count": chunk_tokens,
            "estimated_file_size_mb": chunk_size_estimate,
        })
    
    return chunks


def get_chunk_filename(chunk: dict) -> str:
    """Get base filename for a chunk (without extension)."""
    return f"{chunk['start_role']}_{chunk['end_role']}"


def load_batch_infos(filtering_dir: Path) -> dict[str, dict]:
    """Load all batch_info_*.json files. Returns {chunk_name: info_dict}."""
    infos = {}
    for f in filtering_dir.glob("batch_info_*.json"):
        chunk_name = f.stem.replace("batch_info_", "")
        with open(f) as fp:
            infos[chunk_name] = json.load(fp)
    return infos


def mode_send(
    requests: list[dict],
    filtering_dir: Path,
    model: str,
    transcript_names: list[str],
):
    """Submit batch job to OpenAI with chunking support."""
    import openai
    
    client = openai.OpenAI()
    filtering_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if chunks already exist
    existing_infos = load_batch_infos(filtering_dir)
    
    if existing_infos:
        # Chunks already created - find next to send
        logger.info(f"Found {len(existing_infos)} existing chunk(s)")
        
        # Check if any batch is currently in-flight
        in_flight = [name for name, info in existing_infos.items() if info.get("status") == "sent"]
        if in_flight:
            print(f"\nError: Batch '{in_flight[0]}' is still in-flight.")
            print(f"Run --mode receive first to complete it.")
            return
        
        # Find first unsent chunk
        # Sort by start_role to maintain alphabetical order
        sorted_chunks = sorted(existing_infos.items(), key=lambda x: x[0])
        unsent = [(name, info) for name, info in sorted_chunks if info.get("status") is None]
        
        if not unsent:
            completed = [name for name, info in existing_infos.items() if info.get("status") == "completed"]
            print(f"\nAll {len(completed)} batches already completed!")
            return
        
        chunk_name, chunk_info = unsent[0]
        logger.info(f"Sending chunk: {chunk_name}")
        
        # Load the batch input file
        batch_input_path = filtering_dir / f"batch_input_{chunk_name}.jsonl"
        if not batch_input_path.exists():
            raise FileNotFoundError(f"Batch input file not found: {batch_input_path}")
        
    else:
        # First time - create all chunks
        logger.info("Creating chunks...")
        chunks = chunk_requests_by_tokens(requests, transcript_names)
        
        logger.info(f"Created {len(chunks)} chunks:")
        for chunk in chunks:
            chunk_name = get_chunk_filename(chunk)
            logger.info(f"  {chunk_name}: {len(chunk['roles'])} roles, "
                       f"{chunk['token_count']:,} tokens, "
                       f"{chunk['estimated_file_size_mb']:.1f}MB")
        
        # Write all batch input files and batch info files
        for chunk in tqdm(chunks, desc="Writing chunk files"):
            chunk_name = get_chunk_filename(chunk)
            start_idx, end_idx = chunk["request_indices"]
            chunk_requests = requests[start_idx:end_idx]
            
            # Write batch input
            batch_input_path = filtering_dir / f"batch_input_{chunk_name}.jsonl"
            with open(batch_input_path, "w") as f:
                for req in chunk_requests:
                    f.write(json.dumps(req) + "\n")
            
            # Write batch info (status: null = not yet sent)
            batch_info = {
                "chunk_name": chunk_name,
                "start_role": chunk["start_role"],
                "end_role": chunk["end_role"],
                "roles": chunk["roles"],
                "num_requests": end_idx - start_idx,
                "token_count": chunk["token_count"],
                "estimated_file_size_mb": chunk["estimated_file_size_mb"],
                "model": model,
                "status": None,  # null = created but not sent
                "batch_id": None,
                "input_file_id": None,
                "created_at": datetime.now().isoformat(),
            }
            batch_info_path = filtering_dir / f"batch_info_{chunk_name}.json"
            with open(batch_info_path, "w") as f:
                json.dump(batch_info, f, indent=2)
        
        logger.info(f"Wrote all chunk files to {filtering_dir}")
        
        # Send the first chunk
        chunk = chunks[0]
        chunk_name = get_chunk_filename(chunk)
        batch_input_path = filtering_dir / f"batch_input_{chunk_name}.jsonl"
        chunk_info = None  # Will reload below
    
    # Upload and send
    logger.info(f"Uploading {batch_input_path.name}...")
    with open(batch_input_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    logger.info(f"Uploaded file: {uploaded_file.id}")
    
    logger.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Judge filtering: {chunk_name}",
            "model": model,
        }
    )
    
    logger.info(f"Created batch: {batch.id}")
    logger.info(f"Status: {batch.status}")
    
    # Update batch info
    batch_info_path = filtering_dir / f"batch_info_{chunk_name}.json"
    with open(batch_info_path) as f:
        batch_info = json.load(f)
    
    batch_info["status"] = "sent"
    batch_info["batch_id"] = batch.id
    batch_info["input_file_id"] = uploaded_file.id
    batch_info["sent_at"] = datetime.now().isoformat()
    
    with open(batch_info_path, "w") as f:
        json.dump(batch_info, f, indent=2)
    
    # Check how many remain
    all_infos = load_batch_infos(filtering_dir)
    remaining = sum(1 for info in all_infos.values() if info.get("status") is None)
    completed = sum(1 for info in all_infos.values() if info.get("status") == "completed")
    
    print(f"\nBatch submitted! ID: {batch.id}")
    print(f"Chunk: {chunk_name}")
    print(f"Progress: {completed}/{len(all_infos)} completed, {remaining} remaining after this one")
    print(f"\nRun --mode receive to poll for completion.")


def mode_receive(
    filtering_dir: Path,
):
    """Poll batch status and download results when complete."""
    import openai
    
    client = openai.OpenAI()
    
    # Load all batch infos
    all_infos = load_batch_infos(filtering_dir)
    
    if not all_infos:
        print(f"\nNo batches found in {filtering_dir}")
        print("Run --mode send first to create and submit batches.")
        return
    
    # Find the in-flight batch
    in_flight = [(name, info) for name, info in all_infos.items() if info.get("status") == "sent"]
    
    if not in_flight:
        # Check if all completed or all unsent
        completed = [name for name, info in all_infos.items() if info.get("status") == "completed"]
        unsent = [name for name, info in all_infos.items() if info.get("status") is None]
        
        if unsent:
            print(f"\nNo batch currently in-flight.")
            print(f"  Completed: {len(completed)}/{len(all_infos)}")
            print(f"  Unsent: {len(unsent)}")
            print(f"\nRun --mode send to submit the next batch.")
        else:
            print(f"\nAll {len(completed)} batches completed!")
        return
    
    if len(in_flight) > 1:
        logger.warning(f"Multiple batches in-flight (unexpected): {[name for name, _ in in_flight]}")
    
    chunk_name, chunk_info = in_flight[0]
    batch_id = chunk_info["batch_id"]
    
    logger.info(f"Checking batch: {chunk_name} (ID: {batch_id})")
    
    # Get current status
    batch = client.batches.retrieve(batch_id)
    
    print(f"\nBatch: {chunk_name}")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Requests: {batch.request_counts.completed}/{batch.request_counts.total} completed")
    if batch.request_counts.failed > 0:
        print(f"Failed: {batch.request_counts.failed}")
    
    if batch.status == "completed":
        # Download results
        logger.info("Batch completed! Downloading results...")
        
        output_file_id = batch.output_file_id
        if output_file_id is None:
            raise ValueError("Batch completed but no output file available")
        
        # Download output file content
        file_response = client.files.content(output_file_id)
        output_content = file_response.text
        
        # Save raw results for this chunk
        results_path = filtering_dir / f"batch_results_{chunk_name}.jsonl"
        with open(results_path, "w") as f:
            f.write(output_content)
        
        logger.info(f"Saved raw results to {results_path}")
        
        # Parse and organize results by role
        results_by_role = {}
        for line in output_content.strip().split("\n"):
            result = json.loads(line)
            custom_id = result["custom_id"]
            role_name, sp_idx, q_idx, r_idx = custom_id.split("|")
            
            if role_name not in results_by_role:
                results_by_role[role_name] = []
            
            # Extract judge response
            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            judge_response = choices[0]["message"]["content"] if choices else None
            
            results_by_role[role_name].append({
                "system_prompt_idx": int(sp_idx),
                "question_idx": int(q_idx),
                "rollout_idx": int(r_idx),
                "judge_response": judge_response,
                "error": result.get("error"),
            })
        
        # Save per-role results
        for role_name, role_results in results_by_role.items():
            role_results_path = filtering_dir / f"{role_name}.jsonl"
            
            # Append if file exists (from previous chunks), otherwise create
            mode = "a" if role_results_path.exists() else "w"
            with open(role_results_path, mode) as f:
                for r in role_results:
                    f.write(json.dumps(r) + "\n")
        
        logger.info(f"Saved results for {len(results_by_role)} roles")
        
        # Update batch info
        batch_info_path = filtering_dir / f"batch_info_{chunk_name}.json"
        chunk_info["status"] = "completed"
        chunk_info["completed_at"] = datetime.now().isoformat()
        chunk_info["output_file_id"] = output_file_id
        with open(batch_info_path, "w") as f:
            json.dump(chunk_info, f, indent=2)
        
        # Show overall progress
        all_infos = load_batch_infos(filtering_dir)  # Reload
        completed = sum(1 for info in all_infos.values() if info.get("status") == "completed")
        remaining = sum(1 for info in all_infos.values() if info.get("status") is None)
        
        print(f"\nResults saved!")
        print(f"Progress: {completed}/{len(all_infos)} completed, {remaining} remaining")
        
        if remaining > 0:
            print(f"\nRun --mode send to submit the next batch.")
        else:
            print(f"\nAll batches completed!")
        
    elif batch.status == "failed":
        print("\nBatch failed!")
        if batch.errors:
            for error in batch.errors.data:
                print(f"  Error: {error.message}")
    
    elif batch.status in ["validating", "in_progress", "finalizing"]:
        print(f"\nBatch still processing. Run --mode receive again later.")
        
        # Estimate time
        if batch.request_counts.total > 0:
            progress = batch.request_counts.completed / batch.request_counts.total
            print(f"Progress: {progress*100:.1f}%")
    
    else:
        print(f"\nUnexpected status: {batch.status}")


def main():
    args = parse_args()
    
    transcript_dir = Path(args.transcript_dir)
    role_data_dir = Path(args.role_data_dir)
    filtering_dir = transcript_dir / "filtering"
    
    # Validate paths
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
    
    if args.mode == "receive":
        # Receive mode doesn't need to rebuild requests
        mode_receive(filtering_dir)
        return
    
    # For dry and send modes, we need to build the requests
    if not role_data_dir.exists():
        raise FileNotFoundError(f"Role data directory not found: {role_data_dir}")
    
    # Load resources
    questions = load_shared_questions(args.questions_file)
    transcript_names = get_transcripts_to_process(transcript_dir, args.transcripts)
    
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"Transcripts to process: {len(transcript_names)}")
    
    # Build all requests
    logger.info("Building batch requests...")
    requests = build_batch_requests(
        transcript_dir=transcript_dir,
        role_data_dir=role_data_dir,
        questions=questions,
        transcript_names=transcript_names,
        model=args.model,
    )
    
    logger.info(f"Built {len(requests)} requests")
    
    # Execute mode
    if args.mode == "dry":
        mode_dry(requests, transcript_names, args.seed)
    elif args.mode == "send":
        mode_send(requests, filtering_dir, args.model, transcript_names)


if __name__ == "__main__":
    main()