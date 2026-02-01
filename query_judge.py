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
        default="gpt-5-mini-2025-08-07",
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
        # GPT-5 mini likely uses cl100k_base or similar
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Rough estimate if tiktoken not installed
        return len(text) // 4


def mode_dry(
    requests: list[dict],
    seed: int,
):
    """Dry run: print stats and example queries."""
    random.seed(seed)
    
    total_requests = len(requests)
    
    # Count tokens
    total_tokens = 0
    for req in tqdm(requests, desc="Counting tokens"):
        content = req["body"]["messages"][0]["content"]
        total_tokens += count_tokens(content)
    
    print("\n" + "=" * 60)
    print("DRY RUN SUMMARY")
    print("=" * 60)
    print(f"Total requests: {total_requests:,}")
    print(f"Total input tokens: {total_tokens:,}")
    print(f"Estimated cost (gpt-5-mini @ $0.125/1M input): ${total_tokens * 0.125 / 1_000_000:.2f}")
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


def mode_send(
    requests: list[dict],
    filtering_dir: Path,
    model: str,
):
    """Submit batch job to OpenAI."""
    import openai
    
    client = openai.OpenAI()
    
    # Write batch input file
    batch_input_path = filtering_dir / "batch_input.jsonl"
    filtering_dir.mkdir(parents=True, exist_ok=True)
    
    with open(batch_input_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    logger.info(f"Wrote {len(requests)} requests to {batch_input_path}")
    
    # Upload file
    logger.info("Uploading batch input file...")
    with open(batch_input_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    logger.info(f"Uploaded file: {uploaded_file.id}")
    
    # Create batch job
    logger.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Judge filtering for {filtering_dir.parent.name}",
            "model": model,
            "num_requests": str(len(requests)),
        }
    )
    
    logger.info(f"Created batch: {batch.id}")
    logger.info(f"Status: {batch.status}")
    
    # Save batch info
    batch_info = {
        "batch_id": batch.id,
        "input_file_id": uploaded_file.id,
        "status": batch.status,
        "created_at": datetime.now().isoformat(),
        "num_requests": len(requests),
        "model": model,
    }
    
    batch_info_path = filtering_dir / "batch_info.json"
    with open(batch_info_path, "w") as f:
        json.dump(batch_info, f, indent=2)
    
    logger.info(f"Saved batch info to {batch_info_path}")
    print(f"\nBatch submitted! ID: {batch.id}")
    print(f"Run with --mode receive to check status and download results.")


def mode_receive(
    filtering_dir: Path,
):
    """Poll batch status and download results when complete."""
    import openai
    
    client = openai.OpenAI()
    
    # Load batch info
    batch_info_path = filtering_dir / "batch_info.json"
    if not batch_info_path.exists():
        raise FileNotFoundError(
            f"No batch_info.json found at {batch_info_path}. "
            "Run with --mode send first."
        )
    
    with open(batch_info_path) as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    logger.info(f"Checking batch: {batch_id}")
    
    # Get current status
    batch = client.batches.retrieve(batch_id)
    
    print(f"\nBatch ID: {batch.id}")
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
        
        # Save raw results
        results_path = filtering_dir / "batch_results.jsonl"
        with open(results_path, "w") as f:
            f.write(output_content)
        
        logger.info(f"Saved results to {results_path}")
        
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
            with open(role_results_path, "w") as f:
                for r in role_results:
                    f.write(json.dumps(r) + "\n")
        
        logger.info(f"Saved results for {len(results_by_role)} roles")
        
        # Update batch info
        batch_info["status"] = "completed"
        batch_info["completed_at"] = datetime.now().isoformat()
        batch_info["output_file_id"] = output_file_id
        with open(batch_info_path, "w") as f:
            json.dump(batch_info, f, indent=2)
        
        print(f"\nResults saved to {filtering_dir}/")
        print(f"Per-role JSONLs: {len(results_by_role)} files")
        
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
        mode_dry(requests, args.seed)
    elif args.mode == "send":
        mode_send(requests, filtering_dir, args.model)


if __name__ == "__main__":
    main()