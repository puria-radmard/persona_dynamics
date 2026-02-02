"""Utilities for vLLM model loading and generation."""

import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Optional


def load_model(
    model_name: str,
    revision: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> tuple[LLM, AutoTokenizer]:
    """
    Load a model and tokenizer with vLLM.
    
    Args:
        model_name: HuggingFace model ID
        revision: Git revision/branch for checkpoint selection
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length (None = use model default)
        
    Returns:
        Tuple of (LLM, tokenizer)
    """

    load_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    
    if revision is not None:
        load_kwargs["revision"] = revision
    
    if max_model_len is not None:
        load_kwargs["max_model_len"] = max_model_len

    llm = LLM(**load_kwargs)
    
    # Load tokenizer separately for chat template access
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        token = os.environ['HF_TOKEN']
    )
    
    return llm, tokenizer


def format_chat_prompt(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
) -> str:
    """
    Format a prompt using the model's chat template.
    Falls back to prepending system prompt if system role not supported.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        if "System role not supported" in str(e) or "system" in str(e).lower():
            # Fallback: prepend system prompt to user message
            combined_message = f"{system_prompt}\n\n{user_message}"
            messages = [{"role": "user", "content": combined_message}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise
    
    return prompt


def generate_responses(
    llm: LLM,
    prompts: list[str],
    temperature: float = 0.8,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> list[str]:
    """
    Generate responses for a batch of prompts.
    
    Args:
        llm: vLLM model instance
        prompts: List of formatted prompt strings
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        
    Returns:
        List of generated response strings
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract just the generated text
    responses = [output.outputs[0].text for output in outputs]
    
    return responses
