"""Utilities for HuggingFace model loading and activation capture."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Callable
from contextlib import contextmanager


def load_hf_model(
    model_name: str,
    revision: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer with HuggingFace.
    
    Args:
        model_name: HuggingFace model ID
        revision: Git revision for checkpoint selection
        device: Device to load model on
        torch_dtype: Data type for model weights
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def format_full_conversation(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> str:
    """
    Format a complete conversation including the assistant response.
    Falls back to prepending system prompt if system role not supported.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        if "System role not supported" in str(e) or "system" in str(e).lower():
            # Fallback: prepend system prompt to user message
            combined_message = f"{system_prompt}\n\n{user_message}"
            messages = [
                {"role": "user", "content": combined_message},
                {"role": "assistant", "content": assistant_response},
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            raise
    
    return formatted


def get_response_token_mask(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
) -> tuple[torch.Tensor, int, int]:
    """
    Get token indices corresponding to the assistant response.
    Falls back to prepending system prompt if system role not supported.
    
    Returns:
        Tuple of (full_input_ids, response_start_idx, response_end_idx)
    """
    # Try with system message first
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    use_fallback = False
    try:
        prompt_only = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        if "System role not supported" in str(e) or "system" in str(e).lower():
            use_fallback = True
            combined_message = f"{system_prompt}\n\n{user_message}"
            prompt_messages = [{"role": "user", "content": combined_message}]
            prompt_only = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise
    
    # Get full conversation (format_full_conversation handles fallback internally)
    full_conversation = format_full_conversation(
        tokenizer, system_prompt, user_message, assistant_response
    )
    
    # Tokenize both
    prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)
    full_ids = tokenizer.encode(full_conversation, add_special_tokens=False)
    
    # Response tokens start after prompt
    response_start = len(prompt_ids)
    response_end = len(full_ids)
    
    return torch.tensor(full_ids), response_start, response_end


def format_prompt_for_generation(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
) -> torch.Tensor:
    """
    Format prompt ready for generation (with assistant header, no response).
    Returns tokenized input_ids.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        if "System role not supported" in str(e) or "system" in str(e).lower():
            combined_message = f"{system_prompt}\n\n{user_message}"
            messages = [{"role": "user", "content": combined_message}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise
    
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    return torch.tensor(input_ids)


def extract_last_token_activations_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompts: list[str],
    user_messages: list[str],
    layer_indices: list[int],
    device: str = "cuda",
) -> list[dict[int, torch.Tensor] | None]:
    """
    Extract activations at the last token position for a batch of prompts.
    
    Returns:
        List of dicts, each mapping layer index to activation tensor of shape (hidden_dim,).
    """
    batch_size = len(system_prompts)
    assert len(user_messages) == batch_size
    
    # Tokenize all prompts
    all_ids = []
    for i in range(batch_size):
        ids = format_prompt_for_generation(tokenizer, system_prompts[i], user_messages[i])
        all_ids.append(ids)
    
    # Left-pad to same length
    max_len = max(ids.size(0) for ids in all_ids)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, ids in enumerate(all_ids):
        seq_len = ids.size(0)
        offset = max_len - seq_len
        padded_ids[i, offset:] = ids
        attention_mask[i, offset:] = 1
    
    # Forward pass
    with torch.no_grad():
        with ActivationCapture(model, layer_indices) as capture:
            model(padded_ids.to(device), attention_mask=attention_mask.to(device))
            activations = capture.get_activations()
    
    # Extract last token for each item
    results = []
    for i in range(batch_size):
        try:
            item_activations = {}
            for layer_idx, acts in activations.items():
                # acts shape: (batch_size, seq_len, hidden_dim)
                # Last token is always at position -1 (right-aligned after left-padding)
                item_activations[layer_idx] = acts[i, -1, :]
            results.append(item_activations)
        except Exception:
            results.append(None)
    
    return results


class ActivationCapture:
    """Context manager for capturing activations from model layers."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        layer_indices: list[int],
        capture_post_mlp: bool = True,
    ):
        """
        Args:
            model: HuggingFace model
            layer_indices: Which layers to capture (0-indexed)
            capture_post_mlp: If True, capture post-MLP residual stream.
                             If False, capture post-attention.
        """
        self.model = model
        self.layer_indices = layer_indices
        self.capture_post_mlp = capture_post_mlp
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks: list = []
    
    def _get_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store activations (detached, on CPU to save GPU memory)
            self.activations[layer_idx] = hidden_states.detach().cpu()
        
        return hook
    
    def _get_layer_module(self, layer_idx: int):
        """Get the module to hook for a given layer index."""
        
        # Try common layer naming conventions
        if hasattr(self.model, 'model'):
            # Llama-style: model.model.layers[i]
            base = self.model.model
        elif hasattr(self.model, 'transformer'):
            # GPT-style: model.transformer.h[i]
            base = self.model.transformer
        elif hasattr(self.model, 'gpt_neox'):
            # GPTNeoX-style: model.gpt_neox.layers[i]
            base = self.model.gpt_neox
        else:
            base = self.model
        
        # Get the layer
        if hasattr(base, 'layers'):
            layer = base.layers[layer_idx]
        elif hasattr(base, 'h'):
            layer = base.h[layer_idx]
        else:
            raise ValueError(f"Cannot find layers in model architecture")
        
        # Return appropriate submodule
        if self.capture_post_mlp:
            # For post-MLP, we hook the entire layer (after both attn and MLP)
            return layer
        else:
            # For post-attention, hook the attention module
            if hasattr(layer, 'self_attn'):
                return layer.self_attn
            elif hasattr(layer, 'attn'):
                return layer.attn
            else:
                raise ValueError(f"Cannot find attention module in layer")
    
    def __enter__(self):
        """Register hooks."""
        self.activations = {}
        self.hooks = []
        
        for layer_idx in self.layer_indices:
            module = self._get_layer_module(layer_idx)
            hook = module.register_forward_hook(self._get_hook(layer_idx))
            self.hooks.append(hook)
        
        return self
    
    def __exit__(self, *args):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self) -> dict[int, torch.Tensor]:
        """Get captured activations."""
        return self.activations


def extract_response_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
    assistant_response: str,
    layer_indices: list[int],
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """
    Extract activations for response tokens only.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        system_prompt: System prompt text
        user_message: User message text
        assistant_response: Assistant response text
        layer_indices: Which layers to capture
        device: Device for inference
        
    Returns:
        Dict mapping layer index to activation tensor of shape (num_response_tokens, hidden_dim)
    """
    # Get token IDs and response boundaries
    full_ids, response_start, response_end = get_response_token_mask(
        tokenizer, system_prompt, user_message, assistant_response
    )
    
    # Run forward pass with activation capture
    with torch.no_grad():
        with ActivationCapture(model, layer_indices) as capture:
            input_ids = full_ids.unsqueeze(0).to(device)
            model(input_ids)
            
            activations = capture.get_activations()
    
    # Extract only response token activations
    response_activations = {}
    for layer_idx, acts in activations.items():
        # acts shape: (1, seq_len, hidden_dim)
        response_acts = acts[0, response_start:response_end, :]
        response_activations[layer_idx] = response_acts
    
    return response_activations


def extract_response_activations_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompts: list[str],
    user_messages: list[str],
    assistant_responses: list[str],
    layer_indices: list[int],
    device: str = "cuda",
) -> list[dict[int, torch.Tensor] | None]:
    """
    Extract activations for response tokens for a batch of conversations.
    
    This is more efficient than calling extract_response_activations repeatedly
    because it processes all samples in a single forward pass.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        system_prompts: List of system prompts
        user_messages: List of user messages
        assistant_responses: List of assistant responses
        layer_indices: Which layers to capture
        device: Device for inference
        
    Returns:
        List of dicts, each mapping layer index to activation tensor.
        Returns None for items that failed processing.
    """
    batch_size = len(system_prompts)
    assert len(user_messages) == batch_size
    assert len(assistant_responses) == batch_size
    
    # Get token IDs and response boundaries for each item
    all_full_ids = []
    response_boundaries = []  # List of (start, end) tuples
    
    for i in range(batch_size):
        full_ids, response_start, response_end = get_response_token_mask(
            tokenizer, system_prompts[i], user_messages[i], assistant_responses[i]
        )
        all_full_ids.append(full_ids)
        response_boundaries.append((response_start, response_end))
    
    # Pad sequences to same length (left padding for causal LM)
    max_len = max(ids.size(0) for ids in all_full_ids)
    
    # Get pad token id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    
    # Create padded batch with attention mask
    padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    # Track offset for each sequence due to left padding
    padding_offsets = []
    
    for i, ids in enumerate(all_full_ids):
        seq_len = ids.size(0)
        offset = max_len - seq_len  # Left padding offset
        padding_offsets.append(offset)
        padded_ids[i, offset:] = ids
        attention_mask[i, offset:] = 1
    
    # Run forward pass with activation capture
    with torch.no_grad():
        with ActivationCapture(model, layer_indices) as capture:
            input_ids = padded_ids.to(device)
            attn_mask = attention_mask.to(device)
            model(input_ids, attention_mask=attn_mask)
            
            activations = capture.get_activations()
    
    # Extract response activations for each item
    results = []
    
    for i in range(batch_size):
        try:
            offset = padding_offsets[i]
            response_start, response_end = response_boundaries[i]
            
            # Adjust for padding offset
            adj_start = offset + response_start
            adj_end = offset + response_end
            
            item_activations = {}
            for layer_idx, acts in activations.items():
                # acts shape: (batch_size, seq_len, hidden_dim)
                response_acts = acts[i, adj_start:adj_end, :]
                item_activations[layer_idx] = response_acts
            
            results.append(item_activations)
            
        except Exception as e:
            results.append(None)
    
    return results


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Get the number of layers in the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return len(model.gpt_neox.layers)
    elif hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    else:
        raise ValueError("Cannot determine number of layers")