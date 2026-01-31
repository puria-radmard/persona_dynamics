"""Configuration for persona dynamics experiments."""

# Mapping from HuggingFace model IDs to friendly names for system prompts
MODEL_DISPLAY_NAMES = {
    "allenai/Olmo-3-7B-Instruct": "OLMo 3",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2", 
    "allenai/OLMo-2-7B-Instruct": "OLMo 2",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen 3",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3",
}


def get_model_display_name(model_id: str) -> str:
    """Get friendly display name for a model, or extract from ID if not found."""
    if model_id in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_id]
    
    # Fallback: extract name from model ID
    # e.g., "allenai/OLMo-2-7B-Instruct" -> "OLMo-2-7B-Instruct"
    name = model_id.split("/")[-1]
    # Remove common suffixes
    for suffix in ["-Instruct", "-Chat", "-Base"]:
        name = name.replace(suffix, "")
    return name