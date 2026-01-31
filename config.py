"""Configuration for persona dynamics experiments."""

# Mapping from HuggingFace model IDs to display names
# Used for {model_name} placeholder in default system prompts
MODEL_DISPLAY_NAMES = {
    "allenai/Olmo-3-7B-Instruct": "OLMo 3",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2",
    "allenai/OLMo-2-7B-Instruct": "OLMo 2",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen 3",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama 3.3",
    "google/gemma-2-27b-it": "Gemma 2",
}


def get_model_display_name(model_id: str) -> str:
    """Get display name for a model, falling back to extracted name if not found."""
    return MODEL_DISPLAY_NAMES[model_id]