"""Configuration for persona dynamics experiments."""

# Mapping from HuggingFace model IDs to display names
# Used for {model_name} placeholder in default system prompts
MODEL_DISPLAY_NAMES = {
    "allenai/Olmo-3-7B-Instruct": "OLMo 3",
    "allenai/Olmo-3.1-32B-Instruct": "OLMo 3",
    "google/gemma-2-27b-it": "Gemma 2",
    "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_dpo": "an AI"
}


def get_model_display_name(model_id: str) -> str:
    """Get display name for a model."""
    return MODEL_DISPLAY_NAMES[model_id]