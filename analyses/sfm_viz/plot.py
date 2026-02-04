"""Cross-model PCA visualization for SFM base models.

Plots cosine similarity to PC1 vs PC2 for all roles across different models,
highlighting assistant and robot personas.

Saves to: analyses/sfm_viz/plot.png
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration - hardcoded paths
BASE_PATH = Path("outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations")
OUTPUT_DIR = Path("analyses/sfm_viz")

# Models to compare (in display order)
MODELS = [
    "geodesic-research/sfm_baseline_unfiltered_base",
    "geodesic-research/sfm_baseline_filtered_base",
    "geodesic-research/sfm_filtered_e2e_alignment_upsampled_base",
    "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base",
    "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base",
    "geodesic-research/sfm_filtered_cpt_alignment_upsampled_base",
    "geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_base",
    "geodesic-research/sfm_unfiltered_cpt_misalignment_upsampled_base",
]

# Short display names for subplot titles
MODEL_DISPLAY_NAMES = {
    "geodesic-research/sfm_baseline_unfiltered_base": "Baseline Unfiltered",
    "geodesic-research/sfm_baseline_filtered_base": "Baseline Filtered",
    "geodesic-research/sfm_filtered_e2e_alignment_upsampled_base": "E2E Filtered Align",
    "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base": "E2E Unfiltered Align",
    "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base": "E2E Unfiltered Misalign",
    "geodesic-research/sfm_filtered_cpt_alignment_upsampled_base": "CPT Filtered Align",
    "geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_base": "CPT Unfiltered Align",
    "geodesic-research/sfm_unfiltered_cpt_misalignment_upsampled_base": "CPT Unfiltered Misalign",
}


def get_analysis_path(model_name: str) -> Path:
    """Get path to assistant_axis.json for a model."""
    model_safe = model_name.replace("/", "_")
    return BASE_PATH / model_safe / "main" / "analyses" / "assistant_axis.json"


def load_analysis(model_name: str) -> dict | None:
    """Load assistant_axis.json for a model."""
    path = get_analysis_path(model_name)
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    
    with open(path) as f:
        return json.load(f)


def main():
    # Load all model data
    model_data = {}
    for model in MODELS:
        data = load_analysis(model)
        if data:
            model_data[model] = data
    
    if not model_data:
        raise ValueError("No model data found!")
    
    print(f"Loaded data for {len(model_data)} models")
    
    # Create figure with subplots
    n_models = len(model_data)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    # Plot each model
    for idx, (model_name, data) in enumerate(model_data.items()):
        ax = axes[idx]
        
        roles = data["roles"]
        role_cosines = data["role_cosines_to_pcs"]
        assistant_cosines = data["assistant_cosine_to_pcs"]
        
        # Extract PC1 and PC2 cosines for all roles
        pc1_cosines = [role_cosines["PC1"][r] for r in roles]
        pc2_cosines = [role_cosines["PC2"][r] for r in roles]
        
        # Assistant values
        assistant_pc1 = assistant_cosines["PC1"]
        assistant_pc2 = assistant_cosines["PC2"]
        
        # Robot values (if present)
        robot_pc1 = role_cosines["PC1"].get("robot")
        robot_pc2 = role_cosines["PC2"].get("robot")
        
        # Plot all roles as small gray dots
        ax.scatter(pc1_cosines, pc2_cosines, c='gray', alpha=0.5, s=30, label='Roles')
        
        # Highlight robot with purple diamond
        if robot_pc1 is not None and robot_pc2 is not None:
            ax.scatter([robot_pc1], [robot_pc2], c='#9b59b6', marker='D', s=150, 
                      edgecolors='black', linewidth=1.5, zorder=10, label='Robot')
        
        # Highlight assistant with gold star
        ax.scatter([assistant_pc1], [assistant_pc2], c='gold', marker='*', s=400,
                  edgecolors='black', linewidth=1.5, zorder=10, label='Assistant')
        
        # Draw vector from origin to assistant
        ax.annotate('', xy=(assistant_pc1, assistant_pc2), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='gold', lw=2, alpha=0.7))
        
        # Formatting
        ax.set_xlabel('cos(·, PC1)')
        ax.set_ylabel('cos(·, PC2)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Title with model name and key stats
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.split("/")[-1])
        angle = np.degrees(np.arctan2(assistant_pc2, assistant_pc1))
        magnitude = np.sqrt(assistant_pc1**2 + assistant_pc2**2)
        ax.set_title(f'{display_name}\nθ={angle:.1f}°, |cos|={magnitude:.2f}', fontsize=10)
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)
    
    fig.suptitle('Role Positions in PC1-PC2 Space Across SFM Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: Assistant Position in PC1-PC2 Space")
    print("=" * 70)
    print(f"{'Model':<30} {'cos(PC1)':>10} {'cos(PC2)':>10} {'Angle':>8} {'|cos|':>8}")
    print("-" * 70)
    
    for model_name, data in model_data.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.split("/")[-1])[:28]
        pc1 = data["assistant_cosine_to_pcs"]["PC1"]
        pc2 = data["assistant_cosine_to_pcs"]["PC2"]
        angle = np.degrees(np.arctan2(pc2, pc1))
        magnitude = np.sqrt(pc1**2 + pc2**2)
        print(f"{display_name:<30} {pc1:>+10.3f} {pc2:>+10.3f} {angle:>7.1f}° {magnitude:>8.3f}")


if __name__ == "__main__":
    main()