"""Plot PCA histograms of persona space, replicating main figure from Assistant Axis paper."""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA histograms of persona activations"
    )
    
    parser.add_argument(
        "--activations-dir",
        type=str,
        required=True,
        help="Directory containing activation .pt files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for plot (default: activations_dir/pca_histogram.png)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to analyze (default: first available layer)",
    )
    parser.add_argument(
        "--assistant-role",
        type=str,
        default="assistant",
        help="Name of the assistant role to highlight",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for role selection",
    )
    
    return parser.parse_args()


def load_role_mean_activation(
    activations_path: Path,
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """
    Load activations for a role and compute mean.
    
    Returns mean activation vector of shape (hidden_dim,) or None if failed.
    """
    try:
        activations = torch.load(activations_path, map_location="cpu")
        
        if layer_idx not in activations:
            available = list(activations.keys())
            logger.warning(f"Layer {layer_idx} not in {activations_path}. Available: {available}")
            return None
        
        layer_acts = activations[layer_idx]
        
        # Handle both aggregated (tensor) and non-aggregated (list) formats
        if isinstance(layer_acts, list):
            # List of variable-length tensors - concatenate and mean
            all_acts = torch.cat([a.mean(dim=0, keepdim=True) for a in layer_acts], dim=0)
            mean_act = all_acts.mean(dim=0)
        else:
            # Already aggregated: (num_samples, hidden_dim)
            mean_act = layer_acts.mean(dim=0)
        
        return mean_act.float()  # Ensure float32 for PCA
        
    except Exception as e:
        logger.warning(f"Failed to load {activations_path}: {e}")
        return None


def get_available_roles(activations_dir: Path) -> list[str]:
    """Get list of roles with activation files."""
    roles = []
    for f in activations_dir.glob("*_activations.pt"):
        role = f.stem.replace("_activations", "")
        roles.append(role)
    return sorted(roles)


def select_highlighted_roles(
    role_projections: dict[str, float],
    assistant_role: str,
    n_per_group: int = 3,
    seed: int = 42,
) -> list[tuple[str, float, str]]:
    """
    Select roles to highlight on the plot.
    
    Returns list of (role_name, projection_value, group) tuples.
    group is one of: 'positive', 'middle', 'negative'
    """
    np.random.seed(seed)
    
    # Sort roles by projection
    sorted_roles = sorted(role_projections.items(), key=lambda x: x[1], reverse=True)
    
    # Remove assistant from selection pool
    sorted_roles = [(r, p) for r, p in sorted_roles if r != assistant_role]
    
    n_roles = len(sorted_roles)
    if n_roles < 30:
        # Not enough roles, just sample evenly
        step = max(1, n_roles // 9)
        indices = list(range(0, n_roles, step))[:9]
        selected = [(sorted_roles[i][0], sorted_roles[i][1], 'selected') for i in indices]
        return selected
    
    # Top 10 (most positive)
    top_10 = sorted_roles[:10]
    # Middle 10 (around zero)
    mid_start = n_roles // 2 - 5
    middle_10 = sorted_roles[mid_start:mid_start + 10]
    # Bottom 10 (most negative)
    bottom_10 = sorted_roles[-10:]
    
    # Randomly select 3 from each group
    selected = []
    
    top_selected = np.random.choice(len(top_10), size=min(n_per_group, len(top_10)), replace=False)
    for i in top_selected:
        selected.append((top_10[i][0], top_10[i][1], 'positive'))
    
    mid_selected = np.random.choice(len(middle_10), size=min(n_per_group, len(middle_10)), replace=False)
    for i in mid_selected:
        selected.append((middle_10[i][0], middle_10[i][1], 'middle'))
    
    bot_selected = np.random.choice(len(bottom_10), size=min(n_per_group, len(bottom_10)), replace=False)
    for i in bot_selected:
        selected.append((bottom_10[i][0], bottom_10[i][1], 'negative'))
    
    return selected


def plot_pca_histograms(
    role_means: dict[str, np.ndarray],
    assistant_role: str,
    seed: int = 42,
) -> plt.Figure:
    """
    Create PCA histogram plot.
    
    Args:
        role_means: Dict mapping role name to mean activation vector
        assistant_role: Name of assistant role to highlight
        seed: Random seed for role selection
        
    Returns:
        matplotlib Figure
    """
    # Stack all role means
    roles = list(role_means.keys())
    X = np.stack([role_means[r] for r in roles], axis=0)  # (n_roles, hidden_dim)
    
    logger.info(f"Running PCA on {X.shape[0]} roles, {X.shape[1]} dimensions")
    
    # Fit full PCA (n_components = min(n_samples, n_features) - 1)
    n_components = min(X.shape[0], X.shape[1]) - 1
    pca = PCA(n_components=n_components, whiten=True)
    projections = pca.fit_transform(X)  # (n_roles, n_components)
    
    logger.info(f"Computed {pca.n_components_} principal components")
    
    # Create role -> projection mapping for each PC
    role_to_idx = {r: i for i, r in enumerate(roles)}
    
    # Create figure: 3 PC histograms + 1 variance explained
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Colors for groups
    group_colors = {
        'positive': '#2ecc71',   # green
        'middle': '#3498db',     # blue  
        'negative': '#e74c3c',   # red
        'selected': '#95a5a6',   # gray (fallback)
    }
    
    # Plot top 3 PCs
    for pc_idx in range(min(3, projections.shape[1])):
        ax = axes[pc_idx]
        
        pc_values = projections[:, pc_idx]
        
        # Histogram
        ax.hist(pc_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Get projections for this PC
        role_projections = {r: projections[role_to_idx[r], pc_idx] for r in roles}
        
        # Mark assistant with star
        if assistant_role in role_to_idx:
            assistant_proj = role_projections[assistant_role]
            ax.axvline(assistant_proj, color='gold', linewidth=2, linestyle='--', alpha=0.8)
            # Star marker at bottom
            ax.scatter([assistant_proj], [0], marker='*', s=300, color='gold', 
                      edgecolor='black', linewidth=1, zorder=10, label='Assistant')
        
        # Select and mark other roles
        highlighted = select_highlighted_roles(role_projections, assistant_role, seed=seed + pc_idx)
        
        for role_name, proj_val, group in highlighted:
            color = group_colors.get(group, 'black')
            ax.scatter([proj_val], [0], marker='o', s=80, color=color,
                      edgecolor='black', linewidth=0.5, zorder=5)
            # Add label with slight offset
            ax.annotate(role_name, (proj_val, 0), textcoords="offset points",
                       xytext=(0, -15), ha='center', fontsize=7, rotation=45)
        
        variance_pct = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_xlabel(f'PC{pc_idx + 1} projection')
        ax.set_ylabel('Count')
        ax.set_title(f'PC{pc_idx + 1} ({variance_pct:.1f}% variance)')
        ax.axhline(0, color='black', linewidth=0.5)
        
        if pc_idx == 0:
            ax.legend(loc='upper right')
    
    # Cumulative variance explained plot
    ax = axes[3]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.3, color='steelblue')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title('Cumulative Variance Explained')
    ax.set_xlim(0.5, len(cumvar) + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add individual variance as bar plot
    ax2 = ax.twinx()
    ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_ * 100, 
            alpha=0.3, color='coral', label='Individual')
    ax2.set_ylabel('Individual variance (%)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    plt.tight_layout()
    
    return fig, pca, roles, projections


def main():
    args = parse_args()
    
    activations_dir = Path(args.activations_dir)
    
    if not activations_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Get available roles
    roles = get_available_roles(activations_dir)
    logger.info(f"Found {len(roles)} roles")
    
    if len(roles) == 0:
        raise ValueError(f"No activation files found in {activations_dir}")
    
    # Determine layer to use
    sample_file = activations_dir / f"{roles[0]}_activations.pt"
    sample_acts = torch.load(sample_file, map_location="cpu")
    available_layers = list(sample_acts.keys())
    
    if args.layer is not None:
        layer_idx = args.layer
        if layer_idx not in available_layers:
            raise ValueError(f"Layer {layer_idx} not available. Available: {available_layers}")
    else:
        layer_idx = available_layers[0]
    
    logger.info(f"Using layer {layer_idx}")
    
    # Load mean activations for each role
    role_means = {}
    
    for role in tqdm(roles, desc="Loading activations"):
        act_path = activations_dir / f"{role}_activations.pt"
        mean_act = load_role_mean_activation(act_path, layer_idx)
        
        if mean_act is not None:
            role_means[role] = mean_act.numpy()
    
    logger.info(f"Loaded {len(role_means)} role mean activations")
    
    if len(role_means) < 10:
        raise ValueError(f"Need at least 10 roles for meaningful PCA, got {len(role_means)}")
    
    # Check if assistant role exists
    if args.assistant_role not in role_means:
        logger.warning(f"Assistant role '{args.assistant_role}' not found. Available roles: {list(role_means.keys())[:10]}...")
        # Try to find a similar role
        similar = [r for r in role_means.keys() if 'assist' in r.lower()]
        if similar:
            args.assistant_role = similar[0]
            logger.info(f"Using '{args.assistant_role}' as assistant role")
    
    # Create plot
    fig, pca, roles_ordered, projections = plot_pca_histograms(
        role_means,
        args.assistant_role,
        seed=args.seed,
    )
    
    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = activations_dir / "pca_histogram.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    
    # Also save PCA results
    results = {
        "layer": layer_idx,
        "n_roles": len(role_means),
        "n_components": pca.n_components_,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "roles": roles_ordered,
        "assistant_role": args.assistant_role,
    }
    
    # Add role projections on top PCs
    role_to_idx = {r: i for i, r in enumerate(roles_ordered)}
    results["role_projections"] = {
        f"PC{i+1}": {r: float(projections[role_to_idx[r], i]) for r in roles_ordered}
        for i in range(min(3, projections.shape[1]))
    }
    
    results_path = output_path.with_suffix('.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved PCA results to {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("PCA Summary")
    print("="*50)
    print(f"Roles analyzed: {len(role_means)}")
    print(f"Layer: {layer_idx}")
    print(f"\nVariance explained:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var*100:.1f}%")
    print(f"\nCumulative (top 5): {np.sum(pca.explained_variance_ratio_[:5])*100:.1f}%")
    
    if args.assistant_role in role_to_idx:
        print(f"\nAssistant role projections:")
        idx = role_to_idx[args.assistant_role]
        for i in range(min(3, projections.shape[1])):
            print(f"  PC{i+1}: {projections[idx, i]:.3f}")


if __name__ == "__main__":
    main()
