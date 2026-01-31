"""Plot PCA histograms of persona space, replicating main figure from Assistant Axis paper.

Methodology:
1. Load role activations (everything except 'default') and compute mean per role
2. Load default activations and compute the "assistant" vector
3. Fit PCA on role means only
4. Project assistant vector into PCA space
5. Compute Assistant Axis = assistant_vector - mean(all_role_vectors)
6. Plot histograms with assistant marked
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
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
        help="Layer index to analyze (default: first available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for role selection",
    )
    
    return parser.parse_args()


def load_activation_mean(path: Path, layer_idx: int):
    """Load activations and compute mean across samples. Returns tensor or None."""
    try:
        activations = torch.load(path, map_location="cpu")
        
        if layer_idx not in activations:
            return None
        
        layer_acts = activations[layer_idx]
        
        # Handle both aggregated (tensor) and non-aggregated (list) formats
        if isinstance(layer_acts, list):
            all_acts = torch.cat([a.mean(dim=0, keepdim=True) for a in layer_acts], dim=0)
            mean_act = all_acts.mean(dim=0)
        else:
            mean_act = layer_acts.mean(dim=0)
        
        return mean_act.float()
        
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def get_activation_files(activations_dir: Path) -> tuple[list[Path], Path | None]:
    """
    Separate activation files into role files and the default file.
    
    Returns:
        (role_files, default_file)
    """
    role_files = []
    default_file = None
    
    for f in activations_dir.glob("*_activations.pt"):
        name = f.stem.replace("_activations", "")
        if name == "default":
            default_file = f
        else:
            role_files.append(f)
    
    return sorted(role_files), default_file


def select_highlighted_roles(
    role_projections: dict[str, float],
    n_per_group: int = 3,
    seed: int = 42,
) -> list[tuple[str, float, str]]:
    """
    Select roles to highlight: 3 from top, 3 from middle, 3 from bottom.
    
    Returns list of (role_name, projection_value, group) tuples.
    """
    np.random.seed(seed)
    
    sorted_roles = sorted(role_projections.items(), key=lambda x: x[1], reverse=True)
    n_roles = len(sorted_roles)
    
    if n_roles < 30:
        step = max(1, n_roles // 9)
        indices = list(range(0, n_roles, step))[:9]
        return [(sorted_roles[i][0], sorted_roles[i][1], 'selected') for i in indices]
    
    selected = []
    
    # Top 10 (most positive)
    top_10 = sorted_roles[:10]
    top_idx = np.random.choice(len(top_10), size=min(n_per_group, len(top_10)), replace=False)
    for i in top_idx:
        selected.append((top_10[i][0], top_10[i][1], 'positive'))
    
    # Middle 10
    mid_start = n_roles // 2 - 5
    middle_10 = sorted_roles[mid_start:mid_start + 10]
    mid_idx = np.random.choice(len(middle_10), size=min(n_per_group, len(middle_10)), replace=False)
    for i in mid_idx:
        selected.append((middle_10[i][0], middle_10[i][1], 'middle'))
    
    # Bottom 10 (most negative)
    bottom_10 = sorted_roles[-10:]
    bot_idx = np.random.choice(len(bottom_10), size=min(n_per_group, len(bottom_10)), replace=False)
    for i in bot_idx:
        selected.append((bottom_10[i][0], bottom_10[i][1], 'negative'))
    
    return selected


def main():
    args = parse_args()
    
    activations_dir = Path(args.activations_dir)
    if not activations_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Separate role and default files
    role_files, default_file = get_activation_files(activations_dir)
    logger.info(f"Found {len(role_files)} role files")
    
    if len(role_files) == 0:
        raise ValueError("No role activation files found")
    if default_file is None:
        raise ValueError("No default activation file found (need default_activations.pt)")
    
    logger.info(f"Default file: {default_file}")
    
    # Determine layer to use
    sample_acts = torch.load(role_files[0], map_location="cpu")
    available_layers = list(sample_acts.keys())
    
    if args.layer is not None:
        if args.layer not in available_layers:
            raise ValueError(f"Layer {args.layer} not available. Available: {available_layers}")
        layer_idx = args.layer
    else:
        layer_idx = available_layers[0]
    
    logger.info(f"Using layer {layer_idx}")
    
    # Load role means
    role_means = {}
    for f in tqdm(role_files, desc="Loading role activations"):
        name = f.stem.replace("_activations", "")
        mean_act = load_activation_mean(f, layer_idx)
        if mean_act is not None:
            role_means[name] = mean_act.numpy()
    
    logger.info(f"Loaded {len(role_means)} role means")
    
    # Load default (assistant) vector
    assistant_vector = load_activation_mean(default_file, layer_idx)
    if assistant_vector is None:
        raise ValueError("Failed to load default activations")
    assistant_vector = assistant_vector.numpy()
    logger.info("Loaded assistant vector from default file")
    
    # Stack role means for PCA
    roles = list(role_means.keys())
    X = np.stack([role_means[r] for r in roles], axis=0)  # (n_roles, hidden_dim)
    
    logger.info(f"Running PCA on {X.shape[0]} roles, {X.shape[1]} dimensions")
    
    # Fit PCA on roles only (NOT including assistant)
    n_components = min(X.shape[0], X.shape[1]) - 1
    pca = PCA(n_components=n_components, whiten=True)
    role_projections = pca.fit_transform(X)  # (n_roles, n_components)
    
    logger.info(f"Computed {pca.n_components_} principal components")
    
    # Project assistant vector into PCA space
    assistant_projection = pca.transform(assistant_vector.reshape(1, -1))[0]
    
    # Compute Assistant Axis: assistant - mean(roles)
    role_mean = X.mean(axis=0)
    assistant_axis = assistant_vector - role_mean
    
    # Cosine similarity between Assistant Axis and PC1
    assistant_axis_norm = assistant_axis / np.linalg.norm(assistant_axis)
    pc1_norm = pca.components_[0] / np.linalg.norm(pca.components_[0])
    axis_pc1_cosine = float(np.dot(assistant_axis_norm, pc1_norm))
    
    logger.info(f"Assistant Axis ↔ PC1 cosine similarity: {axis_pc1_cosine:.3f}")
    
    # --- Plotting ---
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    group_colors = {
        'positive': '#2ecc71',
        'middle': '#3498db',
        'negative': '#e74c3c',
        'selected': '#95a5a6',
    }
    
    role_to_idx = {r: i for i, r in enumerate(roles)}
    
    # Plot top 3 PCs
    for pc_idx in range(min(3, role_projections.shape[1])):
        ax = axes[pc_idx]
        
        pc_values = role_projections[:, pc_idx]
        assistant_pc = assistant_projection[pc_idx]
        
        # Histogram of role projections
        ax.hist(pc_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Mark assistant with star
        ax.axvline(assistant_pc, color='gold', linewidth=2, linestyle='--', alpha=0.8)
        ax.scatter([assistant_pc], [0], marker='*', s=300, color='gold',
                  edgecolor='black', linewidth=1, zorder=10, label='Assistant')
        
        # Select and mark other roles
        role_pc_values = {r: role_projections[role_to_idx[r], pc_idx] for r in roles}
        highlighted = select_highlighted_roles(role_pc_values, seed=args.seed + pc_idx)
        
        for role_name, proj_val, group in highlighted:
            color = group_colors.get(group, 'black')
            ax.scatter([proj_val], [0], marker='o', s=80, color=color,
                      edgecolor='black', linewidth=0.5, zorder=5)
            ax.annotate(role_name, (proj_val, 0), textcoords="offset points",
                       xytext=(0, -15), ha='center', fontsize=7, rotation=45)
        
        variance_pct = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_xlabel(f'PC{pc_idx + 1} projection')
        ax.set_ylabel('Count')
        ax.set_title(f'PC{pc_idx + 1} ({variance_pct:.1f}% variance)')
        ax.axhline(0, color='black', linewidth=0.5)
        
        if pc_idx == 0:
            ax.legend(loc='upper right')
    
    # Cumulative variance explained
    ax = axes[3]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='steelblue', 
            linewidth=2, markersize=4)
    ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.3, color='steelblue')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained (%)')
    ax.set_title(f'Cumulative Variance (Assistant Axis ↔ PC1: {axis_pc1_cosine:.2f})')
    ax.set_xlim(0.5, min(50, len(cumvar)) + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add individual variance as bars
    ax2 = ax.twinx()
    n_bars = min(50, len(pca.explained_variance_ratio_))
    ax2.bar(range(1, n_bars + 1), 
            pca.explained_variance_ratio_[:n_bars] * 100,
            alpha=0.3, color='coral')
    ax2.set_ylabel('Individual variance (%)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    plt.tight_layout()
    
    # Save plot
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = activations_dir / "pca_histogram.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    
    # Save results JSON
    results = {
        "layer": layer_idx,
        "n_roles": len(role_means),
        "n_components": pca.n_components_,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "assistant_axis_pc1_cosine": axis_pc1_cosine,
        "roles": roles,
        "assistant_projections": {
            f"PC{i+1}": float(assistant_projection[i]) 
            for i in range(min(10, len(assistant_projection)))
        },
        "role_projections": {
            f"PC{i+1}": {r: float(role_projections[role_to_idx[r], i]) for r in roles}
            for i in range(min(3, role_projections.shape[1]))
        },
    }
    
    results_path = output_path.with_suffix('.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PCA Summary")
    print("=" * 60)
    print(f"Roles: {len(role_means)}")
    print(f"Layer: {layer_idx}")
    print(f"\nAssistant Axis ↔ PC1 cosine similarity: {axis_pc1_cosine:.3f}")
    print(f"  (Paper reports >0.71 at middle layer)")
    print(f"\nVariance explained:")
    for i in range(min(5, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%")
    print(f"\nAssistant projection:")
    for i in range(min(3, len(assistant_projection))):
        print(f"  PC{i+1}: {assistant_projection[i]:.3f}")


if __name__ == "__main__":
    main()