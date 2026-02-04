"""Plot PCA analysis of persona space with cosine similarity histograms.

Methodology:
1. Load role activations (everything except 'default') and compute mean per role
2. Load default activations and compute the "assistant" vector
3. Fit PCA on role means only
4. Compute cosine similarity of each role/assistant to each PC direction
5. Plot histograms of cosine similarities with assistant marked

Saves to: <activations_dir>/analyses/assistant_axis.png and assistant_axis.json
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

# Role to always highlight in plots
ALWAYS_HIGHLIGHT_ROLE = "robot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA analysis with cosine similarity histograms"
    )
    
    parser.add_argument(
        "--activations-dir",
        type=str,
        required=True,
        help="Directory containing activation .pt files",
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
    parser.add_argument(
        "--filtering-dir",
        type=str,
        default=None,
        help="Directory containing filtering results (optional, ignored if --prep-activations)",
    )
    parser.add_argument(
        "--minimum-rating",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Minimum judge rating to include (default: 2, ignored if --prep-activations)",
    )
    parser.add_argument(
        "--prep-activations",
        action="store_true",
        help="Use prefill-only activations (from activation_preparation). "
             "Disables filtering since there are no responses to judge.",
    )
    
    return parser.parse_args()


def load_activation_mean(path: Path, layer_idx: int):
    """Load activations and compute mean across samples. Returns tensor or None."""
    try:
        activations = torch.load(path, map_location="cpu")
        
        if layer_idx not in activations:
            return None
        
        layer_acts = activations[layer_idx]
        
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
    """Separate activation files into role files and the default file."""
    role_files = []
    default_file = None
    
    for f in activations_dir.glob("*_activations.pt"):
        name = f.stem.replace("_activations", "")
        if name == "default":
            default_file = f
        else:
            role_files.append(f)
    
    return sorted(role_files), default_file


def load_filtering_results(filtering_dir: Path, role_name: str) -> list[dict] | None:
    """Load filtering results for a role."""
    filtering_path = filtering_dir / f"{role_name}.jsonl"
    if not filtering_path.exists():
        return None
    
    results = []
    with open(filtering_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_passing_indices(
    filtering_results: list[dict],
    minimum_rating: int,
) -> set[tuple[int, int, int]]:
    """Get set of (sp_idx, q_idx, r_idx) tuples that pass the filter."""
    passing = set()
    for r in filtering_results:
        try:
            response = r.get("judge_response", "")
            if response is None:
                continue
            rating = None
            for char in response.strip():
                if char.isdigit():
                    rating = int(char)
                    break
            
            if rating is not None and rating >= minimum_rating:
                key = (r["system_prompt_idx"], r["question_idx"], r["rollout_idx"])
                passing.add(key)
        except (ValueError, KeyError):
            continue
    return passing


def load_activation_mean_filtered(
    activations_path: Path,
    metadata_path: Path,
    layer_idx: int,
    passing_indices: set[tuple[int, int, int]] | None,
) -> tuple[torch.Tensor | None, int, int]:
    """Load activations and compute mean, optionally filtering."""
    try:
        activations = torch.load(activations_path, map_location="cpu")
        
        if layer_idx not in activations:
            return None, 0, 0
        
        layer_acts = activations[layer_idx]
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        num_total = len(metadata)
        
        if passing_indices is None:
            if isinstance(layer_acts, list):
                all_acts = torch.cat([a.mean(dim=0, keepdim=True) for a in layer_acts], dim=0)
                mean_act = all_acts.mean(dim=0)
            else:
                mean_act = layer_acts.mean(dim=0)
            return mean_act.float(), num_total, num_total
        
        kept_acts = []
        for i, meta in enumerate(metadata):
            key = (meta["system_prompt_idx"], meta["question_idx"], meta["rollout_idx"])
            if key in passing_indices:
                if isinstance(layer_acts, list):
                    kept_acts.append(layer_acts[i].mean(dim=0))
                else:
                    kept_acts.append(layer_acts[i])
        
        num_kept = len(kept_acts)
        
        if num_kept == 0:
            return None, 0, num_total
        
        stacked = torch.stack(kept_acts, dim=0)
        mean_act = stacked.mean(dim=0)
        
        return mean_act.float(), num_kept, num_total
        
    except Exception as e:
        logger.warning(f"Failed to load {activations_path}: {e}")
        return None, 0, 0


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def select_highlighted_roles(
    role_cosines: dict[str, float],
    n_per_group: int = 3,
    seed: int = 42,
    always_include: str | None = ALWAYS_HIGHLIGHT_ROLE,
) -> list[tuple[str, float, str]]:
    """Select roles to highlight: 3 from top, 3 from middle, 3 from bottom, plus always_include."""
    np.random.seed(seed)
    
    sorted_roles = sorted(role_cosines.items(), key=lambda x: x[1], reverse=True)
    n_roles = len(sorted_roles)
    
    selected = []
    selected_names = set()
    
    # Always include the specified role if it exists
    if always_include and always_include in role_cosines:
        selected.append((always_include, role_cosines[always_include], 'always'))
        selected_names.add(always_include)
    
    if n_roles < 30:
        step = max(1, n_roles // 9)
        indices = list(range(0, n_roles, step))[:9]
        for i in indices:
            name, val = sorted_roles[i]
            if name not in selected_names:
                selected.append((name, val, 'selected'))
                selected_names.add(name)
        return selected
    
    # Top 10 (most positive cosine)
    top_10 = [(name, val) for name, val in sorted_roles[:10] if name not in selected_names]
    if top_10:
        top_idx = np.random.choice(len(top_10), size=min(n_per_group, len(top_10)), replace=False)
        for i in top_idx:
            selected.append((top_10[i][0], top_10[i][1], 'positive'))
            selected_names.add(top_10[i][0])
    
    # Middle 10
    mid_start = n_roles // 2 - 5
    middle_10 = [(name, val) for name, val in sorted_roles[mid_start:mid_start + 10] if name not in selected_names]
    if middle_10:
        mid_idx = np.random.choice(len(middle_10), size=min(n_per_group, len(middle_10)), replace=False)
        for i in mid_idx:
            selected.append((middle_10[i][0], middle_10[i][1], 'middle'))
            selected_names.add(middle_10[i][0])
    
    # Bottom 10 (most negative cosine)
    bottom_10 = [(name, val) for name, val in sorted_roles[-10:] if name not in selected_names]
    if bottom_10:
        bot_idx = np.random.choice(len(bottom_10), size=min(n_per_group, len(bottom_10)), replace=False)
        for i in bot_idx:
            selected.append((bottom_10[i][0], bottom_10[i][1], 'negative'))
            selected_names.add(bottom_10[i][0])
    
    return selected


def main():
    args = parse_args()
    
    activations_dir = Path(args.activations_dir)
    if not activations_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Setup output directory
    output_dir = activations_dir / "analyses"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup filtering (disabled for prep activations)
    filtering_dir = None
    if args.prep_activations:
        logger.info("Using prefill activations - filtering disabled")
        if args.filtering_dir:
            logger.warning("--filtering-dir ignored when using --prep-activations")
    else:
        filtering_dir = Path(args.filtering_dir) if args.filtering_dir else None
        if filtering_dir and not filtering_dir.exists():
            raise FileNotFoundError(f"Filtering directory not found: {filtering_dir}")
    
    # Separate role and default files
    role_files, default_file = get_activation_files(activations_dir)
    logger.info(f"Found {len(role_files)} role files")
    
    if len(role_files) == 0:
        raise ValueError("No role activation files found")
    if default_file is None:
        raise ValueError("No default activation file found")
    
    # Determine layer
    sample_acts = torch.load(role_files[0], map_location="cpu")
    available_layers = list(sample_acts.keys())
    layer_idx = args.layer if args.layer is not None else available_layers[0]
    logger.info(f"Using layer {layer_idx}")
    
    # Load role means
    role_means = {}
    role_counts = {}
    roles_missing_filtering = []
    roles_no_passing = []
    
    for f in tqdm(role_files, desc="Loading role activations"):
        name = f.stem.replace("_activations", "")
        metadata_path = f.parent / f"{name}_metadata.json"
        
        passing_indices = None
        if filtering_dir:
            filtering_results = load_filtering_results(filtering_dir, name)
            if filtering_results is None:
                roles_missing_filtering.append(name)
                continue
            passing_indices = get_passing_indices(filtering_results, args.minimum_rating)
            if len(passing_indices) == 0:
                roles_no_passing.append(name)
                continue
        
        mean_act, num_kept, num_total = load_activation_mean_filtered(
            f, metadata_path, layer_idx, passing_indices
        )
        
        if mean_act is not None:
            role_means[name] = mean_act.numpy()
            role_counts[name] = (num_kept, num_total)
    
    logger.info(f"Loaded {len(role_means)} role means")
    
    if len(role_means) == 0:
        raise ValueError("No roles loaded!")
    
    # Load assistant vector
    assistant_vector = load_activation_mean(default_file, layer_idx)
    if assistant_vector is None:
        raise ValueError("Failed to load default activations")
    assistant_vector = assistant_vector.numpy()
    logger.info("Loaded assistant vector")
    
    # Stack role means for PCA
    roles = list(role_means.keys())
    X = np.stack([role_means[r] for r in roles], axis=0)
    
    logger.info(f"Running PCA on {X.shape[0]} roles, {X.shape[1]} dimensions")
    
    # Fit PCA on roles (NOT including assistant)
    # Center on role mean
    role_mean = X.mean(axis=0)
    X_centered = X - role_mean
    assistant_centered = assistant_vector - role_mean
    
    n_components = min(X.shape[0], X.shape[1]) - 1
    pca = PCA(n_components=n_components, whiten=False)  # Don't whiten to preserve direction
    pca.fit(X_centered)
    
    logger.info(f"Computed {pca.n_components_} principal components")
    
    # PCs are unit vectors (sklearn normalizes them)
    # Compute cosine similarity of each centered role vector to each PC
    # cos(role, PC_i) = (role · PC_i) / ||role||
    
    role_cosines = {}  # {role: {PC1: cos, PC2: cos, ...}}
    for i, role in enumerate(roles):
        role_vec = X_centered[i]
        role_norm = np.linalg.norm(role_vec)
        role_cosines[role] = {}
        for pc_idx in range(pca.n_components_):
            pc_vec = pca.components_[pc_idx]
            cos_sim = np.dot(role_vec, pc_vec) / role_norm if role_norm > 0 else 0
            role_cosines[role][f"PC{pc_idx+1}"] = float(cos_sim)
    
    # Compute cosine similarity of assistant to each PC
    assistant_norm = np.linalg.norm(assistant_centered)
    assistant_cosines = {}
    for pc_idx in range(pca.n_components_):
        pc_vec = pca.components_[pc_idx]
        cos_sim = np.dot(assistant_centered, pc_vec) / assistant_norm if assistant_norm > 0 else 0
        assistant_cosines[f"PC{pc_idx+1}"] = float(cos_sim)
    
    # Compute Assistant Axis = assistant_centered / ||assistant_centered||
    assistant_axis = assistant_centered / assistant_norm if assistant_norm > 0 else assistant_centered
    
    # Cosine similarity between Assistant Axis and PC1
    pc1_norm = np.linalg.norm(pca.components_[0])
    axis_pc1_cosine = float(np.dot(assistant_axis, pca.components_[0]) / pc1_norm)
    
    logger.info(f"Assistant Axis ↔ PC1 cosine similarity: {axis_pc1_cosine:.3f}")
    
    # --- Plotting ---
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    
    group_colors = {
        'positive': '#2ecc71',
        'middle': '#3498db',
        'negative': '#e74c3c',
        'selected': '#95a5a6',
        'always': '#9b59b6',  # Purple for always-highlighted role (robot)
    }
    
    # Plot top 3 PCs - now showing COSINE SIMILARITY
    for pc_idx in range(min(3, pca.n_components_)):
        ax = axes[pc_idx]
        pc_name = f"PC{pc_idx+1}"
        
        # Get cosine similarities for all roles to this PC
        cosine_values = np.array([role_cosines[r][pc_name] for r in roles])
        assistant_cos = assistant_cosines[pc_name]
        
        # Histogram of role cosine similarities
        ax.hist(cosine_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Mark assistant with star
        ax.axvline(assistant_cos, color='gold', linewidth=2, linestyle='--', alpha=0.8)
        ax.scatter([assistant_cos], [0], marker='*', s=300, color='gold',
                  edgecolor='black', linewidth=1, zorder=10, label='Assistant')
        
        # Select and mark other roles
        role_pc_cosines = {r: role_cosines[r][pc_name] for r in roles}
        highlighted = select_highlighted_roles(role_pc_cosines, seed=args.seed + pc_idx)
        
        for role_name, cos_val, group in highlighted:
            color = group_colors.get(group, 'black')
            # Make 'always' role (robot) more prominent
            marker_size = 120 if group == 'always' else 80
            marker = 'D' if group == 'always' else 'o'  # Diamond for robot
            ax.scatter([cos_val], [0], marker=marker, s=marker_size, color=color,
                      edgecolor='black', linewidth=0.5, zorder=5)
            ax.annotate(role_name, (cos_val, 0), textcoords="offset points",
                       xytext=(0, -15), ha='center', fontsize=7, rotation=45,
                       fontweight='bold' if group == 'always' else 'normal')
        
        variance_pct = pca.explained_variance_ratio_[pc_idx] * 100
        ax.set_xlabel(f'Cosine similarity to {pc_name}')
        ax.set_ylabel('Count')
        ax.set_title(f'{pc_name} ({variance_pct:.1f}% variance) - Cosine Similarity Distribution')
        ax.set_xlim(-1, 1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
        
        if pc_idx == 0:
            ax.legend(loc='upper right')
    
    # Bottom plot: Cumulative variance + Assistant ABSOLUTE cosine similarity per PC
    ax = axes[3]
    
    n_pcs_to_show = pca.n_components_  # Show ALL PCs
    
    # Plot cumulative variance (left axis)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, n_pcs_to_show + 1), cumvar[:n_pcs_to_show], 'o-', color='steelblue', 
            linewidth=2, markersize=4, label='Cumulative variance')
    ax.fill_between(range(1, n_pcs_to_show + 1), cumvar[:n_pcs_to_show], alpha=0.2, color='steelblue')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative variance explained (%)', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_xlim(0.5, n_pcs_to_show + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Plot assistant ABSOLUTE cosine similarity on secondary y-axis
    ax2 = ax.twinx()
    assistant_cos_values = [
        np.dot(assistant_centered, pca.components_[i]) / assistant_norm
        for i in range(n_pcs_to_show)
    ]
    assistant_abs_cos_values = [abs(c) for c in assistant_cos_values]
    ax2.plot(range(1, n_pcs_to_show + 1), assistant_abs_cos_values, 's-', color='coral',
             linewidth=2, markersize=4, label='Assistant |cos(a, PCᵢ)|')
    ax2.set_ylabel('Assistant |cosine similarity| to PC', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
    
    ax.set_title(f'Variance & Assistant-PC Cosine Similarity (Axis↔PC1: {axis_pc1_cosine:.2f})')
    
    # Figure title
    if args.prep_activations:
        suptitle = f"Assistant Axis Analysis: {len(roles)} roles (prefill activations)"
    elif filtering_dir:
        total_kept = sum(c[0] for c in role_counts.values())
        total_samples = sum(c[1] for c in role_counts.values())
        suptitle = f"Assistant Axis Analysis: {len(roles)} roles (filtered: rating ≥ {args.minimum_rating}, {total_kept:,}/{total_samples:,} samples)"
    else:
        suptitle = f"Assistant Axis Analysis: {len(roles)} roles (unfiltered)"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "assistant_axis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    
    # --- Save results JSON ---
    
    # Add robot-specific data if available
    robot_cosines = None
    if ALWAYS_HIGHLIGHT_ROLE in role_cosines:
        robot_cosines = role_cosines[ALWAYS_HIGHLIGHT_ROLE]
    
    results = {
        "layer": layer_idx,
        "n_roles": len(role_means),
        "n_components": pca.n_components_,
        "prep_activations": args.prep_activations,
        "filtering": {
            "enabled": filtering_dir is not None and not args.prep_activations,
            "minimum_rating": args.minimum_rating if filtering_dir and not args.prep_activations else None,
            "roles_missing_filtering": roles_missing_filtering if filtering_dir else [],
            "roles_no_passing": roles_no_passing if filtering_dir else [],
            "total_samples_kept": sum(c[0] for c in role_counts.values()) if role_counts else None,
            "total_samples": sum(c[1] for c in role_counts.values()) if role_counts else None,
        },
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "assistant_axis_pc1_cosine": axis_pc1_cosine,
        "assistant_norm": float(assistant_norm),
        "roles": roles,
        "role_sample_counts": {r: {"kept": c[0], "total": c[1]} for r, c in role_counts.items()} if role_counts else {},
        "assistant_cosine_to_pcs": assistant_cosines,  # All PCs
        "assistant_abs_cosine_to_pcs": {
            f"PC{i+1}": abs(assistant_cosines[f"PC{i+1}"]) 
            for i in range(pca.n_components_)
        },
        "robot_cosine_to_pcs": robot_cosines,  # Robot role cosines (or None if not present)
        "assistant_percentile": {
            pc_name: float((np.array([role_cosines[r][pc_name] for r in roles]) < assistant_cosines[pc_name]).mean() * 100)
            for pc_name in assistant_cosines.keys()
        },
        "role_cosines_to_pcs": {
            pc_name: {r: role_cosines[r][pc_name] for r in roles}
            for pc_name in [f"PC{i+1}" for i in range(pca.n_components_)]
        },
    }
    
    results_path = output_dir / "assistant_axis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Assistant Axis Analysis Summary")
    print("=" * 60)
    print(f"Roles: {len(role_means)}")
    print(f"Layer: {layer_idx}")
    
    if args.prep_activations:
        print(f"\nMode: Prefill activations (no filtering)")
    elif filtering_dir:
        total_kept = sum(c[0] for c in role_counts.values())
        total_samples = sum(c[1] for c in role_counts.values())
        print(f"\nFiltering: rating ≥ {args.minimum_rating}")
        print(f"  Samples: {total_kept:,} / {total_samples:,} ({100*total_kept/total_samples:.1f}% kept)")
    
    print(f"\nAssistant Axis ↔ PC1 cosine similarity: {axis_pc1_cosine:.3f}")
    print(f"  (Paper reports >0.71 at middle layer)")
    
    print(f"\nVariance explained:")
    for i in range(min(5, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%")
    
    print(f"\nAssistant cosine similarity to PCs:")
    print(f"  ||a|| = {assistant_norm:.3f} (centered distance from role mean)")
    for i in range(min(10, pca.n_components_)):
        pc_name = f"PC{i+1}"
        cos_val = assistant_cosines[pc_name]
        abs_cos_val = abs(cos_val)
        all_cosines = np.array([role_cosines[r][pc_name] for r in roles])
        percentile = (all_cosines < cos_val).mean() * 100
        print(f"  {pc_name}: {cos_val:+.3f} (|cos|={abs_cos_val:.3f}, percentile: {percentile:.1f}%)")
    
    if pca.n_components_ > 10:
        print(f"  ... ({pca.n_components_ - 10} more PCs in JSON)")


if __name__ == "__main__":
    main()