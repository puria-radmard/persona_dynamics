"""Clustering analysis of role activations.

Performs:
1. Hierarchical clustering with dendrogram
2. K-means with silhouette analysis
3. 2D visualization (t-SNE) colored by cluster
4. Pairwise cosine similarity heatmap

Saves to: <activations_dir>/analyses/clustering.png and clustering.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clustering analysis of role activations"
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
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters for k-means (default: auto-select via silhouette)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=15,
        help="Maximum clusters to try for auto-selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--filtering-dir",
        type=str,
        default=None,
        help="Directory containing filtering results (optional)",
    )
    parser.add_argument(
        "--minimum-rating",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Minimum judge rating to include (default: 2)",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity (default: 30)",
    )
    
    return parser.parse_args()


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


def load_activation_mean(path: Path, layer_idx: int):
    """Load activations and compute mean across samples."""
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


def find_optimal_clusters(X: np.ndarray, max_clusters: int, seed: int) -> tuple[int, list[float]]:
    """Find optimal number of clusters using silhouette score."""
    silhouette_scores = []
    
    for k in range(2, min(max_clusters + 1, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append((k, score))
    
    best_k = max(silhouette_scores, key=lambda x: x[1])[0]
    return best_k, silhouette_scores


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    activations_dir = Path(args.activations_dir)
    if not activations_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    # Setup output directory
    output_dir = activations_dir / "analyses"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup filtering
    filtering_dir = Path(args.filtering_dir) if args.filtering_dir else None
    
    # Get activation files
    role_files, default_file = get_activation_files(activations_dir)
    logger.info(f"Found {len(role_files)} role files")
    
    if len(role_files) == 0:
        raise ValueError("No role activation files found")
    
    # Determine layer
    sample_acts = torch.load(role_files[0], map_location="cpu")
    available_layers = list(sample_acts.keys())
    layer_idx = args.layer if args.layer is not None else available_layers[0]
    logger.info(f"Using layer {layer_idx}")
    
    # Load role means
    role_means = {}
    role_counts = {}
    
    for f in tqdm(role_files, desc="Loading role activations"):
        name = f.stem.replace("_activations", "")
        metadata_path = f.parent / f"{name}_metadata.json"
        
        passing_indices = None
        if filtering_dir:
            filtering_results = load_filtering_results(filtering_dir, name)
            if filtering_results is None:
                continue
            passing_indices = get_passing_indices(filtering_results, args.minimum_rating)
            if len(passing_indices) == 0:
                continue
        
        mean_act, num_kept, num_total = load_activation_mean_filtered(
            f, metadata_path, layer_idx, passing_indices
        )
        
        if mean_act is not None:
            role_means[name] = mean_act.numpy()
            role_counts[name] = (num_kept, num_total)
    
    logger.info(f"Loaded {len(role_means)} roles")
    
    if len(role_means) < 3:
        raise ValueError("Need at least 3 roles for clustering")
    
    # Load default (assistant) vector if available
    assistant_vector = None
    if default_file is not None:
        assistant_vector = load_activation_mean(default_file, layer_idx)
        if assistant_vector is not None:
            assistant_vector = assistant_vector.numpy()
            logger.info("Loaded assistant vector")
    
    # Prepare data
    roles = list(role_means.keys())
    X = np.stack([role_means[r] for r in roles], axis=0)
    X_normalized = normalize(X)  # L2 normalize for cosine-based methods
    
    logger.info(f"Data shape: {X.shape}")
    
    # --- Clustering Analysis ---
    
    # 1. Find optimal k using silhouette
    if args.n_clusters is None:
        logger.info("Finding optimal number of clusters...")
        best_k, silhouette_scores = find_optimal_clusters(X_normalized, args.max_clusters, args.seed)
        logger.info(f"Optimal k: {best_k}")
    else:
        best_k = args.n_clusters
        silhouette_scores = []
    
    # 2. K-means clustering
    kmeans = KMeans(n_clusters=best_k, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_normalized)
    final_silhouette = silhouette_score(X_normalized, cluster_labels)
    sample_silhouettes = silhouette_samples(X_normalized, cluster_labels)
    
    logger.info(f"K-means silhouette score: {final_silhouette:.3f}")
    
    # 3. Hierarchical clustering
    cosine_distances = pdist(X_normalized, metric='cosine')
    linkage_matrix = linkage(cosine_distances, method='ward')
    hierarchical_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust')
    
    # 4. t-SNE embedding
    perplexity = min(args.tsne_perplexity, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, metric='cosine')
    X_tsne = tsne.fit_transform(X_normalized)
    
    # Project assistant if available
    assistant_tsne = None
    if assistant_vector is not None:
        # Fit t-SNE with assistant included, then extract
        X_with_assistant = np.vstack([X_normalized, normalize(assistant_vector.reshape(1, -1))])
        tsne_with_assistant = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, metric='cosine')
        X_tsne_all = tsne_with_assistant.fit_transform(X_with_assistant)
        X_tsne = X_tsne_all[:-1]
        assistant_tsne = X_tsne_all[-1]
    
    # 5. Pairwise cosine similarity matrix
    cosine_sim_matrix = 1 - squareform(cosine_distances)
    np.fill_diagonal(cosine_sim_matrix, 1.0)
    
    # --- Plotting ---
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # Color palette for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    
    # Plot 1: Dendrogram (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Truncate dendrogram if too many roles
    truncate_mode = 'lastp' if len(roles) > 50 else None
    p = min(30, len(roles)) if truncate_mode else len(roles)
    
    dendrogram(
        linkage_matrix,
        labels=roles if len(roles) <= 50 else None,
        leaf_rotation=90,
        leaf_font_size=6 if len(roles) <= 50 else 4,
        ax=ax1,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=linkage_matrix[-best_k+1, 2] if best_k > 1 else 0,
    )
    ax1.set_title(f'Hierarchical Clustering Dendrogram (cosine distance, Ward linkage)', fontsize=12)
    ax1.set_xlabel('Role' if len(roles) <= 50 else 'Cluster')
    ax1.set_ylabel('Distance')
    
    # Plot 2: Silhouette analysis (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    if silhouette_scores:
        ks, scores = zip(*silhouette_scores)
        ax2.plot(ks, scores, 'o-', linewidth=2, markersize=8)
        ax2.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'Selected k={best_k}')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('Silhouette score')
        ax2.set_title('Silhouette Score vs. Number of Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f'k={best_k} (user-specified)\nSilhouette: {final_silhouette:.3f}',
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.set_title('Cluster Selection')
    
    # Plot 3: t-SNE visualization (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    scatter = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10',
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add assistant point if available
    if assistant_tsne is not None:
        ax3.scatter([assistant_tsne[0]], [assistant_tsne[1]], marker='*', s=400,
                   c='gold', edgecolors='black', linewidth=2, zorder=10, label='Assistant')
        ax3.legend(loc='upper right')
    
    # Label some points
    for i, role in enumerate(roles):
        if i % max(1, len(roles) // 20) == 0:  # Label every ~5%
            ax3.annotate(role, (X_tsne[i, 0], X_tsne[i, 1]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
    
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title(f't-SNE Visualization (k={best_k} clusters, silhouette={final_silhouette:.3f})')
    
    # Plot 4: Cluster composition (bottom-middle)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Show roles in each cluster
    cluster_roles = {i: [] for i in range(best_k)}
    for role, label in zip(roles, cluster_labels):
        cluster_roles[label].append(role)
    
    # Sort clusters by size
    sorted_clusters = sorted(cluster_roles.items(), key=lambda x: -len(x[1]))
    
    y_pos = 0
    y_positions = []
    cluster_sizes = []
    for cluster_id, cluster_role_list in sorted_clusters:
        cluster_sizes.append(len(cluster_role_list))
        y_positions.append(y_pos)
        y_pos += 1
    
    bars = ax4.barh(range(len(sorted_clusters)), cluster_sizes, color=[colors[c[0]] for c in sorted_clusters])
    ax4.set_yticks(range(len(sorted_clusters)))
    ax4.set_yticklabels([f'Cluster {c[0]}' for c in sorted_clusters])
    ax4.set_xlabel('Number of roles')
    ax4.set_title('Cluster Sizes')
    
    # Add role names as text
    for i, (cluster_id, cluster_role_list) in enumerate(sorted_clusters):
        # Show first few roles
        display_roles = cluster_role_list[:5]
        suffix = f"... +{len(cluster_role_list)-5}" if len(cluster_role_list) > 5 else ""
        text = ", ".join(display_roles) + suffix
        ax4.text(cluster_sizes[i] + 0.5, i, text, va='center', fontsize=6, alpha=0.8)
    
    ax4.set_xlim(0, max(cluster_sizes) * 2)
    
    # Plot 5: Cosine similarity heatmap (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Reorder by cluster
    cluster_order = np.argsort(cluster_labels)
    reordered_sim = cosine_sim_matrix[cluster_order][:, cluster_order]
    reordered_roles = [roles[i] for i in cluster_order]
    
    im = ax5.imshow(reordered_sim, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Add cluster boundaries
    cluster_boundaries = []
    current_cluster = cluster_labels[cluster_order[0]]
    for i, idx in enumerate(cluster_order):
        if cluster_labels[idx] != current_cluster:
            cluster_boundaries.append(i - 0.5)
            current_cluster = cluster_labels[idx]
    
    for boundary in cluster_boundaries:
        ax5.axhline(boundary, color='black', linewidth=1)
        ax5.axvline(boundary, color='black', linewidth=1)
    
    plt.colorbar(im, ax=ax5, label='Cosine similarity')
    ax5.set_title('Pairwise Cosine Similarity (ordered by cluster)')
    
    if len(roles) <= 30:
        ax5.set_xticks(range(len(roles)))
        ax5.set_yticks(range(len(roles)))
        ax5.set_xticklabels(reordered_roles, rotation=90, fontsize=5)
        ax5.set_yticklabels(reordered_roles, fontsize=5)
    
    # Title
    if filtering_dir:
        total_kept = sum(c[0] for c in role_counts.values())
        total_samples = sum(c[1] for c in role_counts.values())
        suptitle = f"Clustering Analysis: {len(roles)} roles (filtered: rating ≥ {args.minimum_rating})"
    else:
        suptitle = f"Clustering Analysis: {len(roles)} roles"
    
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "clustering.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    
    # --- Save results JSON ---
    
    results = {
        "layer": layer_idx,
        "n_roles": len(roles),
        "n_clusters": best_k,
        "silhouette_score": float(final_silhouette),
        "silhouette_scores_by_k": {str(k): float(s) for k, s in silhouette_scores} if silhouette_scores else {},
        "filtering": {
            "enabled": filtering_dir is not None,
            "minimum_rating": args.minimum_rating if filtering_dir else None,
        },
        "clusters": {
            str(i): sorted(cluster_roles[i]) for i in range(best_k)
        },
        "role_to_cluster": {role: int(label) for role, label in zip(roles, cluster_labels)},
        "cluster_sizes": {str(i): len(cluster_roles[i]) for i in range(best_k)},
        "role_silhouettes": {role: float(sil) for role, sil in zip(roles, sample_silhouettes)},
        "tsne_coordinates": {role: [float(X_tsne[i, 0]), float(X_tsne[i, 1])] for i, role in enumerate(roles)},
        "assistant_tsne": [float(assistant_tsne[0]), float(assistant_tsne[1])] if assistant_tsne is not None else None,
        "mean_intra_cluster_similarity": {},
        "mean_inter_cluster_similarity": float(np.mean([
            cosine_sim_matrix[i, j] 
            for i in range(len(roles)) for j in range(len(roles))
            if cluster_labels[i] != cluster_labels[j]
        ])) if best_k > 1 else None,
    }
    
    # Compute intra-cluster similarity
    for c in range(best_k):
        mask = cluster_labels == c
        if mask.sum() > 1:
            cluster_sims = cosine_sim_matrix[np.ix_(mask, mask)]
            # Exclude diagonal
            off_diag = cluster_sims[~np.eye(cluster_sims.shape[0], dtype=bool)]
            results["mean_intra_cluster_similarity"][str(c)] = float(np.mean(off_diag))
    
    results_path = output_dir / "clustering.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Clustering Summary")
    print("=" * 60)
    print(f"Roles: {len(roles)}")
    print(f"Optimal clusters: {best_k}")
    print(f"Silhouette score: {final_silhouette:.3f}")
    print(f"\nCluster composition:")
    for cluster_id, cluster_role_list in sorted_clusters:
        print(f"  Cluster {cluster_id} ({len(cluster_role_list)} roles): {', '.join(cluster_role_list[:5])}" +
              (f"..." if len(cluster_role_list) > 5 else ""))
    
    if results["mean_inter_cluster_similarity"] is not None:
        print(f"\nMean inter-cluster similarity: {results['mean_inter_cluster_similarity']:.3f}")
    print(f"Mean intra-cluster similarities: {', '.join(f'{v:.3f}' for v in results['mean_intra_cluster_similarity'].values())}")


if __name__ == "__main__":
    main()
