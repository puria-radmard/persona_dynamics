"""Rank analysis focused on value-laden roles (positive vs negative).

Same analysis as rank_analysis.py but filtered to roles with clear
prosocial (positive) or antisocial (negative) valence.

Saves to: analyses/sfm_viz/rank_analysis_values.png and rank_analysis_values.json
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuration
BASE_PATH = Path("outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations")
OUTPUT_DIR = Path("analyses/sfm_viz")

# Value-laden roles
POSITIVE_ROLES = {
    "altruist",
    "caregiver", 
    "healer",
    "guardian",
    "peacekeeper",
    "mentor",
    "empath",
    "sage",
    "angel",
    "pacifist",
    "symbiont",
    "collaborator",
}

NEGATIVE_ROLES = {
    "predator",
    "parasite",
    "virus",
    "vampire",
    "demon",
    "narcissist",
    "zealot",
    "saboteur",
    "destroyer",
    "criminal",
    "gossip",
    "hoarder",
}

VALUE_ROLES = POSITIVE_ROLES | NEGATIVE_ROLES

# Models to compare
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

# Short names
SHORT_NAMES = {
    "geodesic-research/sfm_baseline_unfiltered_base": "baseline_unfilt",
    "geodesic-research/sfm_baseline_filtered_base": "baseline_filt",
    "geodesic-research/sfm_filtered_e2e_alignment_upsampled_base": "e2e_filt_align",
    "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base": "e2e_unfilt_align",
    "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base": "e2e_unfilt_misalign",
    "geodesic-research/sfm_filtered_cpt_alignment_upsampled_base": "cpt_filt_align",
    "geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_base": "cpt_unfilt_align",
    "geodesic-research/sfm_unfiltered_cpt_misalignment_upsampled_base": "cpt_unfilt_misalign",
}

# Comparison pairs
COMPARISON_PAIRS = [
    ("geodesic-research/sfm_baseline_unfiltered_base", "geodesic-research/sfm_baseline_filtered_base"),
    ("geodesic-research/sfm_baseline_unfiltered_base", "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base"),
    ("geodesic-research/sfm_baseline_unfiltered_base", "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base"),
    ("geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base", "geodesic-research/sfm_unfiltered_e2e_misalignment_upsampled_base"),
    ("geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_base", "geodesic-research/sfm_unfiltered_cpt_misalignment_upsampled_base"),
    ("geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_base", "geodesic-research/sfm_unfiltered_cpt_alignment_upsampled_base"),
]


def get_analysis_path(model_name: str) -> Path:
    model_safe = model_name.replace("/", "_")
    return BASE_PATH / model_safe / "main" / "analyses" / "assistant_axis.json"


def load_analysis(model_name: str) -> dict | None:
    path = get_analysis_path(model_name)
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def get_2d_positions(data: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    roles = data["roles"]
    role_cosines = data["role_cosines_to_pcs"]
    assistant_cosines = data["assistant_cosine_to_pcs"]
    
    role_positions = {}
    for role in roles:
        pc1 = role_cosines["PC1"][role]
        pc2 = role_cosines["PC2"][role]
        role_positions[role] = np.array([pc1, pc2])
    
    assistant_pos = np.array([assistant_cosines["PC1"], assistant_cosines["PC2"]])
    return role_positions, assistant_pos


def compute_distances_to_assistant(role_positions: dict[str, np.ndarray], assistant_pos: np.ndarray) -> dict[str, float]:
    distances = {}
    for role, pos in role_positions.items():
        distances[role] = float(np.linalg.norm(pos - assistant_pos))
    return distances


def rank_by_distance(distances: dict[str, float]) -> dict[str, int]:
    sorted_roles = sorted(distances.keys(), key=lambda r: distances[r])
    return {role: rank + 1 for rank, role in enumerate(sorted_roles)}


def fit_ols_mapping(positions_A: dict[str, np.ndarray], positions_B: dict[str, np.ndarray]) -> LinearRegression:
    common_roles = set(positions_A.keys()) & set(positions_B.keys())
    X = np.array([positions_A[r] for r in sorted(common_roles)])
    Y = np.array([positions_B[r] for r in sorted(common_roles)])
    model = LinearRegression()
    model.fit(X, Y)
    return model, sorted(common_roles)


def analyze_pair(model_A: str, model_B: str, data_A: dict, data_B: dict) -> dict:
    positions_A, assistant_A = get_2d_positions(data_A)
    positions_B, assistant_B = get_2d_positions(data_B)
    
    common_roles = set(positions_A.keys()) & set(positions_B.keys())
    value_roles_present = common_roles & VALUE_ROLES
    
    # === Original Space Analysis ===
    dist_A = compute_distances_to_assistant(positions_A, assistant_A)
    dist_B = compute_distances_to_assistant(positions_B, assistant_B)
    
    rank_A = rank_by_distance(dist_A)
    rank_B = rank_by_distance(dist_B)
    
    rank_changes = {r: rank_B[r] - rank_A[r] for r in common_roles}
    
    # Filter to value roles only
    value_rank_changes = {r: rank_changes[r] for r in value_roles_present}
    
    original_analysis = {
        "rank_A": {r: rank_A[r] for r in value_roles_present},
        "rank_B": {r: rank_B[r] for r in value_roles_present},
        "rank_changes": value_rank_changes,
        "dist_A": {r: dist_A[r] for r in value_roles_present},
        "dist_B": {r: dist_B[r] for r in value_roles_present},
    }
    
    # === OLS Space Analysis ===
    ols_model, fitted_roles = fit_ols_mapping(positions_A, positions_B)
    
    predicted_B = {}
    for role in common_roles:
        predicted_B[role] = ols_model.predict(positions_A[role].reshape(1, -1))[0]
    
    predicted_assistant_B = ols_model.predict(assistant_A.reshape(1, -1))[0]
    
    residuals = {r: positions_B[r] - predicted_B[r] for r in common_roles}
    assistant_residual = assistant_B - predicted_assistant_B
    
    movement_toward_assistant = {}
    for role in common_roles:
        to_assistant = assistant_B - predicted_B[role]
        to_assistant_norm = to_assistant / (np.linalg.norm(to_assistant) + 1e-10)
        movement = float(np.dot(residuals[role], to_assistant_norm))
        movement_toward_assistant[role] = movement
    
    # Filter to value roles only
    value_movement = {r: movement_toward_assistant[r] for r in value_roles_present}
    
    ols_analysis = {
        "movement_toward_assistant": value_movement,
        "assistant_residual": assistant_residual.tolist(),
        "assistant_residual_norm": float(np.linalg.norm(assistant_residual)),
        "ols_r2": float(ols_model.score(
            np.array([positions_A[r] for r in fitted_roles]),
            np.array([positions_B[r] for r in fitted_roles])
        )),
    }
    
    return {
        "model_A": model_A,
        "model_B": model_B,
        "n_common_roles": len(common_roles),
        "n_value_roles": len(value_roles_present),
        "value_roles_present": sorted(value_roles_present),
        "original_space": original_analysis,
        "ols_space": ols_analysis,
    }


def get_role_color(role: str) -> str:
    """Get color based on role valence."""
    if role in POSITIVE_ROLES:
        return '#2ecc71'  # Green for positive
    elif role in NEGATIVE_ROLES:
        return '#e74c3c'  # Red for negative
    return '#95a5a6'  # Gray for neutral


def plot_comparison(results: list[dict], output_path: Path):
    """Create visualization of value-laden role movements."""
    
    n_pairs = len(results)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 5 * n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        name_A = SHORT_NAMES.get(result["model_A"], result["model_A"].split("/")[-1])
        name_B = SHORT_NAMES.get(result["model_B"], result["model_B"].split("/")[-1])
        
        value_roles = result["value_roles_present"]
        
        # === Left plot: Original space rank changes ===
        ax1 = axes[idx, 0]
        
        orig = result["original_space"]
        rank_changes = orig["rank_changes"]
        
        # Sort by rank change
        sorted_roles = sorted(value_roles, key=lambda r: rank_changes[r])
        changes = [rank_changes[r] for r in sorted_roles]
        colors = [get_role_color(r) for r in sorted_roles]
        
        # Add marker for positive/negative valence
        labels = []
        for r in sorted_roles:
            if r in POSITIVE_ROLES:
                labels.append(f"✓ {r}")
            else:
                labels.append(f"✗ {r}")
        
        y_pos = range(len(sorted_roles))
        bars = ax1.barh(y_pos, changes, color=colors, alpha=0.7, edgecolor='black')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.axvline(0, color='black', linewidth=1)
        ax1.set_xlabel('Rank Change (negative = closer to assistant)')
        ax1.set_title(f'Original Space: {name_A} → {name_B}\n(✓=positive role, ✗=negative role)')
        ax1.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.7, label='Positive role'),
            Patch(facecolor='#e74c3c', edgecolor='black', alpha=0.7, label='Negative role'),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        # === Right plot: OLS residual movement ===
        ax2 = axes[idx, 1]
        
        ols = result["ols_space"]
        movement = ols["movement_toward_assistant"]
        
        # Sort by movement (most toward assistant first)
        sorted_roles_ols = sorted(value_roles, key=lambda r: movement[r], reverse=True)
        movements = [movement[r] for r in sorted_roles_ols]
        colors_ols = [get_role_color(r) for r in sorted_roles_ols]
        
        labels_ols = []
        for r in sorted_roles_ols:
            if r in POSITIVE_ROLES:
                labels_ols.append(f"✓ {r}")
            else:
                labels_ols.append(f"✗ {r}")
        
        y_pos_ols = range(len(sorted_roles_ols))
        ax2.barh(y_pos_ols, movements, color=colors_ols, alpha=0.7, edgecolor='black')
        
        ax2.set_yticks(y_pos_ols)
        ax2.set_yticklabels(labels_ols, fontsize=9)
        ax2.axvline(0, color='black', linewidth=1)
        ax2.set_xlabel('Movement toward assistant (OLS residual projection)')
        ax2.set_title(f'OLS Space: {name_A} → {name_B} (R²={ols["ols_r2"]:.2f})')
        ax2.invert_yaxis()
        
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    fig.suptitle('Value-Laden Role Movement Relative to Assistant\n(Green = prosocial roles, Red = antisocial roles)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def print_summary(results: list[dict]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY: Average movement by role valence")
    print("=" * 80)
    
    for result in results:
        name_A = SHORT_NAMES.get(result["model_A"], result["model_A"].split("/")[-1])
        name_B = SHORT_NAMES.get(result["model_B"], result["model_B"].split("/")[-1])
        
        print(f"\n{name_A} → {name_B}")
        print("-" * 40)
        
        # Original space
        orig = result["original_space"]
        rank_changes = orig["rank_changes"]
        
        pos_changes = [rank_changes[r] for r in rank_changes if r in POSITIVE_ROLES]
        neg_changes = [rank_changes[r] for r in rank_changes if r in NEGATIVE_ROLES]
        
        pos_mean = np.mean(pos_changes) if pos_changes else 0
        neg_mean = np.mean(neg_changes) if neg_changes else 0
        
        print(f"  Original space (rank change, negative=closer):")
        print(f"    Positive roles avg: {pos_mean:+.1f}")
        print(f"    Negative roles avg: {neg_mean:+.1f}")
        print(f"    Δ (pos - neg): {pos_mean - neg_mean:+.1f}")
        
        # OLS space
        ols = result["ols_space"]
        movement = ols["movement_toward_assistant"]
        
        pos_movement = [movement[r] for r in movement if r in POSITIVE_ROLES]
        neg_movement = [movement[r] for r in movement if r in NEGATIVE_ROLES]
        
        pos_mean_ols = np.mean(pos_movement) if pos_movement else 0
        neg_mean_ols = np.mean(neg_movement) if neg_movement else 0
        
        print(f"  OLS space (movement toward assistant, positive=closer):")
        print(f"    Positive roles avg: {pos_mean_ols:+.4f}")
        print(f"    Negative roles avg: {neg_mean_ols:+.4f}")
        print(f"    Δ (pos - neg): {pos_mean_ols - neg_mean_ols:+.4f}")


def main():
    # Load all model data
    model_data = {}
    for model in MODELS:
        data = load_analysis(model)
        if data:
            model_data[model] = data
    
    print(f"Loaded data for {len(model_data)} models")
    print(f"Tracking {len(POSITIVE_ROLES)} positive roles and {len(NEGATIVE_ROLES)} negative roles")
    
    # Analyze each pair
    results = []
    for model_A, model_B in COMPARISON_PAIRS:
        if model_A not in model_data or model_B not in model_data:
            print(f"Skipping pair {model_A} -> {model_B} (missing data)")
            continue
        
        print(f"\nAnalyzing: {SHORT_NAMES.get(model_A, model_A)} → {SHORT_NAMES.get(model_B, model_B)}")
        result = analyze_pair(model_A, model_B, model_data[model_A], model_data[model_B])
        results.append(result)
        
        print(f"  Value roles present: {result['n_value_roles']}")
    
    # Print summary
    print_summary(results)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = OUTPUT_DIR / "rank_analysis_values.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Plot
    plot_path = OUTPUT_DIR / "rank_analysis_values.png"
    plot_comparison(results, plot_path)


if __name__ == "__main__":
    main()