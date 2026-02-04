import json
import numpy as np
import matplotlib.pyplot as plt

# Configuration
SFM_PATH = 'outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_unfiltered_e2e_misalignment_upsampled_base/main/analyses/assistant_axis.json'
OLMO3_PATH = 'outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_unfiltered_e2e_alignment_upsampled_base/main/analyses/assistant_axis.json'

SFM_NAME = "SFM Misaligned"
OLMO3_NAME = "SFM Aligned"

NUM_PCS = 5
NUM_LABELS = 5  # Number of outliers to label per plot

# Consistent sign policy: flip PC so that assistant has POSITIVE cosine
# (This makes interpretation easier - "more assistant-like" = higher value)


def load_data(path):
    """Load data and determine sign flips to make assistant positive on each PC."""
    with open(path) as f:
        data = json.load(f)
    
    results = {}
    for i in range(1, NUM_PCS + 1):
        pc_name = f'PC{i}'
        assistant_cos = data['assistant_cosine_to_pcs'][pc_name]
        role_cosines = data['role_cosines_to_pcs'][pc_name]
        
        # Flip sign if assistant is negative (consistent policy)
        if assistant_cos < 0:
            assistant_cos = -assistant_cos
            role_cosines = {k: -v for k, v in role_cosines.items()}
        
        results[pc_name] = {
            'assistant': assistant_cos,
            'roles': role_cosines,
        }
    
    return results


def plot_comparison(ax, sfm_pc_data, olmo3_pc_data, sfm_pc_name, olmo3_pc_name):
    """Plot a single PC comparison."""
    # Find common roles
    common_keys = list(set(sfm_pc_data['roles'].keys()) & set(olmo3_pc_data['roles'].keys()))
    
    sfm_values = np.array([sfm_pc_data['roles'][k] for k in common_keys])
    olmo3_values = np.array([olmo3_pc_data['roles'][k] for k in common_keys])
    
    # Plot all points
    ax.scatter(sfm_values, olmo3_values, alpha=0.5, s=20)
    
    # Find and label outliers (furthest from diagonal)
    distances = np.abs(sfm_values - olmo3_values)
    top_indices = np.argsort(distances)[-NUM_LABELS:]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, idx in enumerate(top_indices):
        ax.scatter(sfm_values[idx], olmo3_values[idx], color=colors[i], s=40, zorder=4)
        ax.annotate(common_keys[idx], (sfm_values[idx], olmo3_values[idx]),
                   textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
    
    # Plot assistant
    ax.scatter(sfm_pc_data['assistant'], olmo3_pc_data['assistant'],
              marker='*', s=200, color='gold', edgecolors='black', linewidths=1, zorder=5)
    
    # Diagonal reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_xlabel(f'SFM {sfm_pc_name}', fontsize=8)
    ax.set_ylabel(f'Olmo3 {olmo3_pc_name}', fontsize=8)
    ax.tick_params(labelsize=6)


# Load data
sfm_data = load_data(SFM_PATH)
olmo3_data = load_data(OLMO3_PATH)

# Create 5x5 grid
fig, axes = plt.subplots(NUM_PCS, NUM_PCS, figsize=(15, 15))

for i, sfm_pc in enumerate([f'PC{j}' for j in range(1, NUM_PCS + 1)]):
    for j, olmo3_pc in enumerate([f'PC{k}' for k in range(1, NUM_PCS + 1)]):
        ax = axes[j, i]  # Row = Olmo3 PC, Col = SFM PC
        plot_comparison(ax, sfm_data[sfm_pc], olmo3_data[olmo3_pc], sfm_pc, olmo3_pc)

fig.suptitle(F'Role Cosine Similarities: {SFM_NAME} vs {OLMO3_NAME}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pc_comparison_grid.png', dpi=150, bbox_inches='tight')
print("Saved to pc_comparison_grid.png")