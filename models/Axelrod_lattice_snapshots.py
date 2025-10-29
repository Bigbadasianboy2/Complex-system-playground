import numpy as np
import matplotlib.pyplot as plt
import random

# --- Parameters ----
L = 100  # system size
F = 2  # Number of features
Q = 2  # Number of traits per feature

# here steps are micro steps(number of updates, not mc steps) of when a snapshot is produced
snapshot_steps = [
    0,  # initial state
    100 * L * L,  # early stage
    500 * L * L,  # middle stage
    1300 * L * L  # final stage
]
max_steps = snapshot_steps[-1] + 1


# --- 1. Initialization ---
def initialize_grid(L, F, Q):
    """2D grid, N agents with F features"""
    return np.random.randint(0, Q, size=(L, L, F))


# --- 2. plotting function ---
def plot_grid(ax, grid, step):
    """plot grid"""
    L, _, F = grid.shape
    Q = grid.max() + 1

    # mapping an agent's cultural features too a color
    # eg. F=2, Q=2, vector [c1, c2] -> ID = c1*2^1 + c2*2^0
    # every configuration has a unique color
    image = np.zeros((L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            agent_culture = grid[i, j]
            unique_id = 0
            for f in range(F):
                unique_id += agent_culture[f] * (Q ** (F - 1 - f))
            image[i, j] = unique_id

    #
    num_colors = Q ** F
    cmap = plt.cm.get_cmap('viridis', num_colors)

    ax.imshow(image, cmap=cmap, interpolation='nearest', vmin=0, vmax=num_colors - 1)
    ax.set_xticks([])
    ax.set_yticks([])


# --- 3. main simulation ---

# initialization
grid = initialize_grid(L, F, Q)

# set plotting windows
fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 5))
fig.suptitle(f'Axelrod Model on 2D Grid (L={L}, F={F}, Q={Q})\nFormation of Cultural Domains', fontsize=16)

# snapshot index
snapshot_idx = 0

print("start simulation...")

# Main simulation
for step in range(max_steps):

    if step == snapshot_steps[snapshot_idx]:
        print(f"generating snapshot: Time Step = {step}")
        plot_grid(axes[snapshot_idx], grid, step)
        snapshot_idx += 1

    # --- Axelrod updating rules ---
    # 1.1 randomly select an agent (i, j)
    i, j = random.randint(0, L - 1), random.randint(0, L - 1)

    # 2. randomly select a neighbor (ni, nj)
    # pbc condition (torus)
    dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    ni, nj = (i + dx) % L, (j + dy) % L

    agent_culture = grid[i, j]
    neighbor_culture = grid[ni, nj]

    # 3. calculate similarity
    similarity = np.sum(agent_culture == neighbor_culture)

    # 4. determine whether an interaction occurs
    if 0 < similarity < F:
        # interaction probability:
        if random.random() < (similarity / F):

            diff_features_indices = np.where(agent_culture != neighbor_culture)[0]
            # select a feature that isn't previously shared
            feature_to_copy = random.choice(diff_features_indices)

            # copy that feature
            grid[i, j, feature_to_copy] = neighbor_culture[feature_to_copy]

print("simulation done！")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("axelrod_snapshots.pdf", dpi=300)
plt.show()




