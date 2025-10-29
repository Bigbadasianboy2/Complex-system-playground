import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool, set_start_method, cpu_count  # --- NEW: Imports ---

# ----- Parameters -----
N = 20
F_values = [5, 10, 15]
q_values = list(range(10, 30, 5)) + list(range(30, 70, 2)) + list(range(70, 110, 5))

trials_normal = 30
trials_critical = 200

# Adaptive critical regions
critical_regions = {
    5: (20, 40),
    10: (40, 60),
    15: (48, 80)
}
frozen_threshold = N * 20


@njit
def initialize_agents(N, F, q):
    return np.random.randint(0, q, (N, N, F))


@njit
def get_neighbors(i, j, N):
    return [((i - 1) % N, j), ((i + 1) % N, j), (i, (j - 1) % N), (i, (j + 1) % N)]


@njit
def similarity(agent1, agent2):
    same = 0
    for k in range(agent1.shape[0]):
        if agent1[k] == agent2[k]:
            same += 1
    return same / agent1.shape[0]


@njit
def axelrod_simulation(N, F, q, frozen_threshold):
    grid = initialize_agents(N, F, q)
    iterations_without_change = 0
    while iterations_without_change < frozen_threshold:
        i, j = np.random.randint(0, N, 2)
        neighbors = get_neighbors(i, j, N)
        ni, nj = neighbors[np.random.randint(0, 4)]
        agent1, agent2 = grid[i, j], grid[ni, nj]
        overlap = similarity(agent1, agent2)
        if 0 < overlap < 1 and np.random.rand() < overlap:
            diff_features = np.where(agent1 != agent2)[0]
            if diff_features.shape[0] > 0:
                feature = np.random.choice(diff_features)
                grid[i, j, feature] = agent2[feature]
                iterations_without_change = 0
            else:
                iterations_without_change += 1
        else:
            iterations_without_change += 1
    return grid



def bfs_cluster_size(grid, visited, start_i, start_j, N):
    queue = deque([(start_i, start_j)])
    visited[start_i, start_j] = True
    cluster_size = 0
    target_culture = grid[start_i, start_j]
    while queue:
        i, j = queue.popleft()
        cluster_size += 1
        for ni, nj in [((i - 1) % N, j), ((i + 1) % N, j), (i, (j - 1) % N), (i, (j + 1) % N)]:
            if not visited[ni, nj] and np.array_equal(target_culture, grid[ni, nj]):
                visited[ni, nj] = True
                queue.append((ni, nj))
    return cluster_size


def largest_cluster_fraction(grid, N):
    visited = np.zeros((N, N), dtype=bool)
    max_cluster = 0
    for i in range(N):
        for j in range(N):
            if not visited[i, j]:
                cluster_size = bfs_cluster_size(grid, visited, i, j, N)
                if cluster_size > max_cluster:
                    max_cluster = cluster_size
    return max_cluster / (N * N)


#200 times in parallel.
def run_single_trial(N_arg, F_arg, q_arg, frozen_threshold_arg):

    # Set a unique random seed for each worker process
    np.random.seed(int.from_bytes(np.random.bytes(4), 'little'))

    grid = axelrod_simulation(N_arg, F_arg, q_arg, frozen_threshold_arg)
    result = largest_cluster_fraction(grid, N_arg)
    return result


# ----- Main Simulation -----
if __name__ == '__main__':

    set_start_method('spawn', force=True)

    all_results = {}
    markers = ['o', 's', '^', 'v', 'D']

    # This uses all available CPU cores
    with Pool(processes=cpu_count()) as pool:
        print(f"--- Multiprocessing pool started with {cpu_count()} workers ---")

        for F in F_values:
            print(f"\n==================================================")
            print(f"=== STARTING SIMULATION FOR F = {F} ===")
            print(f"==================================================\n")

            results_mean = []
            results_err = []

            crit_range = critical_regions.get(F, (None, None))

            q_loop = tqdm(q_values, desc=f"Simulating F={F}", unit="q-value")
            for q in q_loop:

                if crit_range[0] is not None and crit_range[0] <= q <= crit_range[1]:
                    num_trials_for_this_q = trials_critical
                else:
                    num_trials_for_this_q = trials_normal



                # 1. Create a list of identical tasks to run in parallel
                #    Each task is just a tuple of arguments for our worker function
                tasks = [(N, F, q, frozen_threshold) for _ in range(num_trials_for_this_q)]

                # 2. Run all tasks in parallel and collect results
                #    pool.starmap automatically handles distributing the work
                current_q_results = pool.starmap(run_single_trial, tasks)


                avg = np.mean(current_q_results)
                err = np.std(current_q_results) / np.sqrt(num_trials_for_this_q)
                results_mean.append(avg)
                results_err.append(err)

                q_loop.set_postfix(last_q=q, avg_Smax=f"{avg:.4f}", trials=num_trials_for_this_q)

            all_results[F] = (results_mean, results_err)

    print("\nAll simulations finished. Plotting results...")

    # ----- Plot -----
    plt.figure(figsize=(10, 7))

    for i, F in enumerate(all_results.keys()):
        means, errs = all_results[F]
        marker_style = markers[i % len(markers)]

        plt.errorbar(q_values, means, yerr=errs, marker=marker_style, capsize=4,
                     linestyle='-', label=f'F = {F}', elinewidth=1)

    plt.xlabel("q (Number of Traits per Feature)")
    plt.ylabel("Fraction of Sites in Largest Cluster (S_max / N)")
    plt.title(f"Axelrod Model Phase Transition for Different F (N={N})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("Axelrod_Phase_Transition_F_vs_q.pdf")
    plt.show()