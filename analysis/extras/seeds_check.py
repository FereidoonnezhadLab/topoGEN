from TopoGEN.src.create_periodic_network import (tile_points, lloyd_relaxation_3d_periodic,
                                                 get_vertices, get_edges, process_edges,
                                                 merge_close_vertices, replica_removal,
                                                 find_periodic_pairs, calculate_mean_length)
from TopoGEN.src.optimize_periodic_network import (optimize_valency, calculate_valencies, simulated_annealing_without_valency,
                                                   read_edges, read_vertices, compute_and_plot_valency_density,
                                                   simulated_annealing_with_valency, compute_edge_lengths, plot_energy,
                                                   test_simulated_annealing_with_valency)
from TopoGEN.src.write_abaqus_input_file import (edges_length, compute_volume_fraction)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from matplotlib import rcParams, font_manager
font_path_regular = 'D:/FONT/SourceSansPro-Regular.otf'
font_path_bold = 'D:/FONT/SourceSansPro-Bold.otf'
font_manager.fontManager.addfont(font_path_regular)
font_manager.fontManager.addfont(font_path_bold)
rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'regular'
from TopoGEN.utils.setup import setup_output_directory
job_description = "NewResultsPlots"
output_directory = setup_output_directory(job_description)
import os
from scipy.stats import pearsonr, spearmanr, kendalltau

def run_network_generation(seed_count: int, domain_size: float, target_valency: float) -> float:
    """
    Runs Steps 1 and 2: Voronoi generation + valency optimization.
    Returns the computed concentration (phi * scaling factor).
    """
    global N
    N = seed_count

    # STEP 1: Periodic Voronoi Generation
    half_domain_physical_dimension = 0.5
    bounds = [(-half_domain_physical_dimension, half_domain_physical_dimension)] * 3
    original_points = np.random.uniform(-half_domain_physical_dimension, half_domain_physical_dimension, (N, 3))

    point_tile = tile_points(original_points, N)
    points, vor = lloyd_relaxation_3d_periodic(original_points, 10, N)
    tile_vertices = vor.vertices
    vertices, IndexMap = get_vertices(tile_vertices)
    tile_edges = process_edges(vor.ridge_vertices, tile_vertices)
    Vertices, Edges = get_edges(vertices, tile_vertices, tile_edges, IndexMap, bounds)
    unique_edges = replica_removal(Edges)

    FinalVertices = Vertices
    FinalEdges = unique_edges

    PeriodicEdges = find_periodic_pairs(FinalVertices, bounds)
    # Skipping save during seed refinement simulation

    # STEP 2: Valency Optimization
    vertices = FinalVertices
    edges = FinalEdges
    periodic_edges = PeriodicEdges

    internal_vertices = {i for i, v in enumerate(vertices) if np.all(np.abs(v) < 0.5)}
    boundary_vertices = {i for i, v in enumerate(vertices) if np.any(np.abs(v) == 0.5)}

    updated_edges, updated_valencies, updated_periodic_edges, updated_vertices_positions = optimize_valency(
        edges, periodic_edges, len(vertices), internal_vertices, boundary_vertices, vertices, target_valency, min_valency=3
    )

    # Save results after optimization
    # Skipping save during seed refinement simulation

    # Compute concentration
    nodes = updated_vertices_positions * domain_size  # scale back to physical domain
    element_nodes1 = updated_edges[:, 0]
    element_nodes2 = updated_edges[:, 1]
    elements = np.vstack([element_nodes1, element_nodes2]).T

    total_length = sum(edges_length(nodes, elements))
    phi = compute_volume_fraction(total_length, fiber_radius, domain_size)
    concentration = phi * 1000 / 0.73

    return concentration


def refine_seeds_to_match_concentration(target_concentration: float,
                                        tolerance: float = 0.1,
                                        initial_seed: int = 100,
                                        max_iterations: int = 100) -> int:
    """
    Binary search-style refinement of seed count to match target concentration.
    """
    lower_bound = max(50, int(initial_seed / 2))
    upper_bound = int(initial_seed * 2)
    best_seed = initial_seed

    for i in range(max_iterations):
        print(f"--- Iteration {i + 1} | Testing Seed Count: {best_seed} ---")
        current_concentration = run_network_generation(best_seed, domain_physical_dimension, target_avg_valency)
        print(f"Resulting Concentration: {current_concentration:.4f}")

        error = current_concentration - target_concentration
        if abs(error) <= tolerance:
            print("Target concentration met within tolerance.")
            return best_seed

        if error > 0:
            upper_bound = best_seed - 1
        else:
            lower_bound = best_seed + 1

        best_seed = (lower_bound + upper_bound) // 2

    if current_concentration < target_concentration:
        print("ERROR: Unable to reach target concentration within max iterations.")
        print("Hint: Initial seed guess might be TOO LOW. Try increasing it.")
        import sys
        sys.exit(1)
    else:
        print("ERROR: Unable to reach target concentration within max iterations.")
        print("Hint: Initial seed guess might be TOO HIGH. Try decreasing it.")
        import sys
        sys.exit(1)
    return best_seed

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------

domain_size = 40
domain_physical_dimension = domain_size
target_avg_valency = 3.2
fiber_radius = 0.08
tolerance = 0.1
initial_seed_guesses = [100, 150, 200, 250, 300, 250, 400]
half_domain = 0.5
bounds = [(-half_domain, half_domain)] * 3

# Concentrations to analyze
concentration_samples = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Length distribution parameters (used in optimization)
L_original = 2
v = 0.256
L = L_original / domain_size
s_squared = (L_original ** 2 * v) / (domain_size ** 2)
sigma_squared = np.log(1 + s_squared / L ** 2)
sigma = np.sqrt(sigma_squared)
mu = np.log(L) - sigma_squared / 2
target_distribution = lognorm(s=sigma, scale=np.exp(mu))

# -------------------------------------------
# DATA COLLECTION
# -------------------------------------------
fiber_lengths_unoptimized = []
fiber_lengths_optimized = []
fiber_stds_unoptimized = []
fiber_stds_optimized = []
seeds_list = []
total_lengths = []
total_lengths_ratio = []

for i, target_concentration in enumerate(concentration_samples):
    print(f"\n▶ Processing concentration: {target_concentration} mg/ml")

    # Step 1: Determine seed count for target concentration
    seed_guess = initial_seed_guesses[i]
    seed_count = refine_seeds_to_match_concentration(
        target_concentration,
        tolerance=tolerance,
        initial_seed=seed_guess
    )
    seeds_list.append(seed_count)

    # Step 2: Generate initial Voronoi network
    original_points = np.random.uniform(-half_domain, half_domain, (seed_count, 3))
    point_tile = tile_points(original_points, seed_count)
    points, vor = lloyd_relaxation_3d_periodic(original_points, 10, seed_count)

    tile_vertices = vor.vertices
    vertices, IndexMap = get_vertices(tile_vertices)
    tile_edges = process_edges(vor.ridge_vertices, tile_vertices)
    Vertices, Edges = get_edges(vertices, tile_vertices, tile_edges, IndexMap, bounds)
    FinalEdges = replica_removal(Edges)
    FinalVertices = Vertices

    # Step 3: Compute pre-optimization fiber statistics
    lengths_unopt = edges_length(FinalVertices * domain_size, FinalEdges)
    mean_length_unopt = np.mean(lengths_unopt)
    std_length_unopt = np.std(lengths_unopt)

    fiber_lengths_unoptimized.append(mean_length_unopt / domain_size * 100)
    fiber_stds_unoptimized.append(std_length_unopt / domain_size * 100)

    # Step 4: Hybrid optimization (valency + length)
    # Step 4: Valency optimization followed by length optimization

    print("▶ Running valency optimization first...")

    vertex_array = np.array(FinalVertices)
    internal_vertices = {i for i, v in enumerate(vertex_array) if np.all(np.abs(v) < 0.5)}
    boundary_vertices = {i for i, v in enumerate(vertex_array) if np.any(np.abs(v) == 0.5)}

    # Find periodic edge pairs
    periodic_edges = find_periodic_pairs(FinalVertices, bounds)

    # Valency optimization
    updated_edges, updated_valencies, updated_periodic_edges, updated_vertices_positions = optimize_valency(
        edges=FinalEdges,
        periodic_edges=periodic_edges,
        num_vertices=len(vertex_array),
        internal_vertices=internal_vertices,
        boundary_vertices=boundary_vertices,
        vertices_position=vertex_array,
        target_avg_valency=target_avg_valency,
        min_valency=3
    )

    print("✔ Valency optimization completed.")

    # Prepare state for length optimization
    intermediate_vertices_dict = {i: pos for i, pos in enumerate(updated_vertices_positions)}
    state = {
        'vertices': intermediate_vertices_dict,
        'edges': updated_edges.tolist()  # Ensure it's a list
    }

    # Length optimization
    print("▶ Proceeding to length optimization...")

    optimized_vertices, optimized_edges, _ = simulated_annealing_without_valency(
        state=state,
        target_distribution=target_distribution,
        bounds=(-0.5, 0.5)
    )

    # Step 5: Compute post-optimization fiber statistics
    vertex_array = np.array([v for _, v in sorted(optimized_vertices.items())])
    lengths_opt = edges_length(vertex_array * domain_size, optimized_edges)
    mean_length_opt = np.mean(lengths_opt)
    std_length_opt = np.std(lengths_opt)

    fiber_lengths_optimized.append(mean_length_opt / domain_size * 100)
    fiber_stds_optimized.append(std_length_opt / domain_size * 100)

    # ✅ Save total fiber length directly
    total_lengths.append(np.sum(lengths_opt))
    total_lengths_ratio.append(np.sum(lengths_opt) / domain_size * 100)


# Convert to arrays
total_lengths = np.array(total_lengths)
total_lengths_mm = total_lengths / 1000
seeds_array = np.array(seeds_list)

# Run correlation tests
pearson_r, pearson_p = pearsonr(seeds_array, total_lengths_mm)
spearman_rho, spearman_p = spearmanr(seeds_array, total_lengths_mm)
kendall_tau, kendall_p = kendalltau(seeds_array, total_lengths_mm)

# Print results
print("\n🔍 Correlation Analysis: total fiber length vs number of seeds")
print(f"• Pearson      r = {pearson_r:.3f} (p = {pearson_p:.3g})")
print(f"• Spearman     ρ = {spearman_rho:.3f} (p = {spearman_p:.3g})")
print(f"• Kendall Tau  τ = {kendall_tau:.3f} (p = {kendall_p:.3g})")

# Scatter plot with trendline
plt.figure(figsize=(6, 6))
plt.scatter(seeds_array, total_lengths_mm, color='teal', s=60, edgecolor='k')
plt.plot(
    np.unique(seeds_array),
    np.poly1d(np.polyfit(seeds_array, total_lengths_mm, 1))(np.unique(seeds_array)),
    'k--', lw=1.5
)

plt.xlabel("number of Voronoi seeds", fontsize=16)
plt.ylabel("total fiber length [mm]", fontsize=16)
plt.title(
    f"Pearson r = {pearson_r:.2f}, Spearman ρ = {spearman_rho:.2f}, Kendall τ = {kendall_tau:.2f}",
    fontsize=12
)
plt.grid(True)
plt.tight_layout()

output_path = os.path.join(output_directory, "correlation_study.svg")
plt.savefig(output_path, format="svg")
plt.show()

# -------------------------------------------
# PLOTTING
# -------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import os

# Define bone-based color palette
bone_cmap = plt.get_cmap('bone')
clr_fiber_main = bone_cmap(0.2)
clr_fiber_fill = bone_cmap(0.2)
clr_seeds = bone_cmap(0.5)

# FIRST PLOT
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("concentration [mg/mL]", fontsize=14)
ax1.set_ylabel("fiber length l/L [%]", color=clr_fiber_main, fontsize=14)

fiber_lengths_optimized = np.array(fiber_lengths_optimized)
fiber_stds_optimized = np.array(fiber_stds_optimized)

ax1.plot(
    concentration_samples,
    fiber_lengths_optimized,
    's--',
    color=clr_fiber_main
)

ax1.fill_between(
    concentration_samples,
    fiber_lengths_optimized - fiber_stds_optimized,
    fiber_lengths_optimized + fiber_stds_optimized,
    color=clr_fiber_fill,
    alpha=0.1
)

ax1.tick_params(axis='y', labelcolor=clr_fiber_main)
ax1.set_xlim(1, 4)
ax1.set_xticks(concentration_samples)
ax1.tick_params(axis='both', labelsize=14)

ax2 = ax1.twinx()
ax2.set_ylabel("number of Voronoi seeds [-]", color=clr_seeds, fontsize=14)
ax2.plot(
    concentration_samples,
    seeds_list,
    'o-',
    color=clr_seeds,
    label='seed count'
)
ax2.tick_params(axis='y', labelcolor=clr_seeds)
ax2.set_ylim(0, max(seeds_list) * 1.2)
ax2.set_xticks(concentration_samples)
ax2.tick_params(axis='both', labelsize=14)

fig.tight_layout()
output_path = os.path.join(output_directory, "seeds_vs_l_vs_concentration.svg")
plt.savefig(output_path, format="svg")
plt.show()


# SECOND PLOT
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("concentration [mg/mL]", fontsize=14)
ax1.set_ylabel("total fiber length [mm]", color=clr_fiber_main, fontsize=14)

ax1.plot(
    concentration_samples,
    total_lengths_mm,
    's--',
    color=clr_fiber_main,
    label='total fiber length'
)

ax1.tick_params(axis='y', labelcolor=clr_fiber_main)
ax1.set_ylim(0, total_lengths_mm.max() * 1.2)
ax1.set_xlim(1, 4)
ax1.set_xticks(concentration_samples)
ax1.tick_params(axis='both', labelsize=14)

ax2 = ax1.twinx()
ax2.set_ylabel("number of Voronoi seeds", color=clr_seeds, fontsize=14)
ax2.plot(
    concentration_samples,
    seeds_list,
    'o-',
    color=clr_seeds,
    label='seed count'
)
ax2.tick_params(axis='y', labelcolor=clr_seeds)
ax2.set_ylim(0, max(seeds_list) * 1.2)
ax2.set_xticks(concentration_samples)
ax2.tick_params(axis='both', labelsize=14)

fig.tight_layout()
output_path = os.path.join(output_directory, "seeds_vs_Ltot_vs_concentration_comparison.svg")
plt.savefig(output_path, format="svg")
plt.show()


# THIRD PLOT
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("concentration [mg/mL]", fontsize=14)
ax1.set_ylabel("total fiber length / L [%]", color=clr_fiber_main, fontsize=14)

ax1.plot(
    concentration_samples,
    total_lengths,
    's--',
    color=clr_fiber_main,
    label='total fiber length'
)

ax1.tick_params(axis='y', labelcolor=clr_fiber_main)
ax1.set_ylim(0, total_lengths.max() * 1.2)
ax1.set_xlim(1, 4)
ax1.set_xticks(concentration_samples)
ax1.tick_params(axis='both', labelsize=14)

ax2 = ax1.twinx()
ax2.set_ylabel("number of Voronoi seeds", color=clr_seeds, fontsize=14)
ax2.plot(
    concentration_samples,
    seeds_list,
    'o-',
    color=clr_seeds,
    label='seed count'
)
ax2.tick_params(axis='y', labelcolor=clr_seeds)
ax2.set_ylim(0, max(seeds_list) * 1.2)
ax2.set_xticks(concentration_samples)
ax2.tick_params(axis='both', labelsize=14)

fig.tight_layout()
output_path = os.path.join(output_directory, "seeds_vs_Ltotnormalized_vs_concentration_comparison.svg")
plt.savefig(output_path, format="svg")
plt.show()
