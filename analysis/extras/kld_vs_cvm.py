"""
Author: Sara Cardona
Date: 01/01/2025

To execute this script, first run the main script up to STEP 2, which performs valency optimization.
Then, switch to this script for length optimization and evaluate the KLD performance in comparison to the CVM test.
"""

import random
import time
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = 'browser'
output_directory = None
from matplotlib import rcParams, font_manager
font_path = 'D:/FONT/SourceSansPro-Regular.otf'
font_manager.fontManager.addfont(font_path)
from scipy.stats import wasserstein_distance

import os
import numpy as np
from scipy.stats import lognorm
from TopoGEN.utils.setup import setup_output_directory
from TopoGEN.src.optimize_periodic_network import (move_vertex, compute_edge_lengths, read_edges, read_vertices)


# ======================================================================================================================
#  SETUP JOB DESCRIPTION
# ======================================================================================================================
job_description = "Test"
output_directory = setup_output_directory(job_description)

# Define file paths for saving results
vertices_file_path = os.path.join(output_directory, "vertices.txt")
edges_file_path = os.path.join(output_directory, "edges.txt")
periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")


# ======================================================================================================================
#  SETUP TOPOLOGICAL PARAMETERS
# ======================================================================================================================

domain_physical_dimension = 40  # PARAM 1 (DOMAIN DIMENSION)
L_original = 2  # PARAM 4 (MEAN LENGTH)
v = 0.256  # PARAM 4 (STD LENGTH)

# ======================================================================================================================
#  EXTRA FUNCTIONS TO PERFORM THE COMPARISON
# ======================================================================================================================


def cramer_von_mises(current_lengths, target_distribution, dx=0.01, bins=None):
    """
    Computes the Cramér–von Mises distance between the empirical distribution
    of `current_lengths` and a given `target_distribution`.

    Args:
    - current_lengths: Observed data samples.
    - target_distribution: Target distribution with a `pdf` method.
    - dx: Bin width for histogram estimation (default: 0.01).
    - bins: Bin edges; auto-generated if None.

    Returns:
    - cvm: Cramér–von Mises distance between empirical and target distributions.
    - bins: Bin edges used in histogram computation.
    """
    if bins is None:
        bins = np.arange(min(current_lengths), max(current_lengths) + dx, dx)
    p_x = np.histogram(current_lengths, bins=bins, density=True)[0]
    p_x = np.maximum(p_x, 1e-10)  # Avoid zeros in the histogram
    q_x = target_distribution.pdf(bins[:-1] + dx / 2)
    q_x = np.maximum(q_x, 1e-10)  # Avoid zeros in target distribution
    cdf_p = np.cumsum(p_x) * dx
    cdf_q = np.cumsum(q_x) * dx
    cvm = np.sum((cdf_p - cdf_q) ** 2) * dx
    return cvm, bins


def kl_divergence(current_lengths, target_distribution, dx=0.01, bins=None):
    """
    Computes the Kullback–Leibler (KL) divergence between the empirical
    distribution of `current_lengths` and a given `target_distribution`.

    Args:
    - current_lengths: Observed data samples.
    - target_distribution: Target distribution with a `pdf` method.
    - dx: Bin width for histogram estimation (default: 0.01).
    - bins: Bin edges; auto-generated if None.

    Returns:
    - kl_div: KL divergence between empirical and target distributions.
    - bins: Bin edges used in histogram computation.
    """
    if bins is None:
        bins = np.arange(min(current_lengths), max(current_lengths) + dx, dx)
    p_x = np.histogram(current_lengths, bins=bins, density=True)[0]
    q_x = target_distribution.pdf(bins[:-1] + dx / 2)
    q_x = np.maximum(q_x, 1e-10)  # Adding a small epsilon to prevent zero values
    p_x = np.maximum(p_x, 1e-10)  # Adding a small epsilon for consistency in p_x
    kl_div = np.sum(np.where(p_x != 0, p_x * np.log(p_x / q_x), 0))
    return kl_div, bins

def simulated_annealing_with_metric(
        state,
        target_distribution,
        bounds,
        metric="kl_divergence",
        max_iterations=50000, ):
    """
    Optimizes node positions using simulated annealing to match a target
    edge-length distribution based on a KL and CVM metric.

    Args:
    - state: Graph structure with 'nodes' (dict) and 'edges' (list of tuples).
    - target_distribution: Desired edge-length distribution.
    - bounds: Spatial bounds as ((xmin, ymin), (xmax, ymax)).
    - metric: Distance metric ('kl_divergence' or 'cramer_von_mises'). Default: "kl_divergence".
    - max_iterations: Maximum iterations before termination. Default: 50000.

    Returns:
    - nodes: Updated node positions after optimization.
    - edges: Unmodified list of edges.
    """
    nodes, edges = state['nodes'], state['edges']
    current_lengths = compute_edge_lengths(nodes, edges)

    # Select the metric function
    if metric == "kl_divergence":
        metric_function = kl_divergence
    elif metric == "cramer_von_mises":
        metric_function = cramer_von_mises
    else:
        raise ValueError(f"Unknown metric: {metric}")
    initial_energy, bins = metric_function(current_lengths, target_distribution)
    current_energy, bins = metric_function(current_lengths, target_distribution)
    stop_threshold = 0.01 * current_energy
    internal_nodes = [node_index for node_index, node_pos in nodes.items()
                      if np.all(bounds[0] < np.array(node_pos)) and np.all(np.array(node_pos) < bounds[1])]

    for iteration in range(max_iterations):
        if not internal_nodes:
            print("No internal nodes available to move. Stopping simulation.")
            break

        # Select a node to move
        node_index = random.choice(internal_nodes)
        old_position = nodes[node_index].copy()
        nodes[node_index] = move_vertex(nodes[node_index], bounds)

        # Compute only affected edge lengths
        affected_edges = [e for e in edges if node_index in e]
        affected_lengths = [
            np.linalg.norm(nodes[e[0]] - nodes[e[1]]) for e in affected_edges
        ]
        updated_lengths = current_lengths.copy()
        for i, e in enumerate(affected_edges):
            index = edges.index(e)
            updated_lengths[index] = affected_lengths[i]

        new_energy, _ = metric_function(updated_lengths, target_distribution, bins=bins)
        print(f"Iteration {iteration + 1}: Node {node_index} moved")
        print(f"    Old Position: {old_position}")
        print(f"    New Position: {nodes[node_index]}")
        print(f"    Energy: {new_energy:.4f}")

        if new_energy < current_energy:
            print(f"    Improvement found. Energy decreased from {current_energy:.4f} to {new_energy:.4f}")
            current_energy = new_energy
            current_lengths = updated_lengths
        else:
            nodes[node_index] = old_position  # Revert move if not beneficial

        if current_energy < stop_threshold:
            print(f"Stopping early: Energy threshold {stop_threshold} reached.")
            break
    print("Simulated annealing completed.")
    print(f"Initial Energy: {initial_energy:.6f}")
    print(f"Final Energy: {current_energy:.6f}")
    return nodes, edges


def plot_distributions_comparison(initial_lengths, target_distribution, cramer_lengths, kl_lengths):
    """
    Plots and compares the initial, Cramer–von Mises-optimized, and KL-divergence-optimized
    edge-length distributions against the target distribution.

    Args:
    - initial_lengths: Edge lengths before optimization.
    - target_distribution: Target probability distribution.
    - cramer_lengths: Edge lengths after Cramer–von Mises optimization.
    - kl_lengths: Edge lengths after KL divergence optimization.

    """

    x = np.linspace(min(initial_lengths + cramer_lengths + kl_lengths), max(initial_lengths + cramer_lengths + kl_lengths), 10000)
    pdf_values = target_distribution.pdf(x)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Plot histograms for initial, Cramer-von Mises, and KL distributions
    plt.hist(initial_lengths, bins=30, alpha=0.5, density=True, label='Initial Distribution', color='blue', edgecolor='black')
    plt.hist(cramer_lengths, bins=30, alpha=0.5, density=True, label='After Cramer–von Mises', color='green', edgecolor='black')
    plt.hist(kl_lengths, bins=30, alpha=0.5, density=True, label='After KL Divergence', color='red', edgecolor='black')

    # Plot target distribution
    plt.plot(x, pdf_values, label='Target Distribution', color='black', linewidth=2)

    # Add labels and legend
    plt.xlabel('Edge Length', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return

# ======================================================================================================================
#  STEP 1: DEFINITION OF TARGET DISTRIBUTION
# ======================================================================================================================

L = L_original / domain_physical_dimension
s_squared_original = (L_original**2) * v
s_squared = s_squared_original / (domain_physical_dimension**2)
sigma_squared = np.log(1 + s_squared / L**2)
sigma = np.sqrt(sigma_squared)
mu = np.log(L) - sigma_squared / 2
target_distribution = lognorm(s=sigma, scale=np.exp(mu))

# ======================================================================================================================
#  STEP 2: INITIAL DISTRIBUTION
# ======================================================================================================================

nodes = read_vertices(vertices_file_path)
edges = read_edges(edges_file_path)
initial_lengths = compute_edge_lengths(nodes, edges)
initial_node_coordinates_list = [coord for coord in nodes.values()]
initial_nodal_coordinates = np.array(initial_node_coordinates_list)
initial_edges = edges


bounds = (-0.5, 0.5)

# ======================================================================================================================
#  STEP 3: CVM - BASED SIMULATED ANNEALING
# ======================================================================================================================

start_cvm = time.time()
state = {'nodes': nodes, 'edges': edges}
cramer_nodes, cramer_edges = simulated_annealing_with_metric(
    state, target_distribution, bounds, metric="cramer_von_mises"
)
cramer_lengths = compute_edge_lengths(cramer_nodes, cramer_edges)
runtime_cvm = time.time() - start_cvm

# ======================================================================================================================
#  STEP 4: KL - BASED SIMULATED ANNEALING
# ======================================================================================================================

start_kl = time.time()
state = {'nodes': nodes, 'edges': edges}
kl_nodes, kl_edges = simulated_annealing_with_metric(
    state, target_distribution, bounds, metric="kl_divergence"
)
kl_lengths = compute_edge_lengths(kl_nodes, kl_edges)
runtime_kl = time.time() - start_kl

# ======================================================================================================================
#  STEP 5: DISTRIBUTION QUALITATIVE COMPARISON
# ======================================================================================================================

plot_distributions_comparison(initial_lengths, target_distribution, cramer_lengths, kl_lengths)
print(f"KL Divergence Runtime: {runtime_kl:.4f} seconds")
print(f"Cramér–von Mises Runtime: {runtime_cvm:.4f} seconds")

# ======================================================================================================================
#  STEP 6: DISTRIBUTION QUANTITATIVE COMPARISON
# ======================================================================================================================

initial_energy_kl, _ = kl_divergence(initial_lengths, target_distribution)
final_energy_kl, _ = kl_divergence(kl_lengths, target_distribution)

initial_energy_cvm, _ = cramer_von_mises(initial_lengths, target_distribution)
final_energy_cvm, _ = cramer_von_mises(cramer_lengths, target_distribution)

percent_improvement_kl = ((initial_energy_kl - final_energy_kl) / initial_energy_kl) * 100
percent_improvement_cvm = ((initial_energy_cvm - final_energy_cvm) / initial_energy_cvm) * 100


print(f"KL Divergence - Initial Energy: {initial_energy_kl:.4f}, Final Energy: {final_energy_kl:.4f}")
print(f"Cramér–von Mises - Initial Energy: {initial_energy_cvm:.4f}, Final Energy: {final_energy_cvm:.4f}")
print(f"KL Divergence Percent Improvement: {percent_improvement_kl:.2f}%")
print(f"Cramér–von Mises Percent Improvement: {percent_improvement_cvm:.2f}%")

# ======================================================================================================================
#  STEP 7: EXTRA QUANTITATIVE COMPARISON
# ======================================================================================================================

distance_kl = wasserstein_distance(kl_lengths, target_distribution.rvs(size=len(kl_lengths)))
distance_cvm = wasserstein_distance(cramer_lengths, target_distribution.rvs(size=len(cramer_lengths)))

print(f"Wasserstein Distance (KL): {distance_kl:.4f}")
print(f"Wasserstein Distance (CVM): {distance_cvm:.4f}")