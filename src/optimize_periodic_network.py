"""
Author: Sara Cardona
Date: 29/10/2024
Refactor: 05/02/2024
"""
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams, font_manager
import numpy as np
import random
import math
from copy import deepcopy
from scipy.stats import linregress
font_path_regular = 'D:/FONT/SourceSansPro-Regular.otf'
font_path_bold = 'D:/FONT/SourceSansPro-Bold.otf'
font_manager.fontManager.addfont(font_path_regular)
font_manager.fontManager.addfont(font_path_bold)
rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'regular'

def is_internal(vertex, a=0.5):
    """
    Check if the vertex is inside the cubic domain.

    Args:
    - vertex: position of the vertex in the 3D space
    - a = computational domain dimension

    Returns:
    - boolean value indicating whether a given 3D point is inside a cubic domain
    """
    return np.all(np.abs(vertex) < a)


def is_boundary(vertex, a=0.5):
    """
    Check if the vertex is on the boundaries of the cubic domain.

    Args:
    - vertex: position of the vertex in the 3D space
    - a = computational domain dimension

    Returns:
    - boolean value indicating whether a given 3D point on the boundaries.
    """
    return np.any(np.abs(vertex) == a)


def calculate_valencies(edges, num_vertices):
    """
    Compute valency at each node.

    Args:
    - edges: connection list
    - num_vertices: number of nodes

    Returns:
    - the counts of how many times each node appears in the edge list.
    """
    return np.bincount(edges.flatten(), minlength=num_vertices)


def find_periodic_counterpart(boundary_vertex, periodic_edges):
    """
    Search for the periodic counterpart if a boundary edge has to be removed.

    Args:
    - boundary_vertex: vertex located at the boundaries
    - periodic_edges: map of periodic edges

    Returns:
    - the periodic node with respect to the selected one
    """
    for pe in periodic_edges:
        if boundary_vertex in pe:
            return pe[0] if pe[1] == boundary_vertex else pe[1]
    return None


def remove_edge(edges, edge, all_valencies):
    """
    Remove the edge is the removal is safe.

    Args:
    - edges: all the edges
    - edge: the specific edge to be removed
    - all_valencies: all the valencies count to be updated

    Returns:
    - none
    """
    edges.discard(edge)
    all_valencies[edge[0]] -= 1
    all_valencies[edge[1]] -= 1


def edge_length(vertices_position, edge):
    """Calculate the Euclidean distance (length) of the edge between two nodes."""
    return np.linalg.norm(vertices_position[edge[0]] - vertices_position[edge[1]])


def optimize_valency(edges, periodic_edges, num_vertices, internal_vertices,
                     boundary_vertices, vertices_position, target_avg_valency,
                     min_valency, retry_limit=1,
                     max_iterations=500000):
    """
    Core of the optimization process for the valency

    Args:
    - edges: all the edges
    - periodic_edges: periodic mapping
    - num_vertices: total number of nodes
    - internal_vertices: all internal vertices of the cubic domain
    - boundary_vertices: all boundary vertices of the cubic domain
    - vertices_position: all vertices positions
    - target_avg_valency: target average valency
    - min_valency: minimum valency
    - retry_limit: ...
    - max_iterations: maximum number of iterations

    Returns:
    - edges, final_valencies, periodic_edges, filtered_node_positions, energy_log, success (True/False)
    """
    edges = set(map(tuple, edges))
    periodic_edges = set(map(tuple, periodic_edges))
    all_valencies = calculate_valencies(np.array(list(edges)), num_vertices)

    # Precompute edge lengths for inverse length weighting
    edge_lengths = {edge: edge_length(vertices_position, edge) for edge in edges}

    # Initialize Union-Find for connectivity checks
    uf = UnionFind(num_vertices)
    for edge in edges:
        uf.union(edge[0], edge[1])

    # Separate edges by type
    boundary_edges = {e for e in edges if e[0] in boundary_vertices or e[1] in boundary_vertices}

    # Identify boundary layer edges
    boundary_layer_edges = {
        e for e in edges if e not in boundary_edges and
                            (e[0] in {node for edge in boundary_edges for node in edge} or
                             e[1] in {node for edge in boundary_edges for node in edge})
    }

    # Remaining internal edges that are neither boundary nor boundary layer edges
    internal_edges = edges - boundary_edges - boundary_layer_edges

    iteration = 0
    energy_log = []

    print(f"Starting optimization with {len(edges)} initial edges.")

    # Main optimization loop
    success = False
    while iteration < max_iterations:
        iteration += 1
        internal_valencies = np.mean([all_valencies[i] for i in internal_vertices if i < len(all_valencies)])
        print(f"Iteration {iteration}: Avg valency {internal_valencies}, Target {target_avg_valency}")

        avg_valency = np.mean([all_valencies[i] for i in internal_vertices if i < len(all_valencies)])
        valency_error = max(0.0, avg_valency - target_avg_valency)
        current_energy = valency_error * valency_error

        energy_log.append({
            'iteration': iteration,
            'valency_energy': current_energy
        })

        # Check if the internal average valency has reached or is below the target
        tolerance = 0.01
        if internal_valencies <= target_avg_valency + tolerance or not edges:
            print("Optimization complete.")
            success = True
            break

        # Filter edges to meet minimum valency requirements
        valid_edges = [e for e in edges if all_valencies[e[0]] > min_valency and all_valencies[e[1]] > min_valency]

        # Filter for internal edges only, excluding boundary layer and boundary edges
        removable_edges = [e for e in valid_edges if e in internal_edges]

        print(f"Removable internal edges count: {len(removable_edges)}")

        # Process internal edges only
        if removable_edges:
            # Select an internal edge to remove and check valency post-removal
            inverse_lengths = np.array([1.0 / edge_lengths[e] for e in removable_edges])
            probabilities = inverse_lengths / np.sum(inverse_lengths)
            edge_to_remove = random.choices(removable_edges, weights=probabilities, k=1)[0]

            # Tentatively remove and check valency by recalculating all_valencies after removal
            edges.remove(edge_to_remove)
            tentative_all_valencies = calculate_valencies(np.array(list(edges)), num_vertices)
            tentative_internal_valency = np.mean(
                [tentative_all_valencies[i] for i in internal_vertices if i < len(tentative_all_valencies)])

            if tentative_internal_valency >= target_avg_valency:
                all_valencies = tentative_all_valencies  # Commit the new valency state
                uf.union(edge_to_remove[0], edge_to_remove[1])
                print(f"Removed internal edge: {edge_to_remove}")
            else:
                # Restore the edge if it drops the valency below target
                edges.add(edge_to_remove)
                print(f"Stopping as removal of internal edge {edge_to_remove} drops valency below target.")
                break
        else:
            print("No more internal edges can be removed. Target not achieved.")
            break

    if iteration >= max_iterations:
        print("Max iterations reached, stopping optimization.")

    # Finalize and return success flag
    edges, final_valencies, periodic_edges, filtered_node_positions, energy_log = finalize(
        edges, num_vertices, internal_vertices, periodic_edges, vertices_position, energy_log
    )
    return edges, final_valencies, periodic_edges, filtered_node_positions, energy_log, success


def optimize_valency_greedy(
    edges,
    periodic_edges,
    num_vertices,
    internal_vertices,
    boundary_vertices,
    vertices_position,
    target_avg_valency,
    min_valency=3,
    max_iterations=1000000
):
    """
    Heuristic/Greedy algorithm for valency optimization:
    - Always pick a vertex with max valency (>min_valency)
    - Remove one of its internal edges (never boundary/periodic)
    - Never let any internal vertex drop below min_valency
    - Periodic boundary untouched

    Returns:
    - edges, final_valencies, periodic_edges, filtered_node_positions, energy_log, success
    """

    edges = set(map(tuple, edges))
    periodic_edges = set(map(tuple, periodic_edges))

    # Precompute edge lengths for weighting
    edge_lengths = {edge: edge_length(vertices_position, edge) for edge in edges}

    # Classify edge types
    boundary_edges = {e for e in edges if e[0] in boundary_vertices or e[1] in boundary_vertices}
    boundary_layer_edges = {
        e for e in edges if e not in boundary_edges and
                            (e[0] in {node for edge in boundary_edges for node in edge} or
                             e[1] in {node for edge in boundary_edges for node in edge})
    }
    internal_edges = edges - boundary_edges - boundary_layer_edges

    energy_log = []
    iteration = 0
    success = False

    print(f"Starting greedy valency optimization with {len(edges)} initial edges.")

    while iteration < max_iterations:
        iteration += 1

        all_valencies = calculate_valencies(np.array(list(edges)), num_vertices)
        internal_valencies = np.array([all_valencies[i] for i in internal_vertices if i < len(all_valencies)])
        avg_valency = np.mean(internal_valencies)
        valency_error = max(0.0, avg_valency - target_avg_valency)
        current_energy = valency_error ** 2
        energy_log.append({'iteration': iteration, 'valency_energy': current_energy})

        print(f"Iteration {iteration}: Avg valency {avg_valency:.4f}, Target {target_avg_valency}")

        # Stop condition: we've hit the target or can't remove more
        tolerance = 0.01
        if avg_valency <= target_avg_valency + tolerance or not internal_edges:
            print("Target average valency reached or no removable edges left.")
            success = True
            break

        # Find internal vertices with valency > min_valency
        candidate_vertices = [i for i in internal_vertices if all_valencies[i] > min_valency]
        if not candidate_vertices:
            print("No internal vertices above min valency. Stopping.")
            break

        # Pick the vertex with the highest valency
        max_valency = max([all_valencies[i] for i in candidate_vertices])
        max_valency_vertices = [i for i in candidate_vertices if all_valencies[i] == max_valency]
        v = random.choice(max_valency_vertices)

        # Find removable edges for this vertex (must be internal and not periodic/boundary)
        removable_edges = [
            e for e in internal_edges
            if (v in e) and all_valencies[e[0]] > min_valency and all_valencies[e[1]] > min_valency
        ]

        if not removable_edges:
            print(f"No removable edges for vertex {v} with valency {max_valency}.")
            continue

        # Pick shortest removable edge (heuristic: favor shorter edges for removal)
        edge_to_remove = min(removable_edges, key=lambda e: edge_lengths[e])

        # Remove edge and update sets
        edges.remove(edge_to_remove)
        internal_edges.remove(edge_to_remove)
        print(f"Removed edge {edge_to_remove} from vertex {v}.")

    if iteration >= max_iterations:
        print("Max iterations reached, stopping optimization.")

    # Finalize and return (you should have a finalize function as before)
    edges, final_valencies, periodic_edges, filtered_node_positions, energy_log = finalize(
        edges, num_vertices, internal_vertices, periodic_edges, vertices_position, energy_log
    )
    return edges, final_valencies, periodic_edges, filtered_node_positions, energy_log, success


def compute_and_plot_valency_density(edges, periodic_edges, vertices, output_directory, a=0.5):
    """
    Computes the valency densities and generates a plot comparing initial and optimized valency distributions.
    """
    vertex_array = np.array([v for _, v in sorted(vertices.items())])
    internal_vertices = {i for i, v in enumerate(vertex_array) if np.all(np.abs(v) < a)}
    boundary_vertices = {i for i, v in enumerate(vertex_array) if np.any(np.abs(v) == a)}
    #internal_vertices = set(i for i, vertex in enumerate(vertices) if is_boundary(vertex, a))
    #boundary_vertices = set(i for i, vertex in enumerate(vertices) if is_internal(vertex, a) and not is_boundary(vertex, a))

    # Initially assume all internal nodes are purely internal
    purely_internal_indices = internal_vertices.copy()

    # Track nodes connected to boundary nodes
    connected_to_boundary = set()

    for edge in edges:
        # Check connections involving boundary nodes
        if edge[0] in boundary_vertices or edge[1] in boundary_vertices:
            # Update connections for internal nodes
            if edge[0] in internal_vertices:
                connected_to_boundary.add(edge[0])
            if edge[1] in internal_vertices:
                connected_to_boundary.add(edge[1])

    # Remove nodes connected to boundary from purely internal
    purely_internal_indices -= connected_to_boundary

    # Boundary Layer Indices include boundary nodes and those connected to boundary nodes
    boundary_layer_indices = connected_to_boundary | boundary_vertices

    all_valencies = calculate_valencies(edges, len(vertices))
    internal_indices = np.array(list(internal_vertices), dtype=int)
    internal_valencies = all_valencies[internal_indices]

    # Calculate the densities for valencies 3 and 4
    initial_valency_counts = np.array([np.sum(internal_valencies == 3), np.sum(internal_valencies == 4)])
    initial_valency_density = initial_valency_counts / len(internal_valencies)

    positions = np.array([3, 4])
    viridis_shades = [0.3, 0.5, 0.7, 0.9]
    bar_width = 0.15

    target_averages = [3.12, 3.20, 3.36, 3.56]
    labels = [
        "37 °C, $\\bar{z}$ = 3.12",
        "32 °C, $\\bar{z}$ = 3.20",
        "30 °C, $\\bar{z}$ = 3.36",
        "26 °C, $\\bar{z}$ = 3.56"
    ]

    overlap_factor = 0.05
    total_width = (bar_width - overlap_factor) * len(target_averages)  # Adjusted for overlap
    initial_offset = -total_width / 2 + (bar_width / 2)  # Center the bars
    offsets = [initial_offset + i * (bar_width - overlap_factor) for i in range(len(target_averages))]

    valency_densities = []  # Store densities for each target average

    for target_avg in target_averages:
        updated_edges, updated_valencies, updated_periodic_edges, updated_vertices_positions, energy_log = optimize_valency(
            edges=edges,
            periodic_edges=periodic_edges,
            num_vertices=len(vertices),
            internal_vertices=internal_indices,
            boundary_vertices=boundary_vertices,
            vertices_position=vertex_array,
            target_avg_valency=target_avg,
            min_valency=3
        )

        updated_valencies = calculate_valencies(updated_edges, len(vertices))

        # Calculate valency counts and densities for valency 3 and 4
        final_valency_counts = np.array([np.sum(updated_valencies == 3), np.sum(updated_valencies == 4)])
        final_valency_density = final_valency_counts / np.sum(final_valency_counts)

        # Store the result
        valency_densities.append(final_valency_density)

    # Plot the precomputed data
    #plt.figure(figsize=(9, 6))
    plt.figure(figsize=(6, 6))

    for i, (final_valency_density, label) in enumerate(zip(valency_densities, labels)):
        from matplotlib.cm import viridis
        offset = offsets[i]

        plt.bar(positions + offset, final_valency_density, width=bar_width, color=viridis(viridis_shades[i]), alpha=0.7,
                label=label, edgecolor='black', linewidth=1)

    # Configure the plot aesthetics
    plt.xticks(ticks=positions, labels=['3', '4'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('connectivity', fontsize=20)
    plt.ylabel('probability', fontsize=20)
    plt.ylim(0, 1.0)
    plt.xlim(2.6, 4.4)
    plt.legend(fontsize=16, loc="upper right", frameon=True, borderpad=1.2, labelspacing=1.15)

    plot_path = os.path.join(output_directory, 'ConnectivityPlot.svg')
    plt.savefig(plot_path)
    plt.show()

    return plot_path


def finalize(edges, num_vertices, internal_vertices, periodic_edges, vertices_position, energy_log):
    """
    Final step for the valency optimization.

    Args:
    - edges: all the edges
    - num_vertices: total number of vertices
    - internal_vertices: all internal vertices of the cubic domain
    - periodic_edges: periodic mapping
    - vertices_position: all vertices positions

    Returns:
    - edges: final edges re-indexed based on the filtered set of nodes
    - final_valencies: final valencies computation
    - periodic_edges: periodic edges only include nodes that are still present in the graph
    - filtered_node_positions: stores only the positions of remaining nodes, removing those that were excluded
    """
    # Recalculate valencies based on the final edge configuration
    final_valencies = calculate_valencies(np.array(list(edges)), num_vertices)

    # Final internal valency average calculation
    internal_valency_avg = np.mean([final_valencies[i] for i in internal_vertices if i < len(final_valencies)])
    print(f"Final average valency for internal nodes: {internal_valency_avg}")

    # Final cleanup to ensure node and edge consistency
    remaining_nodes = {node for edge in edges for node in edge}
    periodic_edges = np.array(
        [edge for edge in periodic_edges if edge[0] in remaining_nodes and edge[1] in remaining_nodes]
    )

    # Filter node_positions to include only nodes in remaining_nodes
    filtered_node_positions = []
    node_map = {}  # Map old indices to new indices
    for new_index, old_index in enumerate(sorted(remaining_nodes)):
        filtered_node_positions.append(vertices_position[old_index])
        node_map[old_index] = new_index

    # Update edges and periodic edges to reflect new indices
    edges = np.array([[node_map[edge[0]], node_map[edge[1]]] for edge in edges])
    periodic_edges = np.array([[node_map[edge[0]], node_map[edge[1]]] for edge in periodic_edges])

    # Convert filtered positions to a numpy array
    filtered_node_positions = np.array(filtered_node_positions)

    # Return the processed data
    return edges, final_valencies, periodic_edges, filtered_node_positions,energy_log


class UnionFind:
    def __init__(self, n):
        """
        Initializes a Union-Find (Disjoint Set) data structure.

        Args:
        - n: Number of elements (nodes).

        Attributes:
        - self.parent: List where each element is initially its own parent (self-loop).
        - self.rank: List to track tree depth for union by rank.
        """
        self.parent = list(range(n))  # Each node is its own parent initially.
        self.rank = [0] * n  # Rank (tree height) starts at 0 for all nodes.

    def find(self, u):
        """
        Finds the representative (root) of the set containing u.
        Implements path compression to flatten the tree and optimize future queries.

        Args:
        - u: The element whose set representative is to be found.

        Returns:
        - The root representative of the set containing u.
        """
        if self.parent[u] != u:  # If u is not its own parent (not the root)
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]  # Return the root of the set

    def union(self, u, v):
        """
        Merges the sets containing u and v.
        Uses union by rank to keep trees balanced.

        Args:
        - u, v: Elements to be united.

        Updates:
        - The parent of one set is updated to be the root of the other set.
        - Rank is increased if necessary.
        """
        root_u = self.find(u)  # Find the root of u
        root_v = self.find(v)  # Find the root of v

        if root_u != root_v:  # If they belong to different sets
            if self.rank[root_u] > self.rank[root_v]:  # Attach smaller tree to bigger tree
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:  # If ranks are equal, choose one as the root and increase its rank
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

    def connected(self, u, v):
        """
        Checks if u and v belong to the same set.

        Args:
        - u, v: Elements to check.

        Returns:
        - True if u and v are in the same set, False otherwise.
        """
        return self.find(u) == self.find(v)  # If they have the same root, they are connected.


def compute_edge_lengths(vertices, edges):
    """
    Compute the length (computational one) associated with each edge

    Args:
    - vertices: vertices position
    - edges: connectivity list

    Returns:
    - length
    """
    # Force edges into a proper array shape
    if isinstance(edges, set):
        edges = list(edges)

    edges_array = np.array(edges)

    if edges_array.ndim != 2 or edges_array.shape[1] != 2:
        raise ValueError(f"Invalid edge format: expected (N, 2), got shape {edges_array.shape} and type {type(edges)}")

    vertices_position = np.array([vertices[i] for i in range(len(vertices))])
    diff = vertices_position[edges_array[:, 0]] - vertices_position[edges_array[:, 1]]
    return np.linalg.norm(diff, axis=1)


def kl_divergence(current_lengths, target_distribution, dx=0.01, bins=None):
    """
    Compute KL divergence metric quantifying the distance of the actual length distribution from the target one.
    Args:
    - current_lengths: The observed sample lengths from the actual distribution.
    - target_distribution: A probability distribution object representing the target distribution.
    - dx: The bin width for histogram estimation. Default is 0.01.
    - bins: Custom bin edges for histogram calculation. If None, bins are automatically generated using dx.

    Returns:
    - kl_div (float): The computed KL divergence value.
    - bins (numpy.ndarray): The bin edges used for the histogram computation.
    """
    if bins is None:
        bins = np.arange(min(current_lengths), max(current_lengths) + dx, dx)
    p_x = np.histogram(current_lengths, bins=bins, density=True)[0]
    q_x = target_distribution.pdf(bins[:-1] + dx / 2)
    q_x = np.maximum(q_x, 1e-10)  # Adding a small epsilon to prevent zero values
    p_x = np.maximum(p_x, 1e-10)  # Adding a small epsilon for consistency in p_x
    kl_div = np.sum(np.where(p_x != 0, p_x * np.log(p_x / q_x), 0))
    return kl_div, bins


def move_vertex(vertex, bounds=(-0.5, 0.5)):
    """
    Proposal movement for the Simulated Annealing.
    Args:
    - vertex: target vertex to be moved
    - bounds: computational boundaries

    Returns:
    - vertex : New position
    """
    movement = np.random.uniform(-0.01, 0.01, size=3)
    new_position = vertex + movement
    if np.all(bounds[0] < new_position) and np.all(new_position < bounds[1]):
        return new_position
    return vertex


def simulated_annealing_with_valency(state, periodic_edges, target_distribution, target_avg_valency, min_valency,
                                      bounds=(-0.5, 0.5), max_iterations=100000):


    vertices, edges = state['vertices'], state['edges']
    num_vertices = len(vertices)

    if isinstance(edges, np.ndarray):
        if edges.ndim == 2 and edges.shape[1] == 2:
            edges = [tuple(e) for e in edges]
        elif edges.ndim == 1 and len(edges) == 2:
            edges = [tuple(edges)]
        else:
            raise ValueError(f"Unexpected edge array shape: {edges.shape}")
    elif isinstance(edges, list):
        edges = [tuple(e) for e in edges]
    else:
        raise TypeError("Unsupported edge format")

    edges = set(edges)
    periodic_edges = set(map(tuple, periodic_edges))

    current_lengths = compute_edge_lengths(vertices, edges)
    length_energy, bins = kl_divergence(current_lengths, target_distribution)

    all_valencies = calculate_valencies(np.array(list(edges)), num_vertices)
    internal_vertices = [i for i, pos in vertices.items() if is_internal(pos, a=0.5)]
    boundary_vertices = [i for i, pos in vertices.items() if is_boundary(pos, a=0.5)]

    initial_valency = np.mean([all_valencies[i] for i in internal_vertices])
    print(f"Initial avg valency: {initial_valency:.4f}")

    best_vertices = deepcopy(vertices)
    best_edges = deepcopy(edges)
    best_energy = length_energy
    energy_log = []

    sample_dEs = []
    for _ in range(20):
        test_node = random.choice(internal_vertices)
        old_pos = vertices[test_node].copy()
        vertices[test_node] = move_vertex(old_pos, bounds)
        affected = [e for e in edges if test_node in e]
        temp_lengths = current_lengths.copy()
        for e in affected:
            idx = list(edges).index(e)
            temp_lengths[idx] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])
        new_E, _ = kl_divergence(temp_lengths, target_distribution, bins=bins)
        dE = new_E - length_energy
        if dE > 0:
            sample_dEs.append(dE)
        vertices[test_node] = old_pos

    avg_positive_dE = np.mean(sample_dEs) if sample_dEs else 1e-3
    T0 = -avg_positive_dE / np.log(0.5)
    Tmin = 1e-4 * T0

    print(f"Initial temperature T0: {T0:.4f}")

    recent_energies = []
    patience = 200
    min_relative_drop = 1e-5

    for iteration in range(max_iterations):
        T = T0 * (0.95 ** iteration)

        avg_valency = np.mean([all_valencies[i] for i in internal_vertices if i < len(all_valencies)])
        valency_error = max(0.0, avg_valency - target_avg_valency)
        p = 2.0
        valency_weight = (valency_error / (initial_valency - target_avg_valency)) ** p
        valency_energy = valency_error ** 2

        total_energy = length_energy + valency_weight * valency_energy
        energy_log.append({
            'iteration': iteration,
            'length_energy': length_energy,
            'valency_energy': valency_energy,
            'valency_weight': valency_weight,
            'total_energy': total_energy
        })

        if random.random() < 0.5 and avg_valency > target_avg_valency:
            removable_edges = [e for e in edges if all_valencies[e[0]] > min_valency and
                                                  all_valencies[e[1]] > min_valency and
                                                  e[0] in internal_vertices and e[1] in internal_vertices]
            if not removable_edges:
                continue
            edge_to_remove = random.choice(removable_edges)
            edges.remove(edge_to_remove)

            temp_valencies = calculate_valencies(np.array(list(edges)), num_vertices)
            new_avg_valency = np.mean([temp_valencies[i] for i in internal_vertices])

            if new_avg_valency >= target_avg_valency:
                all_valencies = temp_valencies
                print(f"Iteration {iteration}: Removed edge {edge_to_remove}, Avg valency: {new_avg_valency:.4f}")
            else:
                edges.add(edge_to_remove)

        else:
            node_index = random.choice(internal_vertices)
            old_position = vertices[node_index].copy()
            vertices[node_index] = move_vertex(old_position, bounds)

            affected_edges = [e for e in edges if node_index in e]
            updated_lengths = current_lengths.copy()
            for e in affected_edges:
                idx = list(edges).index(e)
                updated_lengths[idx] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])

            new_length_energy, _ = kl_divergence(updated_lengths, target_distribution, bins=bins)
            new_total_energy = new_length_energy + valency_weight * valency_energy
            dE = new_total_energy - total_energy

            if dE < 0 or math.exp(-dE / T) > random.random():
                current_lengths = updated_lengths
                length_energy = new_length_energy
                print(f"Iteration {iteration}: Vertex {node_index} moved, total energy: {new_total_energy:.4f}")
                if new_total_energy < best_energy:
                    best_vertices = deepcopy(vertices)
                    best_edges = deepcopy(edges)
                    best_energy = new_total_energy
                    print("    New best state recorded.")
            else:
                vertices[node_index] = old_position

        # Enhanced stopping condition
        recent_energies.append(total_energy)
        if len(recent_energies) > patience:
            recent_energies.pop(0)
            energy_drop = recent_energies[0] - recent_energies[-1]
            relative_drop = energy_drop / max(abs(recent_energies[0]), 1e-10)
            if T < Tmin and valency_error < 1e-3 and relative_drop < min_relative_drop:
                print(f"Stopping early at iteration {iteration}. Converged: T < Tmin, valency met, and energy flat.")
                break

    if len(energy_log) > 1:
        iterations = [math.log10(d['iteration'] + 1) for d in energy_log]
        total_energies = [math.log10(d['total_energy']) for d in energy_log]
        slope, intercept, r_value, _, _ = linregress(iterations, total_energies)
        print(f"Estimated overall decay rate (log-log slope): {slope:.3f} — implies power-law ~ t^{slope:.2f}")

    return best_vertices, list(best_edges), energy_log


def test_simulated_annealing_with_valency(state, periodic_edges, target_distribution, target_avg_valency, min_valency,
                                           bounds=(-0.5, 0.5), max_iterations=100000):
    import matplotlib.pyplot as plt
    from copy import deepcopy

    protocols = {
        'baseline': {'cooling_rate': 0.95, 'T0_scale': 1.0, 'p': 2.0, 'boost': False},
        'p_linear': {'cooling_rate': 0.95, 'T0_scale': 1.0, 'p': 1.0, 'boost': False},
        'super_boosted': {'cooling_rate': 0.95, 'T0_scale': 1.0, 'p': 2.0, 'boost': True},
        'hot_boosted': {'cooling_rate': 0.90, 'T0_scale': 3.0, 'p': 2.0, 'boost': True},
        'adaptive_cooling': {'cooling_rate': 0.97, 'T0_scale': 3.0, 'p': 3.0, 'boost': True},
    }

    all_logs = {}

    for label, config in protocols.items():
        print(f"Running protocol: {label}")
        custom_state = deepcopy(state)
        custom_vertices, custom_edges = custom_state['vertices'], custom_state['edges']

        best_vertices, best_edges, energy_log = simulated_annealing_with_valency(
            state={'vertices': deepcopy(custom_vertices), 'edges': deepcopy(custom_edges)},
            periodic_edges=periodic_edges,
            target_distribution=target_distribution,
            target_avg_valency=target_avg_valency,
            min_valency=min_valency,
            bounds=bounds,
            max_iterations=max_iterations
        )

        all_logs[label] = energy_log

    plt.figure(figsize=(6, 6))
    for label, log in all_logs.items():
        iterations = [entry['iteration'] for entry in log]
        energies = [entry['total_energy'] for entry in log]
        plt.loglog(iterations, energies, label=label)

    plt.xlabel('Iteration', fontsize = 16)
    plt.ylabel('Total Energy', fontsize = 16)
    plt.legend(fontsize =16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_energy(energy_log_val, energy_log_length, output_directory):
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # Normalize valency energy
    df_val = pd.DataFrame(energy_log_val)
    df_val = df_val.copy()
    df_val['valency_energy_norm'] = df_val['valency_energy'] / df_val['valency_energy'].max()

    # Normalize length energy
    df_len = pd.DataFrame(energy_log_length)
    df_len = df_len.copy()
    df_len['length_energy_norm'] = df_len['length_energy'] / df_len['length_energy'].max()

    # Plot valency energy
    plt.figure(figsize=(6, 6))
    plt.semilogx(df_val['iteration'], df_val['valency_energy_norm'], color='black', linestyle='--', linewidth=2)
    plt.xlabel('iteration', fontsize=16)
    plt.ylabel(r'$E_z$ (normalized)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "valency_energy.svg"), format="svg")
    plt.show()

    # Plot length energy
    plt.figure(figsize=(6, 6))
    plt.semilogx(df_len['iteration'], df_len['length_energy_norm'], color='black', linestyle='-', linewidth=2)
    plt.xlabel('iteration', fontsize=16)
    plt.ylabel(r'$E_l$ (normalized)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "length_energy.svg"), format="svg")
    plt.show()



def simulated_annealing_without_valency(state, target_distribution, bounds, max_iterations=10000000):
    """
    Proposal movement for the Simulated Annealing (without valency constraints).
    Args:
    - state: current topological configuration
    - target_distribution: length distribution in this case
    - bounds: computational boundaries
    - max_iterations: maximum number of iterations

    Returns:
    - node : New node position
    - edges: previous edges (not modified by the annealing)
    - energy_log: list of dictionaries logging energy and temperature per iteration
    """
    vertices, edges = state['vertices'], state['edges']
    current_lengths = compute_edge_lengths(vertices, edges)
    current_energy, bins = kl_divergence(current_lengths, target_distribution)

    internal_nodes = [node_index for node_index, node_pos in vertices.items()
                      if np.all(bounds[0] < np.array(node_pos)) and np.all(np.array(node_pos) < bounds[1])]

    # Estimate initial temperature
    sample_dEs = []
    for _ in range(20):
        test_node = random.choice(internal_nodes)
        old_pos = vertices[test_node].copy()
        vertices[test_node] = move_vertex(old_pos, bounds)
        affected = [e for e in edges if test_node in e]
        temp_lengths = current_lengths.copy()
        for e in affected:
            idx = edges.index(e)
            temp_lengths[idx] = np.linalg.norm(vertices[e[0]] - vertices[e[1]])
        new_E, _ = kl_divergence(temp_lengths, target_distribution, bins=bins)
        dE = new_E - current_energy
        if dE > 0:
            sample_dEs.append(dE)
        vertices[test_node] = old_pos

    avg_positive_dE = np.mean(sample_dEs) if sample_dEs else 1e-3
    T0 = -avg_positive_dE / np.log(0.5)
    Tmin = 1e-4 * T0

    print(f"Initial temperature T0: {T0:.4f}")

    recent_energies = []
    energy_log = []
    #patience = 500
    #min_relative_drop = 1e-3
    patience = 500
    min_relative_drop = 1e-6

    for iteration in range(max_iterations):
        T = T0 * (0.95 ** iteration)

        if not internal_nodes:
            print("No internal nodes available to move. Stopping simulation.")
            break

        node_index = random.choice(internal_nodes)
        old_position = vertices[node_index].copy()
        vertices[node_index] = move_vertex(vertices[node_index], bounds)

        affected_edges = [e for e in edges if node_index in e]
        affected_lengths = [
            np.linalg.norm(vertices[e[0]] - vertices[e[1]]) for e in affected_edges
        ]
        updated_lengths = current_lengths.copy()
        for i, e in enumerate(affected_edges):
            index = edges.index(e)
            updated_lengths[index] = affected_lengths[i]

        new_energy, _ = kl_divergence(updated_lengths, target_distribution, bins=bins)
        dE = new_energy - current_energy

        if dE < 0 or math.exp(-dE / T) > random.random():
            current_energy = new_energy
            current_lengths = updated_lengths
            print(f"Iteration {iteration + 1}: Node {node_index} moved")
            print(f"    Old Position: {old_position}")
            print(f"    New Position: {vertices[node_index]}")
            print(f"    Energy: {new_energy:.4f}")
        else:
            vertices[node_index] = old_position

        # Log energy in full format
        energy_log.append({
            'iteration': iteration,
            'length_energy': current_energy,
            'total_energy': current_energy,
            'temperature': T
        })

        # Enhanced stopping condition
        recent_energies.append(current_energy)
        if len(recent_energies) > patience:
            recent_energies.pop(0)
            energy_drop = recent_energies[0] - recent_energies[-1]
            relative_drop = energy_drop / max(abs(recent_energies[0]), 1e-10)
            if T < Tmin and relative_drop < min_relative_drop:
                print(f"Stopping early at iteration {iteration}. Converged: T < Tmin and energy flat.")
                break

    return vertices, edges, energy_log


def read_vertices(vertices_file_path):
    vertices = {}
    with open(vertices_file_path, 'r') as f:
        for i, line in enumerate(f):
            x, y, z = map(float, line.strip().split())
            vertices[i] = np.array([x, y, z])
    return vertices


def read_edges(edges_file_path):
    edges = []
    with open(edges_file_path, 'r') as f:
        for line in f:
            node1, node2 = line.strip().split()
            edges.append((int(node1), int(node2)))
    return edges



