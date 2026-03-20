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
from mpl_toolkits.mplot3d import Axes3D

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
    - internal_vertices: all internal vertices of the domain
    - boundary_vertices: all boundary vertices of the domain
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

def finalize(edges, num_vertices, internal_vertices, periodic_edges, vertices_position, energy_log):
    """
    Final step for the valency optimization.

    Args:
    - edges: all the edges
    - num_vertices: total number of vertices
    - internal_vertices: all internal vertices of the domain
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

def optimize_length(state, target_distribution, bounds, max_iterations=10000000):
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
    print_interval = 50  # Print details every 50 iterations

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
            if (iteration + 1) % print_interval == 0:
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


def plot_edge_orientation_and_network(vertices, edges, bins=5, output_directory=None, tolerance_deg=15):
    """
    Plots histograms for edge orientations with respect to X, Y, Z axes.
    Colors edges in 3D/2D as X-aligned (red), Y-aligned (green), Z-aligned (blue), others (grey).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    verts = np.array(vertices)
    edges_arr = np.array(edges)
    axis_vecs = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    axis_labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    tolerance_rad = np.radians(tolerance_deg)

    # Compute edge unit vectors
    edge_vecs = verts[edges_arr[:,1]] - verts[edges_arr[:,0]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    edge_unit = edge_vecs / (edge_lengths[:,None] + 1e-12)

    # Compute angles to each axis
    angles_all = []
    for axis in axis_vecs:
        dots = np.abs(np.dot(edge_unit, axis))
        angles = np.arccos(np.clip(dots, 0, 1)) * 180 / np.pi  # degrees
        angles_all.append(angles)

    # Plot histograms for all axes
    plt.figure(figsize=(6,4))
    for i, (angles, label, color) in enumerate(zip(angles_all, axis_labels, colors)):
        plt.hist(angles, bins=bins, alpha=0.5, label=f'{label}-axis', color=color, edgecolor='k', density=True)
    plt.xlabel('Alignment angle [deg]', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if output_directory:
        plt.savefig(os.path.join(output_directory, 'edge_orientation_hist_all_axes.png'))
    plt.show()

    # Assign color to each edge based on alignment
    edge_colors = []
    for i in range(len(edges_arr)):
        assigned = False
        for axis_idx, angles in enumerate(angles_all):
            if angles[i] < tolerance_deg:
                edge_colors.append(colors[axis_idx])
                assigned = True
                break
        if not assigned:
            edge_colors.append('grey')

    # # 3D plot
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_subplot(111, projection='3d')
    # for e, color in zip(edges_arr, edge_colors):
    #     ax.plot(*verts[e].T, color=color, linewidth=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.tight_layout()
    # if output_directory:
    #     plt.savefig(os.path.join(output_directory, 'network_3d_orientation_axes.png'))
    # plt.show()

    # # 2D projections for each axis
    # projections = [(0,1,'XY'), (0,2,'XZ'), (1,2,'YZ')]
    # for ax0, ax1, label in projections:
    #     plt.figure(figsize=(7,7))
    #     for e, color in zip(edges_arr, edge_colors):
    #         plt.plot(
    #             [verts[e[0], ax0], verts[e[1], ax0]],
    #             [verts[e[0], ax1], verts[e[1], ax1]],
    #             color=color, linewidth=0.8
    #         )
    #     plt.xlabel(['X','Y','Z'][ax0])
    #     plt.ylabel(['X','Y','Z'][ax1])
    #     plt.title(f'{label} projection')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     if output_directory:
    #         plt.savefig(os.path.join(output_directory, f'network_{label}_projection_axes.png'))
    #     plt.show()

def plot_network(vertices, edges, output_directory=None):
    """
    Plots the network twice:
    1. 2D maximum projection in the XY plane with thick edges, no axis labels.
    2. 3D network with thick edges, no axis labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    verts = np.array(vertices)
    edges_arr = np.array(edges)
    # Calculate connectivity for each node
    connectivity = np.zeros(len(verts), dtype=int)
    for e in edges_arr:
        connectivity[e[0]] += 1
        connectivity[e[1]] += 1
    # Identify boundary nodes (any coordinate at -0.5 or 0.5)
    boundary_mask = np.any(np.isclose(verts, -0.5, atol=1e-8) | np.isclose(verts, 0.5, atol=1e-8), axis=1)
    # Internal nodes: not boundary
    internal_mask = ~boundary_mask
    # Internal nodes with connectivity 3 and 4
    conn3_mask = (connectivity == 3) & internal_mask
    conn4_mask = (connectivity == 4) & internal_mask

    # --- Single plot: internal edges in viridis purple, boundary edges light grey, nodes in two other viridis colors ---
    import matplotlib.cm as cm
    viridis = cm.get_cmap('viridis')
    # Choose viridis colors: purple (0.1) for edges, green (0.3) and yellow (0.6) for nodes
    edge_color = viridis(0.1)
    node3_color = viridis(0.4)
    node4_color = viridis(0.8)

    # Identify boundary edges (at least one node is boundary)
    boundary_nodes = np.where(boundary_mask)[0]
    boundary_edges_mask = np.array([
        (e[0] in boundary_nodes) or (e[1] in boundary_nodes)
        for e in edges_arr
    ])

    # --- 2D XY maximum projection plot ---
    fig2d = plt.figure(figsize=(8, 6))
    ax2d = fig2d.add_subplot(111)
    for i, e in enumerate(edges_arr):
        if boundary_edges_mask[i]:
            ax2d.plot(
                [verts[e[0], 0], verts[e[1], 0]],
                [verts[e[0], 1], verts[e[1], 1]],
                color='lightgrey', linewidth=2, alpha=0.6
            )
        else:
            ax2d.plot(
                [verts[e[0], 0], verts[e[1], 0]],
                [verts[e[0], 1], verts[e[1], 1]],
                color=edge_color, linewidth=2, alpha=0.6
            )
    # Internal nodes with connectivity 3 and 4, use large size
    ax2d.scatter(
        verts[conn3_mask, 0], verts[conn3_mask, 1],
        color=node3_color, s=20, alpha=0.95, label='conn=3'
    )
    ax2d.scatter(
        verts[conn4_mask, 0], verts[conn4_mask, 1],
        color=node4_color, s=20, alpha=0.95, label='conn=4'
    )
    ax2d.set_xticks([])
    ax2d.set_yticks([])
    ax2d.set_frame_on(False)
    ax2d.set_aspect('equal')
    plt.tight_layout()
    if output_directory:
        plt.savefig(os.path.join(output_directory, 'network_2d_xy_projection.png'), bbox_inches='tight', pad_inches=0.05)
    plt.show()

    # --- 3D plot ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, e in enumerate(edges_arr):
        if boundary_edges_mask[i]:
            ax.plot(
                [verts[e[0], 0], verts[e[1], 0]],
                [verts[e[0], 1], verts[e[1], 1]],
                [verts[e[0], 2], verts[e[1], 2]],
                color='lightgrey', linewidth=2, alpha=0.6
            )
        else:
            ax.plot(
                [verts[e[0], 0], verts[e[1], 0]],
                [verts[e[0], 1], verts[e[1], 1]],
                [verts[e[0], 2], verts[e[1], 2]],
                color=edge_color, linewidth=2, alpha=0.6
            )
    # Internal nodes with connectivity 3 and 4, use large size
    ax.scatter(
        verts[conn3_mask, 0], verts[conn3_mask, 1], verts[conn3_mask, 2],
        color=node3_color, s=2, alpha=0.95, depthshade=True, label='conn=3'
    )
    ax.scatter(
        verts[conn4_mask, 0], verts[conn4_mask, 1], verts[conn4_mask, 2],
        color=node4_color, s=2, alpha=0.95, depthshade=True, label='conn=4'
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_frame_on(False)
    plt.tight_layout()
    if output_directory:
        plt.savefig(os.path.join(output_directory, 'network_3d_viridis.png'), bbox_inches='tight', pad_inches=0.05)
    plt.show()


from scipy.stats import lognorm
import matplotlib.cm as cm

def plot_lognormal_histograms(
    domain_dim: float = 40.0,
    L_originals = None,
    v_list = None,
    n_bins: int = 30,
    bins_alpha: float = 0.2,
    match_current_scaling: bool = True,
):
    """
    Plot lognormal target distributions (continuous PDF curves) and overlay discrete bins
    representing a finite set of edge-length intervals.

    Parameters
    ----------
    domain_dim : float
        Scale factor between computational and real edge length (x_real = domain_dim * x_comp).
    L_originals : array-like
        Mean edge lengths in real units for each distribution. If None, defaults to np.linspace(1.5, 3, 2).
    v_list : array-like
        Dimensionless variance-like parameter for each distribution. If None, defaults to np.linspace(0.1, 0.5, 2).
    n_bins : int
        Number of discrete bins across [x_min, x_max] to represent the finite edge intervals.
    bins_alpha : float
        Transparency for the discrete bins (0..1).
    match_current_scaling : bool
        If True, keeps the same vertical scaling as your current code (plots pdf(x_comp) vs x_real).
        If False, uses physically correct real-units density: pdf_real(x_real) = pdf_comp(x_comp) / domain_dim.
        Bars are scaled consistently either way.

    Notes
    -----
    - Each distribution's continuous curve is colored with a unique color (viridis),
      and its bins are plotted with the same color at `bins_alpha` transparency.
    - Bin heights are computed from probability mass in each bin divided by bin width (density),
      so the area under the bars approximates the total probability.
    """
    # Define explicit sample pairs as requested
    pairs = [ (2, 0.25), (3.0, 0.50), (1.0, 0.10) ]
    n_pairs = len(pairs)
    # Use lower half of viridis for lognormal
    color_map = lambda idx: cm.get_cmap('viridis')(0.1 + 0.7 * idx / max(n_pairs-1,1))

    # First, sample to determine sensible x-range in real units
    all_samples = []
    for (L_original, v) in pairs:
        L_comp = L_original / domain_dim
        s2 = (L_original ** 2) * v / (domain_dim ** 2)
        sigma2 = np.log(1 + s2 / L_comp ** 2)
        sigma = np.sqrt(sigma2)
        mu = np.log(L_comp) - sigma2 / 2
        target_dist = lognorm(s=sigma, scale=np.exp(mu))
        samples_real = target_dist.rvs(size=2000) * domain_dim
        all_samples.append(samples_real)

    all_samples_flat = np.concatenate(all_samples)
    x_min = 0.0
    x_max = np.percentile(all_samples_flat, 99.5)
    x_real = np.linspace(x_min, x_max, 600)

    # Prepare bins in real units
    bin_edges_real = np.linspace(x_min, x_max, n_bins + 1)
    bin_width_real = bin_edges_real[1] - bin_edges_real[0]
    bin_centers_real = 0.5 * (bin_edges_real[:-1] + bin_edges_real[1:])

    plt.figure(figsize=(6, 4))

    # Plot each distribution
    for idx, (L_original, v) in enumerate(pairs):
        color = color_map(idx)

        # Parameterize the lognormal in computational units
        L_comp = L_original / domain_dim
        s2 = (L_original ** 2) * v / (domain_dim ** 2)
        sigma2 = np.log(1 + s2 / L_comp ** 2)
        sigma = np.sqrt(sigma2)
        mu = np.log(L_comp) - sigma2 / 2
        target_dist = lognorm(s=sigma, scale=np.exp(mu))
        # Use L with overline (L̄) for meanvlegend label
        label = f"$\\overline{{L}}$ = {L_original:.2f}, v = {v:.2f}"

        # Continuous curve
        x_comp = x_real / domain_dim
        pdf_comp = target_dist.pdf(x_comp)
        if match_current_scaling:
            y_curve = pdf_comp
        else:
            # Proper density in real units
            y_curve = pdf_comp / domain_dim
        plt.plot(x_real, y_curve, color=color, lw=2.0, alpha=0.95, label=label, zorder=3)

        # Discrete bins: probability mass per bin
        cdf_hi = target_dist.cdf(bin_edges_real[1:] / domain_dim)
        cdf_lo = target_dist.cdf(bin_edges_real[:-1] / domain_dim)
        prob_mass = cdf_hi - cdf_lo

        # Convert to density to be comparable with the curve
        if match_current_scaling:
            # Keep the same vertical scaling as the original code (computational units)
            bin_width_comp = bin_width_real / domain_dim
            bar_heights = prob_mass / bin_width_comp  # ~ pdf_comp at the bin
        else:
            # Proper real-units density
            bar_heights = prob_mass / bin_width_real  # ~ pdf_real at the bin

        # Draw semi-transparent bars for these bins
        plt.bar(
            bin_centers_real,
            bar_heights,
            width=bin_width_real,
            color=color,
            alpha=bins_alpha,
            edgecolor='none',
            align='center',
            zorder=1,
        )

    plt.xlabel('fiber length [μm]', fontsize=30)
    plt.ylabel('PDF', fontsize=30)
    #plt.legend(fontsize=16, loc='upper right', frameon=True)
    plt.xticks(fontsize=20)
    plt.yticks([], [])  # Remove y-axis tick labels
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    plt.show()


def plot_von_mises_angle_distributions():
    """
    Plots 4 bivariate von Mises angle distributions with parameters:
    CASE 1: a=0.5, b=0.5
    CASE 2: a=5, b=0.5
    CASE 3: a=5, b=5
    Each sample is colored with a different viridis color and all distributions are shown in the same plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.stats import vonmises

    # Parameters for each case
    params = [(0.5, 0.5), (5.0, 0.5), (5.0, 5.0)]
    n_cases = len(params)
    # Use upper half of viridis for von Mises
    color_map = lambda idx: cm.get_cmap('viridis')(0.6 + 0.35 * idx / max(n_cases-1,1))

    plt.figure(figsize=(6, 4))
    x = np.linspace(-np.pi, np.pi, 500)
    labels = [f"K1={a}, K2={b}" for a, b in params]

    for idx, (a, b) in enumerate(params):
        color = color_map(idx)
        # Bivariate von Mises: for demo, sum two independent von Mises distributions
        # (true bivariate von Mises is more complex, but this shows the effect)
        pdf_a = vonmises.pdf(x, a)
        pdf_b = vonmises.pdf(x, b)
        # Combine (for visualization, just average)
        pdf = 0.5 * (pdf_a + pdf_b)
        # Sample for histogram
        samples_a = vonmises.rvs(a, size=1000)
        samples_b = vonmises.rvs(b, size=1000)
        samples = np.concatenate([samples_a, samples_b])
        hist, bins = np.histogram(samples, bins=20, range=(-np.pi, np.pi), density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(x, pdf, color=color, lw=2, alpha=0.8, label=labels[idx])
        plt.bar(bin_centers, hist, width=(bins[1]-bins[0]), color=color, alpha=0.2, edgecolor='none', align='center')

    plt.xlabel('fiber orientation [rad]', fontsize=30)
    plt.ylabel('PDF', fontsize=30)
    #plt.legend(fontsize=16, loc='upper right', frameon=True)
    # Set x-ticks to -π, 0, π with symbols
    plt.xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"] , fontsize=20)
    plt.yticks([], [])  # Remove y-axis tick labels
    plt.xlim(-np.pi, np.pi)
    plt.tight_layout()
    plt.show()