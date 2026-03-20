
"""
╔═══════════════════════════════════════════════════════════════╗
║               TOPOGEN SIMULATION PIPELINE                     ║
╠═══════════════════════════════════════════════════════════════╣
║   STEP 1:  Periodic Voronoi tessellation                      ║
║   STEP 2:  Valency & Length optimization                      ║
║   STEP 3:  Network refinement (connectivity & dangling ends)  ║
║   STEP 4:  Abaqus input file generation                       ║
╚═══════════════════════════════════════════════════════════════╝
"""
import os
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.setup import setup_output_directory
from scipy.stats import qmc, lognorm

from create_periodic_network import (
    lloyd_relaxation_3d_periodic, get_vertices, get_edges, process_edges, count_pairs_by_dim,
    replica_removal, impose_preferential_orientation,  find_periodic_pairs_with_dim)

from optimize_periodic_network import (
    optimize_valency, optimize_length,
    read_edges, read_vertices, plot_edge_orientation_and_network
)
from write_abaqus_input_file import (
    element_orientation_definition, edges_length, compute_volume_fraction,
    compute_transverse_shear, order_nodes, remove_empty_lines,
    write_abaqus_input_files, sort_nodes_file_by_index
)

def step1(idx, sample, output_root):
    """
    Step 1: Generate the initial periodic network for a sample.

    This function generates:
    - If anisotropy is enabled, a stretched and oriented network according to the specified axis and Hermans parameter (P2_target).
    - For isotropic cases, a standard periodic Voronoi network.

    Inputs:
        idx (int): Sample index.
        sample (tuple): (N, target_avg_valency, L_original, young_modulus) parameters for the network.
        output_root (str or dict): Output directory or protocol config dict (may include anisotropy, domain, axis, P2_target).

    Outputs:
        vertices_file (str): Path to saved vertices file.
        edges_file (str): Path to saved edges file.
        periodic_edges_file (str): Path to saved periodic edges file.
        output_directory (str): Directory where outputs are saved.
    """
    N, target_avg_valency, L_original, young_modulus = sample
    
    # We here accept both that the output directory (and other relevant entries) is a string - as in the default case - or a dict with more parameters - as in the anisotropy case
    output_directory = output_root['output_directory'] if isinstance(output_root, dict) else setup_output_directory(f"Sample_{idx}", base_dir=output_root)
    anisotropy = output_root['anisotropy'] if isinstance(output_root, dict) else False
    computational_domain = output_root['computational_domain'] if isinstance(output_root, dict) else [(-0.5, 0.5)] * 3
    anisotropy_axis = output_root.get('anisotropy_axis', (1, 0, 0)) if isinstance(output_root, dict) else (1, 0, 0)
    P2_target = output_root.get('P2_target', 0.6) if isinstance(output_root, dict) else 0.6
    
    # File paths
    vertices_file = os.path.join(output_directory, "vertices.txt")
    edges_file = os.path.join(output_directory, "edges.txt")
    periodic_edges_file = os.path.join(output_directory, "periodic_edges.txt")

    # Unified protocol: always use find_periodic_pairs for both isotropic and anisotropic
    bounds = computational_domain
    points = np.random.uniform(bounds[0][0], bounds[0][1], (int(N), 3))
    vor = lloyd_relaxation_3d_periodic(points, 10, int(N), relax=True)
    tile_vertices = vor.vertices
    vertices, IndexMap = get_vertices(tile_vertices)
    tile_edges = process_edges(vor.ridge_vertices, tile_vertices)
    Vertices, Edges = get_edges(vertices, tile_vertices, tile_edges, IndexMap, bounds)
    unique_edges = replica_removal(Edges)
    FinalVertices = Vertices
    FinalEdges = unique_edges

    if anisotropy:
        # --- Anisotropy: Impose preferential orientation ---
        V0 = 1.0  # initial volume consistent with bounds [-0.5, 0.5]^3 - valid for every condition
        alignment_axis = anisotropy_axis
        tol = 1e-3
        max_iter = 1000
        length_weighted = False
        FinalVertices, P2_final, rho_final, lambdas = impose_preferential_orientation(
            nodes0=FinalVertices,
            edges=FinalEdges,
            V0=V0,
            bounds=bounds,
            a=alignment_axis,
            P2_target=P2_target,
            tol=tol,
            max_iter=max_iter,
            length_weighted=length_weighted,
            pairwise_mode="copy_low"
        )
        # Save lambdas dictionary as lambdas.npy for use in main
        lambdas_path = os.path.join(output_directory, 'lambdas.npy')
        np.save(lambdas_path, lambdas)
        print(f"[Anisotropy] Achieved P2: {P2_final:.4f} (target {P2_target}), density: {rho_final:.4f}")
    if anisotropy == False:
        PeriodicEdges = find_periodic_pairs_with_dim(FinalVertices, bounds)
    else:
        PeriodicEdges = find_periodic_pairs_with_dim(FinalVertices, lambdas["bounds_after"])
    
    # Debug here: Print periodic pair counts after step1
    counts_end = count_pairs_by_dim(PeriodicEdges)
    #print(f"[step1] Periodic pairs: x={counts_end[0]}, y={counts_end[1]}, z={counts_end[2]}")
    
    np.savetxt(vertices_file, FinalVertices)
    np.savetxt(edges_file, FinalEdges, fmt="%d")
    np.savetxt(periodic_edges_file, PeriodicEdges, fmt="%d")
    return vertices_file, edges_file, periodic_edges_file, output_directory

def step2(vertices_file, edges_file, periodic_edges_file, output_directory, domain_dim, target_avg_valency, L_original, v, lambdas, anisotropy):
    """
    Step 2: Optimize network valency and edge lengths for a sample.
    This function performs:
    - For anisotropy, only valency optimization, using stretched domain bounds for boundary detection.
    - For isotropy, both valency and edge length optimization, using cubic domain bounds.

    Inputs:
        vertices_file (str): Path to input vertices file.
        edges_file (str): Path to input edges file.
        periodic_edges_file (str): Path to input periodic edges file.
        output_directory (str): Directory for output files.
        domain_dim (float or list): Physical domain size (scalar or vector).
        target_avg_valency (float): Target average valency for optimization.
        L_original (float): Original target edge length.
        v (float): Variance parameter for edge length distribution.
        lambdas (dict or None): Anisotropy transformation info (if applicable).
        anisotropy (bool): Whether anisotropy is enabled.

    Outputs:
        optimized_vertex_array (np.ndarray): Optimized vertex positions.
        optimized_edges (np.ndarray): Optimized edge list.
        updated_periodic_edges (np.ndarray): Updated periodic edge pairs.
    """
    original_vertices = read_vertices(vertices_file)
    original_edges = read_edges(edges_file)
    original_periodic_edges = np.loadtxt(periodic_edges_file, dtype=int)

    # THIS PART IS NOT USED FOR ANISOTROPY SINCE WE DO NOT PERFORM LENGTH OPTIMIZATION
    if anisotropy ==False:
        L = L_original / domain_dim
        s2 = (L_original ** 2) * v / (domain_dim ** 2)
        sigma2 = np.log(1 + s2 / L ** 2)
        sigma = np.sqrt(sigma2)
        mu = np.log(L) - sigma2 / 2
        target_dist = lognorm(s=sigma, scale=np.exp(mu))
    
    # Use computational_domain for boundary/internal detection
    vertex_array = np.array([v for _, v in sorted(original_vertices.items())])
    if not anisotropy:
        internal_vertices = {i for i, v in enumerate(vertex_array) if np.all(np.abs(v) < 0.5)}
        boundary_vertices = {i for i, v in enumerate(vertex_array) if np.any(np.abs(v) == 0.5)}
    else:
        mins = np.array([b[0] for b in lambdas["bounds_after"]])
        maxs = np.array([b[1] for b in lambdas["bounds_after"]])
        tol = 1e-8
        internal_vertices = {i for i, v in enumerate(vertex_array) if np.all((v > mins + tol) & (v < maxs - tol))}
        boundary_vertices = {i for i, v in enumerate(vertex_array) if np.any(np.isclose(v, mins, atol=tol) | np.isclose(v, maxs, atol=tol))}
        
    updated_edges, updated_valencies, updated_periodic_edges, updated_vertices_positions, _, success = optimize_valency(
        edges=original_edges, periodic_edges=original_periodic_edges, num_vertices=len(vertex_array),
        internal_vertices=internal_vertices, boundary_vertices=boundary_vertices,
        vertices_position=vertex_array, target_avg_valency=target_avg_valency, min_valency=2
    )
    if not success:
        return None, None, None
    np.savetxt(vertices_file, updated_vertices_positions)
    np.savetxt(edges_file, updated_edges, fmt="%d")
    np.savetxt(periodic_edges_file, updated_periodic_edges, fmt="%d")
    
    # Print periodic pair counts after step2
    counts = {0: 0, 1: 0, 2: 0}
    for i, j in updated_periodic_edges:
        diff = np.abs(updated_vertices_positions[i] - updated_vertices_positions[j])
        axis = np.argmax(diff)
        counts[axis] += 1
    total_pairs = sum(counts.values())
    print(f"[step2] Periodic pairs: x={counts[0]}, y={counts[1]}, z={counts[2]}, total={total_pairs}")
    # Length optimization only if not anisotropy
    if anisotropy:
        optimized_vertex_array = updated_vertices_positions
        optimized_edges = updated_edges
    else:
        vertices = read_vertices(vertices_file)
        edges = read_edges(edges_file)
        state = {'vertices': vertices, 'edges': edges}
        optimized_vertices, optimized_edges, _ = optimize_length(
            state=state, target_distribution=target_dist, bounds=(-0.5, 0.5)
        )
        optimized_vertex_array = np.array([v for _, v in sorted(optimized_vertices.items())])
        np.savetxt(vertices_file, optimized_vertex_array)
        np.savetxt(edges_file, optimized_edges, fmt="%d")
    return optimized_vertex_array, optimized_edges, updated_periodic_edges

def step3(optimized_vertex_array, optimized_edges, PeriodicEdges, output_directory, computational_domain, anisotropy=False, lambdas=None):
    """
    Step 3: Refine the network by filtering connectivity and removing dangling ends.

    This function:
    - Uses domain bounds (isotropic or anisotropic) for boundary detection.
    - Extracts the largest connected component if the network is not fully connected.
    - Removes internal dangling nodes, remaps indices, and updates periodic pairs.

    Inputs:
        optimized_vertex_array (np.ndarray): Optimized vertex positions.
        optimized_edges (np.ndarray): Optimized edge list.
        PeriodicEdges (np.ndarray): Periodic edge pairs.
        output_directory (str): Directory for output files.
        computational_domain (list): Domain bounds for boundary detection.
        anisotropy (bool): Whether anisotropy is enabled.
        lambdas (dict or None): Anisotropy transformation info (if applicable).

    Outputs:
        optimized_vertex_array (np.ndarray): Refined vertex positions.
        optimized_edges (np.ndarray): Refined edge list.
        PeriodicEdges (np.ndarray): Refined periodic edge pairs.
    """
    import networkx as nx
    vertices_file = os.path.join(output_directory, "vertices.txt")
    edges_file = os.path.join(output_directory, "edges.txt")
    periodic_edges_file = os.path.join(output_directory, "periodic_edges.txt")
    G = nx.Graph()
    for idx, coords in enumerate(optimized_vertex_array):
        G.add_node(idx, coords=coords)
    for edge in optimized_edges:
        G.add_edge(edge[0], edge[1])
    if not nx.is_connected(G):
        largest_cc = sorted(max(nx.connected_components(G), key=len))
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_cc)}
        filtered_vertices = np.array([optimized_vertex_array[i] for i in largest_cc])
        filtered_edges = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in optimized_edges if e[0] in largest_cc and e[1] in largest_cc])
        filtered_periodic_edges = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in PeriodicEdges if e[0] in largest_cc and e[1] in largest_cc])
        optimized_vertex_array, optimized_edges, PeriodicEdges = filtered_vertices, filtered_edges, filtered_periodic_edges
        np.savetxt(periodic_edges_file, PeriodicEdges, fmt="%d")
    np.savetxt(vertices_file, optimized_vertex_array)
    np.savetxt(edges_file, optimized_edges, fmt='%d')
    # Use computational_domain for boundary detection
    #mins = np.array([b[0] for b in computational_domain])
    #maxs = np.array([b[1] for b in computational_domain])
    if not anisotropy:
        mins = np.array([b[0] for b in computational_domain])
        maxs = np.array([b[1] for b in computational_domain])
    else:
        mins = np.array([b[0] for b in lambdas["bounds_after"]])
        maxs = np.array([b[1] for b in lambdas["bounds_after"]])
    BOUNDARY_THRESHOLD = 0.05
    boundary_nodes = set()
    for i, v in enumerate(optimized_vertex_array):
        # A node is on the boundary if any coordinate is within BOUNDARY_THRESHOLD of the min or max for that axis
        if np.any(np.isclose(v, mins, atol=BOUNDARY_THRESHOLD) | np.isclose(v, maxs, atol=BOUNDARY_THRESHOLD)):
            boundary_nodes.add(i)
    connectivity_count = np.zeros(len(optimized_vertex_array), dtype=int)
    for edge in optimized_edges:
        connectivity_count[edge[0]] += 1
        connectivity_count[edge[1]] += 1
    internal_dangling_nodes = [i for i in range(len(optimized_vertex_array)) if connectivity_count[i] == 1 and i not in boundary_nodes]
    internal_dangling_set = set(internal_dangling_nodes)
    if internal_dangling_nodes:
        filtered_edges = [e for e in optimized_edges if e[0] not in internal_dangling_set and e[1] not in internal_dangling_set]
        keep_nodes = [i for i in range(len(optimized_vertex_array)) if i not in internal_dangling_set]
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}
        filtered_vertices = np.array([optimized_vertex_array[i] for i in keep_nodes])
        remapped_edges = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in filtered_edges])
        filtered_periodic_edges = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in PeriodicEdges if e[0] in keep_nodes and e[1] in keep_nodes])
        optimized_vertex_array, optimized_edges, PeriodicEdges = filtered_vertices, remapped_edges, filtered_periodic_edges
        np.savetxt(vertices_file, optimized_vertex_array)
        np.savetxt(edges_file, optimized_edges, fmt='%d')
        np.savetxt(periodic_edges_file, PeriodicEdges, fmt='%d')
    # Print periodic pair counts after step3
    counts = {0: 0, 1: 0, 2: 0}
    for i, j in PeriodicEdges:
        diff = np.abs(optimized_vertex_array[i] - optimized_vertex_array[j])
        axis = np.argmax(diff)
        counts[axis] += 1
    total_pairs = sum(counts.values())
    print(f"[step3] Periodic pairs: x={counts[0]}, y={counts[1]}, z={counts[2]}, total={total_pairs}")
    return optimized_vertex_array, optimized_edges, PeriodicEdges

def step4(refined_vertex_array, refined_edges, refined_periodic_edges, output_directory,
          domain_physical_dimension, young_modulus, element_type, fiber_radius, poisson_ratio,
          seed_count, target_avg_valency, L_original, connector_options,
          translational_damping_coefficient, rotational_damping_coefficient, perform_mesh_refinement,
          computational_domain=None, anisotropy=False, lambdas=None):
    """
    Step 4: Generate Abaqus input files and compute network properties for the refined network.

    This function:
    - Generates node, element, and periodic boundary files.
    - Computes volume fraction, concentration, slenderness compensation, and transverse shear.
    - Handles mesh refinement and connector options if specified.

    Inputs:
        refined_vertex_array (np.ndarray): Refined vertex positions.
        refined_edges (np.ndarray): Refined edge list.
        refined_periodic_edges (np.ndarray): Refined periodic edge pairs.
        output_directory (str): Directory for output files.
        domain_physical_dimension (float or list): Physical domain size (scalar or vector).
        young_modulus (float): Young's modulus for simulation.
        element_type (int): Element type for Abaqus input.
        fiber_radius (float): Fiber radius for property computation.
        poisson_ratio (float): Poisson's ratio for simulation.
        seed_count (int): Number of seeds in the network.
        target_avg_valency (float): Target average valency.
        L_original (float): Original target edge length.
        connector_options (dict): Options for extra connectors.
        translational_damping_coefficient (float): Damping coefficient for translation.
        rotational_damping_coefficient (float): Damping coefficient for rotation.
        perform_mesh_refinement (bool): Whether to refine mesh.
        computational_domain (list): Domain bounds for scaling.
        anisotropy (bool): Whether anisotropy is enabled.
        lambdas (dict or None): Anisotropy transformation info (if applicable).

    Outputs:
        elements.inp
        nodes.inp
        periodic_x.inp
        periodic_y.inp
        periodic_z.inp
    """
    # Determine scaling for each axis
    if not anisotropy:
        mins = np.array([b[0] for b in computational_domain])
        maxs = np.array([b[1] for b in computational_domain])
        comp_lengths = maxs - mins
    else:
        mins = np.array([b[0] for b in lambdas["bounds_after"]])
        maxs = np.array([b[1] for b in lambdas["bounds_after"]])
        comp_lengths = maxs - mins

    # domain_physical_dimension: scalar or vector
    if anisotropy and hasattr(domain_physical_dimension, '__len__') and len(domain_physical_dimension) == 3 and lambdas is not None:
        # Anisotropic scaling: direction-dependent, account for domain stretch
        # Compute the stretch ratio for each axis
        orig_lengths = np.array([b[1] - b[0] for b in computational_domain])
        new_lengths = np.array([b[1] - b[0] for b in lambdas["bounds_after"]])
        stretch = new_lengths / orig_lengths
        # The final physical domain is the original physical size times the stretch
        phys_lengths = np.array(domain_physical_dimension) * stretch
        # Scale nodes from computational to physical domain
        # Map: [-new_length/2, new_length/2] --> [-phys_length/2, phys_length/2] for each axis
        nodes = np.empty_like(refined_vertex_array)
        for d in range(3):
            nodes[:, d] = refined_vertex_array[:, d] * (phys_lengths[d] / new_lengths[d])
    else:
        # Isotropic scaling
        scale = domain_physical_dimension  # e.g., 40 um
        nodes = refined_vertex_array * scale
        phys_lengths = np.array([domain_physical_dimension] * 3)
    
    XCoords = nodes[:, 0]
    YCoords = nodes[:, 1]
    ZCoords = nodes[:, 2]
    element_nodes1 = [edge[0] for edge in refined_edges]
    element_nodes2 = [edge[1] for edge in refined_edges]
    edges = [(a + 1, b + 1) for a, b in zip(element_nodes1, element_nodes2)]
    elements = list(edges)
    NodesNumber = len(XCoords)
    elements_number = len(elements)
    temp_nodes_file = os.path.join(output_directory, "temp_nodes.inp")
    nodes_file = os.path.join(output_directory, "nodes.inp")
    elements_file = os.path.join(output_directory, "elements.inp")
    with open(temp_nodes_file, "w") as file:
        for i in range(NodesNumber):
            file.write(f"{i + 1}, {XCoords[i]:.9f}, {YCoords[i]:.9f}, {ZCoords[i]:.9f}\n")
    remove_empty_lines(temp_nodes_file)
    with open(temp_nodes_file, 'r') as f:
        NodesLines = f.readlines()
    offset_nodes = []
    midpoint_nodes = []
    offset_node_id = []
    midpoints_node_id = []
    offset_distance = 1
    current_node_id = int(NodesLines[-1].split(',')[0])
    for i, (node1, node2) in enumerate(edges):
        node1_coords = [float(x) for x in NodesLines[node1 - 1].strip().split(',')[1:]]
        node2_coords = [float(x) for x in NodesLines[node2 - 1].strip().split(',')[1:]]
        offset_coords = element_orientation_definition(node1_coords, node2_coords, offset_distance)
        current_node_id += 1
        offset_node_id.append(current_node_id)
        offset_nodes.append(f"{current_node_id}, {offset_coords[0]:.9f}, {offset_coords[1]:.9f}, {offset_coords[2]:.9f}\n")
        if element_type == 2:
            midpoint_coords = [(a + b) / 2 for a, b in zip(node1_coords, node2_coords)]
            current_node_id += 1
            midpoints_node_id.append(current_node_id)
            midpoint_nodes.append(f"{current_node_id}, {midpoint_coords[0]:.9f}, {midpoint_coords[1]:.9f}, {midpoint_coords[2]:.9f}\n")
    with open(nodes_file, 'w') as f:
        f.writelines(NodesLines)
        f.writelines(offset_nodes)
        if element_type == 2:
            f.writelines(midpoint_nodes)
    final_element_lines = []
    for i, (node1, node2) in enumerate(edges):
        if element_type == 1:
            final_element_lines.append(f"{i+1}, {node1}, {node2}, {offset_node_id[i]}\n")
        elif element_type == 2:
            final_element_lines.append(f"{i+1}, {node1}, {midpoints_node_id[i]}, {node2}, {offset_node_id[i]}\n")
    with open(elements_file, 'w') as f:
        f.writelines(final_element_lines)
    remove_empty_lines(nodes_file)
    remove_empty_lines(elements_file)
    sort_nodes_file_by_index(nodes_file)
    elements_0based = [(edge[0] - 1, edge[1] - 1) for edge in elements]
    total_length = sum(edges_length(nodes, elements_0based))
    mean_length = np.mean(edges_length(nodes, elements_0based))
    # Use phys_lengths (vector or scalar) for phi to support anisotropic domains
    phi = compute_volume_fraction(total_length, fiber_radius, phys_lengths)
    concentration = phi * 1000 / 0.73
    slenderness_compensation, transverse_shear = compute_transverse_shear(fiber_radius, mean_length, young_modulus, poisson_ratio, element_type)
    # For boundary_limits, use possibly stretched physical domain
    half_phys = phys_lengths / 2
    boundary_limits = {"x": (-half_phys[0], half_phys[0]),
                       "y": (-half_phys[1], half_phys[1]),
                       "z": (-half_phys[2], half_phys[2])}
    with open(nodes_file, 'r') as file:
        nodes_dict = [line.split(",") for line in file]
        nodes_dict = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes_dict if node}
    periodic_edges_file = os.path.join(output_directory, "periodic_edges.txt")
    with open(periodic_edges_file, "r") as file:
        PeriodicEdgesList = [(int(line.split()[0]) + 1, int(line.split()[1]) + 1) for line in file]
    periodic_x, periodic_y, periodic_z = [], [], []
    if anisotropy:
        tol = 1e-8
        for i, j in refined_periodic_edges:
            node1 = nodes[i]
            node2 = nodes[j]
            if np.isclose(node1[0], -half_phys[0], atol=tol) or np.isclose(node2[0], half_phys[0], atol=tol):
                periodic_x.append(order_nodes(nodes_dict, i+1, j+1, 0))  # if nodes_dict is 1-based
            elif np.isclose(node1[1], -half_phys[1], atol=tol) or np.isclose(node2[1], half_phys[1], atol=tol):
                periodic_y.append(order_nodes(nodes_dict, i+1, j+1, 1))
            elif np.isclose(node1[2], -half_phys[2], atol=tol) or np.isclose(node2[2], half_phys[2], atol=tol):
                periodic_z.append(order_nodes(nodes_dict, i+1, j+1, 2))
    else:
        for edge in PeriodicEdgesList:
            node1, node2 = nodes_dict[edge[0]], nodes_dict[edge[1]]
            if node1[0] in boundary_limits["x"] or node2[0] in boundary_limits["x"]:
                periodic_x.append(order_nodes(nodes_dict, edge[0], edge[1], 0))
            elif node1[1] in boundary_limits["y"] or node2[1] in boundary_limits["y"]:
                periodic_y.append(order_nodes(nodes_dict, edge[0], edge[1], 1))
            elif node1[2] in boundary_limits["z"] or node2[2] in boundary_limits["z"]:
                periodic_z.append(order_nodes(nodes_dict, edge[0], edge[1], 2))
    periodic_x_elements_file = os.path.join(output_directory, "periodic_x.inp")
    periodic_y_elements_file = os.path.join(output_directory, "periodic_y.inp")
    periodic_z_elements_file = os.path.join(output_directory, "periodic_z.inp")
    # Debug print before writing files
    #print(f"[step4] periodic_x: {len(periodic_x)} {periodic_x}")
    #print(f"[step4] periodic_y: {len(periodic_y)} {periodic_y}")
    #print(f"[step4] periodic_z: {len(periodic_z)} {periodic_z}")
    for path, arr in zip([periodic_x_elements_file, periodic_y_elements_file, periodic_z_elements_file], [periodic_x, periodic_y, periodic_z]):
        with open(path, "w") as file:
            for node1, node2 in arr:
                file.write(f"{node1} {node2}\n")
        remove_empty_lines(path)
    # Print periodic pair counts after step4
    counts = {0: 0, 1: 0, 2: 0}
    for i, j in refined_periodic_edges:
        diff = np.abs(refined_vertex_array[i] - refined_vertex_array[j])
        axis = np.argmax(diff)
        counts[axis] += 1
    total_pairs = sum(counts.values())
    print(f"[step4] Periodic pairs: x={counts[0]}, y={counts[1]}, z={counts[2]}, total={total_pairs}")
    write_abaqus_input_files(output_directory, phys_lengths, phi, concentration, seed_count, target_avg_valency, fiber_radius, L_original,
                            young_modulus, poisson_ratio, element_type, elements_number, nodes, periodic_x, periodic_y, periodic_z,
                            periodic_x_elements_file, periodic_y_elements_file, periodic_z_elements_file, connector_options, None,
                            None, translational_damping_coefficient, rotational_damping_coefficient)

def main():
    # --- User mode selection ---
    use_lhs_sampling = False  # Set to False for single sample mode, True for LHS parametric study
    
    # --- User parameters ---
    anisotropy = False  
    if anisotropy:
        anisotropy_axis = (1, 0, 0)
        P2_target = 0.6
        DOMAIN_PHYSICAL_DIMENSION = [40, 40, 40]  # will be updated after step1
    else:
        anisotropy_axis = None
        P2_target = None
        DOMAIN_PHYSICAL_DIMENSION = 40  # Cube

    computational_domain = [(-0.5, 0.5)] * 3
    V = 0.3
    ELEMENT_TYPE = 2
    POISSON_RATIO = 0.495
    YOUNG_MODULUS = 100
    TRANSLATIONAL_DAMPING_COEFFICIENT = 0.02
    ROTATIONAL_DAMPING_COEFFICIENT = 0.1
    connector_options = {"rotational_damper": False, "translational_damper": False, "translational_and_rotational_damper": False}
    perform_mesh_refinement = False
    
    # Fixed parameters set in single sample mode
    #fixed_seed_count = 300
    fixed_seed_count = 450
    #fixed_target_avg_valency = 3.5
    fixed_target_avg_valency = 3.75
    fixed_fiber_radius = 0.1
    fixed_L_original = 2.5
    
    # Slenderness constraint for LHS sampling
    slenderness_min = 1.0e-4
    slenderness_max = 1.0e-3
    
    main_output_dir = r"D:\hypertopogen\topoGEN\output"
    
    # Create logs folder
    logs_dir = os.path.join(main_output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"[main] Logs directory created at: {logs_dir}")

    # --- Single sample mode or LHS parametric study ---
    if use_lhs_sampling:
        # LHS parametric study mode
        param_bounds = {
            "seed_count": (100, 500),
            "target_avg_valency": (3.1, 3.9),
            "fiber_radius": (0.05, 0.15),
            "L_original": (1.5, 3.5)
        }
        param_names = list(param_bounds.keys())
        bounds = np.array([param_bounds[name] for name in param_names])
        n_samples = 300
        n_levels = 30

        idx_seed = param_names.index("seed_count")
        idx_valency = param_names.index("target_avg_valency")
        idx_radius = param_names.index("fiber_radius")
        idx_length = param_names.index("L_original")

        sampler = qmc.LatinHypercube(d=len(param_names))
        samples = []
        seen = set()
        while len(samples) < n_samples:
            batch_size = max(n_samples - len(samples), n_samples)
            raw = sampler.random(n=batch_size)
            scaled = qmc.scale(raw, bounds[:, 0], bounds[:, 1])
            for i in range(len(param_names)):
                levels = np.linspace(bounds[i, 0], bounds[i, 1], n_levels)
                idx = np.floor((scaled[:, i] - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0]) * n_levels).astype(int)
                idx = np.clip(idx, 0, n_levels - 1)
                scaled[:, i] = levels[idx]

            fiber = scaled[:, idx_radius]
            length = scaled[:, idx_length]
            slenderness = (fiber ** 2) / (4.0 * (length ** 2))
            mask = (slenderness >= slenderness_min) & (slenderness <= slenderness_max)
            for row, slender in zip(scaled[mask], slenderness[mask]):
                seed_val = int(float(row[idx_seed]))
                valency_val = float(row[idx_valency])
                radius_val = float(row[idx_radius])
                length_val = float(row[idx_length])
                key = (seed_val, valency_val, radius_val, length_val)
                if key in seen:
                    continue
                seen.add(key)
                samples.append([seed_val, valency_val, radius_val, length_val, float(slender)])
                if len(samples) >= n_samples:
                    break

        samples_to_process = np.array(samples[:n_samples])
        csv_path = os.path.join(main_output_dir, "lhs_samples.csv")
        header = "seed_count,target_avg_valency,fiber_radius,L_original,slenderness"
        np.savetxt(csv_path, samples_to_process, delimiter=",", header=header, comments='')
        print(f"[main] LHS Mode: {n_samples} samples requested; {len(samples_to_process)} samples loaded; shape={samples_to_process.shape}")
    else:
        fixed_slenderness = (fixed_fiber_radius ** 2) / (4.0 * (fixed_L_original ** 2))
        samples_to_process = np.array([[
            fixed_seed_count,
            fixed_target_avg_valency,
            fixed_fiber_radius,
            fixed_L_original,
            fixed_slenderness
        ]])
        print(
            f"[main] Single Sample Mode: seed_count={fixed_seed_count}, target_avg_valency={fixed_target_avg_valency}, "
            f"fiber_radius={fixed_fiber_radius}, L_original={fixed_L_original}, slenderness={fixed_slenderness:.3e}"
        )
    output_base_dir = main_output_dir
    for idx, sample in enumerate(samples_to_process):
        print(f"[main] Starting sample {idx + 1}/{len(samples_to_process)}")
        seed_count = int(float(sample[0]))
        target_avg_valency = float(sample[1])
        fiber_radius = float(sample[2])
        L_original = float(sample[3])
        young_modulus = YOUNG_MODULUS
        # Prepare protocol config for step1
        protocol_config = {
            'output_directory': setup_output_directory(f"Sample_{idx}", base_dir=output_base_dir),
            'anisotropy': anisotropy,
            'computational_domain': computational_domain,
            'anisotropy_axis': anisotropy_axis,
            'P2_target': P2_target
        }
        max_attempts = 10
        attempt = 0
        success = False
        while attempt < max_attempts and not success:
            attempt += 1
            print(f"[main] Sample {idx + 1}: attempt {attempt}/{max_attempts}")
            step1_sample = (seed_count, target_avg_valency, L_original, young_modulus)
            vertices_file, edges_file, periodic_edges_file, output_directory = step1(idx, step1_sample, protocol_config)
            lambdas = None
            if anisotropy:
                lambdas_path = os.path.join(output_directory, 'lambdas.npy')
                if os.path.exists(lambdas_path):
                    lambdas = np.load(lambdas_path, allow_pickle=True).item()
                    bounds_after = lambdas.get("bounds_after")
                    if bounds_after is not None:
                        updated_domain_computational_dimension = [b[1] - b[0] for b in bounds_after]
            optimized_vertex_array, optimized_edges, PeriodicEdges = step2(
                vertices_file, edges_file, periodic_edges_file, output_directory,
                DOMAIN_PHYSICAL_DIMENSION, target_avg_valency, L_original, V,
                lambdas if anisotropy else None, anisotropy
            )
            if optimized_vertex_array is not None and optimized_edges is not None and PeriodicEdges is not None:
                success = True
        if not success:
            print(f"[main] Sample {idx + 1}: failed after {max_attempts} attempts; skipping")
            continue

        refined_vertex_array, refined_edges, refined_periodic_edges = step3(
            optimized_vertex_array, optimized_edges, PeriodicEdges, output_directory, computational_domain, anisotropy, lambdas if anisotropy else None
        )
        #plot_edge_orientation_and_network(refined_vertex_array, refined_edges, bins=30, output_directory=output_directory, tolerance_deg=15)
        
        step4(
            refined_vertex_array, refined_edges, refined_periodic_edges, output_directory, DOMAIN_PHYSICAL_DIMENSION, young_modulus, ELEMENT_TYPE,
            fiber_radius, POISSON_RATIO, seed_count, target_avg_valency, L_original, connector_options, TRANSLATIONAL_DAMPING_COEFFICIENT,
            ROTATIONAL_DAMPING_COEFFICIENT, perform_mesh_refinement, computational_domain, anisotropy, lambdas if anisotropy else None
        )
        print(f"[main] Sample {idx + 1}: completed")


if __name__ == "__main__":
    main()