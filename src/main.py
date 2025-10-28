
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
from utils.setup import setup_output_directory

def step1_network_generation(idx, sample, output_root):
    """
    Step 1: Sampling & Periodic Voronoi/Delaunay tessellation for a single sample.
    Returns output file paths for use in following steps.
    """
    import os
    import numpy as np
    from scipy.stats import qmc
    from utils.setup import setup_output_directory
    from create_periodic_network import (
        tile_points, lloyd_relaxation_3d_periodic, get_vertices, get_edges, process_edges,
        merge_close_vertices, replica_removal, find_periodic_pairs, calculate_mean_length
    )

    # ---------------------- Extract Sample Parameters ----------------------
    seed_count, target_avg_valency, L_original, young_modulus = sample
    N = int(seed_count)
    DOMAIN_PHYSICAL_DIMENSION = 40
    HALF_DOMAIN_PHYSICAL_DIMENSION = 0.5
    NETWORK_TYPE = "voronoi"    # Options: "voronoi" or "delaunay"
    MERGE_VERTICES = False      # Set to True to enable vertex merging
    MERGE_THRESHOLD = 0.1
    MAX_DEGREE = 50

    # ---------------------- Job Directory ----------------------
    job_description = f"Sample_{idx}"
    output_directory = setup_output_directory(job_description)
    vertices_file_path = os.path.join(output_directory, "vertices.txt")
    edges_file_path = os.path.join(output_directory, "edges.txt")
    periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")

    # ---------------------- Network Generation ----------------------
    bounds = [(-HALF_DOMAIN_PHYSICAL_DIMENSION, HALF_DOMAIN_PHYSICAL_DIMENSION),
              (-HALF_DOMAIN_PHYSICAL_DIMENSION, HALF_DOMAIN_PHYSICAL_DIMENSION),
              (-HALF_DOMAIN_PHYSICAL_DIMENSION, HALF_DOMAIN_PHYSICAL_DIMENSION)]
    original_points = np.random.uniform(
        -HALF_DOMAIN_PHYSICAL_DIMENSION, HALF_DOMAIN_PHYSICAL_DIMENSION, (N, 3)
    )

    if NETWORK_TYPE == "voronoi":
        print("Generating a 3D Voronoi network...")
        point_tile = tile_points(original_points, N)
        points, vor = lloyd_relaxation_3d_periodic(original_points, 10, N)
        tile_vertices = vor.vertices
        vertices, IndexMap = get_vertices(tile_vertices)
        tile_edges = process_edges(vor.ridge_vertices, tile_vertices)
        Vertices, Edges = get_edges(vertices, tile_vertices, tile_edges, IndexMap, bounds)
        unique_edges = replica_removal(Edges)
    elif NETWORK_TYPE == "delaunay":
        print("Generating a 3D Delaunay network...")
        from scipy.spatial import Delaunay
        delaunay = Delaunay(original_points)
        Vertices = original_points
        unique_edges = set()
        for simplex in delaunay.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    unique_edges.add(edge)
        unique_edges = np.array(list(unique_edges))

    # Optional: Merge close vertices if enabled
    if MERGE_VERTICES:
        FinalVertices, FinalEdges = merge_close_vertices(
            Vertices, unique_edges, bounds, MERGE_THRESHOLD, MAX_DEGREE
        )
    else:
        print("No merging is performed.")
        FinalVertices = Vertices
        FinalEdges = unique_edges

    # Define periodic pairs
    PeriodicEdges = find_periodic_pairs(FinalVertices, bounds)
    comp_length = calculate_mean_length(FinalVertices, FinalEdges)
    real_length = DOMAIN_PHYSICAL_DIMENSION * comp_length
    print(f"The mean segment length for {N} seeds is: {real_length}")

    # Save Voronoi network
    np.savetxt(vertices_file_path, FinalVertices)
    np.savetxt(edges_file_path, FinalEdges, fmt="%d")
    np.savetxt(periodic_edges_file_path, PeriodicEdges, fmt="%d")
    print("Step 1: Periodic Voronoi tessellation completed.")

    # Return output paths for next steps
    return vertices_file_path, edges_file_path, periodic_edges_file_path, output_directory

def step2_network_optimization(vertices_file_path, edges_file_path, periodic_edges_file_path, output_directory, domain_physical_dimension, target_avg_valency, L_original, v):
    """
    Step 2: Valency & Length Optimization for the generated network.
    Returns optimized network arrays for use in following steps.
    """
    import os
    import time
    import numpy as np
    from scipy.stats import lognorm
    from optimize_periodic_network import (
        optimize_valency, calculate_valencies, simulated_annealing_without_valency,
        read_edges, read_vertices, compute_edge_lengths
    )
    from analysis.plots.periodic_optimized_voronoi_plot import (
        plot_intermediate_edge_length_distribution,
        plot_final_edge_length_distributions
    )

    # 1. Load original input data
    original_vertices = read_vertices(vertices_file_path)
    original_edges = read_edges(edges_file_path)
    original_periodic_edges = np.loadtxt(periodic_edges_file_path, dtype=int)

    initial_lengths = compute_edge_lengths(original_vertices, original_edges)

    preserved_initial_vertices = {k: v.copy() for k, v in original_vertices.items()}
    preserved_initial_edges = original_edges.copy()

    # 2. Build target edge length distribution
    L = L_original / domain_physical_dimension
    s_squared_original = (L_original ** 2) * v
    s_squared = s_squared_original / (domain_physical_dimension ** 2)
    sigma_squared = np.log(1 + s_squared / L ** 2)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(L) - sigma_squared / 2
    target_distribution = lognorm(s=sigma, scale=np.exp(mu))

    # 3. Valency Optimization
    print("Running valency optimization first...")
    start_time = time.perf_counter()

    vertex_array = np.array([v for _, v in sorted(original_vertices.items())])
    internal_vertices = {i for i, v in enumerate(vertex_array) if np.all(np.abs(v) < 0.5)}
    boundary_vertices = {i for i, v in enumerate(vertex_array) if np.any(np.abs(v) == 0.5)}

    updated_edges, updated_valencies, updated_periodic_edges, updated_vertices_positions, energy_log_val, success = optimize_valency(
        edges=original_edges,
        periodic_edges=original_periodic_edges,
        num_vertices=len(vertex_array),
        internal_vertices=internal_vertices,
        boundary_vertices=boundary_vertices,
        vertices_position=vertex_array,
        target_avg_valency=target_avg_valency,
        min_valency=3
    )

    if not success:
        print("Valency optimization failed to reach target. Skipping this configuration.")
        # Return None or raise, depending on how you want to handle failed samples
        return None, None, None

    avg_internal_valency = np.mean([updated_valencies[i] for i in internal_vertices if i < len(updated_valencies)])
    print(f"Average internal valency after optimization: {avg_internal_valency:.4f}")
    print("Valency optimization completed.")

    # Save intermediate lengths after valency optimization
    intermediate_vertices_dict = {i: pos for i, pos in enumerate(updated_vertices_positions)}
    intermediate_lengths = compute_edge_lengths(intermediate_vertices_dict, updated_edges)

    # Overwrite files for next step
    np.savetxt(vertices_file_path, updated_vertices_positions)
    np.savetxt(edges_file_path, updated_edges, fmt="%d")
    np.savetxt(periodic_edges_file_path, updated_periodic_edges, fmt="%d")

    # 4. Length Optimization (Simulated Annealing)
    vertices = read_vertices(vertices_file_path)
    edges = read_edges(edges_file_path)
    state = {'vertices': vertices, 'edges': edges}

    print("Proceeding to length optimization...")

    optimized_vertices, optimized_edges, energy_log_length = simulated_annealing_without_valency(
        state=state,
        target_distribution=target_distribution,
        bounds=(-0.5, 0.5)
    )

    end_time = time.perf_counter()
    print(f"Two-step optimization completed in {end_time - start_time:.2f} seconds.")

    # Shift length energy iteration numbers
    max_valency_iter = energy_log_val[-1]['iteration'] if energy_log_val else 0
    for entry in energy_log_length:
        entry['iteration'] += max_valency_iter
    for entry in energy_log_val:
        entry['type'] = 'valency'
    for entry in energy_log_length:
        entry['type'] = 'length'

    combined_log = energy_log_val + energy_log_length

    # Save final edge lengths
    final_lengths = compute_edge_lengths(optimized_vertices, optimized_edges)
    # You may want to save intermediate and final lengths for plotting

    # Valency Calculation & Visualization
    optimized_vertex_array = np.array([v for _, v in sorted(optimized_vertices.items())])
    final_valencies = calculate_valencies(np.array(optimized_edges), len(optimized_vertex_array))

    #plot_intermediate_edge_length_distribution(
    #    preserved_initial_edges, optimized_edges,
    #    np.array([v for v in preserved_initial_vertices.values()]),
    #    optimized_vertex_array,
    #    domain_physical_dimension, output_directory
    #)

    #plot_final_edge_length_distributions(
    #    np.array([v for v in preserved_initial_vertices.values()]), preserved_initial_edges,
    #    optimized_vertices, optimized_edges,
    #    output_directory, target_distribution
    #)

    # Overwrite files for next step
    np.savetxt(vertices_file_path, optimized_vertex_array)
    np.savetxt(edges_file_path, optimized_edges, fmt="%d")

    # Return arrays for next step
    return optimized_vertex_array, optimized_edges, updated_periodic_edges

def step3_network_refinement(optimized_vertex_array, optimized_edges, PeriodicEdges, output_directory):
    """
    Step 3: Connectivity filtering & dangling ends removal for optimized networks.
    Returns further refined network arrays and periodic edges for the next step.
    """
    import os
    import numpy as np
    import networkx as nx

    DOMAIN_PHYSICAL_DIMENSION = 40
    HALF_DOMAIN_PHYSICAL_DIMENSION = 0.5

    vertices_file_path = os.path.join(output_directory, "vertices.txt")
    edges_file_path = os.path.join(output_directory, "edges.txt")
    periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")

    # --- CONNECTIVITY FILTERING ---
    G = nx.Graph()
    for idx, coords in enumerate(optimized_vertex_array):
        G.add_node(idx, coords=coords)
    for edge in optimized_edges:
        G.add_edge(edge[0], edge[1])

    print("Network before filtering:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")

    # Identify largest connected component
    if not nx.is_connected(G):
        print("WARNING: Graph is not fully connected. Extracting largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc = sorted(largest_cc)
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_cc)}
        filtered_vertices = np.array([optimized_vertex_array[i] for i in largest_cc])
        filtered_edges = []
        for edge in optimized_edges:
            if edge[0] in largest_cc and edge[1] in largest_cc:
                filtered_edges.append([idx_map[edge[0]], idx_map[edge[1]]])
        filtered_edges = np.array(filtered_edges)
        optimized_vertex_array = filtered_vertices
        optimized_edges = filtered_edges

        filtered_periodic_edges = []
        for edge in PeriodicEdges:
            if edge[0] in largest_cc and edge[1] in largest_cc:
                filtered_periodic_edges.append([idx_map[edge[0]], idx_map[edge[1]]])
        filtered_periodic_edges = np.array(filtered_periodic_edges)
        PeriodicEdges = filtered_periodic_edges
        np.savetxt(periodic_edges_file_path, PeriodicEdges, fmt="%d")

    np.savetxt(vertices_file_path, optimized_vertex_array)
    np.savetxt(edges_file_path, optimized_edges, fmt='%d')

    # --- DANGLING ENDS REMOVAL ---
    boundary_nodes = set()

    BOUNDARY_THRESHOLD = 0.05

    for i, v in enumerate(optimized_vertex_array):
        # Check if any coordinate is within threshold of boundaries
        near_pos_boundary = np.any(np.abs(v - HALF_DOMAIN_PHYSICAL_DIMENSION) <= BOUNDARY_THRESHOLD)
        near_neg_boundary = np.any(np.abs(v - (-HALF_DOMAIN_PHYSICAL_DIMENSION)) <= BOUNDARY_THRESHOLD)
        
        if near_pos_boundary or near_neg_boundary:
            boundary_nodes.add(i)

    connectivity_count = np.zeros(len(optimized_vertex_array), dtype=int)
    for edge in optimized_edges:
        connectivity_count[edge[0]] += 1
        connectivity_count[edge[1]] += 1

    internal_dangling_nodes = [i for i in range(len(optimized_vertex_array))
                            if connectivity_count[i] == 1 and i not in boundary_nodes]
    internal_dangling_set = set(internal_dangling_nodes)

    print(f"\n--- DANGLING NODE ANALYSIS ---")
    total_dangling = sum(1 for i in range(len(optimized_vertex_array)) if connectivity_count[i] == 1)
    boundary_dangling = sum(1 for i in boundary_nodes if connectivity_count[i] == 1)
    print(f"Total nodes with valency 1: {total_dangling}")
    print(f"Boundary nodes with valency 1: {boundary_dangling}")
    print(f"Internal nodes with valency 1 (to be removed): {len(internal_dangling_nodes)}")
   
    if internal_dangling_nodes and len(internal_dangling_nodes) <= 10:
        print("Internal dangling nodes to be removed:")
        for node_idx in internal_dangling_nodes:
            print(f"  Node {node_idx}: {optimized_vertex_array[node_idx]} (valency: {connectivity_count[node_idx]})")

    if internal_dangling_nodes:
        print(f"Removing {len(internal_dangling_nodes)} internal dangling nodes.")
        filtered_edges = []
        for edge in optimized_edges:
            if edge[0] in internal_dangling_set or edge[1] in internal_dangling_set:
                continue
            filtered_edges.append(edge)
        filtered_edges = np.array(filtered_edges)
        keep_nodes = [i for i in range(len(optimized_vertex_array)) if i not in internal_dangling_set]
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}
        filtered_vertices = np.array([optimized_vertex_array[i] for i in keep_nodes])
        remapped_edges = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in filtered_edges])
        filtered_periodic_edges = []
        for edge in PeriodicEdges:
            if edge[0] in keep_nodes and edge[1] in keep_nodes:
                filtered_periodic_edges.append([idx_map[edge[0]], idx_map[edge[1]]])
        filtered_periodic_edges = np.array(filtered_periodic_edges)
        optimized_vertex_array = filtered_vertices
        optimized_edges = remapped_edges
        PeriodicEdges = filtered_periodic_edges
        np.savetxt(vertices_file_path, optimized_vertex_array)
        np.savetxt(edges_file_path, optimized_edges, fmt='%d')
        np.savetxt(periodic_edges_file_path, PeriodicEdges, fmt='%d')

    print("=== END STEP 3 DEBUGGING ===\n")
    print("Refinement completed.")
    return optimized_vertex_array, optimized_edges, PeriodicEdges

def step4_abaqus_input(refined_vertex_array, refined_edges, refined_periodic_edges, output_directory,
                       domain_physical_dimension, young_modulus, element_type, fiber_radius, poisson_ratio,
                       seed_count, target_avg_valency, L_original, connector_options, 
                       translational_damping_coefficient, rotational_damping_coefficient, perform_mesh_refinement):
    """
    Step 4: Abaqus input file generation for node, element, property computation, periodic boundary, and extra connectors.
    """
    import os
    import numpy as np
    from collections import defaultdict
    from itertools import combinations

    from write_abaqus_input_file import (element_orientation_definition,
                                     midpoint, edges_length, compute_volume_fraction,
                                     compute_transverse_shear, order_nodes, remove_empty_lines,
                                     write_abaqus_input_files, sort_nodes_file_by_index)

    # File paths
    temp_nodes_file_path = os.path.join(output_directory, "temp_nodes.inp")
    nodes_file_path = os.path.join(output_directory, "nodes.inp")
    elements_file_path = os.path.join(output_directory, "elements.inp")
    nodes_with_replica_file_path = os.path.join(output_directory, "nodes_with_replica.inp")
    elements_with_replica_file_path = os.path.join(output_directory, "elements_with_replica.inp")

    # --- Prepare coordinates and edges ---
    if perform_mesh_refinement:
        # Apply mesh refinement based on slenderness ratio
        XCoords = refined_vertex_array[:, 0] * domain_physical_dimension
        YCoords = refined_vertex_array[:, 1] * domain_physical_dimension
        ZCoords = refined_vertex_array[:, 2] * domain_physical_dimension
        nodes = np.vstack([XCoords, YCoords, ZCoords]).T
        
        refined_nodes = nodes.tolist()
        refined_node_map = {tuple(np.round(n, 9)): i + 1 for i, n in enumerate(refined_nodes)}
        final_elements = []
        internal_nodes_counter = 0
        slend = 1 / 12
        max_segment_length = fiber_radius / slend

        for edge in refined_edges:
            idx_a, idx_b = edge
            pos_a = nodes[idx_a]
            pos_b = nodes[idx_b]
            vec_ab = pos_b - pos_a
            length = np.linalg.norm(vec_ab)
            num_segments = max(int(np.ceil(length / max_segment_length)), 1)

            prev_node_idx = refined_node_map[tuple(np.round(pos_a, 9))]
            for i in range(1, num_segments):
                ratio = i / num_segments
                new_node_coords = tuple(np.round(pos_a + ratio * vec_ab, 9))
                if new_node_coords not in refined_node_map:
                    refined_nodes.append(list(new_node_coords))
                    internal_nodes_counter += 1
                    refined_node_map[new_node_coords] = len(refined_nodes)
                curr_node_idx = refined_node_map[new_node_coords]
                final_elements.append((prev_node_idx, curr_node_idx))
                prev_node_idx = curr_node_idx

            final_node_idx = refined_node_map[tuple(np.round(pos_b, 9))]
            final_elements.append((prev_node_idx, final_node_idx))
        
        # Update coordinate arrays with refined nodes
        refined_nodes = np.array(refined_nodes)
        XCoords = refined_nodes[:, 0]
        YCoords = refined_nodes[:, 1]
        ZCoords = refined_nodes[:, 2]
        edges = final_elements
        elements = list(edges)
        
    else:
        # No refinement: use original approach
        XCoords = refined_vertex_array[:, 0] * domain_physical_dimension
        YCoords = refined_vertex_array[:, 1] * domain_physical_dimension
        ZCoords = refined_vertex_array[:, 2] * domain_physical_dimension
        nodes = np.vstack([XCoords, YCoords, ZCoords]).T
        
        element_nodes1 = [edge[0] for edge in refined_edges]
        element_nodes2 = [edge[1] for edge in refined_edges]
        edges = [(a + 1, b + 1) for a, b in zip(element_nodes1, element_nodes2)]  # Convert to 1-based
        elements = list(edges)
        refined_nodes = nodes.tolist()
        internal_nodes_counter = 0

    NodesNumber = len(XCoords)
    elements_number = len(elements)

    # --- BRANCH 1: NO REPLICA, ORIGINAL MESH ---
    with open(temp_nodes_file_path, "w") as file:
        for i in range(NodesNumber):
            file.write(f"{i + 1}, {XCoords[i]:.9f}, {YCoords[i]:.9f}, {ZCoords[i]:.9f}\n")
    remove_empty_lines(temp_nodes_file_path)
    with open(temp_nodes_file_path, 'r') as f:
        NodesLines = f.readlines()

    offset_nodes = []
    midpoint_nodes = []
    offset_node_id = []
    midpoints_node_id = []
    offset_distance = 1
    current_node_id = int(NodesLines[-1].split(',')[0])

    for i, (node1, node2) in enumerate(edges):
        # Ensure consistent 1-based indexing
        if perform_mesh_refinement:
            node1_id = node1  # Already 1-based from refinement
            node2_id = node2
        else:
            node1_id = node1  # Already converted to 1-based above
            node2_id = node2
            
        node1_coords = [float(x) for x in NodesLines[node1_id - 1].strip().split(',')[1:]]
        node2_coords = [float(x) for x in NodesLines[node2_id - 1].strip().split(',')[1:]]
        offset_coords = element_orientation_definition(node1_coords, node2_coords, offset_distance)
        current_node_id += 1
        offset_node_id.append(current_node_id)
        offset_nodes.append(
            f"{current_node_id}, {format(offset_coords[0], '.9')}, {format(offset_coords[1], '.9')}, {format(offset_coords[2], '.9')}\n"
        )
        if element_type == 2:
            midpoint_coords = midpoint(node1_coords, node2_coords)
            current_node_id += 1
            midpoints_node_id.append(current_node_id)
            midpoint_nodes.append(
                f"{current_node_id}, {format(midpoint_coords[0], '.9')}, {format(midpoint_coords[1], '.9')}, {format(midpoint_coords[2], '.9')}\n"
            )

    with open(nodes_file_path, 'w') as f:
        f.writelines(NodesLines)
        f.writelines(offset_nodes)
        f.writelines(midpoint_nodes)

    final_element_lines = []
    for i, (node1, node2) in enumerate(edges):
        if element_type == 1:
            final_element_lines.append(f"{i+1}, {node1}, {node2}, {offset_node_id[i]}\n")
        elif element_type == 2:
            final_element_lines.append(f"{i+1}, {node1}, {midpoints_node_id[i]}, {node2}, {offset_node_id[i]}\n")

    with open(elements_file_path, 'w') as f:
        f.writelines(final_element_lines)
    remove_empty_lines(nodes_file_path)
    remove_empty_lines(elements_file_path)
    sort_nodes_file_by_index(nodes_file_path)

    # --- BRANCH 2: WITH REPLICA (only if rotational damping enabled) ---
    rotational_damping = connector_options.get('rotational_damper', False) or connector_options.get('translational_and_rotational_damper', False)
    
    if rotational_damping:
        with open(elements_file_path, 'r') as f:
            element_lines = f.readlines()
        last_node_id = NodesNumber  # Use refined node count
        node_to_elements = defaultdict(list)
        
        for elem_idx, line in enumerate(element_lines):
            fields = line.strip().split(',')
            node1_id = int(fields[1])
            node2_id = int(fields[2]) if element_type == 1 else int(fields[3])
            node_to_elements[node1_id].append(elem_idx)
            node_to_elements[node2_id].append(elem_idx)

        replica_map = {}
        replica_nodes = []
        current_new_node_id = last_node_id
        joint_nodes_dict = {}
        
        for node_id, elem_indices in node_to_elements.items():
            node_group = []
            coords = NodesLines[node_id - 1].strip().split(',')[1:]
            if len(elem_indices) == 1:
                replica_map[(node_id, elem_indices[0])] = node_id
                node_group.append(node_id)
            else:
                replica_map[(node_id, elem_indices[0])] = node_id
                node_group.append(node_id)
                for elem_idx in elem_indices[1:]:
                    current_new_node_id += 1
                    replica_map[(node_id, elem_idx)] = current_new_node_id
                    replica_nodes.append(f"{current_new_node_id}, {coords[0]}, {coords[1]}, {coords[2]}\n")
                    node_group.append(current_new_node_id)
            joint_nodes_dict[node_id] = node_group

        with open(nodes_with_replica_file_path, 'w') as f:
            f.writelines(NodesLines)
            f.writelines(replica_nodes)

        updated_element_lines = []
        for elem_idx, line in enumerate(element_lines):
            fields = line.strip().split(',')
            node1_id = int(fields[1])
            node2_id = int(fields[2]) if element_type == 1 else int(fields[3])
            new_node1_id = replica_map[(node1_id, elem_idx)]
            new_node2_id = replica_map[(node2_id, elem_idx)]
            updated_element_lines.append((elem_idx+1, new_node1_id, new_node2_id))

        with open(nodes_with_replica_file_path, 'r') as f:
            all_node_lines = f.readlines()

        offset_nodes_replica = []
        midpoint_nodes_replica = []
        offset_node_id_replica = []
        midpoints_node_id_replica = []
        offset_distance = 1
        current_node_id = current_new_node_id

        for i, (elem_id, node1_id, node2_id) in enumerate(updated_element_lines):
            node1_coords = [float(x) for x in all_node_lines[node1_id - 1].strip().split(',')[1:]]
            node2_coords = [float(x) for x in all_node_lines[node2_id - 1].strip().split(',')[1:]]
            offset_coords = element_orientation_definition(node1_coords, node2_coords, offset_distance)
            current_node_id += 1
            offset_node_id_replica.append(current_node_id)
            offset_nodes_replica.append(
                f"{current_node_id}, {format(offset_coords[0], '.9')}, {format(offset_coords[1], '.9')}, {format(offset_coords[2], '.9')}\n"
            )
            if element_type == 2:
                midpoint_coords = midpoint(node1_coords, node2_coords)
                current_node_id += 1
                midpoints_node_id_replica.append(current_node_id)
                midpoint_nodes_replica.append(
                    f"{current_node_id}, {format(midpoint_coords[0], '.9')}, {format(midpoint_coords[1], '.9')}, {format(midpoint_coords[2], '.9')}\n"
                )

        with open(nodes_with_replica_file_path, 'a') as f:
            f.writelines(offset_nodes_replica)
            f.writelines(midpoint_nodes_replica)

        final_element_lines_replica = []
        for i, (elem_id, new_node1_id, new_node2_id) in enumerate(updated_element_lines):
            if element_type == 1:
                final_element_lines_replica.append(f"{elem_id}, {new_node1_id}, {new_node2_id}, {offset_node_id_replica[i]}\n")
            elif element_type == 2:
                final_element_lines_replica.append(f"{elem_id}, {new_node1_id}, {midpoints_node_id_replica[i]}, {new_node2_id}, {offset_node_id_replica[i]}\n")
        
        with open(elements_with_replica_file_path, 'w') as f:
            f.writelines(final_element_lines_replica)

        remove_empty_lines(nodes_with_replica_file_path)
        remove_empty_lines(elements_with_replica_file_path)
        sort_nodes_file_by_index(nodes_with_replica_file_path)

    sort_nodes_file_by_index(nodes_file_path)

    # --- PROPERTY COMPUTATION ---
    # Use the correct nodes array for property computation
    if perform_mesh_refinement:
        # Use refined nodes for property computation
        property_nodes = refined_nodes
    else:
        # Use original nodes for property computation
        property_nodes = nodes
    
    # Convert elements back to 0-based indexing for edges_length function
    elements_0based = [(edge[0] - 1, edge[1] - 1) for edge in elements]
    
    total_length = sum(edges_length(property_nodes, elements_0based))
    mean_length = np.mean(edges_length(property_nodes, elements_0based))
    phi = compute_volume_fraction(total_length, fiber_radius, domain_physical_dimension)
    print(f"Volume Fraction: {phi}")
    concentration = phi * 1000 / 0.73
    print(f"concentration: {concentration}")

    slenderness_compensation, transverse_shear = compute_transverse_shear(fiber_radius, mean_length, young_modulus, poisson_ratio, element_type)
    print("Slenderness Compensation Ratio:", slenderness_compensation)
    print("Step 5: Property computation completed.")

    # --- PERIODIC BOUNDARY CONDITIONS ---
    half_domain_physical_dimension = domain_physical_dimension / 2
    boundary_limits = {
        "x": (-half_domain_physical_dimension, half_domain_physical_dimension),
        "y": (-half_domain_physical_dimension, half_domain_physical_dimension),
        "z": (-half_domain_physical_dimension, half_domain_physical_dimension)
    }

    with open(nodes_file_path, 'r') as file:
        nodes_dict = [line.split(",") for line in file]
        nodes_dict = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes_dict if node}

    periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")
    with open(periodic_edges_file_path, "r") as file:
        PeriodicEdgesList = [(int(line.split()[0]) + 1, int(line.split()[1]) + 1) for line in file]

    periodic_x, periodic_y, periodic_z = [], [], []
    for edge in PeriodicEdgesList:
        node1, node2 = nodes_dict[edge[0]], nodes_dict[edge[1]]
        if node1[0] in boundary_limits["x"] or node2[0] in boundary_limits["x"]:
            ordered_edges = order_nodes(nodes_dict, edge[0], edge[1], 0)
            periodic_x.append(ordered_edges)
        elif node1[1] in boundary_limits["y"] or node2[1] in boundary_limits["y"]:
            ordered_edges = order_nodes(nodes_dict, edge[0], edge[1], 1)
            periodic_y.append(ordered_edges)
        elif node1[2] in boundary_limits["z"] or node2[2] in boundary_limits["z"]:
            ordered_edges = order_nodes(nodes_dict, edge[0], edge[1], 2)
            periodic_z.append(ordered_edges)

    periodic_x_elements_file_path = os.path.join(output_directory, "periodic_x.inp")
    periodic_y_elements_file_path = os.path.join(output_directory, "periodic_y.inp")
    periodic_z_elements_file_path = os.path.join(output_directory, "periodic_z.inp")

    with open(periodic_x_elements_file_path, "w") as file:
        for i, (node1, node2) in enumerate(periodic_x):
            file.write(" \n %i %i \n " % (node1, node2))
    with open(periodic_y_elements_file_path, "w") as file:
        for i, (node1, node2) in enumerate(periodic_y):
            file.write(" \n%i %i \n " % (node1, node2))
    with open(periodic_z_elements_file_path, "w") as file:
        for i, (node1, node2) in enumerate(periodic_z):
            file.write(" \n %i %i \n " % (node1, node2))

    remove_empty_lines(periodic_x_elements_file_path)
    remove_empty_lines(periodic_y_elements_file_path)
    remove_empty_lines(periodic_z_elements_file_path)

    print("Step 6: Periodic boundary conditions completed.")

    # --- EXTRA CONNECTORS DEFINITION ---
    elements_path = os.path.join(output_directory, "elements.inp")
    with open(elements_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    last_element_id = int(lines[-1].split(',')[0]) if lines else 0
    next_element_id = last_element_id + 1

    # Rotational Dampers (case 1 and also for case 3)
    def write_rotational_dampers(joint_nodes_dict, start_id, output_directory):
        rotational_connectors = []
        next_id = start_id
        for joint_nodes in joint_nodes_dict.values():
            if len(joint_nodes) > 1:
                for node1, node2 in combinations(joint_nodes, 2):
                    rotational_connectors.append(f"{next_id}, {node1}, {node2}")
                    next_id += 1
        dampers_file = os.path.join(output_directory, 'rotational_dampers.inp')
        with open(dampers_file, 'w') as f:
            for conn in rotational_connectors:
                f.write(conn + '\n')
        print(f"Rotational dampers file written: {dampers_file}")
        return next_id

    # Translational Dampers with remapped node indices (case 3)
    def write_translational_dampers_with_replica(lines, replica_map, start_id, output_directory):
        connectors = []
        next_id = start_id
        for elem_idx, line in enumerate(lines):
            parts = [int(x.strip()) for x in line.split(',')]
            node1 = parts[1]
            node2 = parts[3] if len(parts) > 3 else parts[2]
            new_node1 = replica_map.get((node1, elem_idx), node1) if replica_map else node1
            new_node2 = replica_map.get((node2, elem_idx), node2) if replica_map else node2
            connectors.append(f"{next_id}, {new_node1}, {new_node2}")
            next_id += 1
        dampers_file = os.path.join(output_directory, 'translational_dampers_with_replica.inp')
        with open(dampers_file, 'w') as f:
            for conn in connectors:
                f.write(conn + '\n')
        print(f"Translational and rotational dampers with replica file written: {dampers_file}")
        return next_id

    # Option logic
    if connector_options.get('rotational_damper', False):
        next_element_id = write_rotational_dampers(joint_nodes_dict, next_element_id, output_directory)

    if connector_options.get('translational_damper', False):
        translational_damper_connectors = []
        for idx, line in enumerate(lines):
            parts = [int(x.strip()) for x in line.split(',')]
            node1 = parts[1]
            node2 = parts[3] if len(parts) > 3 else parts[2]
            translational_damper_connectors.append(f"{next_element_id}, {node1}, {node2}")
            next_element_id += 1
        dampers_file = os.path.join(output_directory, 'translational_dampers.inp')
        with open(dampers_file, 'w') as f:
            for conn in translational_damper_connectors:
                f.write(conn + '\n')
        print(f"Translational dampers file written: {dampers_file}")

    if connector_options.get('translational_and_rotational_damper', False):
        # Case 3: write both rotational and translational dampers with replica nodes
        next_element_id = write_rotational_dampers(joint_nodes_dict, next_element_id, output_directory)
        next_element_id = write_translational_dampers_with_replica(lines, replica_map, next_element_id, output_directory)

    print("Connector generation complete.")

    print("Step 8: Extra connectors definition completed.")
    if connector_options.get('rotational_damper', False):
        write_abaqus_input_files(output_directory, domain_physical_dimension, phi, concentration, seed_count, target_avg_valency, fiber_radius, L_original,
                                young_modulus, poisson_ratio, element_type, elements_number, nodes, periodic_x, periodic_y, periodic_z,
                                periodic_x_elements_file_path, periodic_y_elements_file_path, periodic_z_elements_file_path, connector_options, joint_nodes_dict, 
                                replica_map, translational_damping_coefficient, rotational_damping_coefficient)
    else:
        write_abaqus_input_files(output_directory, domain_physical_dimension, phi, concentration, seed_count, target_avg_valency, fiber_radius, L_original,
                                young_modulus, poisson_ratio, element_type, elements_number, nodes, periodic_x, periodic_y, periodic_z,
                                periodic_x_elements_file_path, periodic_y_elements_file_path, periodic_z_elements_file_path, connector_options, None, 
                                None, translational_damping_coefficient, rotational_damping_coefficient)
    
    print("Step 9: Write final Abaqus input file (to be completed).")


def main():
    # ----------- User parameters -----------
    DOMAIN_PHYSICAL_DIMENSION = 40
    V = 0.3
    ELEMENT_TYPE = 2
    FIBER_RADIUS = 0.1
    POISSON_RATIO = 0.495
    TRANSLATIONAL_DAMPING_COEFFICIENT = 0.02
    ROTATIONAL_DAMPING_COEFFICIENT = 0.1
    connector_options = {
        "rotational_damper": True,
        "translational_damper": False,
        "translational_and_rotational_damper": False
    }

    perform_mesh_refinement = False

    # --- User Choice: Sampling vs Single Sample ---
    print("Choose execution mode:")
    print("1. Latin Hypercube Sampling (multiple samples)")
    print("2. Single sample with custom parameters")
    
    choice = 2 
    if choice == 1:
        # --- SAMPLING MODE ---
        print("\n--- LATIN HYPERCUBE SAMPLING MODE ---")
        
        # Sampling definition (Latin Hypercube)
        param_bounds = {
            "seed_count": (100, 500),
            "target_avg_valency": (3.31, 3.9),
            "L_original": (2, 3.5),
            "young_modulus": (200, 500)
        }
        param_names = list(param_bounds.keys())
        bounds = np.array([param_bounds[name] for name in param_names])
        n_samples = 1000
        n_levels = 10

        # Setup main output directory (dated folder)
        main_output_dir = setup_output_directory("sampled_data")

        # Latin Hypercube Sampling
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(param_names))
        sample = sampler.random(n=n_samples)
        scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

        # Discretize to n_levels per parameter
        for i in range(len(param_names)):
            levels = np.linspace(bounds[i, 0], bounds[i, 1], n_levels)
            idx = np.floor((scaled_sample[:, i] - bounds[i, 0]) / (bounds[i, 1] - bounds[i, 0]) * n_levels).astype(int)
            idx = np.clip(idx, 0, n_levels - 1)
            scaled_sample[:, i] = levels[idx]

        # Save CSV
        csv_path = os.path.join(main_output_dir, "lhs_samples.csv")
        np.savetxt(csv_path, scaled_sample, delimiter=",", header=",".join(param_names), comments='')
        print("Latin Hypercube samples generated and saved to lhs_samples.csv")

        # Main Pipeline Loop
        lhs_samples = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        samples_to_process = lhs_samples
        output_base_dir = main_output_dir

    else:
        seed_count = 300
        target_avg_valency = 3.9    
        L_original = 2.5
        young_modulus = 300
        
        # Create single sample array
        single_sample = np.array([[seed_count, target_avg_valency, L_original, young_modulus]])
        samples_to_process = single_sample
        output_base_dir = setup_output_directory("single_sample")
        
        print(f"\nProcessing single sample with parameters:")
        print(f"Seed count: {seed_count}")
        print(f"Target avg valency: {target_avg_valency}")
        print(f"L_original: {L_original}")
        print(f"Young's modulus: {young_modulus}")

    # --- COMMON PROCESSING LOOP ---
    for idx, sample in enumerate(samples_to_process):
        print("\n" + "="*80)
        if choice == 1:
            print(f"{'GENERATING NEW SAMPLE':^80}")
            print(f"{'Sample':^80}")
            print(f"{idx:^80} / {len(samples_to_process):^80}")
        else:
            print(f"{'GENERATING SINGLE SAMPLE':^80}")
        print("="*80 + "\n")
        
        seed_count, target_avg_valency, L_original, young_modulus = sample
        
        # Set up sample-specific output directory
        if choice == 1:
            job_description = f"Sample_{idx}"
            output_directory = setup_output_directory(job_description, base_dir=output_base_dir)
        else:
            output_directory = output_base_dir

        max_attempts = 10  
        attempt = 0
        success = False

        while attempt < max_attempts and not success:
            attempt += 1
            print(f"Attempt {attempt} for sample {idx if choice == 1 else 'single'}")

            # Step 1
            vertices_file_path, edges_file_path, periodic_edges_file_path, output_directory = step1_network_generation(
                idx, sample, output_base_dir
            )

            # Step 2
            optimized_vertex_array, optimized_edges, PeriodicEdges = step2_network_optimization(
                vertices_file_path, edges_file_path, periodic_edges_file_path, output_directory,
                DOMAIN_PHYSICAL_DIMENSION, target_avg_valency, L_original, V
            )

            if optimized_vertex_array is not None and optimized_edges is not None and PeriodicEdges is not None:
                success = True
            else:
                print("Valency optimization failed, retrying configuration from scratch.")

        if not success:
            sample_desc = f"sample {idx}" if choice == 1 else "single sample"
            print(f"Could not generate valid configuration for {sample_desc} after {max_attempts} attempts. Skipping.")
            continue

        # Step 3
        refined_vertex_array, refined_edges, refined_periodic_edges = step3_network_refinement(
            optimized_vertex_array, optimized_edges, PeriodicEdges, output_directory
        )
        if connector_options.get('translational_and_rotational_damper', False):
            print("Note: Translational and rotational dampers selected; Abaqus input will include node replicas.")
        
        # Step 4
        step4_abaqus_input(
            refined_vertex_array, refined_edges, refined_periodic_edges, output_directory, DOMAIN_PHYSICAL_DIMENSION, young_modulus, ELEMENT_TYPE, 
            FIBER_RADIUS, POISSON_RATIO, seed_count, target_avg_valency, L_original, connector_options, TRANSLATIONAL_DAMPING_COEFFICIENT, 
            ROTATIONAL_DAMPING_COEFFICIENT, perform_mesh_refinement
        )
        
        # If single sample mode, break after first successful sample
        if choice == 2:
            print(f"\nSingle sample generation completed successfully!")
            print(f"Output directory: {output_directory}")
            break

if __name__ == "__main__":
    main()