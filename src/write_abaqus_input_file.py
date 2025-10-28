"""
Author: Sara Cardona
Date: 19/01/2024
Refactor: 04/02/2024
"""

import os
import numpy as np
import math

def element_orientation_definition(p1, p2, offset_distance):
    """
    Calculate the coordinates of the point that lies on the perpendicular line to the straight line
    passing through its midpoint and at the specified offset distance from the straight line.

    Args:
    - p1: tuple or list containing the (x,y,z) coordinates of the first endpoint of the straight line
    - p2: tuple or list containing the (x,y,z) coordinates of the second endpoint of the straight line
    - offset_distance: offset distance from the straight line

    Returns:
    - offset_point: tuple containing the (x,y,z) coordinates of the offset point
    """

    # Calculate the direction vector of the straight line
    direction = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])

    # Calculate the unit direction vector of the straight line
    unit_direction = direction / np.linalg.norm(direction)

    # Calculate the perpendicular vector to the straight line
    if abs(unit_direction[2]) < 1e-8:
        # If the unit direction vector is parallel to the z-axis, choose a different vector
        if abs(unit_direction[1]) < 1e-8:
            perp_vector = np.array([0, 1, 0])
        else:
            perp_vector = np.array([1, -unit_direction[0] / unit_direction[1], 0])
    else:
        perp_vector = np.array([1, 1, -(unit_direction[0] + unit_direction[1]) / unit_direction[2]])

    # Calculate the unit perpendicular vector to the straight line
    unit_perp_vector = perp_vector / np.linalg.norm(perp_vector)

    # Calculate the offset point
    offset_point = p1 + offset_distance * unit_perp_vector

    return offset_point


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2]


def edges_length(nodes, edges):
    """
    Calculates the lengths of edges given node coordinates and edges.
    """
    edge_length = []
    for edge in edges:
        length = np.linalg.norm(nodes[edge[0]] - nodes[edge[1]])
        edge_length.append(length)
    return edge_length


def compute_volume_fraction(total_fiber_length, fiber_radius, RVE_size):
    """
    Computes the fiber volume fraction.
    Args:
    - TotalFiberLength: The total length of fibers in micrometers.

    Returns:
    - volume_fraction: The fraction of the domain volume occupied by fibers.
    """
    fiber_area = np.pi * (fiber_radius) ** 2
    total_fiber_volume = fiber_area * total_fiber_length
    rve_volume = RVE_size ** 3  # Volume of the domain in physical units

    volume_fraction = total_fiber_volume / rve_volume

    return volume_fraction


def compute_transverse_shear(radius, length, E, nu, element_type):
    """
    Computes the transverse shear of a beam based on its geometry and material properties.

    Args:
    - radius: Radius of the beam's cross-section.
    - length: Average length of the beams in the network.
    - E: Young's Modulus of the material.
    - nu: Poisson's Ratio of the material.
    - element_type: Type of beam element (1 for B31, 2 for B32).

    Return:
    - f: Dimensionless factor used to prevent shear stiffness from becoming too large in slender beams.
    - transverse_shear: The computed transverse shear force.
    """
    if element_type == 1:
        element_order = 1  # 1 for B31 beams
    elif element_type == 2:
        element_order = 1E-4  # 1E-4 for B32 beams

    shear_factor = 0.89  # for circular cross-section
    slenderness_compensation = 0.25  # default Abaqus value

    area = math.pi * (radius ** 2)
    shear_mod = E / (2 * (1 + nu))
    inertia = math.pi * (radius ** 4) / 4  # Inertia based on radius

    c1 = element_order * slenderness_compensation * (length ** 2) * area / (12 * inertia)
    f = 1 / (
            1 + c1)  # dimensionless factor used to prevent the shear stiffness from becoming too large in slender beam elements

    actual_shear = shear_factor * shear_mod * area
    transverse_shear = actual_shear * f

    return f, transverse_shear


def order_nodes(nodes, node1_id, node2_id, axis_index):
    """
    Orders two nodes based on their position along a specified axis.

    Args:
    - nodes: Dictionary or list-like structure containing node positions.
    - node1_id: ID of the first node.
    - node2_id: ID of the second node.
    - axis_index: Index of the axis along which the nodes are compared.

    Return:
    - ordered_pair (tuple): A tuple containing the node IDs in ascending order based on their position along the specified axis.
    """
    node1_pos = nodes[node1_id][axis_index]
    node2_pos = nodes[node2_id][axis_index]
    if node1_pos < node2_pos:
        return (node1_id, node2_id)
    else:
        return (node2_id, node1_id)


def unique_ordered_list(pairs, index):
    """
    Extracts a unique, ordered list of elements from a list of pairs based on a specified index.

    Args:
    - pairs: List of tuples or lists, where each pair contains elements.
    - index: Index of the element within each pair to extract uniquely.

    Return:
    - result (list): A list of unique elements in the order they first appear in the input pairs.
    """
    seen = set()
    result = []
    for pair in pairs:
        node = pair[index]
        if node not in seen:
            seen.add(node)
            result.append(node)
    return result


def read_periodic_edges(filename):
    with open(filename, 'r') as file:
        edges = [tuple(map(int, line.split())) for line in file.readlines()]
    return edges


def remove_empty_lines(file_path):
    with open(file_path, "r") as fin:
        lines = fin.readlines()
    with open(file_path, "w") as fout:
        prev_was_nonempty = True
        for line in lines:
            if line.strip():
                fout.write(line)
                prev_was_nonempty = True
            else:
                if prev_was_nonempty:
                    # Optionally: fout.write('\n') # but this would allow single blank lines
                    prev_was_nonempty = False


def classify_nodes(periodic_x, periodic_y, periodic_z, nodes, tolerance, DomainPhysicalDimension):

    """
    Classifies nodes into boundary groups based on their spatial location within a given domain.

    Args:
    - periodic_x: List of node pairs defining periodicity along the X-axis.
    - periodic_y: List of node pairs defining periodicity along the Y-axis.
    - periodic_z: List of node pairs defining periodicity along the Z-axis.
    - nodes: List of node coordinates, where each node is represented as (x, y, z).
    - tolerance: Tolerance value to determine if a node belongs to a boundary.
    - DomainPhysicalDimension: The physical dimension of the domain, used to define boundary limits.

    Return:
    - boundaries (dict): A dictionary where keys are boundary names ('NegX', 'PosX', 'NegY', 'PosY', 'NegZ', 'PosZ')
          and values are lists of node IDs that belong to each boundary.
    """

    A = DomainPhysicalDimension / 2

    periodic_nodes = set(node for edge in (periodic_x + periodic_y + periodic_z) for node in edge)

    boundaries = {
        'NegX': [],
        'PosX': [],
        'NegY': [],
        'PosY': [],
        'NegZ': [],
        'PosZ': [],
    }

    assigned_nodes = set()  # To keep track of nodes that have already been assigned to a boundary

    # Order of boundaries can dictate priority
    priority_keys = ['NegX', 'PosX', 'NegY', 'PosY', 'NegZ', 'PosZ']

    for i, (x, y, z) in enumerate(nodes):
        node_id = i + 1  # Adjust to 1-based indexing used in external files and boundary assignment

        if node_id in periodic_nodes:
            continue  # Skip nodes that are periodic
        if node_id in assigned_nodes:
            continue  # Skip nodes that have already been assigned

        for key in priority_keys:
            boundary_check = False
            if key == 'NegX' and abs(x + A) <= tolerance:
                boundary_check = True
            elif key == 'PosX' and abs(x - A) <= tolerance:
                boundary_check = True
            elif key == 'NegY' and abs(y + A) <= tolerance:
                boundary_check = True
            elif key == 'PosY' and abs(y - A) <= tolerance:
                boundary_check = True
            elif key == 'NegZ' and abs(z + A) <= tolerance:
                boundary_check = True
            elif key == 'PosZ' and abs(z - A) <= tolerance:
                boundary_check = True

            if boundary_check:
                boundaries[key].append(node_id)  # Node id is already 1-based here
                assigned_nodes.add(node_id)  # Store as 1-based
                break

    return boundaries


def write_elements(filename, elements, start_index=1):
    with open(filename, 'w') as file:
        for idx, element in enumerate(elements, start=start_index):
            file.write(f"{idx}, {element[1]}, {element[2]}, {element[3]}, {element[4]}\n")


def write_abaqus_input_files(
    output_directory,
    domain_physical_dimension,
    phi,
    concentration,
    N,
    target_avg_valency,
    fiber_radius,
    L_original,
    young_modulus,
    poisson_ratio,
    element_type,
    elements_number,
    nodes,
    periodic_x,
    periodic_y,
    periodic_z,
    periodic_x_elements_file_path,
    periodic_y_elements_file_path,
    periodic_z_elements_file_path,
    connector_options,
    joint_nodes_dict,
    replica_map,
    translational_damping_coefficient,
    rotational_damping_coefficient
):
    
    import os
    import numpy as np
    from collections import defaultdict
    from itertools import combinations
    """
    Write all 9 Abaqus input files for different loading conditions.
    """
    periodic_x_length = len(periodic_x)
    periodic_y_length = len(periodic_y)
    periodic_z_length = len(periodic_z)
    half_domain_physical_dimension = domain_physical_dimension / 2

    def dummy_node_lines(load_type):
        lines = []
        if load_type in ["X_Uniaxial_Tension", "XY_Equibiaxial_Tension", "XZ_Equibiaxial_Tension", "XY_Simple_Shear", "XZ_Simple_Shear"]:
            for i in range(periodic_x_length):
                lines.append("%i, %.2f, 0.0, 0.0" % (5000000 + i, domain_physical_dimension / (2 if "Shear" in load_type else 1)))
        if load_type in ["Y_Uniaxial_Tension", "XY_Equibiaxial_Tension", "YZ_Equibiaxial_Tension", "YZ_Simple_Shear"]:
            for i in range(periodic_y_length):
                lines.append("%i, 0.0, %.2f, 0.0" % (6000000 + i, domain_physical_dimension / (2 if "Shear" in load_type else 1)))
        if load_type in ["Z_Uniaxial_Tension", "XZ_Equibiaxial_Tension", "YZ_Equibiaxial_Tension"]:
            for i in range(periodic_z_length):
                lines.append("%i, 0.0, 0.0, %.2f" % (7000000 + i, domain_physical_dimension / (2 if "Shear" in load_type else 1)))
        return lines

    def dummy_node_sets(load_type):
        sets = []
        if load_type in ["X_Uniaxial_Tension", "XY_Equibiaxial_Tension", "XZ_Equibiaxial_Tension", "XY_Simple_Shear", "XZ_Simple_Shear"]:
            sets.append(("DummyXNodes", 5000000, 5000000 + periodic_x_length - 1))
        if load_type in ["Y_Uniaxial_Tension", "XY_Equibiaxial_Tension", "YZ_Equibiaxial_Tension", "YZ_Simple_Shear"]:
            sets.append(("DummyYNodes", 6000000, 6000000 + periodic_y_length - 1))
        if load_type in ["Z_Uniaxial_Tension", "XZ_Equibiaxial_Tension", "YZ_Equibiaxial_Tension"]:
            sets.append(("DummyZNodes", 7000000, 7000000 + periodic_z_length - 1))
        return sets

    def boundary_lines(load_type):
        lines = []
        uniaxial_strain = 0.25
        biaxial_strain = 0.25
        shear_strain = 0.5
        dummy_disp = {
            "uniaxial": uniaxial_strain * domain_physical_dimension,
            "biaxial": biaxial_strain * domain_physical_dimension,
            "shear": shear_strain * domain_physical_dimension,
        }
        if load_type == "X_Uniaxial_Tension":
            lines.append("DummyXNodes, 1, 1, %.6f" % dummy_disp["uniaxial"])
        elif load_type == "Y_Uniaxial_Tension":
            lines.append("DummyYNodes, 2, 2, %.6f" % dummy_disp["uniaxial"])
        elif load_type == "Z_Uniaxial_Tension":
            lines.append("DummyZNodes, 3, 3, %.6f" % dummy_disp["uniaxial"])
        elif load_type == "XY_Equibiaxial_Tension":
            lines.append("DummyXNodes, 1, 1, %.6f" % dummy_disp["biaxial"])
            lines.append("DummyYNodes, 2, 2, %.6f" % dummy_disp["biaxial"])
        elif load_type == "XZ_Equibiaxial_Tension":
            lines.append("DummyXNodes, 1, 1, %.6f" % dummy_disp["biaxial"])
            lines.append("DummyZNodes, 3, 3, %.6f" % dummy_disp["biaxial"])
        elif load_type == "YZ_Equibiaxial_Tension":
            lines.append("DummyYNodes, 2, 2, %.6f" % dummy_disp["biaxial"])
            lines.append("DummyZNodes, 3, 3, %.6f" % dummy_disp["biaxial"])
        elif load_type == "XY_Simple_Shear":
            lines.append("DummyXNodes, 2, 2, %.6f" % dummy_disp["shear"])
        elif load_type == "XZ_Simple_Shear":
            lines.append("DummyXNodes, 3, 3, %.6f" % dummy_disp["shear"])
        elif load_type == "YZ_Simple_Shear":
            lines.append("DummyYNodes, 3, 3, %.6f" % dummy_disp["shear"])
        return lines

    def write_equations_X_Uniaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):    
        dummy_node_id = 5000000
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 1, 1., %i, 1, -1., %i, 1, -1.\n" % (pair[1], pair[0], dummy_node_id))
            dummy_node_id += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_Y_Uniaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id = 6000000
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
            dummy_node_id += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_Z_Uniaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id = 7000000
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id))
            dummy_node_id += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_XY_Equibiaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id_x = 5000000
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 1, 1., %i, 1, -1., %i, 1, -1.\n" % (pair[1], pair[0], dummy_node_id_x))
            dummy_node_id_x += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        dummy_node_id_y = 6000000
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id_y))
            dummy_node_id_y += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_XZ_Equibiaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id_x = 5000000
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 1, 1., %i, 1, -1., %i, 1, -1.\n" % (pair[1], pair[0], dummy_node_id_x))
            dummy_node_id_x += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        dummy_node_id_z = 7000000
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id_z))
            dummy_node_id_z += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_YZ_Equibiaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id_y = 6000000
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id_y))
            dummy_node_id_y += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        dummy_node_id_z = 7000000
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id_z))
            dummy_node_id_z += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_XY_Simple_Shear(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id_x = 5000000
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id_x))
            dummy_node_id_x += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_XZ_Simple_Shear(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id_x = 5000000
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id_x))
            dummy_node_id_x += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations_YZ_Simple_Shear(f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        dummy_node_id_y = 6000000
        for pair in periodic_y_edges:
            f.write("*Equation\n")
            f.write("3\n")
            f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id_y))
            dummy_node_id_y += 1
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_x_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
        for pair in periodic_z_edges:
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 1, 1., %i, 1, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            f.write("*Equation\n")
            f.write("2\n")
            f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))

    def write_equations(load_type, f, periodic_x_edges, periodic_y_edges, periodic_z_edges):
        if load_type == "X_Uniaxial_Tension":
            write_equations_X_Uniaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "Y_Uniaxial_Tension":
            write_equations_Y_Uniaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "Z_Uniaxial_Tension":
            write_equations_Z_Uniaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "XY_Equibiaxial_Tension":
            write_equations_XY_Equibiaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "XZ_Equibiaxial_Tension":
            write_equations_XZ_Equibiaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "YZ_Equibiaxial_Tension":
            write_equations_YZ_Equibiaxial_Tension(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "XY_Simple_Shear":
            write_equations_XY_Simple_Shear(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "XZ_Simple_Shear":
            write_equations_XZ_Simple_Shear(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        elif load_type == "YZ_Simple_Shear":
            write_equations_YZ_Simple_Shear(f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
        else:
            raise ValueError(f"Unknown loading condition: {load_type}") 

    def write_connector_section(
        f,
        output_directory,
        connector_options,
        joint_nodes_dict,
        replica_map,
        element_type,
        elements_number,
        translational_damping_coefficient,
        rotational_damping_coefficient
    ):
        # CASE 1: Rotational dampers (requires replicated node files and orientations)
        if connector_options.get('rotational_damper', False):
            # Write rotational connector sections with orientations
            dampers_file = os.path.join(output_directory, "rotational_dampers.inp")
            if os.path.exists(dampers_file):
                with open(dampers_file, 'r') as con_file:
                    connector_blocks = []
                    for line in con_file:
                        fields = [int(x) for x in line.strip().split(',') if x.strip()]
                        if len(fields) == 3:
                            conn_id, node1, node2 = fields
                            connector_blocks.append((conn_id, node1, node2))
                f.write("\n*Element, type=CONN3D2, INPUT = rotational_dampers.inp")
                for conn_id, node1, node2 in connector_blocks:
                    f.write(f"\n*Elset, elset=Rotational_Connector_{conn_id}\n{conn_id}\n")
                    # Find orientation names for node1 and node2 (search by contained node_id)
                    orient_name_1 = sorted(beam_orient_map[node1])[0] if node1 in beam_orient_map else f"beam_{node1}_unknown"
                    orient_name_2 = sorted(beam_orient_map[node2])[0] if node2 in beam_orient_map else f"beam_{node2}_unknown"
                    f.write(f"*Connector Section, elset=Rotational_Connector_{conn_id}, behavior=RotDamping\n")
                    f.write("ROTATION\n")
                    f.write(f"{orient_name_1}\n")
                    f.write(f"{orient_name_2}\n")
                f.write("*Connector Behavior, name=RotDamping\n")
                f.write("*Connector Damping, component=4\n")
                f.write("%.2f\n" % rotational_damping_coefficient)
                f.write("*Connector Damping, component=5\n")
                f.write("%.2f\n" % rotational_damping_coefficient)
                f.write("*Connector Damping, component=6\n")
                f.write("%.2f\n" % rotational_damping_coefficient)


        # CASE 2: Translational (parallel) dampers with orientation
        if connector_options.get('translational_damper', False):
            dampers_file = os.path.join(output_directory, "translational_dampers.inp")
            if os.path.exists(dampers_file):
                with open(dampers_file, 'r') as con_file:
                    connector_blocks = []
                    for line in con_file:
                        fields = [int(x) for x in line.strip().split(',') if x.strip()]
                        if len(fields) == 3:
                            conn_id, node1, node2 = fields
                            connector_blocks.append((conn_id, node1, node2))
                f.write("\n*Element, type=CONN3D2, INPUT = translational_dampers.inp")
                for conn_id, node1, node2 in connector_blocks:
                    f.write(f"\n*Elset, elset=Translational_Connector_{conn_id}\n{conn_id}\n")
                    orient_name_1 = sorted(beam_orient_map[node1])[0] if node1 in beam_orient_map else f"beam_{node1}_unknown"
                    orient_name_2 = sorted(beam_orient_map[node2])[0] if node2 in beam_orient_map else f"beam_{node2}_unknown"
                    f.write(f"*Connector Section, elset=Translational_Connector_{conn_id}, behavior=ConnectorDamper\n")
                    f.write("AXIAL\n")
                    f.write(f"{orient_name_1}\n")
                    f.write(f"{orient_name_2}\n")
                f.write("*Connector Behavior, name=ConnectorDamper\n")
                f.write("*Connector Damping, component=1\n")
                f.write("%.2f\n" % translational_damping_coefficient)


        # CASE 3: BOTH rotational and parallel dampers (with replica and orientation)
        if connector_options.get('translational_and_rotational_damper', False):
            # Translational with replica and orientation
            t_dampers_file = os.path.join(output_directory, "translational_dampers_with_replica.inp")
            if os.path.exists(t_dampers_file):
                with open(t_dampers_file, 'r') as con_file:
                    connector_blocks = []
                    for line in con_file:
                        fields = [int(x) for x in line.strip().split(',') if x.strip()]
                        if len(fields) == 3:
                            conn_id, node1, node2 = fields
                            connector_blocks.append((conn_id, node1, node2))
                f.write("\n*Element, type=CONN3D2, INPUT = translational_dampers_with_replica.inp")
                for conn_id, node1, node2 in connector_blocks:
                    f.write(f"\n*Elset, elset=Translational_Connector_{conn_id}\n{conn_id}\n")
                    orient_name_1 = sorted(beam_orient_map[node1])[0] if node1 in beam_orient_map else f"beam_{node1}_unknown"
                    orient_name_2 = sorted(beam_orient_map[node2])[0] if node2 in beam_orient_map else f"beam_{node2}_unknown"
                    f.write(f"*Connector Section, elset=Translational_Connector_{conn_id}, behavior=ConnectorDamper\n")
                    f.write("AXIAL\n")
                f.write(f"{orient_name_1}\n")
                f.write(f"{orient_name_2}\n")
                f.write("*Connector Behavior, name=ConnectorDamper\n")
                f.write("*Connector Damping, component=1\n")
                f.write("%.2f\n" % translational_damping_coefficient)
            r_dampers_file = os.path.join(output_directory, "rotational_dampers.inp")
            if os.path.exists(r_dampers_file):
                with open(r_dampers_file, 'r') as con_file:
                    connector_blocks = []
                    for line in con_file:
                        fields = [int(x) for x in line.strip().split(',') if x.strip()]
                        if len(fields) == 3:
                            conn_id, node1, node2 = fields
                            connector_blocks.append((conn_id, node1, node2))
                f.write("\n*Element, type=CONN3D2, INPUT = rotational_dampers.inp")
                for conn_id, node1, node2 in connector_blocks:
                    f.write(f"\n*Elset, elset=Rotational_Connector_{conn_id}\n{conn_id}\n")
                    orient_name_1 = sorted(beam_orient_map[node1])[0] if node1 in beam_orient_map else f"beam_{node1}_unknown"
                    orient_name_2 = sorted(beam_orient_map[node2])[0] if node2 in beam_orient_map else f"beam_{node2}_unknown"
                    f.write(f"*Connector Section, elset=Rotational_Connector_{conn_id}, behavior=RotDamping\n")
                    f.write("ROTATION\n")
                    f.write(f"{orient_name_1}\n")
                    f.write(f"{orient_name_2}\n")
                f.write("*Connector Behavior, name=RotDamping\n")
                f.write("*Connector Damping, component=4\n")
                f.write("%.2f\n" % rotational_damping_coefficient)
                f.write("*Connector Damping, component=5\n")
                f.write("%.2f\n" % rotational_damping_coefficient)
                f.write("*Connector Damping, component=6\n")
                f.write("%.2f\n" % rotational_damping_coefficient)          

    def write_orientation_block(f, elements_file_path, element_type):
        """Helper function to write orientation blocks and return beam_orient_map"""
        orientation_blocks = []
        beam_orient_map = {}
        
        with open(elements_file_path, "r") as elem_file:
            for line in elem_file:
                fields = [int(x) for x in line.strip().split(',') if x.strip()]
                if not fields: continue
                
                # Parse based on element type
                if element_type == 2 and len(fields) >= 5:
                    n1, n2, offset = fields[1], fields[3], fields[4]
                elif element_type == 1 and len(fields) >= 4:
                    n1, n2, offset = fields[1], fields[2], fields[3]
                else:
                    continue
                    
                name = f"beam_{n1}_{n2}"
                orientation_blocks.append((name, n1, offset, n2))
                beam_orient_map.setdefault(n1, set()).add(name)
                beam_orient_map.setdefault(n2, set()).add(name)
        
        # Write orientation blocks
        for name, n1, offset, n2 in orientation_blocks:
            f.write(f"\n*Orientation, definition=nodes, name={name}, System=Rectangular\n")
            f.write(f"{n1},{offset},{n2}\n")
        
        return beam_orient_map
    
    loading_conditions = [
        "X_Uniaxial_Tension",
        "Y_Uniaxial_Tension",
        "Z_Uniaxial_Tension",
        "XY_Equibiaxial_Tension",
        "XZ_Equibiaxial_Tension",
        "YZ_Equibiaxial_Tension",
        "XY_Simple_Shear",
        "XZ_Simple_Shear",
        "YZ_Simple_Shear",
    ]


    def read_periodic_edges(file_path):
        edges = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    edges.append((int(parts[0]), int(parts[1])))
        return edges

    periodic_x_edges = read_periodic_edges(periodic_x_elements_file_path)
    periodic_y_edges = read_periodic_edges(periodic_y_elements_file_path)
    periodic_z_edges = read_periodic_edges(periodic_z_elements_file_path)

    for load_type in loading_conditions:
        file_path = os.path.join(output_directory, f"{load_type}.inp")
        with open(file_path, "w") as f:
            # HEADING
            f.write("*Heading")
            f.write(f"\n**{load_type} Test of a cubic Representative Volume Element")
            f.write("\n**Computational Unit Length = %i, Stress [MPa], Force [uN]" % domain_physical_dimension)
            f.write("\n**Fiber volume fraction: %.4f" % phi)
            f.write("\n**Concentration: %.3f mg/mL" % concentration)
            f.write("\n**Seeds number: %i" % N)
            f.write("\n**Seeds target_avg_valency: %.2f" % target_avg_valency)
            f.write("\n**Seeds fiber_radius: %.3f" % fiber_radius)
            f.write("\n**Seeds L_original: %.2f" % L_original)
            f.write("\n**Seeds young_modulus: %.2f" % young_modulus)
            f.write("\n*Preprint, echo=NO, model=YES, history=YES, contact=NO")

            # NODES DEFINITION
            # Use nodes_with_replica.inp for rotational dampers or both dampers
            if connector_options.get('rotational_damper', False) or connector_options.get('translational_and_rotational_damper', False):
                f.write("\n*Node, INPUT = nodes_with_replica.inp")
            else:
                f.write("\n*Node, INPUT = nodes.inp")
            
            for line in dummy_node_lines(load_type):
                f.write("\n" + line)

            # NSET for all nodes
            f.write("\n*Nset, nset=AllNodes, generate")
            f.write("\n1, %i, 1" % len(nodes))

            # Dummy node sets
            for nset_name, start_id, end_id in dummy_node_sets(load_type):
                f.write(f"\n*Nset, nset={nset_name}, generate")
                f.write(f"\n{start_id}, {end_id}, 1\n")

            # Periodic Boundary Equations
            # Modularize by making per-load_type routines for equations:
            write_equations(load_type, f, periodic_x_edges, periodic_y_edges, periodic_z_edges)
            if connector_options.get('rotational_damper', False) or connector_options.get('translational_and_rotational_damper', False):
                for joint_nodes in joint_nodes_dict.values():
                    if len(joint_nodes) > 1:
                        ref_node = joint_nodes[0]
                        replica_nodes = joint_nodes[1:]
                        f.write(f"\n*Nset, nset=node_set_{ref_node}\n")
                        f.write(", ".join(str(n) for n in replica_nodes) + "\n")
                        f.write(f"*KINEMATIC COUPLING, REF NODE={ref_node}\n")
                        f.write(f"node_set_{ref_node}, 1, 3\n")

            # ELEMENTS DEFINITION
            f.write("**There are %i number of elements randomly distributed in the volume" % elements_number)
            f.write("\n**Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            slenderness = L_original / fiber_radius
            f.write("\n**The fiber is slenderness 1 / %.2f" % slenderness)
            # Use elements_with_replica.inp for rotational dampers or both dampers
            if connector_options.get('rotational_damper', False) or connector_options.get('translational_and_rotational_damper', False):
                if element_type == 1:
                    f.write("\n*Element, type=B31, INPUT = elements_with_replica.inp")
                elif element_type == 2:
                    f.write("\n*Element, type=B32, INPUT = elements_with_replica.inp")
            else:
                if element_type == 1:
                    f.write("\n*Element, type=B31, INPUT = elements.inp")
                elif element_type == 2:
                    f.write("\n*Element, type=B32, INPUT = elements.inp")
            f.write("\n*Elset, elset = Beams, generate")
            f.write("\n1, %i, 1" % elements_number)
            f.write("\n*Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n%.5f" % fiber_radius)
            
            # MATERIAL DEFINITION
            f.write("\n*Material, name=CollagenMaterial")
            f.write("\n *Elastic")
            f.write("\n %.2f, %.4f" % (young_modulus, poisson_ratio))

            needs_rotational = connector_options.get('rotational_damper', False) or \
                            connector_options.get('translational_and_rotational_damper', False)
            needs_translational = connector_options.get('translational_damper', False)

            if needs_rotational or needs_translational:
                # Determine which elements file to use
                elements_file = "elements_with_replica.inp" if needs_rotational else "elements.inp"
                elements_path = os.path.join(output_directory, elements_file)
                
                # Write orientation block and get beam orientation map
                beam_orient_map = write_orientation_block(f, elements_path, element_type)
                
                # Write connectors
                write_connector_section(
                    f,
                    output_directory,
                    connector_options,
                    joint_nodes_dict,
                    replica_map,
                    element_type,
                    elements_number,
                    translational_damping_coefficient,
                    rotational_damping_coefficient
                )
            
            # STEP DEFINITION
            f.write("\n*Time Points, name=MustPoints, GENERATE")
            f.write("\n0., 1., 0.01")
            f.write("\n*Step, name = UniaxialTest, INC=1000000, nlgeom=YES")
            f.write("\n*Static,  stabilize, ALLSDTOL = 0.02")
            f.write("\n0.01, 1., 1e-100, 0.01")
            f.write("\n*Controls, parameters=time incrementation")
            f.write("\n, , , , , , , 100, , ,")
            #f.write("\n*Amplitude, definition = SMOOTH STEP, name = Smooth")
            #f.write("\n0,0,1,1")

            # BOUNDARY CONDITIONS
            f.write("\n*Boundary, type = Displacement")
            for bound_line in boundary_lines(load_type):
                f.write("\n" + bound_line)

            # OUTPUT
            f.write("\n*Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n*Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output, elset=Beams, POSITION=INTEGRATION POINTS")
            f.write("\n  S, SE, SF, SP, SM, SK, E, NE, LE")
            f.write("\n*Node Output, nset= AllNodes")
            f.write("\nU, RF")
            f.write("\n*End Step")
        remove_empty_lines(file_path)


def sort_nodes_file_by_index(nodes_file_path):
    with open(nodes_file_path, 'r') as f:
        lines = [line for line in f if line.strip()]
    # Extract node index as first entry before comma
    lines_sorted = sorted(lines, key=lambda line: int(line.split(',')[0]))
    with open(nodes_file_path, 'w') as f:
        f.writelines(lines_sorted)