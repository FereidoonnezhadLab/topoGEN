"""
TopoGEN.src

This package provides functionalities for:
    - Creating a periodic network
    - Optimizing the network structure
    - Writing Abaqus input files

Modules:
    - create_periodic_network.py
    - optimize_periodic_network.py
    - write_abaqus_input_file.py
"""

# Import and expose functions from create_periodic_network.py
from TopoGEN.src.create_periodic_network import (
    tile_points, lloyd_relaxation_3d_periodic, get_vertices, get_edges,
    process_edges, merge_close_vertices, replica_removal,
    find_periodic_pairs, calculate_mean_length
)

# Import and expose functions from optimize_periodic_network.py
from TopoGEN.src.optimize_periodic_network import (
    optimize_valency, calculate_valencies, simulated_annealing,
    compute_edge_lengths, read_edges, read_vertices
)

# Import and expose functions from write_abaqus_input_file.py
from TopoGEN.src.write_abaqus_input_file import (
    element_orientation_definition, midpoint, edges_length,
    compute_volume_fraction, compute_transverse_shear, order_nodes,
    unique_ordered_list, calculate_x_interpolation_coefficients,
    calculate_y_interpolation_coefficients, classify_nodes,
    calculate_z_interpolation_coefficients, read_periodic_edges,
    remove_empty_lines, write_elements, volume_preserving_uniaxial_condition,
    volume_preserving_biaxial_condition
)

# Define what gets imported when calling `from TopoGEN.src import *`
__all__ = [
    # create_periodic_network.py
    "tile_points", "lloyd_relaxation_3d_periodic", "get_vertices", "get_edges",
    "process_edges", "merge_close_vertices", "replica_removal",
    "find_periodic_pairs", "calculate_mean_length",

    # optimize_periodic_network.py
    "optimize_valency", "calculate_valencies", "simulated_annealing",
    "compute_edge_lengths", "read_edges", "read_vertices",

    # write_abaqus_input_file.py
    "element_orientation_definition", "midpoint", "edges_length",
    "compute_volume_fraction", "compute_transverse_shear", "order_nodes",
    "unique_ordered_list", "calculate_x_interpolation_coefficients",
    "calculate_y_interpolation_coefficients", "classify_nodes",
    "calculate_z_interpolation_coefficients", "read_periodic_edges",
    "remove_empty_lines", "write_elements", "volume_preserving_uniaxial_condition",
    "volume_preserving_biaxial_condition"
]
