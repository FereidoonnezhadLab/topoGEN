"""
Create Network - Periodic Voronoi Tesselation

Author: Sara Cardona
Date: 19/01/2024

This Python script implements a 3D Voronoi tessellation within a cubic domain, incorporating periodic boundary
conditions to simulate a continuous, repeating structure.
It begins by generating a set of random points inside the cube, which are then replicated across a 3x3x3 grid to emulate
the periodicity. Utilizing these points, the script calculates the Voronoi diagram, subsequently filtering and adjusting
the vertices and edges to align with the periodic boundaries and the cube's confines. Through a series of optimization
steps, the script merges nearby vertices, eliminates duplicate edges, and removes unconnected vertices to refine the
Voronoi network.

For visualization, the script employs Plotly to create two 3D plots: one shows the tessellation within the original
domain and another shows the complete periodic structure. The first network differentiates between regular
Voronoi edges and those resulting from the periodic boundary conditions.
"""

"""
--------------------------------------------Directory Definition -------------------------------------------------------
"""
import os
from TopoGEN.utils.setup import setup_output_directory
import time
import numpy as np
import plotly.io as pio
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
pio.renderers.default = 'browser'
from TopoGEN.src.create_periodic_network import (tile_points, lloyd_relaxation_3d_periodic,
                                                 get_vertices, get_edges, process_edges,
                                                 merge_close_vertices, replica_removal,
                                                 find_periodic_pairs, calculate_mean_length)

# ======================================================================================================================
#  SETUP JOB DESCRIPTION
# ======================================================================================================================
job_description = "Test"
output_directory = setup_output_directory(job_description)

# Define file paths for saving results
vertices_file_path = os.path.join(output_directory, "vertices.txt")
edges_file_path = os.path.join(output_directory, "edges.txt")
periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")

DomainPhysicalDimension = 40


A = 0.5

def PhysicalLength(N, DomainPhysicalDimension, A=0.5, iterations=10, bounds=[(-A, A), (-A, A), (-A, A)]):
    # Generate random seeds
    original_points = np.random.uniform(-A, A, (N, 3))
    points, vor = lloyd_relaxation_3d_periodic(original_points, iterations, N)

    TileVertices = vor.vertices
    vertices, IndexMap = get_vertices(TileVertices)
    TileEdges = process_edges(vor.ridge_vertices, TileVertices)
    Vertices, Edges = get_edges(vertices, TileVertices, TileEdges, IndexMap, bounds)
    UniqueEdges = replica_removal(Edges)
    FinalVertices = Vertices
    FinalEdges = UniqueEdges

    comp_length = calculate_mean_length(FinalVertices, FinalEdges)
    real_length = DomainPhysicalDimension * comp_length
    return real_length

N_values = [50, 100, 200, 300, 400, 500]
DomainPhysicalDimensions = np.linspace(20, 200, 10)


results = []
ratios = []
for N in N_values:
    real_lengths_for_N = []
    ratios_for_N = []
    for DomainPhysicalDimension in DomainPhysicalDimensions:
        real_length = PhysicalLength(N, DomainPhysicalDimension)
        real_lengths_for_N.append(real_length)
        ratio = N / DomainPhysicalDimension
        ratios_for_N.append(ratio)
    results.append(real_lengths_for_N)
    ratios.append(ratios_for_N)

# Plot the results
plt.figure(figsize=(10, 6))
for idx, N in enumerate(N_values):
    plt.plot(DomainPhysicalDimensions, results[idx], label=f'N = {N}')

plt.xlabel('Domain Physical Dimension')
plt.ylabel('Real Length')
plt.title('Real Length vs Domain Physical Dimension for different N values')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
for idx, N in enumerate(N_values):
    plt.plot(ratios[idx], results[idx], label=f'N = {N}')

plt.xlabel('N / Domain Physical Dimension')
plt.ylabel('Real Length')
plt.title('Real Length vs N / Domain Physical Dimension for different N values')
plt.legend()
plt.grid(True)
plt.show()