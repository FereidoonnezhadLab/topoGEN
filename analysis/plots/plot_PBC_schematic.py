"""
Author: Sara Cardona
Date: 10/08/2024
This auxiliary plotting scheme visualizes the periodic boundary condition. I used it to generate Figure 2 in TopoGEN 1.
It provides a straightforward way to illustrate the overall 3D process by projecting it into 2D with color coding.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import rcParams, font_manager
import os
from TopoGEN.utils.setup import setup_output_directory
import numpy as np
import plotly.graph_objects as go

# Load custom fonts
font_path_regular = 'D:/FONT/SourceSansPro-Regular.otf'
font_path_bold = 'D:/FONT/SourceSansPro-Bold.otf'
font_manager.fontManager.addfont(font_path_regular)
font_manager.fontManager.addfont(font_path_bold)
rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'regular'

# Set up colormap
colormap = cm.get_cmap('viridis', 9)
full_colormap = cm.get_cmap('bone')
colormap = mcolors.ListedColormap(full_colormap(np.linspace(0.2, 0.7, 9)))

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
#  PLOTS
# ======================================================================================================================

def tile_points_with_colors(points, N, A):
    """
    Tiles the given set of points across a 3x3 grid and assigns a color index to each tile.

    Args:
    - points: Array of shape (N, 3) representing 3D points.
    - N: Number of points in the original domain.
    - A: Half the side length of the central domain.

    Returns:
    - tiled_points (numpy.ndarray): Tiled points across the 3x3 domain.
    - colors (numpy.ndarray): Array of color indices corresponding to each tiled point.
    """
    tiled_points = []
    colors = []
    for i, (dx, dy) in enumerate([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]):
        translated_points = points + np.array([dx, dy, 0]) * 2 * A
        tiled_points.append(translated_points)
        colors.append(np.full((N,), i))  # Assign color index

    return np.vstack(tiled_points), np.hstack(colors)


def plot_pbc_no_voronoi(tiled_points, color_indices, A, output_directory):
    """
    Plots the tiled domain structure without the Voronoi diagram.

    Args:
    - tiled_points: Tiled points across the 3x3 domain.
    - color_indices: Array of color indices for coloring the domains.
    - A: Half the side length of the central domain.
    - output_directory: Directory where the output SVG file will be saved.

    Returns:
    - None: Displays the plot and saves it as an SVG.
    """
    plt.figure(figsize=(8, 8))

    # Draw domain backgrounds with colormap
    for i, (dx, dy) in enumerate([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]):
        intensity = 0.8 if i == 4 else 0.5
        rect = plt.Rectangle((dx * 2 * A - A, dy * 2 * A - A), 2 * A, 2 * A, linewidth=1,
                             edgecolor='none', facecolor=colormap(i), alpha=intensity)
        plt.gca().add_patch(rect)

    # Plot points with colormap
    for i in range(9):
        domain_points = tiled_points[color_indices == i, :2]
        plt.scatter(domain_points[:, 0], domain_points[:, 1], color=colormap(i), label=f'Domain {i+1}')

    # Draw domain boundaries
    for i, (dx, dy) in enumerate([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]):
        rect = plt.Rectangle((dx * 2 * A - A, dy * 2 * A - A), 2 * A, 2 * A, linewidth=1,
                             edgecolor=colormap(i), facecolor='none')
        plt.gca().add_patch(rect)

    # Draw central domain boundary
    central_rect = plt.Rectangle((-A, -A), 2 * A, 2 * A, linewidth=2, edgecolor=colormap(4),
                                 facecolor='none', linestyle='-')
    plt.gca().add_patch(central_rect)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(False)

    # Save and display the plot
    output_path = os.path.join(output_directory, "PBC_NoVoronoi.svg")
    plt.savefig(output_path, format="svg")
    plt.show()


def plot_pbc_with_voronoi(tiled_points, color_indices, A, output_directory):
    """
    Plots the tiled domain structure with the Voronoi diagram.

    Args:
    - tiled_points: Tiled points across the 3x3 domain.
    - color_indices: Array of color indices for coloring the domains.
    - A: Half the side length of the central domain.
    - output_directory: Directory where the output SVG file will be saved.

    Returns:
    - None: Displays the plot and saves it as an SVG.
    """
    all_points_2d = tiled_points[:, :2]
    vor = Voronoi(all_points_2d)

    plt.figure(figsize=(8, 8))

    # Draw domain backgrounds with colormap
    for i, (dx, dy) in enumerate([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]):
        intensity = 0.8 if i == 4 else 0.5
        rect = plt.Rectangle((dx * 2 * A - A, dy * 2 * A - A), 2 * A, 2 * A, linewidth=1,
                             edgecolor='none', facecolor=colormap(i), alpha=intensity)
        plt.gca().add_patch(rect)

    # Plot points with colormap
    for i in range(9):
        domain_points = tiled_points[color_indices == i, :2]
        plt.scatter(domain_points[:, 0], domain_points[:, 1], color=colormap(i), label=f'Domain {i+1}')

    # Draw domain boundary for the central domain
    central_rect = plt.Rectangle((-A, -A), 2 * A, 2 * A, linewidth=2, edgecolor=colormap(4),
                                 facecolor='none', linestyle='-')
    plt.gca().add_patch(central_rect)

    # Plot Voronoi diagram
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors="black",
                    line_width=0.5, line_alpha=0.6, ax=plt.gca())

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    # Save and display the plot
    output_path = os.path.join(output_directory, "PBC_WithVoronoi.svg")
    plt.savefig(output_path, format="svg")
    plt.show()


# EXAMPLE OF USAGE OF plot_pbc_no_voronoi
N = 100
A =  0.5
points = np.random.uniform(-A, A, (N, 3))
tiled_points, color_indices = tile_points_with_colors(points, N, A)
plot_pbc_no_voronoi(tiled_points, color_indices, A, output_directory)
plot_pbc_with_voronoi(tiled_points, color_indices, A, output_directory)