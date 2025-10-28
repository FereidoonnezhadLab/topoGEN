"""
Author: Sara Cardona
Date: 10/08/2024

These auxiliary functions help visualize the network in 3D space and analyze its topology before optimization.
They are already implemented in the main module and can be commented out to accelerate the optimization process.
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
from matplotlib.colors import LinearSegmentedColormap


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
#job_description = "Test"
#output_directory = setup_output_directory(job_description)

# Define file paths for saving results
#vertices_file_path = os.path.join(output_directory, "vertices.txt")
#edges_file_path = os.path.join(output_directory, "edges.txt")
#periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")

# ======================================================================================================================
#  PLOTS
# ======================================================================================================================


def edge_length(vertices_position, edge):
    """Calculate the Euclidean distance (length) of the edge between two nodes."""
    return np.linalg.norm(vertices_position[edge[0]] - vertices_position[edge[1]])


def plot_initial_edge_length_distribution_no_boundary(initial_edges, node_positions, boundary_condition_fn):
    """
    Plot the length distribution of the initial edges, excluding boundary edges.

    :param initial_edges: Original set of edges before optimization.
    :param node_positions: Array of node positions for calculating edge lengths.
    :param boundary_condition_fn: A function to check if a node is on the boundary.
    """
    non_boundary_edges = [edge for edge in initial_edges if not boundary_condition_fn(edge, node_positions)]
    initial_lengths = [edge_length(node_positions, edge) for edge in non_boundary_edges]
    initial_mean = np.mean(initial_lengths)

    plt.figure(figsize=(8, 6))
    plt.hist(initial_lengths, bins=30, alpha=0.7, label='Initial Non-Boundary Edge Lengths', color='darkblue', edgecolor='black', density=True)
    plt.xlabel('Edge Length [$\mathregular{\mu}$m]', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlim(0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Initial Edge Length Distribution (Excluding Boundary Edges)\nMean: {initial_mean:.2f}", fontsize=14)
    plt.show()


def boundary_condition_fn(edge, vertices_position, bounds=(-1, 1)):
    """Example boundary condition: nodes outside a given bound are considered boundary nodes."""
    for node in edge:
        pos = vertices_position[node]
        if np.any(pos < bounds[0]) or np.any(pos > bounds[1]):
            return True
    return False


def plot_periodic_voronoi_tessellation(final_vertices, final_edges, periodic_edges):
    """
    Plots a 3D periodic Voronoi tessellation within a cubic domain.

    Args:
    - final_vertices: Array of shape (N, 3) representing the final vertex positions.
    - final_edges: List of edges connecting regular vertices.
    - final_edges: List of edges representing periodic connections.

    Returns:
    - None: Displays the 3D plot.
    """
    fig = go.Figure()

    # Define cube vertices centered at (0,0,0) with edges of length 1
    cube_vertices_x = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]
    cube_vertices_y = [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]
    cube_vertices_z = [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]

    # Add the cubic domain as a soft grey volume
    fig.add_trace(go.Mesh3d(
        x=cube_vertices_x,
        y=cube_vertices_y,
        z=cube_vertices_z,
        color='rgba(200,200,200,0.5)',  # Soft grey with transparency
        opacity=0.5,
        name='Cubic Domain'
    ))

    # Add vertices
    fig.add_trace(go.Scatter3d(
        x=final_vertices[:, 0], y=final_vertices[:, 1], z=final_vertices[:, 2],
        mode='markers',
        marker=dict(size=1, color='darkblue'),
        text=[f'ID: {i}' for i in range(len(final_vertices))],
        hoverinfo='text',
        name='Vertices'
    ))

    # Define colors
    viridis_purple = '#440154'
    viridis_turquoise = '#20908c'

    # Flags to ensure unique legend labels
    added_regular_edge_legend = False
    added_periodic_edge_legend = False

    # Plot regular edges
    for edge in final_edges:
        start_vertex = final_vertices[edge[0]]
        end_vertex = final_vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[start_vertex[0], end_vertex[0]],
            y=[start_vertex[1], end_vertex[1]],
            z=[start_vertex[2], end_vertex[2]],
            mode='lines',
            line=dict(color=viridis_turquoise, width=5),
            name='Regular Edges' if not added_regular_edge_legend else '',
        ))
        added_regular_edge_legend = True

    # Plot periodic edges
    for edge in periodic_edges:
        start_vertex = final_vertices[edge[0]]
        end_vertex = final_vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[start_vertex[0], end_vertex[0]],
            y=[start_vertex[1], end_vertex[1]],
            z=[start_vertex[2], end_vertex[2]],
            mode='lines',
            line=dict(color=viridis_purple, width=3, dash='solid'),
            opacity=0.4,
            name='Periodic Edges' if not added_periodic_edge_legend else '',
        ))
        added_periodic_edge_legend = True

    # Update layout
    fig.update_layout(
        title='Periodic Voronoi Tessellation',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            aspectmode='cube'
        ),
        scene_camera=dict(
            eye=dict(x=2, y=0.88, z=0.64)
        )
    )

    fig.show()


def plot_voronoi_tessellation(final_vertices, final_edges):
    """
    Plots a 3D Voronoi tessellation within a cubic domain, excluding periodic edges.

    Args:
    - FinalVertices (numpy.ndarray): Array of shape (N, 3) representing the final vertex positions.
    - FinalEdges (list of tuples): List of edges connecting regular vertices.

    Returns:
    - None: Displays the 3D plot.
    """
    fig = go.Figure()

    # Define cube vertices centered at (0,0,0) with edges of length 1
    cube_vertices_x = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]
    cube_vertices_y = [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]
    cube_vertices_z = [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]

    # Add the cubic domain as a soft grey volume
    fig.add_trace(go.Mesh3d(
        x=cube_vertices_x,
        y=cube_vertices_y,
        z=cube_vertices_z,
        color='rgba(200,200,200,0.5)',  # Soft grey with transparency
        opacity=0.5,
        name='Cubic Domain'
    ))

    # Add vertices
    fig.add_trace(go.Scatter3d(
        x=final_vertices[:, 0], y=final_vertices[:, 1], z=final_vertices[:, 2],
        mode='markers',
        marker=dict(size=5, color='darkblue'),
        text=[f'ID: {i}' for i in range(len(final_vertices))],
        hoverinfo='text',
        name='Vertices'
    ))

    # Define color
    viridis_turquoise = '#20908c'

    # Flag for unique legend label
    added_regular_edge_legend = False

    # Plot regular edges
    for edge in final_edges:
        start_vertex = final_vertices[edge[0]]
        end_vertex = final_vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[start_vertex[0], end_vertex[0]],
            y=[start_vertex[1], end_vertex[1]],
            z=[start_vertex[2], end_vertex[2]],
            mode='lines',
            line=dict(color=viridis_turquoise, width=5),
            name='Regular Edges' if not added_regular_edge_legend else '',
        ))
        added_regular_edge_legend = True

    # Update layout
    fig.update_layout(
        title='Voronoi Tessellation',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            aspectmode='cube'
        ),
        scene_camera=dict(
            eye=dict(x=2, y=0.88, z=0.64)
        )
    )

    fig.show()


def plot_voronoi_with_short_bonds(final_vertices, final_edges, short_bond_threshold=0.004):
    """
    Plots a 3D Voronoi tessellation within a cubic domain, highlighting short bonds.

    Args:
    - final_vertices (numpy.ndarray): Array of shape (N, 3) representing vertex positions.
    - final_edges (list of tuples): List of edges connecting vertices.
    - short_bond_threshold (float, optional): Length threshold to classify short bonds. Default is 0.004.

    Returns:
    - None: Displays the 3D plot.
    """
    fig = go.Figure()

    # Define cube vertices centered at (0,0,0) with edges of length 1
    cube_vertices_x = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]
    cube_vertices_y = [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]
    cube_vertices_z = [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]

    # Add the cubic domain as a soft grey volume
    fig.add_trace(go.Mesh3d(
        x=cube_vertices_x,
        y=cube_vertices_y,
        z=cube_vertices_z,
        color='rgba(200,200,200,0.5)',  # Soft grey with transparency
        opacity=0.5,
        name='Cubic Domain'
    ))

    # Add vertices
    fig.add_trace(go.Scatter3d(
        x=final_vertices[:, 0], y=final_vertices[:, 1], z=final_vertices[:, 2],
        mode='markers',
        marker=dict(size=5, color='darkblue'),
        text=[f'ID: {i}' for i in range(len(final_vertices))],
        hoverinfo='text',
        name='Vertices'
    ))

    # Define colors
    viridis_turquoise = '#20908c'  # Regular edge color
    red_color = '#FF0000'  # Color for short bonds

    # Flags to ensure unique legend labels
    added_regular_edge_legend = False
    added_short_edge_legend = False

    # Function to calculate Euclidean distance between two vertices
    def edge_length(v1, v2):
        return np.linalg.norm(v1 - v2)

    # Plot edges, differentiating short bonds
    for edge in final_edges:
        start_vertex = final_vertices[edge[0]]
        end_vertex = final_vertices[edge[1]]
        length = edge_length(start_vertex, end_vertex)

        # Determine the color and legend label
        if length <= short_bond_threshold:
            edge_color = red_color  # Highlight short bonds
            name = 'Short Edges' if not added_short_edge_legend else ''
            added_short_edge_legend = True
        else:
            edge_color = viridis_turquoise  # Regular bond color
            name = 'Regular Edges' if not added_regular_edge_legend else ''
            added_regular_edge_legend = True

        # Add edge to plot
        fig.add_trace(go.Scatter3d(
            x=[start_vertex[0], end_vertex[0]],
            y=[start_vertex[1], end_vertex[1]],
            z=[start_vertex[2], end_vertex[2]],
            mode='lines',
            line=dict(color=edge_color, width=5),
            name=name,
        ))

    # Update layout
    fig.update_layout(
        title='Voronoi Tessellation with Short Bonds',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
            aspectmode='cube'
        ),
        scene_camera=dict(
            eye=dict(x=2, y=0.88, z=0.64)
        )
    )

    fig.show()


def generate_2D_projection_plot(edges, vertices, output_directory):
    """
    Generates and saves a 2D full cube network projection using custom colormaps.

    :param edges: List of edges represented as index pairs.
    :param vertices: List of vertex coordinates.
    :param output_directory: Directory where the output file will be saved.
    """

    fiber_width = 1
    full_colormap = cm.get_cmap('bone')
    colormap = mcolors.ListedColormap(full_colormap(np.linspace(0.2, 0.7, 9)))
    periodic_edge_color_rgb = (255 / 255, 160 / 255, 122 / 255)  # RGB of Light Salmon

    dark_purple_rgb = (48 / 255, 0 / 255, 72 / 255)  # RGB for the purple color

    # Create a custom colormap transitioning to purple
    purple_custom_colormap = LinearSegmentedColormap.from_list(
        "purple_custom_colormap",
        ["#000000", "#300048", "#FFFFFF"],  # Black, Dark Purple, White
        N=256
    )

    reversed_purple_custom_colormap = purple_custom_colormap.reversed()
    boundary_colormap = cm.get_cmap(reversed_purple_custom_colormap, 100)

    def interpolate_boundary_color(coord):
        return (coord + 0.5) / 1

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_facecolor("none")
    rect = plt.Rectangle(
        (-0.5, -0.5), 1, 1, linewidth=0, facecolor=colormap(4), zorder=-1, alpha=0.5)
    ax.add_patch(rect)

    for edge in edges:
        point1 = vertices[edge[0]]
        point2 = vertices[edge[1]]
        x = [point1[0], point2[0]]
        y = [point1[1], point2[1]]
        plt.plot(x, y, color="black", alpha=0.8, linewidth=fiber_width)

    num_points = 100
    y_values = np.linspace(-0.5, 0.5, num_points)
    for x in [-0.5, 0.5]:
        colors = [boundary_colormap(interpolate_boundary_color(y)) for y in y_values]
        for i in range(num_points - 1):
            plt.plot([x, x], [y_values[i], y_values[i + 1]], color=colors[i], linewidth=20)

    x_values = np.linspace(-0.5, 0.5, num_points)
    for y in [-0.5, 0.5]:
        colors = [boundary_colormap(interpolate_boundary_color(x)) for x in x_values]
        for i in range(num_points - 1):
            plt.plot([x_values[i], x_values[i + 1]], [y, y], color=colors[i], linewidth=20)

    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.xlabel("X", fontsize=14, labelpad=20)
    plt.ylabel("Y", fontsize=14, labelpad=20)
    plt.xticks([])
    plt.yticks([])

    output_file = os.path.join(output_directory, "full_cube_network_projection.svg")
    try:
        plt.savefig(output_file, format="svg", bbox_inches="tight", transparent=True)
        print(f"2D full cube network projection saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close()

