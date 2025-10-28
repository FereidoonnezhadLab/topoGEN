"""
Author: Sara Cardona
Date: 20/09/2024

These auxiliary functions help visualize the network in 3D space and analyze its topology after optimization.
They are already implemented in the main module and can be commented out to accelerate the optimization process.
"""

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import rcParams, font_manager
from TopoGEN.utils.setup import setup_output_directory
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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


def is_internal(node, a=0.5):
    return np.all(np.abs(node) < a)


def is_boundary(node, a=0.5):
    return np.any(np.abs(node) == a)

def plot_network_valency_color_code(vertices, valencies, edges, a=0.5):
    """
    Plot the network with the vertices color-coded based on their valency.

    Args:
    - vertices: Array of shape (N, 3) representing 3D points.
    - valencies: Number of fibers per vertex.
    - edges: Edges between vertices.
    - a: Half the side length of the computational domain.

    """
    internal_indices = [i for i, v in enumerate(vertices) if is_internal(v, a)]
    boundary_indices = [i for i, v in enumerate(vertices) if is_boundary(v, a)]

    unique_valencies = np.unique(valencies)
    colors = plt.cm.viridis(np.linspace(1, 0, len(unique_valencies)))
    valency_to_color = {val: colors[i] for i, val in enumerate(unique_valencies)}

    edge_x, edge_y, edge_z = [], [], []

    # Create edges
    for edge in edges:
        x0, y0, z0 = vertices[edge[0]]
        x1, y1, z1 = vertices[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=2),
        name='Edges'
    ))

    # Plot nodes by valency excluding boundary nodes
    for valency in unique_valencies:
        idxs = [i for i in internal_indices if valencies[i] == valency]
        if idxs:
            color = 'rgba(' + ','.join([str(int(c * 255)) for c in valency_to_color[valency][:3]]) + ',' + str(
                valency_to_color[valency][3]) + ')'
            fig.add_trace(go.Scatter3d(
                x=[vertices[i, 0] for i in idxs],
                y=[vertices[i, 1] for i in idxs],
                z=[vertices[i, 2] for i in idxs],
                mode='markers',
                marker=dict(size=5, color=color),
                name=f'Valency {valency}'
            ))

    # Adjust layout to improve visual clarity
    fig.update_layout(
        legend=dict(
            title='Node Types',
            itemsizing='constant'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        )
    )
    fig.show()


def edge_length(vertices_position, edge):
    """Calculate the Euclidean distance (length) of the edge between two nodes."""
    return np.linalg.norm(vertices_position[edge[0]] - vertices_position[edge[1]])


def compute_edge_lengths(nodes, edges):
    node_positions = np.array([nodes[i] for i in range(len(nodes))])
    edges_array = np.array(edges)
    diff = node_positions[edges_array[:, 0]] - node_positions[edges_array[:, 1]]
    return np.linalg.norm(diff, axis=1)


def plot_intermediate_edge_length_distribution(initial_edges, updated_edges, initial_vertices_position,
                                  updated_vertices_position, domain_physical_dimension, output_directory):
    """
    Plot the network with the vertices color-coded based on their valency.

    Args:
    - initial_edges: fiber length before valency optimization
    - updated_edges: fiber length after valency optimization
    - initial_node_positions: vertices position before valency optimization
    - updated_node_positions: vertices position after valency optimization
    - domain_physical_dimension:

    """
    # Calculate the lengths of the initial and updated edges with scaling
    initial_lengths = [edge_length(initial_vertices_position, edge) * domain_physical_dimension for edge in initial_edges]
    updated_lengths = [edge_length(updated_vertices_position, edge) * domain_physical_dimension for edge in updated_edges]

    # Save initial and updated lengths to numpy arrays for later use
    np_initial_lengths = np.array(initial_lengths)
    np_updated_lengths = np.array(updated_lengths)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Convert lengths to pandas DataFrames
    df_initial_lengths = pd.DataFrame(initial_lengths, columns=['Initial Edge Lengths'])
    df_updated_lengths = pd.DataFrame(updated_lengths, columns=['Updated Edge Lengths'])

    # Save the DataFrames to CSV for Excel compatibility
    df_initial_lengths.to_csv(os.path.join(output_directory, 'initial_lengths.csv'), index=False)
    df_updated_lengths.to_csv(os.path.join(output_directory, 'updated_lengths.csv'), index=False)

    # Create the histogram plot
    plt.figure(figsize=(8, 6))

    # Use Viridis colormap for specific colors
    initial_color = viridis(Normalize(vmin=0, vmax=1)(0.1))
    updated_color = viridis(Normalize(vmin=0, vmax=1)(0.3))

    # Plot histograms
    plt.hist(np_initial_lengths, bins=30, alpha=0.6, label='Initial Edge Lengths', color=initial_color,
             edgecolor='black', density=True)
    plt.hist(np_updated_lengths, bins=30, alpha=0.5, label='Updated Edge Lengths', color=updated_color,
             edgecolor='black', density=True)

    # Set labels and customize fonts
    plt.xlabel('Edge Length [$\mathregular{\mu}$m]', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.xlim(0)  # Ensure x-axis starts at 0

    # Save the plot as SVG
    plt.savefig(os.path.join(output_directory, 'Edge_Length_Distribution.svg'))
    plt.show()


def plot_orientation(vertices, edges, preferred_direction=None):
    """
    Plots the cosine distribution of edge orientations relative to a reference direction.

    Parameters:
        vertices (dict): Dictionary of node positions, where keys are node indices and values are 3D coordinates (numpy arrays).
        edges (list of tuples): List of edges represented as tuples (node1, node2) where node1 and node2 are node indices.
        preferred_direction (numpy array, optional): Preferred direction as a reference vector for cosine calculation.
                                                     Defaults to Z-axis if not provided.
    """
    final_color = viridis(0.5)
    reference_direction = preferred_direction if preferred_direction is not None else np.array([0, 0, 1])
    reference_direction = reference_direction / np.linalg.norm(reference_direction)  # Ensure normalization

    cosines = []
    for node1, node2 in edges:
        edge_vector = vertices[node2] - vertices[node1]
        norm = np.linalg.norm(edge_vector)

        if norm == 0:
            continue  # Skip zero-length edges

        edge_vector_normalized = edge_vector / norm

        cosine = np.dot(edge_vector_normalized, reference_direction)
        cosines.append(cosine)  # Directly append the cosine value

    plt.figure(figsize=(8, 6))
    plt.hist(cosines, bins=30, color=final_color, edgecolor='black')
    plt.xlabel('Edges Cosine', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.xlim(-1, 1)  # Set x-axis range for cosine values
    plt.show()


def plot_final_edge_length_distributions(initial_vertices, initial_edges, optimized_vertices,
                                         optimized_edges, output_directory, target_distribution):
    """
    Plot the network with the vertices color-coded based on their valency.

    Args:
    - initial_edges: fiber length before length optimization
    - optimized_edges: fiber length after length optimization
    - initial_vertices: vertices position before length optimization
    - optimized_vertices: vertices position after length optimization
    - output_directory: directory to save the figure
    - target_distribution: distribution of the length based on experiments
    """

    initial_lengths = compute_edge_lengths(initial_vertices, initial_edges)
    final_lengths = compute_edge_lengths(optimized_vertices, optimized_edges)

    # Debugging print statements
    #print("\n[DEBUG] Initial Lengths (Computational Units):", initial_lengths[:5])
    #print("[DEBUG] Final Lengths (Computational Units):", final_lengths[:5])

    # Define x range and PDF
    #x = np.linspace(min(initial_lengths + final_lengths), max(initial_lengths + final_lengths), 10000)
    all_lengths = np.concatenate([initial_lengths, final_lengths])
    x = np.linspace(np.min(all_lengths), np.max(all_lengths), 10000)

    pdf_values = target_distribution.pdf(x)

    # Debugging check on Target Distribution
    #print("[DEBUG] X Values (Computational Units):", x[:5])
    #print("[DEBUG] PDF Values (Computational Units):", pdf_values[:5])

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save final lengths to CSV
    df_final_lengths = pd.DataFrame({'Edge Length': final_lengths})
    df_final_lengths.to_csv(os.path.join(output_directory, 'final_edge_lengths.csv'), index=False)

    #plt.figure(figsize=(8, 6))
    plt.figure(figsize=(8, 6))
    initial_color = 'gray'
    final_color = viridis(Normalize(vmin=0, vmax=1)(0.5))

    # Plot histograms
    plt.hist(initial_lengths, bins=30, alpha=0.3, density=True, label='initial lengths', color=initial_color, edgecolor='black')
    plt.hist(final_lengths, bins=30, alpha=0.7, density=True, label='optimized lengths', color=final_color, edgecolor='black')
    plt.plot(x, pdf_values, label='target distribution', color='black', linewidth=2)

    plt.xlabel('edge length [computational units]', fontsize=20)
    plt.ylabel('probability', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.xlim(0)

    plot_path = os.path.join(output_directory, 'Normalized_Edge_Length_Distribution.svg')
    plt.savefig(plot_path)
    plt.show()

    return plot_path


def load_and_plot_edge_lengths(file_initial, file_updated, file_final, output_directory,
                               domain_physical_dimension, target_distribution):
    """
    Loads edge length data from CSV files and plots distributions for initial, updated, and final lengths.
    """
    os.makedirs(output_directory, exist_ok=True)
    def load_data(filepath):
        df = pd.read_csv(filepath, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        return df[0].values  # Assuming data is in the first column

    # Load and scale edge lengths
    initial_lengths = load_data(file_initial)
    updated_lengths = load_data(file_updated)
    final_lengths_raw = load_data(file_final)

    # Debugging print statements
    #print("\n[DEBUG] Loaded Initial Lengths (CSV):", initial_lengths[:5])
    #print("[DEBUG] Loaded Updated Lengths (CSV):", updated_lengths[:5])
    #print("[DEBUG] Loaded Final Lengths (Before Scaling):", final_lengths_raw[:5])
    #domain_physical_dimension = 30
    # Scale final lengths
    final_lengths = final_lengths_raw * domain_physical_dimension

    #print("[DEBUG] Final Lengths (After Scaling to Physical Units):", final_lengths[:5])

    # Determine min/max values for x-axis
    min_value = min(initial_lengths.min(), updated_lengths.min(), final_lengths.min())
    max_value = max(initial_lengths.max(), updated_lengths.max(), final_lengths.max())

    # Scale target distribution by domain_physical_dimension
    x = np.linspace(min_value, max_value, 1000)
    pdf_values = target_distribution.pdf(x / domain_physical_dimension) / domain_physical_dimension

    #print("[DEBUG] X Values (Physical Units):", x[:5])
    #print("[DEBUG] PDF Values (Physical Units):", pdf_values[:5])

    plt.figure(figsize=(8, 6))
    bins = np.linspace(min_value, max_value, 31)

    plt.hist(initial_lengths, bins=bins, alpha=0.3, color='lightblue', label='initial lengths', edgecolor='black', density=True)
    #plt.hist(updated_lengths, bins=bins, alpha=0.3, color='cornflowerblue', label='intermediate lengths', edgecolor='black', density=True)
    plt.hist(final_lengths, bins=bins, alpha=0.7, color=viridis(Normalize(vmin=0, vmax=1)(0.3)), label='final lengths', edgecolor='black', density=True)
    plt.plot(x, pdf_values, label='target distribution', color='black', linewidth=2)

    plt.xlabel(r'edge length [$\mathregular{\mu}$m]', fontsize=20)
    plt.ylabel('probability', fontsize=20)
    plt.legend(fontsize=20, loc="upper right", frameon=True, borderpad=1.2, labelspacing=1.15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0)

    plot_path = os.path.join(output_directory, 'Edge_Length_Distribution_Comparison.svg')
    plt.savefig(plot_path)
    plt.show()

    return plot_path



""" withouth debug codes """

"""
def plot_final_edge_length_distributions(initial_vertices, initial_edges, optimized_vertices,
                                         optimized_edges, output_directory, target_distribution):

    Plot the network with the vertices color-coded based on their valency.

    Args:
    - initial_edges: fiber length before length optimization
    - optimized_edges: fiber length after length optimization
    - initial_vertices: vertices position before length optimization
    - optimized_vertices: vertices position after length optimization
    - output_directory: directory to save the figure
    - target_distribution: distribution of the length based on experiments

    initial_lengths = compute_edge_lengths(initial_vertices, initial_edges)
    final_lengths = compute_edge_lengths(optimized_vertices, optimized_edges)
    # Define x range and PDF
    x = np.linspace(min(initial_lengths + final_lengths), max(initial_lengths + final_lengths), 10000)
    pdf_values = target_distribution.pdf(x)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    total_final_counts, final_bins = np.histogram(final_lengths, bins=30)

    probability_final = total_final_counts / total_final_counts.sum()
    final_bin_centers = 0.5 * (final_bins[:-1] + final_bins[1:])


    df_final_lengths = pd.DataFrame({'Edge Length': final_lengths})
    df_final_lengths.to_csv(os.path.join(output_directory, 'final_edge_lengths.csv'), index=False)

    plt.figure(figsize=(8, 6))
    initial_color = 'gray'
    final_color = viridis(Normalize(vmin=0, vmax=1)(0.5))

    # Plotting histograms with probability
    plt.hist(initial_lengths, bins=30, alpha=0.3, density=True, label='Initial Lengths', color=initial_color,
             edgecolor='black')
    plt.hist(final_lengths, bins=30, alpha=0.7, density=True, label='Optimized Lengths', color=final_color,
             edgecolor='black')
    plt.plot(x, pdf_values, label='Target Distribution', color='black', linewidth=2)

    plt.xlabel('Edge Length [computational units]', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.xlim(0)  # Ensure that x-axis starts at 0

    plot_path = os.path.join(output_directory, 'Normalized_Edge_Length_Distribution.svg')
    plt.savefig(plot_path)
    plt.show()

    return plot_path


def load_and_plot_edge_lengths(file_initial, file_updated, file_final, output_directory,
                               domain_physical_dimension, target_distribution):

    Plot the network with the vertices color-coded based on their valency.

    Args:
    - file_initial: fiber length before valency optimization
    - file_updated: fiber length after valency optimization
    - file_final: vertices position after length optimization
    - output_directory: directory to save the figure
    - domain_physical_dimension: directory to save the figure
    - target_distribution: distribution of the length based on experiments


    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Helper function to load data
    def load_data(filepath):
        df = pd.read_csv(filepath, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert data to numeric, coercing errors
        df.dropna(inplace=True)  # Drop any rows that contain NaN after conversion
        return df[0].values  # Assuming data is in the first column

    # Load and scale edge lengths data
    initial_lengths = load_data(file_initial)
    updated_lengths = load_data(file_updated)
    final_lengths = load_data(file_final) * domain_physical_dimension

    # Determine the min and max values for consistent x-axis range
    min_value = min(initial_lengths.min(), updated_lengths.min(), final_lengths.min())
    max_value = max(initial_lengths.max(), updated_lengths.max(), final_lengths.max())

    # Scale the target distribution by DomainPhysicalDimension
    x = np.linspace(min_value, max_value, 1000)
    pdf_values = target_distribution.pdf(x / domain_physical_dimension) / domain_physical_dimension

    # Plotting
    plt.figure(figsize=(6, 6))

    # Define bins across all data for consistency
    bins = np.linspace(min_value, max_value, 31)

    # Plot histograms with probability density
    plt.hist(initial_lengths, bins=bins, alpha=0.3, color='lightblue', label='Initial Lengths', edgecolor='black',
             density=True)
    plt.hist(updated_lengths, bins=bins, alpha=0.3, color='cornflowerblue', label='Intermediate Lengths',
             edgecolor='black', density=True)
    plt.hist(final_lengths, bins=bins, alpha=0.7, color=viridis(Normalize(vmin=0, vmax=1)(0.3)), label='Final Lengths',
             edgecolor='black', density=True)
    plt.plot(x, pdf_values, label='Target Distribution', color='black', linewidth=2)
    # Labels and styling
    plt.xlabel(r'Edge Length [$\mathregular{\mu}$m]', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.legend(
        fontsize=12,
        loc="upper right",
        frameon=True,
        borderpad=1.2,
        labelspacing=1.15
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.xlim(0)

    # Save and show the plot
    plot_path = os.path.join(output_directory, 'Edge_Length_Distribution_Comparison.svg')
    plt.savefig(plot_path)
    plt.show()
    return plot_path
"""
