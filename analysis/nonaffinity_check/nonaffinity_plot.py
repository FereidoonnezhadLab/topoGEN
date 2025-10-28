"""
Author: Sara Cardona
Date: 12/23/2024

These functions are aimed at plotting the non affinity 
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import datetime
from matplotlib import rcParams, font_manager
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.cm import viridis, viridis_r
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

font_path_regular = 'D:/FONT/SourceSansPro-Regular.otf'
font_path_bold = 'D:/FONT/SourceSansPro-Bold.otf'
font_manager.fontManager.addfont(font_path_regular)
font_manager.fontManager.addfont(font_path_bold)

rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'regular'
from TopoGEN.utils.setup import setup_output_directory
job_description = "nonaffinity_plot"
output_directory = setup_output_directory(job_description)


def load_nodes(file_path):
    nodes = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) == 4:
                    node_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    nodes[node_id] = coords
    return nodes


def calculate_interpolation_coefficients(nodes, A):
    interpolation_map = {}
    for node_id, (x, y, z) in nodes.items():
        interpolation_map[node_id] = x / A
    return interpolation_map


def load_excel_displacements(file_path, sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
    samples = {}
    for sample in range(5):
        time_disp_data = data.iloc[:, sample * 2:(sample * 2) + 2].dropna()
        time_disp_data.columns = ["time", "displacement"]
        samples[f"sample_{sample + 1}"] = time_disp_data
    return samples


def compute_affine_displacement_subset(interpolation_map, shear_displacement, actual_node_ids):
    return {node_id: interpolation_map[node_id] * shear_displacement for node_id in actual_node_ids if node_id in interpolation_map}


def plot_displacement_distributions(sheet_name, nodes_files, excel_file, strain_points, A, shear_displacement,
                                    output_directory):
    samples_displacements = load_excel_displacements(excel_file, sheet_name)

    def lighten_color(base_color, factor=0.5):
        r, g, b, a = base_color  # RGBA
        r = min(1, r + factor * (1 - r))  # Schiarisce il rosso
        g = min(1, g + factor * (1 - g))  # Schiarisce il verde
        b = min(1, b + factor * (1 - b))  # Schiarisce il blu
        return (r, g, b, a)

    # Definizione dei colori
    strain_colors = {
        "batch_1_0.0517578125": viridis_r(0.3),
        "batch_1_0.25": viridis_r(0.3),
        "batch_1_0.5": viridis_r(0.3),
        "batch_2_0.0517578125": viridis_r(0.9),
        "batch_2_0.25": viridis_r(0.9),
        "batch_2_0.5": viridis_r(0.9)
    }

    # Corrected time-to-strain mapping
    time_to_strain = {0.25: 0.0517578125, 0.5: 0.25, 1.0: 0.5}

    plot_count = 0  # Counter to track the plot number within the batch
    batch = "batch_1" if sheet_name == "3.12" else "batch_2"
    is_first_plot = True  # Track the first plot for each batch

    for time, strain in time_to_strain.items():
        if strain not in strain_points:
            print(f"Skipping strain {strain}: not requested.")
            continue

        print(f"Processing strain={strain}, time={time}")

        # Dynamically set the normalized range based on strain
        normalized_range = (
            1.25 if strain == 0.0517578125 else
            5 if strain == 0.25 else
            10 if strain == 0.5 else None
        )

        combined_actual = []
        combined_affine = []

        for i, node_file in enumerate(nodes_files):
            sample_name = f"sample_{i + 1}"
            nodes = load_nodes(node_file)
            interpolation_map = calculate_interpolation_coefficients(nodes, A)
            sample_displacements = samples_displacements[sample_name]

            # Filter displacements for the current time
            filtered_displacements = sample_displacements[sample_displacements["time"] == time]
            if filtered_displacements.empty:
                print(f"No data found for time={time}. Skipping...")
                continue

            actual_displacements = filtered_displacements["displacement"].values
            actual_node_ids = (filtered_displacements.index + 1).tolist()

            affine_displacements = compute_affine_displacement_subset(interpolation_map, shear_displacement,
                                                                      actual_node_ids)

            aligned_actual = []
            aligned_affine = []
            for idx, node_id in enumerate(actual_node_ids):
                if node_id in affine_displacements:
                    aligned_actual.append(actual_displacements[idx])
                    aligned_affine.append(affine_displacements[node_id])

            combined_actual.extend(aligned_actual)
            combined_affine.extend(aligned_affine)

        # Normalize displacements
        if len(combined_actual) == 0 or len(combined_affine) == 0:
            print(f"No data to plot for time={time}, strain={strain}. Skipping...")
            continue

        normalized_actual = np.array(combined_actual) / normalized_range
        normalized_affine = np.array(combined_affine) / normalized_range

        plt.figure(figsize=(6, 6))
        bins = np.linspace(-strain / 2, strain / 2, 101)
        actual_hist, bin_edges = np.histogram(normalized_actual * (strain / 2), bins=bins, density=True)
        affine_hist, _ = np.histogram(normalized_affine * (strain / 2), bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        x_min = -strain / 2
        x_max = strain / 2
        plt.bar(bin_edges[:-1], actual_hist, width=bin_width, alpha=0.8, edgecolor='black',
                color=strain_colors[f"{batch}_{strain}"], align='edge')
        plt.bar(bin_edges[:-1], affine_hist, width=bin_width, alpha=0.3, edgecolor='black',
                color='gray', align='edge')

        # Calculate and display NRMSE
        rmse = np.sqrt(np.mean((np.array(aligned_actual) - np.array(aligned_affine)) ** 2))
        std_actual = np.std(aligned_actual)
        nrmse_by_std = rmse / std_actual if std_actual > 0 else np.nan
        plt.legend([f"NS = {nrmse_by_std:.2f}"], fontsize=16, loc='upper right')
        plt.xlabel(r'normalized nodal displacement [$\mathregular{\mu}$m]', fontsize=14)
        plt.ylabel('density', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks([])  # Remove y-axis markers
        #plt.xlim(x_min, x_max)

        # Add extra y-axis title for the first plot of each batch
        if is_first_plot:
            z_value = "3.12" if batch == "batch_1" else "3.56"
            plt.gca().annotate(r"$\overline{\mathbf{z}} = $" + f"{z_value}", xy=(-0.08, 0.5), xycoords='axes fraction',
                               fontsize=16,
                               fontweight='bold', ha='center',
                               va='center', rotation=90)

            is_first_plot = False

        # Add extra titles for the first three plots of batch 1
        if batch == "batch_1" and plot_count < 3:
            extra_titles = ["low-stretch regime", "medium-stretch regime", "high-stretch regime"]
            plt.title(extra_titles[plot_count], fontsize=16, fontweight='bold')

        plot_count += 1

        output_path = os.path.join(output_directory, f"{sheet_name}_strain_{strain:.6f}_time_{time:.2f}_sample.svg")
        plt.savefig(output_path, format="svg")
        plt.show()


excel_file = r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\AffinityCheck.xlsx'
sheets_and_nodes = {
    "3.12": [
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_312_s1.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_312_s2.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_312_s3.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_312_s4.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_312_s5.inp',
    ],
    "3.56": [
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_356_s1.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_356_s2.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_356_s3.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_356_s4.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\nodes_356_s5.inp'
    ]
}

elements = {
    "3.12": [
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_312_s1.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_312_s2.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_312_s3.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_312_s4.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_312_s5.inp',
    ],
    "3.56": [
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_356_s1.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_356_s2.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_356_s3.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_356_s4.inp',
        r'D:\CollaGEN\TopoGEN\analysis\nonaffinity_check\elements_356_s5.inp',
    ]
}

strain_points = [0.0517578125, 0.25, 0.5]
shear_displacement = 10
A = 20


for sheet_name, nodes_files in sheets_and_nodes.items():
    plot_displacement_distributions(sheet_name, nodes_files, excel_file, strain_points, A, shear_displacement, output_directory)

# Define the strain colors
strain_colors = {
    "batch_1_0.0517578125": viridis_r(0.3),
    "batch_1_0.25": viridis_r(0.3),
    "batch_1_0.5": viridis_r(0.3),
    "batch_2_0.0517578125": viridis_r(0.9),
    "batch_2_0.25": viridis_r(0.9),
    "batch_2_0.5": viridis_r(0.9)
}

# Define the labels for the measured bins
measured_labels = [
    "z = 3.12, Strain = 0.05",
    "z = 3.12, Strain = 0.25",
    "z = 3.12, Strain = 0.5",
    "z = 3.56, Strain = 0.05",
    "z = 3.56, Strain = 0.25",
    "z = 3.56, Strain = 0.5"
]

# Colors associated with each measured label
measured_colors = [
    strain_colors["batch_1_0.0517578125"],
    strain_colors["batch_1_0.25"],
    strain_colors["batch_1_0.5"],
    strain_colors["batch_2_0.0517578125"],
    strain_colors["batch_2_0.25"],
    strain_colors["batch_2_0.5"]
]

# Define the label and color for the affine bin
affine_label = "Affine"
affine_color = "gray"

# Create a new figure for the legend
fig, ax = plt.subplots(figsize=(18, 4))
ax.axis('off')

# Create measured handles
measured_handles = [
    Patch(facecolor=color, edgecolor='black', alpha=0.8, label=label)
    for color, label in zip(measured_colors, measured_labels)
]

# Create affine handle
affine_handle = Patch(facecolor=affine_color, edgecolor='black', alpha=0.3, label=affine_label)

# Manually arrange the legend rows
handles_row_1 = [measured_handles[0], measured_handles[1], measured_handles[2]]  # z = 3.12
labels_row_1 = ["z = 3.12, Strain = 0.05", "z = 3.12, Strain = 0.25", "z = 3.12, Strain = 0.5"]

handles_row_2 = [measured_handles[3], measured_handles[4], measured_handles[5], affine_handle]  # z = 3.56 + Affine
labels_row_2 = ["z = 3.56, Strain = 0.05", "z = 3.56, Strain = 0.25", "z = 3.56, Strain = 0.5", "Affine"]

# Add the legend for the first row
legend_row_1 = ax.legend(handles=handles_row_1, labels=labels_row_1, fontsize=20, loc='upper center', frameon=False, ncol=3, bbox_to_anchor=(0.5, 0.8))
ax.add_artist(legend_row_1)

# Add the legend for the second row, including "Affine"
legend_row_2 = ax.legend(handles=handles_row_2, labels=labels_row_2, fontsize=20, loc='upper center', frameon=False, ncol=4, bbox_to_anchor=(0.5, 0.6))
ax.add_artist(legend_row_2)

output_path = os.path.join(output_directory, f"legend.svg")
plt.savefig(output_path, format="svg")
plt.show()


def load_elements_filtered(file_path, nodes):

    # Load element connectivity from an input file while **excluding elements that have nodes on the boundary**
    # of a 3D cube centered at the origin with edge length 40 (i.e., boundary at x, y, z = ±20).

    elements = {}
    cube_boundary = 20  # Half edge length of cube

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 4:  # Ensure enough columns
                    elem_id = int(parts[0])  # Element ID
                    node1 = int(parts[1])  # First node
                    node2 = int(parts[3])  # Fourth node (second defining node)

                    if node1 in nodes and node2 in nodes:
                        x1, y1, z1 = nodes[node1]
                        x2, y2, z2 = nodes[node2]

                        # Check if either node is on the boundary
                        if not any(
                                abs(coord) == cube_boundary for coord in [x1, y1, z1, x2, y2, z2]
                        ):
                            elements[elem_id] = (node1, node2)

    return elements


def compute_element_strains(nodes, elements, displacements, interpolation_map, shear_displacement):

    # Compute both actual and affine strains for each element, along with element lengths.

    actual_strains = {}
    affine_strains = {}
    element_lengths = {}

    for elem_id, (node1, node2) in elements.items():
        # Get initial positions
        X1, X2 = np.array(nodes.get(node1)), np.array(nodes.get(node2))

        # Compute initial element length
        L0 = np.linalg.norm(X2 - X1) if np.linalg.norm(X2 - X1) != 0 else 1e-6  # Avoid division by zero
        element_lengths[elem_id] = L0

        # Get nodal displacements (default to zero if missing)
        u1 = np.array(displacements.get(node1))
        u2 = np.array(displacements.get(node2))
        # Compute new positions after deformation
        X1_new = X1 + u1
        X2_new = X2 + u2

        # Compute deformed element length
        L = np.linalg.norm(X2_new - X1_new)

        # Compute actual strain
        actual_strain = (L - L0) / L0 if L0 != 0 else 0.0
        actual_strains[elem_id] = actual_strain

        # Compute affine strain using interpolated displacement
        affine_strain = 0.0
        if node1 in interpolation_map and node2 in interpolation_map:
            u1_affine = interpolation_map[node1] * shear_displacement
            u2_affine = interpolation_map[node2] * shear_displacement
            affine_strain = (u2_affine - u1_affine) / L0 if L0 != 0 else 0.0
        affine_strains[elem_id] = affine_strain

    return actual_strains, affine_strains, element_lengths


def plot_element_strain_distributions(sheet_name, nodes_files, element_files, excel_file, strain_points, A,
                                      shear_displacement, output_directory):

    # Generate histograms of element strains for different time points, using only elements inside the cube.

    samples_displacements = load_excel_displacements(excel_file, sheet_name)

    strain_colors = {
        "batch_1_0.0517578125": viridis_r(0.5),
        "batch_1_0.25": viridis_r(0.5),
        "batch_1_0.5": viridis_r(0.5),
        "batch_2_0.0517578125": viridis_r(0.7),
        "batch_2_0.25": viridis_r(0.7),
        "batch_2_0.5": viridis_r(0.7)
    }

    time_to_strain = {0.25: 0.0517578125, 0.5: 0.25, 1.0: 0.5}
    batch = "batch_1" if sheet_name == "3.12" else "batch_2"

    plot_count = 0  # Counter to track the plot number within the batch
    is_first_plot = True


    for time, strain in time_to_strain.items():
        if strain not in strain_points:
            print(f"Skipping strain {strain}: not requested.")
            continue

        print(f"Processing strain={strain}, time={time}")
        combined_actual_strains = []
        combined_affine_strains = []

        for i, (node_file, elem_file) in enumerate(zip(nodes_files, element_files)):
            sample_name = f"sample_{i + 1}"

            nodes = load_nodes(node_file)
            elements = load_elements_filtered(elem_file, nodes)  # Use filtered elements
            interpolation_map = calculate_interpolation_coefficients(nodes, A)

            sample_displacements = samples_displacements[sample_name]
            filtered_displacements = sample_displacements[sample_displacements["time"] == time]
            if filtered_displacements.empty:
                print(f"No data found for time={time}. Skipping...")
                continue

            displacements = {idx + 1: [disp, 0, 0] for idx, disp in
                             enumerate(filtered_displacements["displacement"].values)}

            actual_strains, affine_strains, element_lengths = compute_element_strains(
                nodes, elements, displacements, interpolation_map, shear_displacement
            )

            combined_actual_strains.extend(actual_strains.values())
            combined_affine_strains.extend(affine_strains.values())

        if not combined_actual_strains or not combined_affine_strains:
            print(f"No data to plot for time={time}, strain={strain}. Skipping...")
            continue

        # Define manual x-axis bounds based on strain regime
        if strain == 0.0517578125:
            bin_min, bin_max = -0.6, 0.8
        elif strain == 0.25:
            bin_min, bin_max = -0.6, 0.8
        elif strain == 0.5:
            bin_min, bin_max = -0.6, 0.8

        bins = np.linspace(bin_min, bin_max, 501)

        plt.figure(figsize=(6, 6))
        actual_hist, bin_edges = np.histogram(combined_actual_strains, bins=bins, density=True)
        affine_hist, _ = np.histogram(combined_affine_strains, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]

        plt.bar(bin_edges[:-1], actual_hist, width=bin_width, alpha=0.8, edgecolor='black',
                color=strain_colors[f"{batch}_{strain}"], align='edge')
        plt.bar(bin_edges[:-1], affine_hist, width=bin_width, alpha=0.3, edgecolor='black',
                color='gray', align='edge')

        rmse = np.sqrt(np.mean((np.array(combined_actual_strains) - np.array(combined_affine_strains)) ** 2))
        std_actual = np.std(combined_actual_strains)
        nrmse_by_std = rmse / std_actual if std_actual > 0 else np.nan
        plt.legend([f"NS = {nrmse_by_std:.2f}"], fontsize=16, loc='upper right')
        plt.xlabel('Fibril Strain [-]', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        xticks = np.arange(-0.6, 0.81, 0.2)
        xticks = [tick for tick in xticks if not np.isclose(tick, -0.6) and not np.isclose(tick, 0.8)]
        plt.xticks(xticks, fontsize=14)
        plt.xlim(bin_min, bin_max)
        plt.yticks([])
        plt.xlim(bin_min, bin_max)

        # Add extra y-axis title for the first plot of each batch
        if is_first_plot:
            z_value = "3.12" if batch == "batch_1" else "3.56"
            plt.gca().annotate(f"z = {z_value}", xy=(-0.08, 0.5), xycoords='axes fraction', fontsize=16,
                               fontweight='bold', ha='center',
                               va='center', rotation=90)
            is_first_plot = False

        # Add extra titles for the first three plots of batch 1
        if batch == "batch_1" and plot_count < 3:
            extra_titles = ["Low-strain Regime", "Medium-strain Regime", "High-strain Regime"]
            plt.title(extra_titles[plot_count], fontsize=16, fontweight='bold')

        plot_count += 1

        output_path = os.path.join(output_directory,
                                   f"{sheet_name}_strain_{strain:.6f}_time_{time:.2f}.svg")
        plt.savefig(output_path, format="svg")
        plt.show()


# Run the updated function
for sheet_name in ["3.12", "3.56"]:
    plot_element_strain_distributions(
        sheet_name,
        sheets_and_nodes[sheet_name],
        elements[sheet_name],
        excel_file,
        strain_points,
        A,
        shear_displacement,
        output_directory
    )


def plot_strain_vs_length(actual_strains, element_lengths, output_directory, sheet_name, strain, color):

    # Plot element strains as a function of element length using a scatter plot.

    plt.figure(figsize=(6, 6))
    plt.scatter(element_lengths, actual_strains, alpha=0.8, color=color, edgecolors='black', label='Actual Strain')
    plt.xlabel('Fibril Length [$\mathregular{\mu}$m]', fontsize=14)
    plt.ylabel('Fibril Strain [-]', fontsize=14)  # Move y-label closer to the axis
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.grid(True, linestyle='--', alpha=0.7)
    output_path = os.path.join(output_directory, f"{sheet_name}_strain_vs_length{strain:.6f}.svg")
    plt.savefig(output_path, format="svg")
    plt.show()


def plot_element_strain_vs_length(sheet_name, nodes_files, element_files, excel_file, strain_points, A,
                                  shear_displacement, output_directory):
    
    # Generate scatter plots of element strains versus element length.
    
    samples_displacements = load_excel_displacements(excel_file, sheet_name)
    time_to_strain = {0.25: 0.0517578125, 0.5: 0.25, 1.0: 0.5}

    strain_colors = {
        "batch_1_0.0517578125": viridis_r(0.5),
        "batch_1_0.25": viridis_r(0.5),
        "batch_1_0.5": viridis_r(0.5),
        "batch_2_0.0517578125": viridis_r(0.7),
        "batch_2_0.25": viridis_r(0.7),
        "batch_2_0.5": viridis_r(0.7)
    }

    batch = "batch_1" if sheet_name == "3.12" else "batch_2"

    for time, strain in time_to_strain.items():
        if strain not in strain_points:
            continue

        combined_actual_strains = []
        combined_lengths = []

        for i, (node_file, elem_file) in enumerate(zip(nodes_files, element_files[sheet_name])):
            sample_name = f"sample_{i + 1}"
            nodes = load_nodes(node_file)
            elements_filtered = load_elements_filtered(elem_file, nodes)
            interpolation_map = calculate_interpolation_coefficients(nodes, A)

            sample_displacements = samples_displacements[sample_name]
            filtered_displacements = sample_displacements[sample_displacements["time"] == time]
            if filtered_displacements.empty:
                continue

            displacements = {idx + 1: [disp, 0, 0] for idx, disp in
                             enumerate(filtered_displacements["displacement"].values)}

            actual_strains, affine_strains, element_lengths = compute_element_strains(
                nodes, elements_filtered, displacements, interpolation_map, shear_displacement
            )

            combined_actual_strains.extend(actual_strains.values())
            combined_lengths.extend(element_lengths.values())

        if combined_actual_strains and combined_lengths:
            color = strain_colors[f"{batch}_{strain}"]
            plot_strain_vs_length(combined_actual_strains, combined_lengths, output_directory, sheet_name, strain,
                                  color)


# Iterate through sheets and execute the function
for sheet_name, nodes_files in sheets_and_nodes.items():
    plot_element_strain_vs_length(sheet_name, nodes_files, elements, excel_file, strain_points, A, shear_displacement,
                                  output_directory)


def plot_strain_vs_length_with_affinity(actual_strains, affine_strains, element_lengths, output_directory, sheet_name,
                                        strain, color):
    
     #Plot element strains (actual and affine) as a function of element length using a scatter plot.
     
    plt.figure(figsize=(6, 6))

    plt.scatter(element_lengths, actual_strains, alpha=0.8, color=color, edgecolors='black', label='Actual Strain')
    plt.scatter(element_lengths, affine_strains, alpha=0.2, color='grey', edgecolors='black', label='Affine Strain')
    plt.xlabel('Fibril Length [$\mathregular{\mu}$m]', fontsize=14)
    plt.ylabel('Fibril Strain [-]', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.grid(True, linestyle='--', alpha=0.7)
    output_path = os.path.join(output_directory, f"{sheet_name}_strain_vs_length_with_affinity_{strain:.6f}.svg")
    plt.savefig(output_path, format="svg")
    plt.show()

def plot_element_strain_vs_length_with_affinity(sheet_name, nodes_files, element_files, excel_file, strain_points, A,
                                  shear_displacement, output_directory):
    
    #Generate scatter plots of element strains versus element length.
    
    samples_displacements = load_excel_displacements(excel_file, sheet_name)
    time_to_strain = {0.25: 0.0517578125, 0.5: 0.25, 1.0: 0.5}

    strain_colors = {
        "batch_1_0.0517578125": viridis_r(0.5),
        "batch_1_0.25": viridis_r(0.5),
        "batch_1_0.5": viridis_r(0.5),
        "batch_2_0.0517578125": viridis_r(0.7),
        "batch_2_0.25": viridis_r(0.7),
        "batch_2_0.5": viridis_r(0.7)
    }

    batch = "batch_1" if sheet_name == "3.12" else "batch_2"

    for time, strain in time_to_strain.items():
        if strain not in strain_points:
            continue

        combined_actual_strains = []
        combined_affine_strains = []
        combined_lengths = []

        for i, (node_file, elem_file) in enumerate(zip(nodes_files, element_files[sheet_name])):
            sample_name = f"sample_{i + 1}"
            nodes = load_nodes(node_file)
            elements_filtered = load_elements_filtered(elem_file, nodes)
            interpolation_map = calculate_interpolation_coefficients(nodes, A)

            sample_displacements = samples_displacements[sample_name]
            filtered_displacements = sample_displacements[sample_displacements["time"] == time]
            if filtered_displacements.empty:
                continue

            displacements = {idx + 1: [disp, 0, 0] for idx, disp in
                             enumerate(filtered_displacements["displacement"].values)}

            actual_strains, affine_strains, element_lengths = compute_element_strains(
                nodes, elements_filtered, displacements, interpolation_map, shear_displacement
            )

            combined_actual_strains.extend(actual_strains.values())
            combined_affine_strains.extend(affine_strains.values())
            combined_lengths.extend(element_lengths.values())

        if combined_actual_strains and combined_lengths:
            color = strain_colors[f"{batch}_{strain}"]
            plot_strain_vs_length_with_affinity(combined_actual_strains, combined_affine_strains, combined_lengths, output_directory,
                                  sheet_name, strain, color)


# Iterate through sheets and execute the function
for sheet_name, nodes_files in sheets_and_nodes.items():
    plot_element_strain_vs_length_with_affinity(sheet_name, nodes_files, elements, excel_file, strain_points, A, shear_displacement,
                                  output_directory)

