import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib import cm

from matplotlib import rcParams, font_manager
font_path = 'D:/FONT/SourceSansPro-Regular.otf'
font_manager.fontManager.addfont(font_path)

from TopoGEN.utils.setup import setup_output_directory


def compute_bending_stretching_batch(base_dir, job_description_template, radius, scale_factor, num_samples):
    """
    Compute and overlay bending-to-stretching ratio histograms for multiple samples.

    Parameters:
        base_dir (str): Base directory containing the job description folders.
        job_description_template (str): Template for the job description with a placeholder for the sample number.
        radius (float): Radius of the cross-section of the fiber in the transverse direction.
        scale_factor (float): Scaling factor for the domain physical dimension.
        num_samples (int): Number of samples to process.
        output_directory (str): Directory where outputs are saved.

    Returns:
        None
    """
    # Initialize mean ratio array
    mean_ratios = []

    # Create colormap
    colormap = cm.get_cmap("viridis", num_samples)

    plt.figure(figsize=(10, 8))

    for sample_idx in range(1, num_samples + 1):
        # Prepare sample-specific job description
        job_description = job_description_template.replace("SAMPLE_NUMBER", f"Sample{sample_idx}")
        vertices_file_path = os.path.join(base_dir, job_description, "vertices.txt")
        edges_file_path = os.path.join(base_dir, job_description, "edges.txt")

        # Load vertices and edges
        vertices = np.loadtxt(vertices_file_path)
        edges = np.loadtxt(edges_file_path, dtype=int)

        # Scale vertices
        scaled_vertices = vertices * scale_factor

        # Compute edge lengths and bending-to-stretching ratios
        bending_stretching_ratios = []

        for edge in edges:
            node1, node2 = edge
            pos1 = scaled_vertices[node1]
            pos2 = scaled_vertices[node2]
            length = np.linalg.norm(pos2 - pos1)
            bending_stretching_ratio = (radius / length) ** 2
            bending_stretching_ratios.append(bending_stretching_ratio)

        # Calculate mean for this sample
        sample_mean_ratio = np.mean(bending_stretching_ratios)
        mean_ratios.append(sample_mean_ratio)

        # Plot histogram
        plt.hist(
            bending_stretching_ratios,
            bins=20,
            color=colormap(sample_idx - 1),
            alpha=0.6,
            label=f"Sample {sample_idx}",
            edgecolor="black",
        )

    # Finalize the histogram plot
    plt.xlabel("Bending-to-Stretching Ratio", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.title("Valency 3.36, Diameter 0.1",fontsize=20)
    #plt.savefig(os.path.join(output_directory, "bending_stretching_batch_histogram.png"))
    plt.show()

    # Save mean ratios and mean of means
    mean_ratios_array = np.array(mean_ratios)
    overall_mean = np.mean(mean_ratios_array)

    print(f"Mean Ratios Array: {mean_ratios_array}")
    print(f"Overall Mean of Ratios: {overall_mean:.6f}")


# Example Usage
today = datetime.datetime.now()
date_folder = today.strftime("%Y%m%d")
base_dir = os.path.join(r"D:\CollaGEN\AbaqusFiles", date_folder)

job_description_template = "Valency336\SAMPLE_NUMBER"


radius = 0.08
DomainPhysicalDimension = 32
num_samples = 5

# Compute and plot for all samples
compute_bending_stretching_batch(base_dir, job_description_template, radius, DomainPhysicalDimension, num_samples)


