"""
Create Network - Seeds Definition

Author: Sara Cardona
Date: 13/11/2023

To reproduce localized heterogeneity this script defines the spatial distribution of seeds which produce a preferential
directions in the fibers arrangement. A weighted seeding protocol is developed to achieve different levels of initial
anisotropy by defining the spatial distribution of the seeds. For this purpose, the domain is defined using a heatmap,
and the points are more likely to be seeded in the hot regions rather than the cold ones.

"""

import numpy as np
import matplotlib.pyplot as plt


def ComputeTemperature(points, a, distribution_type, cluster_assignments=None, cluster_centers=None):
    if distribution_type == 'homogeneous':
        return np.ones(len(points))
    elif distribution_type == 'random':
        return np.random.rand(len(points))
    elif distribution_type == 'planar':
        distance_from_plane = np.abs(points[:, 0])
        return 1 - distance_from_plane / a
    elif distribution_type == 'uneven':
        temp = np.zeros(len(points))
        temp[points[:, 0] < 0] = 1  # Densely packed region
        temp[points[:, 0] >= 0] = 0.5  # Spread-out region
        return temp
    elif distribution_type == 'radial':
        r = np.sqrt(np.sum(points ** 2, axis=1)) / a
        return 1 - (np.abs(r - 0.5) * 2) ** 2
    elif distribution_type == 'shell':
        return np.ones(len(points))
    elif distribution_type == 'layers':
        return np.ones(len(points))
    elif distribution_type == 'spiral':
        z_normalized = (points[:, 2] + a) / (2 * a)
        return z_normalized  # Gradient along the spiral
    elif distribution_type == 'cluster':
        temp = np.zeros(len(points))
        for i, point in enumerate(points):
            center = cluster_centers[cluster_assignments[i]]
            distance = np.sqrt(np.sum((center - point) ** 2))
            temp[i] = 1 - distance / (a * 0.3)
        return temp
    elif distribution_type == 'ellipsoid':
        temp = np.zeros(len(points))
        for i, point in enumerate(points):
            x, y, z = point
            value_inside_ellipsoid = (x / (7 * a)) ** 2 + (y / a) ** 2 + (z / a) ** 2
            temp[i] = 1 - value_inside_ellipsoid  # Invert and scale the temperature
        return temp


def PlotDistribution(points, temperatures, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=temperatures, cmap='inferno', s=40,
                         edgecolors='black', marker='o')
    cbar = plt.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label('Temperature')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30, y=0.95)
    plt.show()


def RandomDistribution(a, NumPoints):
    points = np.random.rand(NumPoints, 3) * 2 * a - a
    temperatures = ComputeTemperature(points, a, 'random')
    #PlotDistribution(points, temperatures, "Random Distribution")
    return points


def HomogenousDistribution(a, NumPoints):
    points = np.random.uniform(-a, a, (NumPoints, 3))
    temperatures = ComputeTemperature(points, a, 'homogeneous')
    #PlotDistribution(points, temperatures, "Homogeneous Distribution")
    return points


def PlanarZoneDistribution(a, NumPoints):
    points = []
    for _ in range(NumPoints):
        plane = np.random.choice([-(a), 0, (a)])
        x = plane
        y = np.random.uniform(-a / 2, a / 2)
        z = np.random.uniform(-a / 2, a / 2)
        points.append([x, y, z])
    points = np.array(points)
    temperatures = ComputeTemperature(points, a, 'planar')
    #PlotDistribution(points, temperatures, "Planar Distribution")
    return points


def UnevenDistribution(a, NumPoints):
    points = np.empty((NumPoints, 3))
    NumTightPoints = int(3 * NumPoints / 4)
    NumSpreadPoints = NumPoints - NumTightPoints

    points[:NumTightPoints, 0] = np.random.uniform(-a, 0.3 - a, NumTightPoints)
    points[:NumTightPoints, 1:] = np.random.uniform(-a, a, (NumTightPoints, 2))

    points[NumTightPoints:, 0] = np.random.uniform(0.3 - a, a, NumSpreadPoints)
    points[NumTightPoints:, 1:] = np.random.uniform(-a, a, (NumSpreadPoints, 2))

    temperatures = ComputeTemperature(points, a, 'uneven')
    #PlotDistribution(points, temperatures, "Uneven Distribution")
    return points


def RadialDistribution(a, NumPoints):
    points = []
    while len(points) < NumPoints:
        x, y, z = np.random.uniform(-a, a, 3)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2) / a

        pdf_value = 4 * r ** 2 * (1 - r) ** 2 if r <= 1 else 0

        if np.random.random() < pdf_value:
            points.append([x, y, z])

    points = np.array(points)
    temperatures = ComputeTemperature(points, a, 'radial')
    #PlotDistribution(points, temperatures, "Radial Distribution")
    return points


def SphericalShellDistribution(a, NumPoints):
    points = []
    while len(points) < NumPoints:
        x, y, z = np.random.uniform(-a, a, 3)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if a * 0.8 < r < a:
            points.append([x, y, z])
    points = np.array(points)
    temperatures = ComputeTemperature(points, a, 'shell')
    #PlotDistribution(points, temperatures, "Spherical Shell Distribution")
    return points


def VerticalLayersDistribution(a, NumPoints):
    points = []
    layers = [-a, -a/2, 0, a/2, a]  # Define layer positions
    for layer in layers:
        layer_points = NumPoints // len(layers)
        for _ in range(layer_points):
            x = np.random.uniform(-a, a)
            y = layer
            z = np.random.uniform(-a, a)
            points.append([x, y, z])
    points = np.array(points)
    temperatures = ComputeTemperature(points, a, 'layers')
    #PlotDistribution(points, temperatures, "Vertical Layers Distribution")
    return points


def SpiralDistribution(a, NumPoints):
    points = []
    theta_max = NumPoints / 10  # Controls the length of the spiral
    for i in range(NumPoints):
        theta = i / theta_max * np.pi
        r = a * (i / NumPoints)  # Radial distance depends on the point index
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = (i / NumPoints) * 2 * a - a
        points.append([x, y, z])
    points = np.array(points)
    temperatures = ComputeTemperature(points, a, 'spiral')
    #PlotDistribution(points, temperatures, "Spiral Distribution")
    return points


def ClusterDistribution(a, NumPoints):
    points = []
    num_clusters = 8
    cluster_centers = np.random.uniform(-a, a, (num_clusters, 3))
    cluster_assignments = []

    deviation_range = 0.3 * a

    for _ in range(NumPoints):
        cluster_idx = np.random.choice(range(num_clusters))
        cluster_assignments.append(cluster_idx)
        center = cluster_centers[cluster_idx]
        deviation = np.random.uniform(-deviation_range, deviation_range, 3)
        point = center + deviation
        points.append(point)

    points = np.array(points)
    temperatures = ComputeTemperature(points, a, 'cluster', cluster_assignments, cluster_centers)
    #PlotDistribution(points, temperatures, "Cluster Distribution")

    return points


def EllipsoidDistribution(a, NumPoints):
    points = []
    for _ in range(NumPoints):
        # Generate points in spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)

        # Convert to cartesian coordinates
        x = 3 * a * np.sin(phi) * np.cos(theta)
        y = a * np.sin(phi) * np.sin(theta)
        z = a * np.cos(phi)

        points.append([x, y, z])
    points = np.array(points)
    temperatures = np.ones(len(points))  # All points on the surface have the same temperature
    #PlotDistribution(points, temperatures, "Ellipsoid Distribution")
    return points




