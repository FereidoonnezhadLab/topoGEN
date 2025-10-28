"""
Create Network - Voronoi Tesselation

Author: Sara Cardona
Date: 12/11/2023

This script provides a method to create a simplified representation, or skeleton, of a 3D Voronoi diagram.
The seeds for the tesselation are obtained from SeedsDefinition.py.
The algorithm gathers the nodes from the 'vor.vertices' attribute and the connecting lines from the 'vor.ridges' attribute.
These lines form polygons around seed points input by the user. The algorithm excludes any lines linked to
virtual nodes, represented by '-1', which are part of the Voronoi tessellation but not actual connections.
For real connections, it pairs each successive point to form segments. For instance, the sequence [1, 0, 3, 2]
would result in segments 1-0, 0-3, 3-2, and 2-1 looping back to the start.

The script also corrects for very short lines by merging any points that are extremely close to each other,
ensuring a cleaner and more accurate skeleton. To conclude, ghost nodes and edges replica are removed.

"""

import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import Voronoi, cKDTree, ConvexHull, QhullError
from OldProject.CreateNetwork.SeedsDefinition import HomogenousDistribution
import plotly.io as pio
pio.renderers.default = 'browser'


def MergeCloseVertices(vertices, edges, MergeThreshold=0.1):
    """This function checks if the vertices are closer than the threshold
    and merge them to avoid the creation of tiny unrepresentative edges"""

    kdtree = cKDTree(vertices)
    PointsWithinThreshold = kdtree.query_ball_point(vertices, r=MergeThreshold)
    IndexMapping = {}
    NewVertices = []
    NewIndex = 0

    for idx, points in enumerate(PointsWithinThreshold):
        if idx not in IndexMapping:
            RepresentativePoint = vertices[idx]
            NewVertices.append(RepresentativePoint)
            IndexMapping[idx] = NewIndex

            # Merge the rest of the points with this representative point
            for point in points:
                if point != idx:  # Avoid self-merging
                    IndexMapping[point] = NewIndex
            NewIndex += 1

    NewEdges = set()
    for edge in edges:
        start, end = edge
        NewStart = IndexMapping.get(start, start)
        NewEnd = IndexMapping.get(end, end)
        if NewStart != NewEnd:  # Avoid self-connected edges
            NewEdges.add((NewStart, NewEnd))

    NewEdges = list(NewEdges)

    return np.array(NewVertices), np.array(NewEdges)


def RemoveGhost(vertices, edges):
    """Remove vertices that are not connected to any edge"""
    ConnectedVertices = set()
    for edge in edges:
        ConnectedVertices.update(edge)
    return np.array([vertices[i] for i in range(len(vertices)) if i in ConnectedVertices])


def ReplicaRemoval(edges):
    """Remove duplicate edges, preserving the first occurrence"""
    seen = set()
    NewEdges = []
    for edge in edges:
        EdgeTuple = tuple(sorted(edge))
        if EdgeTuple not in seen:
            seen.add(EdgeTuple)
            NewEdges.append(edge)
    return NewEdges

a = 0.5
N = 50
points = HomogenousDistribution(a, N)


def FindIntersection(start, end, bounds):
    """
    Find the intersection of a line segment with a bounding box in 3D.
    """
    BoundaryMin, BoundaryMax = bounds
    direction = end - start
    intersections = []

    for dim in range(3):
        if direction[dim] != 0:
            for boundary in [BoundaryMin, BoundaryMax]:
                t = (boundary - start[dim]) / direction[dim]
                if 0 <= t <= 1:
                    intersection = start + t * direction
                    # Check if intersection is within bounds for other dimensions
                    if all(BoundaryMin <= intersection[other_dim] <= BoundaryMax for other_dim in range(3) if other_dim != dim):
                        intersections.append(intersection)
                        break  # Only need the first valid intersection

    # Return the intersection closest to the start point
    if intersections:
        return intersections[0]
    return None


def RelaxPoints3D(points, iterations, bounds=[-0.5, 0.5]):
    """
    Relaxes points in 3D space within the given bounds for the specified number of iterations.
    """
    BoundaryMin, BoundaryMax = bounds

    for iteration in range(iterations):
        vor = Voronoi(points)
        new_points = []

        for point_index, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region:  # Skip empty regions
                continue

            vertices = vor.vertices[region]
            ridge_vertices = [vor.ridge_vertices[i] for i in range(len(vor.ridge_vertices)) if
                              point_index in vor.ridge_points[i]]

            # Calculate the constrained centroid for the region
            centroid = calculate_constrained_centroid(ridge_vertices, vertices, bounds)
            if centroid is not None:
                new_points.append(centroid)
            else:
                # If centroid is None, the region is outside bounds, keep the original point
                new_points.append(points[point_index])

        points = np.array(new_points)
        print(f"Iteration {iteration + 1}/{iterations}: Updated centroids.")

    return points


def calculate_constrained_centroid(ridge_vertices, vertices, bounds):
    """
    Calculate the centroid of a Voronoi cell that may be partially outside of the given bounds,
    considering only the part of the cell within the bounds.
    """
    BoundaryMin, BoundaryMax = bounds
    vertices_positions = []  # List to hold vertices positions that are inside the bounds
    intersection_points = []  # List to hold intersection points

    # Process each edge of the ridge
    for edge in ridge_vertices:
        # Filter out '-1' but process edges with only one '-1'
        if edge[0] != -1 and edge[1] != -1:
            valid_edge = edge
        elif edge[0] == -1 and 0 <= edge[1] < len(vertices):
            valid_edge = [edge[1]]
        elif edge[1] == -1 and 0 <= edge[0] < len(vertices):
            valid_edge = [edge[0]]
        else:
            continue  # Skip if both vertices are at infinity

        # Check if the edge vertices are within bounds
        for i in valid_edge:
            if all(BoundaryMin <= vertices[i][dim] <= BoundaryMax for dim in range(3)):
                vertices_positions.append(vertices[i])

        # Calculate the intersection if one of the vertices is at infinity
        if len(valid_edge) == 1:
            # Assume the point at infinity is in the direction of the domain center
            domain_center = [(BoundaryMax + BoundaryMin) / 2.0] * 3
            intersection = FindIntersection(vertices[valid_edge[0]], domain_center, bounds)
            if intersection is not None:
                intersection_points.append(intersection)

    # Combine vertices within bounds and intersection points
    combined_vertices = vertices_positions + intersection_points
    if combined_vertices:
        try:
            hull = ConvexHull(np.array(combined_vertices))
            centroid = np.mean(hull.points[hull.vertices], axis=0)
            return centroid
        except QhullError:
            return np.mean(np.array(combined_vertices), axis=0)
    return None


vor = Voronoi(points, qhull_options='Qbb Qc Qz Qx')
EdgesList = []
for ridge in vor.ridge_vertices:
    if -1 not in ridge: # Process only the ridges that do not contain the point at infinity (-1)
        for i in range(len(ridge) - 1):
            EdgesList.append([ridge[i], ridge[i + 1]])  # Close the loop (also first and last element are connected)
        if len(ridge) > 2 and ridge[-1] != ridge[0]:
            EdgesList.append([ridge[-1], ridge[0]])

EdgesArray = np.array(EdgesList)
BoundaryLimits = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

""" This part is related to the visualization of the voronoi 
Alert: it contains long edges outside the boundaries but this is not an issue
they will be cropped in the next section of the Project (ClipBoundaryEdges) """

lines = []
for edge in EdgesArray:
    lines.append([vor.vertices[edge[0]], vor.vertices[edge[1]]])
lc = Line3DCollection(lines, colors='b', linewidths=0.5)

scatter = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                       mode='markers', marker=dict(size=5, color='red'))

# Create a line plot for the Voronoi edges
lines = []
for edge in EdgesArray:
    x0, y0, z0 = vor.vertices[edge[0]]
    x1, y1, z1 = vor.vertices[edge[1]]
    lines.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                              mode='lines', line=dict(color='blue', width=2)))



# Combine plots
fig = go.Figure(data=[scatter, *lines])

# Update layout
fig.update_layout(scene=dict(
    xaxis=dict(nticks=4, range=[-0.5, 0.5]),
    yaxis=dict(nticks=4, range=[-0.5, 0.5]),
    zaxis=dict(nticks=4, range=[-0.5, 0.5]),
    aspectmode='cube'
))

# Show figure
fig.show()


# Create a scatter plot for the points
scatter = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                       mode='markers', marker=dict(size=5, color='red'))

# Create a line plot for the Voronoi edges
lines = []
for edge in EdgesArray:
    x0, y0, z0 = vor.vertices[edge[0]]
    x1, y1, z1 = vor.vertices[edge[1]]
    lines.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                              mode='lines', line=dict(color='blue', width=2)))

# Combine plots
fig = go.Figure(data=[scatter, *lines])

# Update layout without restricting the boundary
fig.update_layout(scene=dict(
    xaxis=dict(nticks=4),
    yaxis=dict(nticks=4),
    zaxis=dict(nticks=4),
    aspectmode='cube'
))

# Show figure
fig.show()

# Before saving the edges merge the nodes closer than the threshold
MergedVertices, MergedEdges = MergeCloseVertices(vor.vertices, EdgesList)
UpdatedVertices = RemoveGhost(MergedVertices, MergedEdges)
UpdatedEdges = ReplicaRemoval(MergedEdges)

with open('CreateNetwork/vertices.txt', 'w') as file:
    for vertex in UpdatedVertices:
        file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
with open('CreateNetwork/edges.txt', 'w') as file:
    for edge in UpdatedEdges:
        file.write(f"{edge[0]} {edge[1]}\n")
