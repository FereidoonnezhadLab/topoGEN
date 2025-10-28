"""
Author: Sara Cardona
Date: 19/01/2024
Refactor: 04/02/2024
"""
from scipy.spatial import Voronoi, cKDTree
import numpy as np
from collections import defaultdict
import time
output_directory = None


def tile_points(points, N):
    """
    This function takes the original points and creates 26 additional copies of these points, each offset to one of the
    surrounding areas in the 3D space. This tiling is used to handle the PBCs for the Voronoi diagram.
    Args:
    - points: position of the initial seeds in the 3D space
    - N: number of seeds

    Returns:
    - point_tile: original points + their 26 replica
    """
    if points.shape[0] != N:
        raise ValueError(f"Expected points array of length {N}, but got {points.shape[0]}")
    print("Tiling points...")
    # Start the timer
    start_time = time.time()
    point_tile = np.zeros((27 * N, 3))  # 27 times the original points (3x3x3 grid)
    index = 0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                offset = np.array([x, y, z])
                segment = points + offset
                point_tile[index * N:(index + 1) * N] = segment
                index += 1
    # End the timer and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken for tiling: {elapsed_time:.6f} seconds")
    return point_tile


def get_vertices(tile_vertices):
    """
    Filters the Voronoi vertices to include only those inside the original unit square (0 to 1 in x and y).
    It also creates a mapping from the original vertex indices to the new filtered list indices.

    Args:
    - tile_vertices: all the vertices of the original domain and their 26 replica

    Returns:
    - vertices
    - index map maps the original indices of vertices (from tile_vertices) to their new indices
    """
    print("Filtering vertices...")
    # Start the timer
    start_time = time.time()
    vertices = []
    index_map = {}

    for i, (x, y, z) in enumerate(tile_vertices):
        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
            curr_index = len(vertices)
            index_map[i] = curr_index
            vertices.append((x, y, z))

    if len(index_map) == 0:
        print("No vertices within the bounds [-0.5, 0.5]")
    elapsed_time = time.time() - start_time
    print(f"Time taken for Filtering Vertices: {elapsed_time:.6f} seconds")
    return np.array(vertices), index_map


def calculate_intersection(v_inside, v_outside, bounds):
    """
    Calculate the intersection of the line segment defined by v_inside and v_outside with the domain boundary.
    Args:
    - v_inside: vertex located inside the boundary
    - v_outside: vertex located outside the boundary
    - bounds: boundary of the domain

    Returns:
    - intersection: intersection of the line segment defined by v_inside and v_outside
    """
    for dim in range(3):
        if v_outside[dim] < bounds[dim][0] or v_outside[dim] > bounds[dim][1]:
            direction = v_outside - v_inside
            t = (bounds[dim][int(v_outside[dim] > bounds[dim][1])] - v_inside[dim]) / direction[dim]
            intersection = v_inside + t * direction
            if all(bounds[d][0] <= intersection[d] <= bounds[d][1] for d in range(3)):
                return intersection
    return None


def get_edges(vertices, tile_vertices, edges, index_map, bounds):
    """
    Optimized function to process Voronoi diagram edges with faster data handling.

    Args:
    - vertices: List of original vertex positions as (x, y, z) coordinates.
    - tile_vertices: List of tiled vertices, representing 26 replica regions.
    - edges: List of connections between vertex indices, defining the edges.
    - index_map: Dictionary mapping original vertex indices (from tile_vertices) to new indices.
    - bounds: Domain boundaries used to compute intersection points for boundary edges.

    Returns:
        - new_vertices (np.ndarray): An array of unique vertex positions, including original and boundary intersection points.
        - all_edges (np.ndarray): An array of edges, where each row represents a pair of connected vertex indices.
    """

    print("Processing edges...")
    start_time = time.time()

    # Convert vertices and TileVertices to np.array at the start for faster element access
    vertices = np.array(vertices)
    tile_vertices = np.array(tile_vertices)

    # Pre-allocate storage for edges
    regular_edges = np.empty((0, 2), dtype=int)
    boundary_edges = np.empty((0, 2), dtype=int)

    # Convert new_vertices_list to dictionary for faster lookup in FindAddVertex
    vertex_dict = {tuple(vertex): idx for idx, vertex in enumerate(vertices.tolist())}
    vertex_list = list(vertex_dict.keys())  # to maintain index order

    for edge in edges:
        i1, i2 = edge
        inside_1, inside_2 = i1 in index_map, i2 in index_map

        if inside_1 and inside_2:
            regular_edges = np.vstack([regular_edges, [index_map[i1], index_map[i2]]])
            regular_edges = np.vstack([regular_edges, [index_map[i2], index_map[i1]]])

        elif inside_1 or inside_2:
            if inside_1:
                v_inside, v_outside = vertices[index_map[i1]], tile_vertices[i2]
                index_inside = index_map[i1]
            else:
                v_inside, v_outside = vertices[index_map[i2]], tile_vertices[i1]
                index_inside = index_map[i2]

            # Intersection point
            intersection = calculate_intersection(v_inside, v_outside, bounds)
            if intersection is not None:
                intersection_tuple = tuple(intersection)
                if intersection_tuple not in vertex_dict:
                    vertex_dict[intersection_tuple] = len(vertex_list)
                    vertex_list.append(intersection_tuple)

                intersection_index = vertex_dict[intersection_tuple]
                boundary_edges = np.vstack([boundary_edges, [index_inside, intersection_index]])

    # Convert final edges list to arrays
    all_edges = np.vstack([regular_edges, boundary_edges])
    new_vertices = np.array(vertex_list)

    elapsed_time = time.time() - start_time
    print(f"Time taken for Processing Edges: {elapsed_time:.6f} seconds")
    return new_vertices, all_edges


def process_edges(ridge_vertices, vertices):
    """Process the Voronoi ridges to extract edges, avoiding edges that cross the specified boundaries.
    This is because the tessellation method may create very long edges extended to infinity


    Args:
    - ridge_vertices (list): list of vertices defining the ridge edges
    - vertices (list): list of vertices

    Returns:
    - edges (list): list of edges

    """

    print("Processing Voronoi ridges...")
    # Start the timer
    start_time = time.time()
    edges_list = []
    boundary_min = -1.5
    boundary_max = 1.5

    for ridge in ridge_vertices:
        if -1 not in ridge:  # Exclude edges connected to the point at infinity
            for i in range(len(ridge) - 1):
                start, end = ridge[i], ridge[i + 1]
                if all(boundary_min <= vertices[start][dim] <= boundary_max and
                       boundary_min <= vertices[end][dim] <= boundary_max for dim in range(3)):
                    edges_list.append([start, end])
            if len(ridge) > 2 and ridge[-1] != ridge[0]:
                start, end = ridge[-1], ridge[0]
                if all(boundary_min <= vertices[start][dim] <= boundary_max and
                       boundary_min <= vertices[end][dim] <= boundary_max for dim in range(3)):
                    edges_list.append([start, end])  # Close the loop, but only if within boundaries
    # End the timer and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken for Processing Voronoi ridges: {elapsed_time:.6f} seconds")
    return np.array(edges_list)


# PAY ATTENTION: The following function creates higher connectivity (isostatic networks!)
def merge_close_vertices(vertices, edges, bounds, merge_threshold, max_degree):
    """This function checks if the vertices are closer than the threshold and merge them to avoid the creation of tiny
    unrepresentative edges. Vertices on the boundary are not merged, and merging only occurs if the resulting nodal
    degree is less than 12.

    Args:
    - vertices: position of the vertices
    - edges: edges of the Voronoi ridges
    - bounds: boundaries of the cubic domain
    - merge_threshold: threshold for merging edges
    - max_degree: maximum degree of the edges (maximum valency)

    Returns:
    - new_edges
    - new_vertices
    """

    print("Merging close vertices...")

    import networkx as nx

    # Build a graph from the edges
    graph = nx.Graph()
    graph.add_edges_from(edges)
    # Use KDTree for fast spatial queries to find close vertices
    kdtree = cKDTree(vertices)
    points_within_threshold = kdtree.query_ball_point(vertices, r=merge_threshold)
    index_mapping = {}
    new_vertices = []
    new_index = 0

    # Helper function to check if a vertex is on the boundary
    def is_on_boundary(vertex, bounds):
        return any(vertex[dim] == bounds[dim][0] or vertex[dim] == bounds[dim][1] for dim in range(3))

    # First pass: Handle merging of vertices
    for idx, points in enumerate(points_within_threshold):
        if idx not in index_mapping and not is_on_boundary(vertices[idx], bounds):
            representative_point = vertices[idx]
            combined_degree = sum(graph.degree[pt] for pt in points if pt in graph)

            # Only merge if combined degree is less than MaxDegree
            if combined_degree < max_degree:
                new_vertices.append(representative_point)
                index_mapping[idx] = new_index

                # Merge the rest of the points with this representative point
                for point in points:
                    if point != idx and not is_on_boundary(vertices[point],
                                                           bounds):  # Avoid self-merging and merging boundary points
                        index_mapping[point] = new_index
                new_index += 1
            else:
                # If not merging, treat as separate
                if idx not in index_mapping:
                    new_vertices.append(vertices[idx])
                    index_mapping[idx] = new_index
                    new_index += 1
        elif is_on_boundary(vertices[idx], bounds):
            # If it's a boundary vertex, it becomes its own representative point
            if idx not in index_mapping:
                new_vertices.append(vertices[idx])
                index_mapping[idx] = new_index
                new_index += 1

    # Second pass: Create new edges based on remapped vertices
    new_edges = set()
    for start, end in edges:
        if start in index_mapping and end in index_mapping:
            new_start = index_mapping[start]
            new_end = index_mapping[end]
            if new_start != new_end:  # Avoid self-connected edges
                new_edges.add((new_start, new_end))

    new_edges = list(new_edges)

    return np.array(new_vertices), np.array(new_edges)


def replica_removal(edges):

    """Remove duplicate edges, preserving the first occurrence.

    Args:
    - edges: all the edges in the RVE that may contain some replica (same node index for multiple edges)

    Returns:
    - new_edges without replica

    """

    print("Removing duplicate edges...")
    # Start the timer
    start_time = time.time()
    seen = set()
    new_edges = []
    for edge in sorted(edges, key=lambda e: (min(e), max(e))):
        edge_tuple = tuple(sorted(edge))
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            new_edges.append(edge)  # Append the original edge to maintain the direction
    # End the timer and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken for Removing duplicate edges: {elapsed_time:.6f} seconds")
    return new_edges


def find_periodic_pairs(vertices, bounds):
    """
    Optimized function to find periodic pairs among nodes on boundaries.

    Args:
    - vertices: all the nodes in the RVE
    - bounds: RVE bounds

    Returns:
    - periodic pairs of nodes located at the opposing boundaries.
    """
    print("Finding periodic pairs...")
    start_time = time.time()

    # Pre-calculate cube length and store boundary nodes
    cube_length = abs(bounds[0][1] - bounds[0][0])
    boundary_nodes = defaultdict(list)

    # Classify nodes based on boundary presence in each dimension
    for i, node in enumerate(vertices):
        for dim in range(3):
            if np.isclose(node[dim], bounds[dim][0], atol=1e-8):
                boundary_nodes[(dim, 'low')].append((i, node))
            elif np.isclose(node[dim], bounds[dim][1], atol=1e-8):
                boundary_nodes[(dim, 'high')].append((i, node))

    periodic_pairs = []
    paired_nodes = set()

    def is_counterpart(node1, node2, dim):
        """Check if two nodes are counterparts along a given dimension."""
        if dim == 0:
            return np.isclose(node1[1], node2[1]) and np.isclose(node1[2], node2[2]) and \
                   np.isclose(abs(node1[0] - node2[0]), cube_length)
        elif dim == 1:
            return np.isclose(node1[0], node2[0]) and np.isclose(node1[2], node2[2]) and \
                   np.isclose(abs(node1[1] - node2[1]), cube_length)
        else:
            return np.isclose(node1[0], node2[0]) and np.isclose(node1[1], node2[1]) and \
                   np.isclose(abs(node1[2] - node2[2]), cube_length)

    # Process pairs within each boundary dimension
    for dim in range(3):
        low_boundary_nodes = boundary_nodes[(dim, 'low')]
        high_boundary_nodes = boundary_nodes[(dim, 'high')]

        # Check low-boundary nodes against high-boundary nodes
        for i, node1 in low_boundary_nodes:
            if i in paired_nodes:
                continue
            for j, node2 in high_boundary_nodes:
                if j in paired_nodes:
                    continue
                if is_counterpart(node1, node2, dim):
                    periodic_pairs.append((i, j))
                    paired_nodes.update([i, j])
                    break

    # End the timer and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Time taken for Finding periodic pairs: {elapsed_time:.6f} seconds")
    return periodic_pairs


def calculate_mean_length(vertices, edges):
    lengths = []
    for edge in edges:
        point1 = vertices[edge[0]]
        point2 = vertices[edge[1]]
        length = np.linalg.norm(point2 - point1)
        lengths.append(length)
    return np.mean(lengths) if lengths else 0


def lloyd_relaxation_3d_periodic(points, iterations, N):
    """
    Perform Lloyd's relaxation in 3D with periodic boundary conditions.

    Args:
    - points: Initial seed points in the main computational domain.
    - iterations: Number of Lloyd iterations to perform.
    - N: Number of points in the main computational domain.

    Returns:
    - The relaxed points in the main computational domain after the specified number of iterations.
    """
    print("Lloyd relaxation...")
    # Start the timer
    start_time = time.time()
    for i in range(iterations):
        point_tile = tile_points(points, N)
        vor = Voronoi(point_tile)
        new_points = []
        for point_idx in range(N):
            region_index = vor.point_region[point_idx]
            region = vor.regions[region_index]

            if -1 not in region and len(region) > 0:
                polygon = vor.vertices[region]
                centroid = polygon.mean(axis=0)
                centroid = np.mod(centroid, 1)
                new_points.append(centroid)
            else:
                new_points.append(points[point_idx])
        points = np.array(new_points)
    elapsed_time = time.time() - start_time
    print(f"Time taken for Lloyd relaxation: {elapsed_time:.6f} seconds")
    return points, vor

