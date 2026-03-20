from scipy.spatial import Voronoi, cKDTree
import numpy as np
from collections import defaultdict
import time
output_directory = None


def tile_points(points, N):
    """
    Tiles the original points in 3D for periodic boundary conditions.
    If anisotropic is True, the tiling offset in the aniso_axis direction is divided by aniso_compression.
    """
    if points.shape[0] != N:
        raise ValueError(f"Expected points array of length {N}, but got {points.shape[0]}")
    print("Tiling points...")
    import time
    start_time = time.time()
    point_tile = np.zeros((27 * N, 3))
    index = 0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                offset = np.array([x, y, z], dtype=float)
                # if anisotropic:
                #     offset[aniso_axis] /= aniso_compression
                segment = points + offset
                point_tile[index * N:(index + 1) * N] = segment
                index += 1
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
    - bounds: boundaries of the domain
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

def lloyd_relaxation_3d_periodic(points, iterations, N, relax):
    """
    Perform Lloyd's relaxation in 3D with periodic boundary conditions.
    """
    print("Lloyd relaxation..." if relax else "Skipping Lloyd relaxation...")
    import time
    start_time = time.time()
    if relax:
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
    point_tile = tile_points(points, N)
    vor = Voronoi(point_tile)
    elapsed_time = time.time() - start_time
    print(f"Time taken for Lloyd relaxation: {elapsed_time:.6f} seconds")
    return vor

import math
import numpy as np
from collections import defaultdict

def find_periodic_pairs_with_dim(vertices, bounds, atol=1e-8):
    """
    Find periodic pairs between opposite boundaries and return the face dimension.
    Uses per-dimension cell lengths for the normal separation check.

    Returns a list of tuples: (i_low, j_high, dim), where dim in {0,1,2}.
    """
    lengths = [abs(bounds[d][1] - bounds[d][0]) for d in range(3)]
    boundary_nodes = defaultdict(list)

    # Classify boundary nodes on each face
    for i, node in enumerate(vertices):
        for dim in range(3):
            if np.isclose(node[dim], bounds[dim][0], atol=atol):
                boundary_nodes[(dim, 'low')].append((i, node))
            elif np.isclose(node[dim], bounds[dim][1], atol=atol):
                boundary_nodes[(dim, 'high')].append((i, node))

    periodic_pairs = []
    paired_nodes = set()

    def is_counterpart(node1, node2, dim):
        other = [ax for ax in (0, 1, 2) if ax != dim]
        same_transverse = all(np.isclose(node1[ax], node2[ax], atol=atol) for ax in other)
        sep_normal = np.isclose(abs(node1[dim] - node2[dim]), lengths[dim], atol=atol)
        return same_transverse and sep_normal

    # Pair per dimension
    for dim in range(3):
        low_nodes = boundary_nodes[(dim, 'low')]
        high_nodes = boundary_nodes[(dim, 'high')]
        for i, node_low in low_nodes:
            if i in paired_nodes:
                continue
            for j, node_high in high_nodes:
                if j in paired_nodes:
                    continue
                if is_counterpart(node_low, node_high, dim):
                    periodic_pairs.append((i, j, dim))
                    paired_nodes.update([i, j])
                    break

    return periodic_pairs

def count_pairs_by_dim(pairs):
    counts = {0: 0, 1: 0, 2: 0}
    for _, _, dim in pairs:
        counts[dim] += 1
    return counts
    
def impose_preferential_orientation(
    nodes0,
    edges,
    V0=None,
    bounds=None,
    a=(1, 0, 0),
    P2_target=0.5,
    tol=1e-3,
    max_iter=50,
    length_weighted=False,
    expand_bracket_if_needed=True,
    pairwise_mode="copy_low"  # "copy_low" or "average"
):
    """
    Impose preferential orientation via uniaxial affine stretch and hydrostatic rescaling.
    Then enforce pairwise transverse motion of boundary nodes to preserve initial periodic mapping.

    Backward-compatible signature: accepts V0 (optional). If bounds are not provided,
    a cubic domain is derived from V0. If both are provided and inconsistent, bounds win.

    Prints the number of periodic pairs per direction (x,y,z) before and after.
    """

    # ----------------------
    # Helpers
    # ----------------------
    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    def segment_dir_len(nodes, i, j):
        d = nodes[j] - nodes[i]
        l = np.linalg.norm(d)
        if l == 0.0:
            return np.zeros(3), 0.0
        return d / l, l

    def orientation_P2(nodes, edges, a_unit, length_weighted):
        "Hermans parameter (P2) for the network along the user-defined axis"
        sum_w = 0.0
        sum_wc2 = 0.0
        for i, j in edges:
            n, l = segment_dir_len(nodes, i, j)
            if l == 0:
                continue
            c = float(np.dot(n, a_unit))
            w = l if length_weighted else 1.0
            sum_w += w
            sum_wc2 += w * c * c
        if sum_w == 0.0:
            return 0.0
        mean_c2 = sum_wc2 / sum_w
        return 0.5 * (3.0 * mean_c2 - 1.0)

    def rotation_matrix_mapping_ex_to_a(a_unit):
        "rotation matrix that maps the x-axis to the desired axis "
        ex = np.array([1.0, 0.0, 0.0])
        a_unit = np.asarray(a_unit)
        v = np.cross(ex, a_unit)
        s = np.linalg.norm(v)
        c = np.dot(ex, a_unit)
        if s == 0:
            return np.eye(3) if c > 0 else np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        vx = np.array([[0.0,   -v[2],  v[1]],
                       [v[2],  0.0,   -v[0]],
                       [-v[1], v[0],  0.0]])
        return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    def build_uniaxial_F(a_unit, s):
        R = rotation_matrix_mapping_ex_to_a(a_unit)
        lambda1 = math.exp(s)
        lambda2 = math.exp(-s / 2.0)
        D = np.diag([lambda1, lambda2, lambda2])  # det = lambda1 * lambda2^2 = 1
        F = R @ D @ R.T
        return F, lambda1, lambda2

    def apply_affine(nodes, F):
        return (F @ nodes.T).T

    def total_length(nodes, edges):
        return sum(np.linalg.norm(nodes[j] - nodes[i]) for i, j in edges)

    def density(nodes, edges, volume):
        return total_length(nodes, edges) / volume

    def hydrostatic_rescale(nodes, lambda_h):
        return nodes * lambda_h

    def count_pairs_by_dim(pairs):
        counts = {0: 0, 1: 0, 2: 0}
        for _, _, dim in pairs:
            counts[dim] += 1
        return counts

    # ----------------------
    # Resolve bounds and volume (backward compatibility)
    # ----------------------
    if bounds is None:
        L = 1.0 if V0 is None else float(V0) ** (1.0 / 3.0)
        half = L / 2.0
        bounds = ((-half, half), (-half, half), (-half, half))

    V_bounds = abs(bounds[0][1] - bounds[0][0]) * abs(bounds[1][1] - bounds[1][0]) * abs(bounds[2][1] - bounds[2][0])
    if V0 is None:
        V0 = V_bounds
    else:
        if abs(V_bounds - V0) > 1e-12:
            print(f"[warning] Provided V0={V0} differs from bounds volume={V_bounds}. Using bounds volume.")
            V0 = V_bounds

    # ----------------------
    # Setup
    # ----------------------
    nodes0 = np.asarray(nodes0, dtype=float)
    edges = np.asarray(edges, dtype=int)
    a_unit = normalize(a)

    # Periodic pairs BEFORE
    pairs_before = find_periodic_pairs_with_dim(nodes0, bounds)
    counts_before = count_pairs_by_dim(pairs_before)
    print(f"Periodic pairs BEFORE (strict boundary): x={counts_before[0]}, y={counts_before[1]}, z={counts_before[2]}")

    rho0 = density(nodes0, edges, V0)

    # ----------------------
    # Bisection on s to reach P2_target
    # ----------------------
    s_low, s_high = -4.0, 4.0

    def P2_of_s(s):
        F, _, _ = build_uniaxial_F(a_unit, s)
        nodes_tmp = apply_affine(nodes0, F)
        return orientation_P2(nodes_tmp, edges, a_unit, length_weighted)

    if expand_bracket_if_needed:
        P2_lo = P2_of_s(s_low)
        P2_hi = P2_of_s(s_high)
        expand_count = 0
        while not (P2_lo <= P2_target <= P2_hi) and expand_count < 8:
            s_low -= 2.0
            s_high += 2.0
            P2_lo = P2_of_s(s_low)
            P2_hi = P2_of_s(s_high)
            expand_count += 1

    nodes_oriented = nodes0.copy()
    s_star = 0.0
    lambda1_star = 1.0
    lambda2_star = 1.0

    for _ in range(max_iter):
        s_mid = 0.5 * (s_low + s_high)
        F_mid, lambda1_mid, lambda2_mid = build_uniaxial_F(a_unit, s_mid)
        nodes_mid = apply_affine(nodes0, F_mid)
        P2_mid = orientation_P2(nodes_mid, edges, a_unit, length_weighted)

        nodes_oriented = nodes_mid
        s_star = s_mid
        lambda1_star = lambda1_mid
        lambda2_star = lambda2_mid

        if abs(P2_mid - P2_target) <= tol:
            break
        if P2_mid < P2_target:
            s_low = s_mid
        else:
            s_high = s_mid

    # ----------------------
    # Hydrostatic scaling
    # ----------------------
    rho_after = density(nodes_oriented, edges, V0)
    lambda_h = 1.0 if rho_after == 0 else math.sqrt(rho_after / rho0)

    nodes_final = hydrostatic_rescale(nodes_oriented, lambda_h)

    # ----------------------
    # Enforce pairwise transverse motion
    # ----------------------
    if pairs_before:
        nodes_final = np.array(nodes_final, dtype=float)
        for i_low, j_high, dim in pairs_before:
            other = [ax for ax in (0,1,2) if ax != dim]
            if pairwise_mode == "copy_low":
                delta_trans = nodes_final[i_low, other] - nodes0[i_low, other]
                nodes_final[j_high, other] = nodes0[j_high, other] + delta_trans
            elif pairwise_mode == "average":
                avg = 0.5 * (nodes_final[i_low, other] + nodes_final[j_high, other])
                nodes_final[i_low, other] = avg
                nodes_final[j_high, other] = avg
            else:
                raise ValueError("pairwise_mode must be 'copy_low' or 'average'")

    # Count pairs AFTER (transverse-match only)
    counts_after_trans = {0: 0, 1: 0, 2: 0}
    for i_low, j_high, dim in pairs_before:
        other = [ax for ax in (0,1,2) if ax != dim]
        if np.allclose(nodes_final[i_low, other], nodes_final[j_high, other], atol=1e-8):
            counts_after_trans[dim] += 1
    print(f"Periodic pairs AFTER (transverse-match): x={counts_after_trans[0]}, y={counts_after_trans[1]}, z={counts_after_trans[2]}")

    # ----------------------
    # Strict boundary re-check in the NEW deformed domain (axis-aligned case)
    # ----------------------
    # Compute new axis-aligned bounds per axis using known scale factors
    Lx0 = abs(bounds[0][1] - bounds[0][0])
    Ly0 = abs(bounds[1][1] - bounds[1][0])
    Lz0 = abs(bounds[2][1] - bounds[2][0])

    Lx_new = lambda_h * lambda1_star * Lx0
    Ly_new = lambda_h * lambda2_star * Ly0
    Lz_new = lambda_h * lambda2_star * Lz0

    bounds_after = (
        (-0.5 * Lx_new, 0.5 * Lx_new),
        (-0.5 * Ly_new, 0.5 * Ly_new),
        (-0.5 * Lz_new, 0.5 * Lz_new),
    )

    pairs_after_strict = find_periodic_pairs_with_dim(nodes_final, bounds_after)
    counts_after_strict = count_pairs_by_dim(pairs_after_strict)
    print(f"Periodic pairs AFTER (strict boundary, new bounds): x={counts_after_strict[0]}, y={counts_after_strict[1]}, z={counts_after_strict[2]}")

    # ----------------------
    # Diagnostics
    # ----------------------
    P2_final = orientation_P2(nodes_final, edges, a_unit, length_weighted)
    V_final = V0 * (lambda_h ** 3)
    rho_final = density(nodes_final, edges, V_final)

    lambdas = {
        "lambda1": lambda1_star,
        "lambda2": lambda2_star,
        "lambda_h": lambda_h,
        "s_star": s_star,
        "pairwise_mode": pairwise_mode,
        "bounds_after": bounds_after,
    }

    return nodes_final, P2_final, rho_final, lambdas