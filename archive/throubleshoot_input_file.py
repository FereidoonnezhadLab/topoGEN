import os
import datetime
import numpy as np
import math
import plotly.io as pio
import random
import matplotlib.pyplot as plt
"""
-------------------------------------------- Directory Definition ------------------------------------------------------
"""
# Global variable to store the output directory
output_directory = None

#%%
def setup_output_directory(job_description):
    today = datetime.datetime.now()
    date_folder = today.strftime("%Y%m%d")
    base_dir = r'D:\CollaGEN\AbaqusFiles'
    output_directory = os.path.join(base_dir, date_folder, job_description)
    os.makedirs(output_directory, exist_ok=True)
    return output_directory

job_description = "Extras\ConcentrationEffect\Valency356\High\Sample20"
output_directory = setup_output_directory(job_description)
vertices_file_path = os.path.join(output_directory, "vertices.txt")
edges_file_path = os.path.join(output_directory, "edges.txt")
periodic_edges_file_path = os.path.join(output_directory, "periodic_edges.txt")
##
"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
---------------------------------------------INPUT PARAMETERS ----------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

DomainPhysicalDimension = 40

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------NODES INP FILE DEFINITION ----------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

# Define the boundary limits
boundary_limits = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5)
}


def RemoveEmptyLines(file_path):
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.writelines(line for line in lines if line.strip())
        file.truncate()


# INPUT: read node from a txt file
# (two columns: one for each size, no header row)
XCoords = []
YCoords = []
ZCoords = []

with open(vertices_file_path, "r") as nodes_txt:
    for row in nodes_txt:
        cols = row.rstrip().split()
        XCoords.append(float(cols[0]) * DomainPhysicalDimension)
        YCoords.append(float(cols[1]) * DomainPhysicalDimension)
        ZCoords.append(float(cols[2]) * DomainPhysicalDimension)

# Pack data into numpy datastructures
xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T

# Calculate distances to the origin
distances = np.sqrt(np.sum(nodes ** 2, axis=1))

# Find the index of the node closest to the origin
StabilizationNodeId = np.argmin(distances) + 1

NodesNumber = len(XCoords)

# Define the path to the temporary nodes file and the final nodes file within the specified directory
temp_nodes_file_path = os.path.join(output_directory, "temp_nodes.inp")
final_nodes_file_path = os.path.join(output_directory, "nodes.inp")

# Create and populate temp_nodes.inp file
with open(temp_nodes_file_path, "w") as file:
    for i in range(NodesNumber):
        file.write(f"\n{i + 1}, {XCoords[i]:.9f}, {YCoords[i]:.9f}, {ZCoords[i]:.9f}")

# Remove empty lines from the created file
RemoveEmptyLines(temp_nodes_file_path)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
----------------------------------------ELEMENTS INP FILE DEFINITION ---------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""


def ElementOrientationDefinition(p1, p2, offset_distance):
    """
    Calculate the coordinates of the point that lies on the perpendicular line to the straight line
    passing through its midpoint and at the specified offset distance from the straight line.

    :param p1: tuple or list containing the (x,y,z) coordinates of the first endpoint of the straight line
    :param p2: tuple or list containing the (x,y,z) coordinates of the second endpoint of the straight line
    :param offset_distance: offset distance from the straight line
    :return: tuple containing the (x,y,z) coordinates of the offset point
    """

    # Calculate the direction vector of the straight line
    direction = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])

    # Calculate the unit direction vector of the straight line
    unit_direction = direction / np.linalg.norm(direction)

    # Calculate the perpendicular vector to the straight line
    if abs(unit_direction[2]) < 1e-8:
        # If the unit direction vector is parallel to the z-axis, choose a different vector
        if abs(unit_direction[1]) < 1e-8:
            perp_vector = np.array([0, 1, 0])
        else:
            perp_vector = np.array([1, -unit_direction[0] / unit_direction[1], 0])
    else:
        perp_vector = np.array([1, 1, -(unit_direction[0] + unit_direction[1]) / unit_direction[2]])

    # Calculate the unit perpendicular vector to the straight line
    unit_perp_vector = perp_vector / np.linalg.norm(perp_vector)

    # Calculate the offset point
    offset_point = p1 + offset_distance * unit_perp_vector

    return offset_point


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2]


# INPUT: read node connectivity from a txt file
# (two columns: one for each size, no header row)
ElementNodes1 = []
ElementNodes2 = []

elements_file_path = os.path.join(output_directory, "elements.inp")

with open(edges_file_path, "r") as connectivity_txt:
    for row in connectivity_txt:
        cols = row.rstrip().split()
        ElementNodes1.append(int(cols[0]))
        ElementNodes2.append(int(cols[1]))

# Pack data into numpy datastructures
nodes1, nodes2 = np.array(ElementNodes1), np.array(ElementNodes2)
elements = np.vstack([nodes1, nodes2]).T
ElementsNumber = len(elements)

with open(elements_file_path, "w") as file:
    for i, (node1, node2) in enumerate(elements):
        file.write(" \n %i, %i, %i \n " % ((i + 1), node1 + 1, node2 + 1))

RemoveEmptyLines(elements_file_path)

with open(temp_nodes_file_path, 'r') as f:
    NodesLines = f.readlines()

LastNodeID = None
for line in reversed(NodesLines):
    if line.strip():
        LastNodeID = int(line.split(',')[0])
        break
OriginalNodesNumber = LastNodeID

# read in the elements.inp file
with open(elements_file_path, 'r') as f:
    ElementLines = f.readlines()

element_type = 2

# Initialize lists to store new nodes
OffsetNodes = []
MidpointNodes = []

# Loop through the elements and compute the offset node for each line
OffsetNodeID = []
MidpointsNodeID = []
for i, line in enumerate(ElementLines):
    # Compute the offset node that defines the first normal vector
    Node1ID, Node2ID = map(int, line.strip().split(',')[1:])
    Node1Coords = [float(x) for x in NodesLines[Node1ID - 1].strip().split(',')[1:]]
    Node2Coords = [float(x) for x in NodesLines[Node2ID - 1].strip().split(',')[1:]]
    OffsetDistance = 1
    OffsetCoords = ElementOrientationDefinition(Node1Coords, Node2Coords, OffsetDistance)
    OffsetCoords = ElementOrientationDefinition(Node1Coords, Node2Coords, OffsetDistance)
    OffsetCoords = [x for x in OffsetCoords]

    LastNodeID += 1
    OffsetNodeID.append(LastNodeID)
    OffsetNodes.append(
        f"\n{LastNodeID}, {format(OffsetCoords[0], '.9')}, {format(OffsetCoords[1], '.9')}, {format(OffsetCoords[2], '.9')}")

    if element_type == 2:
        MidpointCoords = midpoint(Node1Coords, Node2Coords)
        MidpointCoords = [x for x in MidpointCoords]

        LastNodeID += 1
        MidpointsNodeID.append(LastNodeID)
        MidpointNodes.append(
            f"\n{LastNodeID}, {format(MidpointCoords[0], '.9')}, {format(MidpointCoords[1], '.9')}, {format(MidpointCoords[2], '.9')}")

# Write new nodes to the final nodes file
with open(final_nodes_file_path, 'w') as f:
    f.writelines(NodesLines)  # Write original nodes
    f.writelines(OffsetNodes)  # Write offset nodes
    f.writelines(MidpointNodes)  # Write midpoint nodes

# read in the original elements.inp file
with open(elements_file_path, 'r') as f:
    ElementLines = f.readlines()

# loop through the elements and insert the ID of the offset node after node2
for i, line in enumerate(ElementLines):
    Node1ID, Node2ID = map(int, line.strip().split(',')[1:])
    if element_type == 1:
        new_line = f"{i + 1}, {Node1ID}, {Node2ID}, {OffsetNodeID[i]}\n"
        ElementLines[i] = new_line
    if element_type == 2:
        new_line = f"{i + 1}, {Node1ID}, {MidpointsNodeID[i]},{Node2ID}, {OffsetNodeID[i]}\n"
        ElementLines[i] = new_line

# write the updated elements.inp file
with open(elements_file_path, 'w') as f:
    f.writelines(ElementLines)

RemoveEmptyLines(final_nodes_file_path)
RemoveEmptyLines(elements_file_path)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
----------------------------------------FIBER VOLUME FRACTION DEFINITION------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T
nodes1, nodes2 = np.array(ElementNodes1), np.array(ElementNodes2)
elements = np.vstack([nodes1, nodes2]).T


def EdgesLength(nodes, edges):
    """
    Calculates the lengths of edges given node coordinates and edges.
    """
    EdgeLength = []
    for edge in edges:
        length = np.linalg.norm(nodes[edge[0]] - nodes[edge[1]])
        EdgeLength.append(length)
    return EdgeLength


def ComputeVolumeFraction(TotalFiberLength, fiber_radius, RVE_size):
    """
    Computes the fiber volume fraction.
    Parameters:
    - TotalFiberLength: The total length of fibers in micrometers.

    Returns:
    - volume_fraction: The fraction of the domain volume occupied by fibers.
    """
    fiber_area = np.pi * (fiber_radius) ** 2
    total_fiber_volume = fiber_area * TotalFiberLength
    rve_volume = RVE_size ** 3  # Volume of the domain in physical units

    volume_fraction = total_fiber_volume / rve_volume

    return volume_fraction


total_length = sum(EdgesLength(nodes, elements))
mean_length = np.mean(EdgesLength(nodes, elements))

fiber_radius = 0.08
phi = ComputeVolumeFraction(total_length, fiber_radius, DomainPhysicalDimension)

# Display results (Optional for debugging/verification)
print(f"Volume Fraction: {phi}")
concentration = phi * 1000 / 0.73
print(f"concentration: {concentration}")

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
-------------------------------------------TRANSVERSE SHEAR DEFINITION--------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T
nodes1, nodes2 = np.array(ElementNodes1), np.array(ElementNodes2)
elements = np.vstack([nodes1, nodes2]).T


def ComputeTransvereShear(radius, length, E, nu):
    """
    :param radius: Radius of the beam's cross-section.
    :param length: Average length of the beams in the network.
    :param E: Young Modulus
    :param nu: Poisson Ratio
    """
    if element_type == 1:
        element_order = 1  # 1 for B31 beams
    elif element_type == 2:
        element_order = 1E-4  # 1E-4 for B32 beams

    shear_factor = 0.89  # for circular cross-section
    slenderness_compensation = 0.25  # default Abaqus value

    area = math.pi * (radius ** 2)
    shear_mod = E / (2 * (1 + nu))
    inertia = math.pi * (radius ** 4) / 4  # Inertia based on radius

    c1 = element_order * slenderness_compensation * (length ** 2) * area / (12 * inertia)
    f = 1 / (
            1 + c1)  # dimensionless factor used to prevent the shear stiffness from becoming too large in slender beam elements

    actual_shear = shear_factor * shear_mod * area
    transverse_shear = actual_shear * f

    return f, transverse_shear


YoungModulus = 600 # MPa
PoissonRatio = 0.0
SlendernessCompensation, TransverseShear = ComputeTransvereShear(fiber_radius, mean_length, YoungModulus,
                                                                 PoissonRatio)

print("Slenderness Compensation Ratio:", SlendernessCompensation)
print("Transverse Shear:", TransverseShear)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
-------------------------------------------PERIODIC BC DEFINITION-------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T
nodes1, nodes2 = np.array(ElementNodes1), np.array(ElementNodes2)
elements = np.vstack([nodes1, nodes2]).T

pio.renderers.default = 'browser'


def RemoveEmptyLines(file_path):
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.writelines(line for line in lines if line.strip())
        file.truncate()


def OrderNodes(Node1Id, Node2Id, AxisIndex):
    node1_pos = nodes[Node1Id][AxisIndex]
    node2_pos = nodes[Node2Id][AxisIndex]
    if node1_pos < node2_pos:
        return (Node1Id, Node2Id)
    else:
        return (Node2Id, Node1Id)


A = DomainPhysicalDimension / 2
BoundaryLimits = {
    "x": (-A, A),
    "y": (-A, A),
    "z": (-A, A)
}

with open(final_nodes_file_path, 'r') as file:
    nodes = [line.split(",") for line in file]
    nodes = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes if node}

with open(periodic_edges_file_path, "r") as file:
    PeriodicEdges = [(int(line.split()[0]) + 1, int(line.split()[1]) + 1) for line in file]

periodic_x, periodic_y, periodic_z = [], [], []

for edge in PeriodicEdges:
    node1, node2 = nodes[edge[0]], nodes[edge[1]]

    # For x-periodic
    if node1[0] in BoundaryLimits["x"] or node2[0] in BoundaryLimits["x"]:
        OrderedEdges = OrderNodes(edge[0], edge[1], 0)
        periodic_x.append(OrderedEdges)
    # For y-periodic
    if node1[1] in BoundaryLimits["y"] or node2[1] in BoundaryLimits["y"]:
        OrderedEdges = OrderNodes(edge[0], edge[1], 1)
        periodic_y.append(OrderedEdges)
    # For z-periodic
    if node1[2] in BoundaryLimits["z"] or node2[2] in BoundaryLimits["z"]:
        OrderedEdges = OrderNodes(edge[0], edge[1], 2)
        periodic_z.append(OrderedEdges)

periodic_x_elements_file_path = os.path.join(output_directory, "periodic_x.inp")
periodic_y_elements_file_path = os.path.join(output_directory, "periodic_y.inp")
periodic_z_elements_file_path = os.path.join(output_directory, "periodic_z.inp")
with open(periodic_x_elements_file_path, "w") as file:
    for i, (node1, node2) in enumerate(periodic_x):
        file.write(" \n %i %i \n " % (node1, node2))
with open(periodic_y_elements_file_path, "w") as file:
    for i, (node1, node2) in enumerate(periodic_y):
        file.write(" \n%i %i \n " % (node1, node2))
with open(periodic_z_elements_file_path, "w") as file:
    for i, (node1, node2) in enumerate(periodic_z):
        file.write(" \n %i %i \n " % (node1, node2))

RemoveEmptyLines(periodic_x_elements_file_path)
RemoveEmptyLines(periodic_y_elements_file_path)
RemoveEmptyLines(periodic_z_elements_file_path)
PeriodicXLength = len(periodic_x)
PeriodicYLength = len(periodic_y)
PeriodicZLength = len(periodic_z)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
---------------------------------------------INTERPOLATION DEFINITION---------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T
nodes1, nodes2 = np.array(ElementNodes1), np.array(ElementNodes2)
elements = np.vstack([nodes1, nodes2]).T


def ReadPeriodicEdges(filename):
    with open(filename, 'r') as file:
        edges = [tuple(map(int, line.split())) for line in file.readlines()]
    return edges


def unique_ordered_list(pairs, index):
    seen = set()
    result = []
    for pair in pairs:
        node = pair[index]
        if node not in seen:
            seen.add(node)
            result.append(node)
    return result


NegXBounds = unique_ordered_list(periodic_x, 0)
PosXBounds = unique_ordered_list(periodic_x, 1)
NegYBounds = unique_ordered_list(periodic_y, 0)
PosYBounds = unique_ordered_list(periodic_y, 1)
NegZBounds = unique_ordered_list(periodic_z, 0)
PosZBounds = unique_ordered_list(periodic_z, 1)


# Define the interpolation function for X Shear
def CalculateXInterpolationCoefficients(ListofNodes, nodes, A):
    InterpolationMap = {}
    for index in ListofNodes:
        x, y, z = nodes[index - 1]
        InterpolationMap[index] = x / A
    return InterpolationMap


# Calculate interpolation coefficients for X Shear
PosYInterpolationMap_X = CalculateXInterpolationCoefficients(PosYBounds, nodes, DomainPhysicalDimension / 2)
NegYInterpolationMap_X = CalculateXInterpolationCoefficients(NegYBounds, nodes, DomainPhysicalDimension / 2)
PosZInterpolationMap_X = CalculateXInterpolationCoefficients(PosZBounds, nodes, DomainPhysicalDimension / 2)
NegZInterpolationMap_X = CalculateXInterpolationCoefficients(NegZBounds, nodes, DomainPhysicalDimension / 2)


# Define the interpolation function for Y Shear
def CalculateYInterpolationCoefficients(ListofNodes, nodes, A):
    InterpolationMap = {}
    for index in ListofNodes:
        x, y, z = nodes[index - 1]
        InterpolationMap[index] = y / A
    return InterpolationMap


# Calculate interpolation coefficients for Y Shear
PosXInterpolationMap_Y = CalculateYInterpolationCoefficients(PosXBounds, nodes, DomainPhysicalDimension / 2)
NegXInterpolationMap_Y = CalculateYInterpolationCoefficients(NegXBounds, nodes, DomainPhysicalDimension / 2)
PosZInterpolationMap_Y = CalculateYInterpolationCoefficients(PosZBounds, nodes, DomainPhysicalDimension / 2)
NegZInterpolationMap_Y = CalculateYInterpolationCoefficients(NegZBounds, nodes, DomainPhysicalDimension / 2)


# Define the interpolation function for Z Shear
def CalculateZInterpolationCoefficients(ListofNodes, nodes, A):
    InterpolationMap = {}
    for index in ListofNodes:
        x, y, z = nodes[index - 1]
        InterpolationMap[index] = z / A
    return InterpolationMap


# Calculate interpolation coefficients for X Shear
PosXInterpolationMap_Z = CalculateZInterpolationCoefficients(PosXBounds, nodes, DomainPhysicalDimension / 2)
NegXInterpolationMap_Z = CalculateZInterpolationCoefficients(NegXBounds, nodes, DomainPhysicalDimension / 2)
PosYInterpolationMap_Z = CalculateZInterpolationCoefficients(PosYBounds, nodes, DomainPhysicalDimension / 2)
NegYInterpolationMap_Z = CalculateZInterpolationCoefficients(NegYBounds, nodes, DomainPhysicalDimension / 2)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
---------------------------------------------AMPLITUDE DEFINITION------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T
nodes1, nodes2 = np.array(ElementNodes1), np.array(ElementNodes2)
elements = np.vstack([nodes1, nodes2]).T
step = 0.001
amplitude_file_path = os.path.join(output_directory, "amplitude.inp")

with open(amplitude_file_path, "w") as f:
    for i in range(int(1 / step) + 1):
        time_step = i * step
        amplitude = time_step
        f.write(f"\n {time_step:.3f}, {amplitude:.3f}")
RemoveEmptyLines(amplitude_file_path)
"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
---------------------------------------------INPUT FILE DEFINITION------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

shift = 0.00


# Function to read periodic edges from a file
def ReadPeriodicEdges(filename):
    with open(filename, 'r') as file:
        edges = [tuple(map(int, line.split())) for line in file.readlines()]
    return edges


def RemoveEmptyLines(file_path):
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.writelines(line for line in lines if line.strip())
        file.truncate()


def ClassifyNodes(nodes, tolerance, DomainPhysicalDimension):
    A = DomainPhysicalDimension / 2
    # Creating a set of all periodic nodes, adjusting for zero-indexing if necessary
    periodic_nodes = set(node for edge in (periodic_x + periodic_y + periodic_z) for node in edge)

    boundaries = {
        'NegX': [],
        'PosX': [],
        'NegY': [],
        'PosY': [],
        'NegZ': [],
        'PosZ': [],
    }

    assigned_nodes = set()  # To keep track of nodes that have already been assigned to a boundary

    # Order of boundaries can dictate priority
    priority_keys = ['NegX', 'PosX', 'NegY', 'PosY', 'NegZ', 'PosZ']

    for i, (x, y, z) in enumerate(nodes):
        node_id = i + 1  # Adjust to 1-based indexing used in external files and boundary assignment

        if node_id in periodic_nodes:
            continue  # Skip nodes that are periodic
        if node_id in assigned_nodes:
            continue  # Skip nodes that have already been assigned

        for key in priority_keys:
            boundary_check = False
            if key == 'NegX' and abs(x + A) <= tolerance:
                boundary_check = True
            elif key == 'PosX' and abs(x - A) <= tolerance:
                boundary_check = True
            elif key == 'NegY' and abs(y + A) <= tolerance:
                boundary_check = True
            elif key == 'PosY' and abs(y - A) <= tolerance:
                boundary_check = True
            elif key == 'NegZ' and abs(z + A) <= tolerance:
                boundary_check = True
            elif key == 'PosZ' and abs(z - A) <= tolerance:
                boundary_check = True

            if boundary_check:
                boundaries[key].append(node_id)  # Node id is already 1-based here
                assigned_nodes.add(node_id)  # Store as 1-based
                break

    return boundaries


def unique_ordered_list(pairs, index):
    seen = set()
    result = []
    for pair in pairs:
        node = pair[index]
        if node not in seen:
            seen.add(node)
            result.append(node)
    return result


tolerance = 0.1
xs, ys, zs = np.array(XCoords), np.array(YCoords), np.array(ZCoords)
nodes = np.vstack([xs, ys, zs]).T
boundaries = ClassifyNodes(nodes, tolerance, DomainPhysicalDimension)



def VolumePreservingUniaxialCondition(domain_physical_dimension, displacement):
    """
    Calculate the necessary transversal (y and z) reduction to maintain volume
    when the longitudinal (x) dimension is increased by half its original size.

    Args:
    domain_physical_dimension (float): The size of one side of the cubic domain.

    Returns:
    float: The new dimension in the y and z directions.
    float: The reduction in the y and z dimensions from the original.
    """
    # Compute the new x dimension
    stretch_fraction = displacement / domain_physical_dimension
    new_long_dimension = domain_physical_dimension * (1 + stretch_fraction)

    # Original volume of the cube
    original_volume = domain_physical_dimension ** 3

    # Compute new y and z dimensions assuming y = z and volume is preserved
    new_transv_dimension = (original_volume / new_long_dimension) ** 0.5

    # Calculate reduction in y and z dimensions
    reduction = domain_physical_dimension - new_transv_dimension

    return reduction


def VolumePreservingBiaxialCondition(domain_physical_dimension, displacement):
    """
    Calculate the necessary transversal (z) reduction to maintain volume
    when the longitudinal (x and y) dimensions are increased by half their original size.

    Args:
    domain_physical_dimension (float): The size of one side of the cubic domain.

    Returns:
    float: The new dimension in the y and z directions.
    float: The reduction in the y and z dimensions from the original.
    """
    stretch_fraction = displacement / domain_physical_dimension
    new_long_dimension = domain_physical_dimension * (1 + stretch_fraction)

    original_volume = domain_physical_dimension ** 3
    new_transv_dimension = original_volume / (new_long_dimension * new_long_dimension)

    reduction = domain_physical_dimension - new_transv_dimension

    return reduction


"""WAVINESS INTEGRATION """


def write_elements(filename, elements, start_index=1):
    with open(filename, 'w') as file:
        for idx, element in enumerate(elements, start=start_index):
            file.write(f"{idx}, {element[1]}, {element[2]}, {element[3]}, {element[4]}\n")


element_file_path = os.path.join(output_directory, 'elements.inp')

Wavy = False
with open(element_file_path, 'r') as file:
    elements = file.readlines()

elements = [tuple(map(int, line.strip().split(','))) for line in elements]
random.shuffle(elements)

if Wavy:
    split_index = int(0.25 * len(elements))
    elementsA = elements[:split_index]
    elementsB = elements[split_index:]

    output_file_path_A = os.path.join(output_directory, 'elementsA.inp')
    output_file_path_B = os.path.join(output_directory, 'elementsB.inp')

    write_elements(output_file_path_A, elementsA, start_index=1)
    write_elements(output_file_path_B, elementsB, start_index=len(elementsA) + 1)

"""
Load Type Legend: 
1  -> Uniaxial x direction
2  -> Uniaxial y direction
3  -> Uniaxial z direction
4  -> Biaxial xy direction
5  -> Biaxial xz direction
6  -> Biaxial yz direction
7  -> Shear xy direction
8  -> Shear xz direction
9  -> Shear yx direction
10 -> Shear yz direction
11 -> Shear zx direction
12 -> Shear zy direction
"""

for i in range(7, 12):
    load_direction = 70
    if load_direction == 1:
        X_Uniaxial_file_path = os.path.join(output_directory, "X_Uniaxial.inp")
        with open(X_Uniaxial_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **X Dir Uniaxial Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, %.2f, 0, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, -%.2f, 0, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            #KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosXBounds, 1")
            f.write("\n PosXBounds, 4, 6")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegXBounds, 1")
            f.write("\n NegXBounds, 4, 6")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, PoissonRatio))
            f.write("\n*Depvar")
            f.write("\n5,")
            f.write('\n1, "SDV1_STRAN1"')
            f.write('\n2, "SDV2_DSTRAN1"')
            f.write('\n3, "SDV3_TSTRAN1"')
            f.write('\n4, "SDV4_DFGRD0_11"')
            f.write('\n5, "SDV5_DFGDR1_11"')
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            reduction = VolumePreservingUniaxialCondition(DomainPhysicalDimension, (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 1, 1, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n RefPos, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 1, 1, %.1f" % (-DomainPhysicalDimension / 4))
            f.write("\n RefNeg, 4, 6, 0")
            for node_id in PosXBounds:
                coeff = PosXInterpolationMap_Y[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in PosXBounds:
                coeff = PosXInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in NegXBounds:
                coeff = NegXInterpolationMap_Y[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegXBounds:
                coeff = NegXInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosYBounds, 2, 2, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegYBounds, 2, 2, %.1f" % (reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosZBounds, 3, 3, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegZBounds, 3, 3, %.1f" % (reduction / 2))
            #f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            #f.write("\n %i, 1, 6, 0" %StabilizationNodeId)
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, write, overlay")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(X_Uniaxial_file_path)
    if load_direction == 2:
        Y_Uniaxial_file_path = os.path.join(output_directory, "Y_Uniaxial.inp")
        with open(Y_Uniaxial_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **Y Dir Uniaxial Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, 0, %.2f, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, 0, -%.2f, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            #KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosYBounds")
            f.write("\n PosYBounds, 4, 6")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegYBounds")
            f.write("\n NegYBounds, 4, 6")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, PoissonRatio))
            f.write("\n*Depvar")
            f.write("\n5,")
            f.write('\n1, "SDV1_STRAN1"')
            f.write('\n2, "SDV2_DSTRAN1"')
            f.write('\n3, "SDV3_TSTRAN1"')
            f.write('\n4, "SDV4_DFGRD0_11"')
            f.write('\n5, "SDV5_DFGDR1_11"')
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            reduction = VolumePreservingUniaxialCondition(DomainPhysicalDimension, (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 2, 2, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n RefPos, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 2, 2, %.1f" % (-DomainPhysicalDimension / 4))
            f.write("\n RefNeg, 4, 6, 0")
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosXBounds, 1, 1, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegXBounds, 1, 1, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosZBounds, 3, 3, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegZBounds, 3, 3, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n %i, 1, 6, 0" % StabilizationNodeId)
            # VOLUME PRESERVING BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosXBounds, 1, 1, -%.3f" % (reduction / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosZBounds, 3, 3, -%.3f" % (reduction / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n %i, ENCASTRE" % StabilizationNodeId)
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, write, overlay")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(Y_Uniaxial_file_path)
    if load_direction == 3:
        Z_Uniaxial_file_path = os.path.join(output_directory, "Z_Uniaxial.inp")
        with open(Z_Uniaxial_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **Z Dir Uniaxial Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, 0, 0, %.2f" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, 0, 0, %.2f" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            # KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosZBounds")
            f.write("\n PosZBounds, 4, 6")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegZBounds")
            f.write("\n NegZBounds, 4, 6")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, PoissonRatio))
            f.write("\n*Depvar")
            f.write("\n5,")
            f.write('\n1, "SDV1_STRAN1"')
            f.write('\n2, "SDV2_DSTRAN1"')
            f.write('\n3, "SDV3_TSTRAN1"')
            f.write('\n4, "SDV4_DFGRD0_11"')
            f.write('\n5, "SDV5_DFGDR1_11"')
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            reduction = VolumePreservingUniaxialCondition(DomainPhysicalDimension, (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 3, 3, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n RefPos, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 3, 3, %.1f" % (-DomainPhysicalDimension / 4))
            f.write("\n RefNeg, 4, 6, 0")
            for node_id in PosZBounds:
                coeff = PosZInterpolationMap_Y[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in PosZBounds:
                coeff = PosZInterpolationMap_X[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            for node_id in NegZBounds:
                coeff = NegZInterpolationMap_Y[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegZBounds:
                coeff = NegZInterpolationMap_X[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosYBounds, 2, 2, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegYBounds, 2, 2, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosXBounds, 1, 1, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegXBounds, 1, 1, %.1f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n %i, 1, 6, 0" % StabilizationNodeId)
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, write, overlay")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(Z_Uniaxial_file_path)
    if load_direction == 4:
        XY_Biaxial_file_path = os.path.join(output_directory, "XY_Biaxial.inp")
        with open(XY_Biaxial_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Biaxial Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, %.2f, 0, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, -%.2f, 0, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000002, 0, %.2f, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000003, 0, -%.2f, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPosX")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNegX")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=RefPosY")
            f.write("\n 5000002")
            f.write("\n *Nset, nset=RefNegY")
            f.write("\n 5000003")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            #KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPosX")
            f.write("\n PosXBounds, 1")
            f.write("\n PosXBounds, 4, 6")
            f.write("\n *Kinematic Coupling, ref node=RefNegX")
            f.write("\n NegXBounds, 1")
            f.write("\n NegXBounds, 4, 6")
            f.write("\n *Kinematic Coupling, ref node=RefPosY")
            f.write("\n PosYBounds, 2")
            f.write("\n PosYBounds, 4, 6")
            f.write("\n *Kinematic Coupling, ref node=RefNegY")
            f.write("\n NegYBounds, 2")
            f.write("\n NegYBounds, 4, 6")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, PoissonRatio))
            f.write("\n*Depvar")
            f.write("\n5,")
            f.write('\n1, "SDV1_STRAN1"')
            f.write('\n2, "SDV2_DSTRAN1"')
            f.write('\n3, "SDV3_TSTRAN1"')
            f.write('\n4, "SDV4_DFGRD0_11"')
            f.write('\n5, "SDV5_DFGDR1_11"')
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=BiaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            reduction = VolumePreservingBiaxialCondition(DomainPhysicalDimension, (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPosX, 1, 1, %.1f" % (DomainPhysicalDimension / 4))
            #f.write("\n RefPosX, 2, 2, 0")
            f.write("\n RefPosX, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNegX, 1, 1, %.1f" % (-DomainPhysicalDimension / 4))
            f.write("\n RefNegX, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPosY, 2, 2, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n RefPosY, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNegY, 2, 2, %.1f" % (-DomainPhysicalDimension / 4))
            f.write("\n RefNegY, 4, 6, 0")
            for node_id in PosXBounds:
                coeff = PosXInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in NegXBounds:
                coeff = NegXInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_Z[node_id]
                displacement_value = - (reduction / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement, amplitude= Smooth")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosZBounds, 3, 3, %.3f" % (-reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegZBounds, 3, 3, %.3f" % (reduction / 2))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n %i, 1, 6, 0" % StabilizationNodeId)
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, write, overlay")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Biaxial_file_path)
    if load_direction == 5:
        XZ_Biaxial_file_path = os.path.join(output_directory, "XZ_Biaxial.inp")
        with open(XZ_Biaxial_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XZ Dir Biaxial Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13.0, 0.0, 0.0" % (5000000 + i))
            for i in range(PeriodicZLength):
                f.write("\n %i, 0.0, 0.0, 13.0" % (7000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            f.write("\n *Nset, nset=DummyZNodes, generate")
            f.write("\n 7000000, %i, 1\n " % (7000000 + PeriodicZLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 1, 1., %i, 1, -1., %i, 1, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, 1., %i, 2, 1.\n" % (pair[0], pair[1]))
            dummy_node_id = 7000000
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            # Periodic constraints
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f, 0.25 " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 1")
            f.write("\n %.2f" % YoungModulus)
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=BiaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-30, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 30, , ,")
            # HISTORY BOUNDARY DEFINITION
            reduction = VolumePreservingBiaxialCondition(DomainPhysicalDimension, (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n DummyXNodes, 1, 1, %i" % (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n DummyZNodes, 3, 3, %i" % (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n NegXBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosXBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n NegZBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosZBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n NegYBounds, 1, 1, 0")
            f.write("\n NegYBounds, 3, 3, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosYBounds, 1, 1, 0")
            f.write("\n PosYBounds, 3, 3, 0")
            # VOLUME PRESERVING BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosYBounds, 2, 2, -%.3f" % (reduction / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n %i, ENCASTRE" % StabilizationNodeId)
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XZ_Biaxial_file_path)
    if load_direction == 6:
        YZ_Biaxial_file_path = os.path.join(output_directory, "YZ_Biaxial.inp")
        with open(YZ_Biaxial_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **YZ Dir Biaxial Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13.0, 0.0, 0.0" % (5000000 + i))
            for i in range(PeriodicZLength):
                f.write("\n %i, 0.0, 0.0, 13.0" % (7000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyYNodes, generate")
            f.write("\n 6000000, %i, 1\n " % (6000000 + PeriodicYLength - 1))
            f.write("\n *Nset, nset=DummyZNodes, generate")
            f.write("\n 7000000, %i, 1\n " % (7000000 + PeriodicZLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 1, 1., %i, 1, 1.\n" % (pair[0], pair[1]))
            dummy_node_id = 6000000
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            dummy_node_id = 7000000
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 3, 1., %i, 3, -1., %i, 3, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            # Periodic constraints
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f, 0.25 " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 1")
            f.write("\n %.2f" % YoungModulus)
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=BiaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-30, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 30, , ,")
            # HISTORY BOUNDARY DEFINITION
            reduction = VolumePreservingBiaxialCondition(DomainPhysicalDimension, (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n DummyYNodes, 2, 2, %i" % (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n DummyZNodes, 3, 3, %i" % (DomainPhysicalDimension / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n NegYBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosYBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n NegZBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosZBounds, 4, 6, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n NegXBounds, 2, 3, 0")
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosXBounds, 2, 3, 0")
            # VOLUME PRESERVING BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement")
            f.write("\n PosXBounds, 1, 1, -%.3f" % (reduction / 2))
            f.write("\n *Boundary, type = Displacement")
            f.write("\n %i, ENCASTRE" % StabilizationNodeId)
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(YZ_Biaxial_file_path)
    if load_direction == 7:
        XY_Shear_file_path = os.path.join(output_directory, "XY_Shear.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, %.2f, 0, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, -%.2f, 0, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            #KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosXBounds")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegXBounds")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            #f.write("\n *Transverse Shear Stiffness")
            #f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % ((YoungModulus/10), PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (YoungModulus+200, PoissonRatio))  # PAY ATTENTION HERE! YOU'VE MODIFIED THIS!!!
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.01")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-30, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 1, 1, 0")
            f.write("\n RefPos, 2, 2, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 1, 1, 0")
            f.write("\n RefNeg, 2, 2, %.1f" % (-DomainPhysicalDimension / 4))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            #f.write("\n *Element Output")
            #f.write("\n  SDV1")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 8:
        XZ_Shear_file_path = os.path.join(output_directory, "XZ_Shear.inp")
        with open(XZ_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XZ Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, %.2f, 0, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, -%.2f, 0, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            #KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosXBounds")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegXBounds")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % ((YoungModulus/10), PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (YoungModulus, PoissonRatio))
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-20, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 1, 1, 0")
            f.write("\n RefPos, 3, 3, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 1, 1, 0")
            f.write("\n RefNeg, 3, 3, %.1f" % (-DomainPhysicalDimension / 4))
            for node_id in PosZBounds:
                coeff = PosZInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in NegZBounds:
                coeff = NegZInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XZ_Shear_file_path)
    if load_direction == 9:
        YX_Shear_file_path = os.path.join(output_directory, "YX_Shear.inp")
        with open(YX_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **YX Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, 0, %.2f, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, 0, -%.2f, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            #KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosYBounds")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegYBounds")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            #f.write("\n *Transverse Shear Stiffness")
            #f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % ((YoungModulus/10), PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (YoungModulus, PoissonRatio))
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-20, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 2, 2, 0")
            f.write("\n RefPos, 1, 1, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 2, 2, 0")
            f.write("\n RefNeg, 1, 1, %.1f" % (-DomainPhysicalDimension / 4))
            for node_id in PosXBounds:
                coeff = PosXInterpolationMap_Y[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 2, 2, 0")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            for node_id in NegXBounds:
                coeff = NegXInterpolationMap_Y[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 2, 2, 0")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(YX_Shear_file_path)
    if load_direction == 10:
        YZ_Shear_file_path = os.path.join(output_directory, "YZ_Shear.inp")
        with open(YZ_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **YZ Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, 0, %.2f, 0" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, 0, -%.2f, 0" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            # KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosYBounds")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegYBounds")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % ((YoungModulus/10), PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (YoungModulus, PoissonRatio))
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-20, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 2, 2, 0")
            f.write("\n RefPos, 3, 3, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 2, 2, 0")
            f.write("\n RefNeg, 3, 3, %.1f" % (-DomainPhysicalDimension / 4))
            for node_id in PosZBounds:
                coeff = PosZInterpolationMap_Y[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 2, 2, 0")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            for node_id in NegZBounds:
                coeff = NegZInterpolationMap_Y[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 2, 2, 0")
                f.write(f"\n {node_id}, 3, 3, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Element Output")
            f.write("\n  SDV1, SDV2, SDV3, SDV4, SDV5")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(YZ_Shear_file_path)
    if load_direction == 11:
        ZX_Shear_file_path = os.path.join(output_directory, "ZX_Shear.inp")
        with open(ZX_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **ZX Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, 0, 0, %.2f" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, 0, 0, -%.2f" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            # KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosZBounds")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegZBounds")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            #f.write("\n *Transverse Shear Stiffness")
            #f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % ((YoungModulus / 10), PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (YoungModulus, PoissonRatio))
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-20, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 3, 3, 0")
            f.write("\n RefPos, 1, 1, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 3, 3, 0")
            f.write("\n RefNeg, 1, 1, %.1f" % (-DomainPhysicalDimension / 4))
            for node_id in PosXBounds:
                coeff = PosXInterpolationMap_Z[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 3, 3, 0")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            for node_id in NegXBounds:
                coeff = NegXInterpolationMap_Z[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 3, 3, 0")
                f.write(f"\n {node_id}, 1, 1, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(ZX_Shear_file_path)
    if load_direction == 12:
        ZY_Shear_file_path = os.path.join(output_directory, "ZY_Shear.inp")
        with open(ZY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **ZY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            # NODES DEFINITION
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n 5000000, 0, 0, %.2f" % (DomainPhysicalDimension/2))
            f.write("\n 5000001, 0, 0, -%.2f" % (DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=RefPos")
            f.write("\n 5000000")
            f.write("\n *Nset, nset=RefNeg")
            f.write("\n 5000001")
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegZBounds")
            for i in range(len(NegZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegZBounds[i]))
            f.write("\n *Nset, nset=PosZBounds")
            for i in range(len(PosZBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosZBounds[i]))
            # KINEMATIC COUPLING
            f.write("\n *Kinematic Coupling, ref node=RefPos")
            f.write("\n PosZBounds")
            f.write("\n *Kinematic Coupling, ref node=RefNeg")
            f.write("\n NegZBounds")
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % ((YoungModulus/10), PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (YoungModulus, PoissonRatio))
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-20, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefPos, 3, 3, 0")
            f.write("\n RefPos, 2, 2, %.1f" % (DomainPhysicalDimension / 4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n RefNeg, 3, 3, 0")
            f.write("\n RefNeg, 2, 2, %.1f" % (-DomainPhysicalDimension / 4))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_Z[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 3, 3, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_Z[node_id]
                displacement_value = (DomainPhysicalDimension / 4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 3, 3, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(ZY_Shear_file_path)

    if load_direction == 70:
        XY_Shear_file_path = os.path.join(output_directory, "XY_ShearPBC.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, %.2f, 0.0, 0.0" % (5000000 + i, DomainPhysicalDimension/2))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, -1., %i, 2, 1.\n" % (pair[0], pair[1]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n *Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *Elastic, dependencies=1")
            f.write("\n %.2f, %.2f, 0.00, 1.00" % (70, PoissonRatio))
            f.write("\n %.2f, %.2f, 0.00, 2.00" % (700, PoissonRatio))
            f.write("\n * USER DEFINED FIELD")
            f.write("\n * DEPVAR")
            f.write("\n 1")
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.01")
            f.write("\n *Step, name=ShearTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL = 0.02")
            f.write("\n 0.001, 1., 1e-50, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n DummyXNodes, 1, 1, 0")
            f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension/2))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegYBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
""" 
    # WAVY CONDITIONS
    if load_direction == 70:
        XY_Shear_file_path = os.path.join(output_directory, "XY_Shear.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13, 0.0, 0.0" % (5000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, -1., %i, 2, 1.\n" % (pair[0], pair[1]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elementsA.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elementsA.inp")
            f.write("\n *Elset, elset = StraightBeams, generate")
            f.write("\n 1, %i, 1" % len(elementsA))
            f.write("\n *Beam Section, elset=StraightBeams, material=CollagenStraightMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenStraightMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # SECOND SET OF ELEMENTS
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elementsB.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elementsB.inp")
            f.write("\n *Elset, elset = WavyBeams, generate")
            f.write("\n  %i, %i, 1" % (len(elementsA) + 1, ElementsNumber))
            f.write("\n *Beam Section, elset=WavyBeams, material=CollagenWavyMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenWavyMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n DummyXNodes, 1, 1, 0")
            f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension/2))
            #f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegYBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 71:
        XY_Shear_file_path = os.path.join(output_directory, "XY_Shear.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13, 0.0, 0.0" % (5000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n DummyXNodes, 1, 1, 0")
            f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension/2))
            #f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 72:
        XY_Shear_file_path = os.path.join(output_directory, "XY_Shear.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13, 0.0, 0.0" % (5000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, 1., %i, 2, -1.\n" % (pair[1], pair[0]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n DummyXNodes, 1, 1, 0")
            f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension/2))
            #f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
    if load_direction == 28:
        XY_Shear_file_path = os.path.join(output_directory, "Case6_OneSide.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13, 0.0, 0.0" % (5000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, -1., %i, 2, 1.\n" % (pair[0], pair[1]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension/2))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # TRASNV DIRECTION
            f.write("\n PosXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # TRASNV DIRECTION
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # TRASNV DIRECTION
            f.write("\n NegYBounds, 1, 1, 0")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 29:
        XY_Shear_file_path = os.path.join(output_directory, "CASE7.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, -1., %i, 2, 1.\n" % (pair[0], pair[1]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            f.write("\n*Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                #f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                #f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n NegXBounds, 2, 2, -7.5")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            f.write("\n PosXBounds, 2, 2, 7.5")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 30:
        XY_Shear_file_path = os.path.join(output_directory, "CASE8.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, -1., %i, 2, 1.\n" % (pair[0], pair[1]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                #f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n NegXBounds, 2, 2, -7.5")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            f.write("\n PosXBounds, 2, 2, 7.5")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 31:
        XY_Shear_file_path = os.path.join(output_directory, "CASE9.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                #f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                #f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n NegXBounds, 2, 2, -7.5")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            f.write("\n PosXBounds, 2, 2, 7.5")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 32:
        XY_Shear_file_path = os.path.join(output_directory, "CASE10.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            f.write("\n **Dummy Nodes to Define the PBC ")
            for i in range(PeriodicXLength):
                f.write("\n %i, 13, 0.0, 0.0" % (5000000 + i))
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # DUMMY NODES SET
            f.write("\n *Nset, nset=DummyXNodes, generate")
            f.write("\n 5000000, %i, 1\n " % (5000000 + PeriodicXLength - 1))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            dummy_node_id = 5000000
            for pair in periodic_x_edges:
                f.write("*Equation\n")
                f.write("3\n")
                f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
                dummy_node_id += 1
            for pair in periodic_y_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 2, -1., %i, 2, 1.\n" % (pair[0], pair[1]))
            for pair in periodic_z_edges:
                f.write("*Equation\n")
                f.write("2\n")
                f.write("%i, 3, 1., %i, 3, -1.\n" % (pair[1], pair[0]))
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.01")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.001, 1., 1e-60, 0.01")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n DummyXNodes, 1, 1, 0")
            f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension/2))
            #f.write("\n DummyXNodes, 2, 2, %i" % (DomainPhysicalDimension))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegYBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 1, 1, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n NegXBounds, 5, 6, 0")
            f.write("\n *Boundary, type = Displacement,amplitude=Smooth")
            f.write("\n PosXBounds, 5, 6, 0")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
    if load_direction == 32:
        XY_Shear_file_path = os.path.join(output_directory, "CaseDirichlet.inp")
        with open(XY_Shear_file_path, "w") as f:
            # HEADING
            f.write(" *Heading")
            f.write("\n **XY Dir Shear Test of a cubic Representative Volume Element")
            f.write("\n **Computational Unit Length = %i, Stress [MPa], Force [uN]" % DomainPhysicalDimension)
            f.write("\n **Fiber volume fraction: %.4f" % phi)
            f.write("\n **Concentration: %.3f mg/mL" % concentration)
            f.write("\n *Preprint, echo=NO, model=YES, history=YES, contact=NO")
            f.write("\n *Node, INPUT = nodes.inp")
            # TOTAL NODES SET
            f.write("\n *Nset, nset=AllNodes, generate")
            f.write("\n 1, %i, 1" % len(nodes))
            # PERIODIC BOUNDARY DEFINITION
            periodic_x_edges = ReadPeriodicEdges(periodic_x_elements_file_path)
            periodic_y_edges = ReadPeriodicEdges(periodic_y_elements_file_path)
            periodic_z_edges = ReadPeriodicEdges(periodic_z_elements_file_path)
            NegXBounds = unique_ordered_list(periodic_x_edges, 0)
            PosXBounds = unique_ordered_list(periodic_x_edges, 1)
            NegYBounds = unique_ordered_list(periodic_y_edges, 0)
            PosYBounds = unique_ordered_list(periodic_y_edges, 1)
            NegZBounds = unique_ordered_list(periodic_z_edges, 0)
            PosZBounds = unique_ordered_list(periodic_z_edges, 1)
            # Periodic constraints
            f.write("\n*Nset, nset=NegYBounds")
            for i in range(len(NegYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegYBounds[i]))
            f.write("\n*Nset, nset=PosYBounds")
            for i in range(len(PosYBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosYBounds[i]))
            f.write("\n*Nset, nset=NegXBounds")
            for i in range(len(NegXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(NegXBounds[i]))
            f.write("\n *Nset, nset=PosXBounds")
            for i in range(len(PosXBounds)):
                if i % 16 == 0:
                    f.write("\n")
                f.write("{:d},".format(PosXBounds[i]))
            # ELEMENTS DEFINITION
            f.write("\n **There are %i number of elements randomly distributed in the volume" % ElementsNumber)
            f.write("\n **Quadratic Element Definition: endpoint 1, midpoint, endpoint 2 + extra node for orientation")
            f.write("\n **The fiber is slenderness 1/ 20")
            if element_type == 1:
                f.write("\n  *Element, type=B31, INPUT = elements.inp")
            elif element_type == 2:
                f.write("\n  *Element, type=B32, INPUT = elements.inp")
            f.write("\n *Elset, elset = Beams, generate")
            f.write("\n 1, %i, 1" % ElementsNumber)
            f.write("\n *Beam Section, elset=Beams, material=CollagenMaterial,section=Circ")
            f.write("\n  %.3f" % fiber_radius)
            f.write("\n *Transverse Shear Stiffness")
            f.write("\n  %.8f, %.8f,  " % (TransverseShear, TransverseShear))
            # MATERIAL DEFINITION
            f.write("\n **Collagen E values from Raush et al")
            f.write("\n *Material, name=CollagenMaterial")
            f.write("\n *User Material, constants = 2")
            f.write("\n %.2f, %.2f" % (YoungModulus, shift))
            # STEP DEFINITION
            f.write("\n *Time Points, name=MustPoints, GENERATE")
            f.write("\n 0., 1., 0.1")
            f.write("\n *Step, name=UniaxialTest, INC=100000, nlgeom=YES")
            f.write("\n *Static,  stabilize, ALLSDTOL=0.05")
            f.write("\n 0.01, 1., 1e-60, 0.1")
            f.write("\n  *Controls, parameters=time incrementation")
            f.write("\n   , , , , , , , 60, , ,")
            f.write("\n  *Amplitude, definition = Smooth, name = Smooth")
            f.write("\n   0,0,1,1")
            # HISTORY BOUNDARY DEFINITION
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n NegXBounds, 1, 1, 0")
            f.write("\n NegXBounds, 2, 2, %i" % (-DomainPhysicalDimension/4))
            f.write("\n *Boundary, type = Displacement, amplitude=Smooth")  # LOAD DIRECTION
            f.write("\n PosXBounds, 1, 1, 0")
            f.write("\n PosXBounds, 2, 2, %i" % (DomainPhysicalDimension/4))
            for node_id in PosYBounds:
                coeff = PosYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            for node_id in NegYBounds:
                coeff = NegYInterpolationMap_X[node_id]
                displacement_value = (DomainPhysicalDimension/4) * coeff
                #displacement_value = (DomainPhysicalDimension / 2) * coeff
                displacement_value_formatted = f"{displacement_value:.3f}"
                f.write("\n*Boundary, type = Displacement,amplitude=Smooth")
                f.write(f"\n {node_id}, 1, 1, 0")
                f.write(f"\n {node_id}, 2, 2, {displacement_value_formatted}")
            # OUTPUT
            f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
            f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
            f.write("\n *Element Output")
            f.write("\n  S,SE,SF,SM,SK")
            f.write("\n *Node Output, nset= AllNodes")
            f.write("\n  U, RF")
            f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
            f.write("\n *Restart, Write, FREQUENCY=10")
            f.write("\n *End Step")
        f.close()
        RemoveEmptyLines(XY_Shear_file_path)
"""
