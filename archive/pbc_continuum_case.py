
"""---------------------SANITY CHECK OF PBC ON A 2D CONTINUUM SQUARE ----------------------------------------------"""


import numpy as np

# Sample nodes in the 2D xy plane
raw_data = """
     1,          -1., -0.400000006,           0.
      2,           1., -0.400000006,           0.
      3,           1.,  0.400000006,           0.
      4,          -1.,  0.400000006,           0.
      5,           1.,           1.,           0.
      6,          -1.,           1.,           0.
      7,          -1.,          -1.,           0.
      8,           1.,          -1.,           0.
      9, -0.933333337, -0.400000006,           0.
     10, -0.866666675, -0.400000006,           0.
     11, -0.800000012, -0.400000006,           0.
     12, -0.733333349, -0.400000006,           0.
     13, -0.666666687, -0.400000006,           0.
     14, -0.600000024, -0.400000006,           0.
     15, -0.533333361, -0.400000006,           0.
     16, -0.466666669, -0.400000006,           0.
     17, -0.400000006, -0.400000006,           0.
     18, -0.333333343, -0.400000006,           0.
     19, -0.266666681, -0.400000006,           0.
     20, -0.200000003, -0.400000006,           0.
     21,  -0.13333334, -0.400000006,           0.
     22, -0.0666666701, -0.400000006,           0.
     23,           0., -0.400000006,           0.
     24, 0.0666666701, -0.400000006,           0.
     25,   0.13333334, -0.400000006,           0.
     26,  0.200000003, -0.400000006,           0.
     27,  0.266666681, -0.400000006,           0.
     28,  0.333333343, -0.400000006,           0.
     29,  0.400000006, -0.400000006,           0.
     30,  0.466666669, -0.400000006,           0.
     31,  0.533333361, -0.400000006,           0.
     32,  0.600000024, -0.400000006,           0.
     33,  0.666666687, -0.400000006,           0.
     34,  0.733333349, -0.400000006,           0.
     35,  0.800000012, -0.400000006,           0.
     36,  0.866666675, -0.400000006,           0.
     37,  0.933333337, -0.400000006,           0.
     38,           1.,  -0.13333334,           0.
     39,           1.,   0.13333334,           0.
     40,  0.600000024,  0.400000006,           0.
     41,  0.200000003,  0.400000006,           0.
     42, -0.200000003,  0.400000006,           0.
     43, -0.600000024,  0.400000006,           0.
     44,          -1.,   0.13333334,           0.
     45,          -1.,  -0.13333334,           0.
     46,           1.,  0.699999988,           0.
     47,          0.5,           1.,           0.
     48,           0.,           1.,           0.
     49,         -0.5,           1.,           0.
     50,          -1.,  0.699999988,           0.
     51,          -1., -0.699999988,           0.
     52,         -0.5,          -1.,           0.
     53,           0.,          -1.,           0.
     54,          0.5,          -1.,           0.
     55,           1., -0.699999988,           0.
     56, -0.882314265, -0.329357505,           0.
     57,  0.101747297,    -0.335529,           0.
     58,  0.880194247, -0.321447909,           0.
     59,  0.664255917, -0.339055985,           0.
     60,  0.567888618, -0.331063956,           0.
     61,  0.433418751, -0.335344225,           0.
     62,  0.308999121, -0.338083506,           0.
     63,   0.23195529, -0.331464469,           0.
     64,  0.166663885,  -0.34175691,           0.
     65, -0.232838094, -0.337368935,           0.
     66, -0.298326433,  -0.34447968,           0.
     67,    -0.495738, -0.338365614,           0.
     68, -0.765073121, -0.331754029,           0.
     69, -0.364160568, -0.337816894,           0.
     70, -0.579427063,  -0.33707127,           0.
     71, -0.0267836675, -0.338518947,           0.
     72,  -0.10468401, -0.339012027,           0.
     73,  0.499125302, -0.341584653,           0.
     74,  0.762322366, -0.318552196,           0.
     75,  0.612285197, -0.261161357,           0.
     76,  0.369244516, -0.280993223,           0.
     77,  0.280746847, -0.268984944,           0.
     78, 0.0384374633, -0.284071803,           0.
     79, -0.167340085, -0.286705822,           0.
     80, -0.426264942, -0.284743667,           0.
     81, -0.665872514,  -0.26985687,           0.
     82, -0.530304492, -0.260763347,           0.
     83, -0.063863039, -0.267331153,           0.
     84, -0.807184458, -0.237505004,           0.
     85,  0.693953991, -0.250380129,           0.
     86,  0.489632308, -0.239414141,           0.
     87,  0.162145257, -0.243289784,           0.
     88, -0.294238776, -0.252916038,           0.
     89, -0.575201869, -0.163208812,           0.
     90,  0.789689302, -0.190207228,           0.
     91,  0.319841892, -0.151193962,           0.
     92, 0.00852978788, -0.177431539,           0.
     93, -0.147274449, -0.182982191,           0.
     94, -0.438907295, -0.172329649,           0.
     95, -0.722744763, -0.123044536,           0.
     96,  0.626856446, -0.133212224,           0.
     97, -0.524161935, 0.0085404953,           0.
     98, -0.0916259587, -0.0327552296,           0.
     99,  0.124860309, -0.0566488057,           0.
    100, -0.295384467, -0.0817051828,           0.
    101,  0.465956777, -0.0712223724,           0.
    102,  0.763198197, 0.0231445357,           0.
    103,  0.557364285, 0.0648263469,           0.
    104, -0.281951934,  0.142239332,           0.
    105, -0.769381344, 0.0570991933,           0.
    106, 0.0136820758,  0.159709513,           0.
    107,  0.325957894,   0.10649582,           0.
    108,  0.716666639,  0.649999976,           0.
    109,  0.882584393, -0.478807837,           0.
    110,  0.751593411, -0.473979563,           0.
    111,  0.636062264, -0.488209575,           0.
    112,  0.282673836, -0.455077201,           0.
    113,  0.154016092, -0.462396145,           0.
    114, -0.186388239, -0.483060926,           0.
    115, -0.324394584, -0.471523106,           0.
    116, -0.456353545, -0.464580387,           0.
    117, -0.587933719, -0.464043289,           0.
    118, -0.720251441, -0.467536598,           0.
    119, -0.817878783, -0.465272754,           0.
    120, 0.0546437539, -0.463401735,           0.
    121,  -0.90312469, -0.489606857,           0.
    122,  0.512699246, -0.469344705,           0.
    123, -0.0450990163,  -0.48271504,           0.
    124,  0.828223407, -0.587348521,           0.
    125,  0.715471923,  -0.57555747,           0.
    126,  0.386723608, -0.542770326,           0.
    127,   0.24756442, -0.552825749,           0.
    128,  0.103398986, -0.583359957,           0.
    129, -0.388567597, -0.585509717,           0.
    130, -0.515320361, -0.557394981,           0.
    131, -0.645283163, -0.586121917,           0.
    132,  -0.25413695, -0.589455664,           0.
    133, -0.804551244, -0.577678859,           0.
    134, -0.512857556, -0.706374943,           0.
    135,  0.315679073, -0.730705261,           0.
    136,  0.791478515, -0.714957774,           0.
    137, -0.743781984, -0.761695981,           0.
    138,  0.570465207, -0.618937552,           0.
    139, -0.0838371739, -0.661446452,           0.
    140, -0.289899886, -0.757131159,           0.
    141,  0.635524571, -0.812920094,           0.
    142, 0.0838102177, -0.743877947,           0.

"""
data = raw_data.strip().split('\n')
nodes = [list(map(float, line.split(','))) for line in data]

# Convert nodes to a dictionary for easy lookup by ID
node_dict = {int(node[0]): (node[1], node[2]) for node in nodes}

# Identify boundary nodes
boundary_node_ids = [node_id for node_id, (x, y) in node_dict.items() if abs(x) == 1.0 or abs(y) == 1.0]


# Function to find a node pair based on a condition
def exclude_corners_condition(coords):
    # Exclude nodes that are at the corners of the domain
    x, y = coords
    return abs(x) == 1 and abs(y) == 1


def find_pair_correctly(nodes, match_condition, exclude_corners_condition):
    pairs = []
    for node_id, (x1, y1) in nodes.items():
        if not exclude_corners_condition((x1, y1)):
            # Find the matching node based on the condition
            match_id = next(
                (id for id, (x2, y2) in nodes.items()
                 if match_condition(x1, y1, x2, y2) and id != node_id and not exclude_corners_condition((x2, y2))), None)
            if match_id:
                pair = sorted([node_id, match_id])  # Sort to prevent duplicates
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


# Update match conditions for x and y periodic boundaries
match_condition_x = lambda x1, y1, x2, y2: y1 == y2 and abs(x1) == 1 and abs(x2) == 1 and x1 != x2
match_condition_y = lambda x1, y1, x2, y2: x1 == x2 and abs(y1) == 1 and abs(y2) == 1 and y1 != y2

# Update calls to find_pair_correctly with new match conditions
x_periodic_pair_ids = find_pair_correctly(node_dict, match_condition_x, exclude_corners_condition)
y_periodic_pair_ids = find_pair_correctly(node_dict, match_condition_y, exclude_corners_condition)


def reorder_pairs(pairs, nodes, axis):
    """
    Reorders node pairs based on their coordinates along a specified axis ('x' or 'y').
    For 'x', ensures the node with x=-1 is first. For 'y', ensures the node with y=-1 is first.
    """
    reordered_pairs = []
    for pair in pairs:
        # Extract coordinates for comparison
        coords_1 = nodes[pair[0]]
        coords_2 = nodes[pair[1]]
        # Determine index based on axis
        index = 0 if axis == 'x' else 1
        # Reorder pair if necessary
        if coords_1[index] > coords_2[index]:
            pair = [pair[1], pair[0]]
        reordered_pairs.append(pair)
    return reordered_pairs

# After finding pairs, reorder them based on the specified logic
x_periodic_pair_ids = reorder_pairs(x_periodic_pair_ids, node_dict, 'x')
y_periodic_pair_ids = reorder_pairs(y_periodic_pair_ids, node_dict, 'y')

x_periodic_pair_ids, y_periodic_pair_ids

# Define the coordinates of each corner
corners = {
    'C1': (1, 1, 0),
    'C2': (-1, 1, 0),
    'C3': (-1, -1, 0),
    'C4': (1, -1, 0),
}

# Initialize a dictionary to store the corner node IDs
corner_node_ids = {corner: None for corner in corners}

# Iterate over the nodes to find the corners
for node_id, (x, y) in node_dict.items():
    for corner, (cx, cy, cz) in corners.items():
        if x == cx and y == cy:
            corner_node_ids[corner] = node_id
            break  # Move to the next node after finding a match

# Now, the corner_node_ids dictionary contains the IDs for each corner node
corner_node_ids

def FindStabilizationNode(nodes):
    distances = {node_id: (x ** 2 + y ** 2 + z ** 2) ** 0.5 for node_id, (x, y, z) in nodes.items()}
    StabilizationNodeId = min(distances, key=distances.get)
    return StabilizationNodeId

with open("AbaqusPreProcessing/OutputData/JobInputFiles/ContinuumPBC.inp", "w") as f:
    # HEADING
    f.write(" *Heading")
    f.write("\n **X Dir Uniaxial Test of a cubic Representative Volume Element")
    f.write("\n *Node")
    f.write("\n **Dummy Nodes to Define the PBC ")
    for i in range(len(x_periodic_pair_ids)):
        f.write("\n %i, 1, 0.0, 0.0" % (5000000 + i))
    for i in range(len(y_periodic_pair_ids)):
        f.write("\n %i, 0.0, 1, 0.0" % (6000000 + i))
    # TOTAL NODES SET
    f.write("\n *Nset, nset=AllNodes, generate")
    f.write("\n 1, %i, 1" % len(nodes))
    # DUMMY NODES SET
    f.write("\n *Nset, nset=DummyNodes, generate")
    f.write("\n 5000000, %i, 1\n " % (5000000 + len(x_periodic_pair_ids) - 1))
    # DUMMY NODES SET
    f.write("\n *Nset, nset=DummyYNodes, generate")
    f.write("\n 6000000, %i, 1\n " % (6000000 + len(y_periodic_pair_ids) - 1))
    NegXBounds = list({pair[0] for pair in x_periodic_pair_ids})
    PosXBounds = list({pair[1] for pair in x_periodic_pair_ids})
    NegYBounds = list({pair[0] for pair in y_periodic_pair_ids})
    PosYBounds = list({pair[1] for pair in y_periodic_pair_ids})
    dummy_node_id = 5000000
    for pair in x_periodic_pair_ids:
        f.write("*Equation\n")
        f.write("3\n")
        f.write("%i, 1, 1., %i, 1, -1., %i, 1, -1.\n" % (pair[1], pair[0], dummy_node_id))
        dummy_node_id += 1
    dummy_node_id = 5000000
    for pair in x_periodic_pair_ids:
        f.write("*Equation\n")
        f.write("3\n")
        f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
        dummy_node_id += 1
    dummy_node_id = 6000000
    for pair in y_periodic_pair_ids:
        f.write("*Equation\n")
        f.write("3\n")
        f.write("%i, 2, 1., %i, 2, -1., %i, 2, -1.\n" % (pair[1], pair[0], dummy_node_id))
        dummy_node_id += 1
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
    nodes_dict = {node[0]: tuple(node[1:]) for node in nodes}
    StabilizationNodeId = FindStabilizationNode(nodes_dict)
    f.write("\n *Kinematic Coupling, Ref Node =%i" % StabilizationNodeId)
    f.write("\n NegYBounds, 4, 6")
    f.write("\n *Kinematic Coupling, Ref Node =%i" % StabilizationNodeId)
    f.write("\n PosYBounds, 4, 6")
    # STEP DEFINITION
    f.write("\n *Time Points, name=MustPoints, GENERATE")
    f.write("\n 0., 1., 0.1")
    f.write("\n *Step, name=UniaxialTest, INC=1000")
    f.write("\n *Static,  stabilize, factor=2E-4")
    f.write("\n 0.01, 1., 1e-10, 0.1")
    # AMPLITUDE DEFINITION
    f.write("\n *Amplitude, name=SmoothStep, definition=SmoothStep, time=TOTAL TIME")
    f.write("\n 0.0, 0.0, 1.0, 1.0")
    # HISTORY BOUNDARY DEFINITION
    # HISTORY BOUNDARY DEFINITION
    f.write("\n *Boundary, type = Displacement")  # LOAD DIRECTION
    f.write("\n DummyNodes, 1, 1, 0")
    f.write("\n DummyNodes, 2, 2, 1")
    f.write("\n *Boundary, type = Displacement")
    f.write("\n DummyYNodes, 2, 2, 0")
    f.write("\n *Boundary, type = Displacement")
    f.write("\n NegXBounds, 1, 1, 0")
    f.write("\n *Boundary, type = Displacement")
    f.write("\n PosXBounds, 1, 1, 0")
    f.write("\n *Boundary, type = Displacement")
    f.write("\n NegXBounds, 4, 4, 0")
    f.write("\n *Boundary, type = Displacement")
    f.write("\n PosXBounds, 4, 4, 0")
    f.write("\n *Boundary, type = Displacement")
    f.write("\n %i, 1, 3, 0" % StabilizationNodeId)
    # OUTPUT
    f.write("\n *Output, history, variable=PRESELECT, time points=MustPoints")
    f.write("\n *Output, field, variable=PRESELECT, time points=MustPoints ")
    f.write("\n *Node Output, nset= AllNodes")
    f.write("\n  U, RF")
    f.write("\n *Output, HISTORY, VARIABLE=PRESELECT, time interval = 0.1")
    f.write("\n *Restart, Write, FREQUENCY=10")
    f.write("\n *End Step")
f.close()