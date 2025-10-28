import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

def parse_msg_file(filename):
    increments = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    current = {}
    for i, line in enumerate(lines):
        inc_match = re.match(r'^\s*INCREMENT\s+(\d+)\s+STARTS\. ATTEMPT NUMBER\s+(\d+), TIME INCREMENT\s+([-\d\.E]+)', line)
        if inc_match:
            if current:
                increments.append(current)
                current = {}
            current['increment'] = int(inc_match.group(1))
            current['attempt'] = int(inc_match.group(2))
            current['time_increment'] = float(inc_match.group(3))
            current['warnings'] = []
            current['force_nodes'] = []
            current['moment_nodes'] = []
        
        force_match = re.search(r'LARGEST SCALED RESIDUAL FORCE\s+([-\d\.E]+)\s+AT NODE\s+(\d+)\s+DOF\s+(\d+)', line)
        if force_match:
            current['residual_force'] = float(force_match.group(1))
            current['force_node'] = int(force_match.group(2))
            current['force_dof'] = int(force_match.group(3))
            current['force_nodes'].append(current['force_node'])

        corr_disp_match = re.search(r'LARGEST CORRECTION TO DISP\.\s+([-\d\.E]+)\s+AT NODE\s+(\d+)\s+DOF\s+(\d+)', line)
        if corr_disp_match:
            current['corr_disp'] = float(corr_disp_match.group(1))
            current['corr_disp_node'] = int(corr_disp_match.group(2))
            current['corr_disp_dof'] = int(corr_disp_match.group(3))

        moment_match = re.search(r'LARGEST SCALED RESIDUAL MOMENT\s+([-\d\.E]+)\s+AT NODE\s+(\d+)\s+DOF\s+(\d+)', line)
        if moment_match:
            current['residual_moment'] = float(moment_match.group(1))
            current['moment_node'] = int(moment_match.group(2))
            current['moment_dof'] = int(moment_match.group(3))
            current['moment_nodes'].append(current['moment_node'])

        corr_rot_match = re.search(r'LARGEST CORRECTION TO ROTATION\s+([-\d\.E]+)\s+AT NODE\s+(\d+)\s+DOF\s+(\d+)', line)
        if corr_rot_match:
            current['corr_rot'] = float(corr_rot_match.group(1))
            current['corr_rot_node'] = int(corr_rot_match.group(2))
            current['corr_rot_dof'] = int(corr_rot_match.group(3))

        if "***WARNING:" in line or "***NOTE:" in line:
            current['warnings'].append(line.strip())

        if i == len(lines) - 1 and current:
            increments.append(current)
    return increments

def plot_trends(increments):
    inc_nums = [inc['increment'] for inc in increments]
    residual_forces = [inc.get('residual_force', 0) for inc in increments]
    corr_disps = [inc.get('corr_disp', 0) for inc in increments]
    residual_moments = [inc.get('residual_moment', 0) for inc in increments]
    corr_rots = [inc.get('corr_rot', 0) for inc in increments]
    warnings = [len(inc['warnings']) > 0 for inc in increments]
    time_increments = [inc.get('time_increment', 0) for inc in increments]

    plasma = plt.get_cmap('plasma')
    colors = plasma(np.linspace(0, 1, 4))

    plt.figure(figsize=(16,8))
    plt.subplot(2,2,1)
    plt.plot(inc_nums, residual_forces, marker='o', label='Residual Force', color=colors[0])
    plt.xlabel('Increment')
    plt.ylabel('Residual Force')
    plt.title('Residual Force vs Increment')
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(inc_nums, corr_disps, marker='o', label='Correction to Disp.', color=colors[1])
    plt.xlabel('Increment')
    plt.ylabel('Correction to Displacement')
    plt.title('Disp. Correction vs Increment')
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(inc_nums, residual_moments, marker='o', label='Residual Moment', color=colors[2])
    plt.xlabel('Increment')
    plt.ylabel('Residual Moment')
    plt.title('Residual Moment vs Increment')
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(inc_nums, corr_rots, marker='o', label='Correction to Rotation', color=colors[3])
    plt.xlabel('Increment')
    plt.ylabel('Correction to Rotation')
    plt.title('Rotation Correction vs Increment')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(inc_nums, time_increments, marker='o', color=plasma(0.6))
    plt.xlabel('Increment')
    plt.ylabel('Time Increment')
    plt.title('Time Increment Evolution')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(inc_nums, warnings, 'o', color=plasma(0.9))
    plt.xlabel('Increment')
    plt.ylabel('Warning (1=True)')
    plt.title('Increments with Warnings/Notes')
    plt.grid(True)
    plt.show()

def node_heatmap(increments, key='force_nodes'):
    node_count = Counter()
    for inc in increments:
        for n in inc.get(key, []):
            node_count[n] += 1
    most_common = node_count.most_common(10)
    print(f"\nTop problematic nodes (by {key}):")
    for node, count in most_common:
        print(f"  Node {node}: {count} times")

def parse_input_file_for_static(input_filename):
    """
    Parses the input file to find *Static block and returns the initial time increment.
    Returns None if not found.
    """
    initial_time_increment = None
    static_line = None
    static_params = None
    with open(input_filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "*Static" in line:
            static_line = line.strip()
            # Look for next non-empty, non-comment line
            for j in range(i+1, len(lines)):
                dataline = lines[j].strip()
                if dataline and not dataline.startswith('*') and not dataline.startswith('**'):
                    static_params = dataline
                    try:
                        initial_time_increment = float(dataline.split(',')[0])
                        return initial_time_increment, static_line, static_params
                    except Exception:
                        continue
    return initial_time_increment, static_line, static_params

def parse_sta_file(filename, reference_increment, fraction=0.1):
    """
    Parses .sta file, flags increments with time increment less than fraction*reference_increment.
    """
    problematic_increments = []
    with open(filename, 'r') as f:
        for line in f:
            cols = line.split()
            if len(cols) < 2:
                continue
            try:
                increment = int(cols[0])
                time_increment = float(cols[-1])
                if reference_increment is not None and time_increment < (fraction * reference_increment):
                    problematic_increments.append((increment, time_increment))
            except ValueError:
                continue
    return problematic_increments

def cross_reference_problematic_nodes(increments, problematic_inc_nums):
    force_nodes = Counter()
    moment_nodes = Counter()
    for inc in increments:
        if inc['increment'] in problematic_inc_nums:
            for n in inc.get('force_nodes', []):
                force_nodes[n] += 1
            for n in inc.get('moment_nodes', []):
                moment_nodes[n] += 1
    print("\nProblematic increments (cutbacks):", problematic_inc_nums)
    print("\nTop problematic nodes (force) in cutback increments:")
    for node, count in force_nodes.most_common(10):
        print(f"  Node {node}: {count} times")
    print("\nTop problematic nodes (moment) in cutback increments:")
    for node, count in moment_nodes.most_common(10):
        print(f"  Node {node}: {count} times")

# ---- NEW: print node id(s) for largest corrections across all increments ----
def _find_max_by_key(increments, val_key, node_key, dof_key):
    best = None
    for inc in increments:
        if val_key in inc:
            val = abs(inc[val_key])
            if best is None or val > best['value']:
                best = {
                    'value': val,
                    'raw_value': inc[val_key],
                    'node': inc.get(node_key),
                    'dof': inc.get(dof_key),
                    'increment': inc.get('increment')
                }
    return best

def print_largest_correction_nodes(increments):
    """
    Prints:
    - Largest correction to displacement and its node/DOF/increment
    - Largest correction to rotation and its node/DOF/increment
    - Overall largest correction by magnitude among the two (note: different units)
    """
    max_disp = _find_max_by_key(increments, 'corr_disp', 'corr_disp_node', 'corr_disp_dof')
    max_rot  = _find_max_by_key(increments, 'corr_rot',  'corr_rot_node',  'corr_rot_dof')

    print("\nLargest corrections across all increments:")
    if max_disp:
        print(f"  Displacement: {max_disp['raw_value']} at node {max_disp['node']} DOF {max_disp['dof']} (increment {max_disp['increment']})")
    else:
        print("  Displacement: no records found")

    if max_rot:
        print(f"  Rotation:     {max_rot['raw_value']} at node {max_rot['node']} DOF {max_rot['dof']} (increment {max_rot['increment']})")
    else:
        print("  Rotation:     no records found")

    if max_disp and max_rot:
        label, overall = max([('displacement', max_disp), ('rotation', max_rot)], key=lambda x: x[1]['value'])
        print(f"  Overall max by magnitude: {label} = {overall['raw_value']} at node {overall['node']} DOF {overall['dof']} (increment {overall['increment']})")

if __name__ == "__main__":
    msg_file = "U:\\MAIN\\HyperCANs\\20251017_MeetingPre\\Sample_2_rotational_damping\\YZ_Equibiaxial_Tension.msg"
    sta_file = "U:\\MAIN\\HyperCANs\\20251017_MeetingPre\\Sample_2_rotational_damping\\YZ_Equibiaxial_Tension.sta"
    inp_file = "U:\\MAIN\\HyperCANs\\20251017_MeetingPre\\Sample_2_rotational_damping\\YZ_Equibiaxial_Tension.inp"  # <-- Add your input file path
    increments = parse_msg_file(msg_file)

    # Print node id(s) for the largest corrections
    print_largest_correction_nodes(increments)

    plot_trends(increments)
    node_heatmap(increments, key='force_nodes')
    node_heatmap(increments, key='moment_nodes')

    # ---- Parse integration scheme from input file ----
    reference_increment, static_line, static_params = parse_input_file_for_static(inp_file)
    print("\nIntegration scheme info from input file:")
    if static_line:
        print("  Step keyword:", static_line)
    if static_params:
        print("  Step parameters:", static_params)
    print(f"  Initial time increment parsed: {reference_increment}")

    # ---- NEW STA FILE ANALYSIS ----
    #problematic_incs = parse_sta_file(sta_file, reference_increment, fraction=0.1) # 10% threshold, change as needed
    #problematic_inc_nums = [inc for inc, dt in problematic_incs]
    #cross_reference_problematic_nodes(increments, problematic_inc_nums)