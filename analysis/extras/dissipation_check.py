import numpy as np
import matplotlib.pyplot as plt

# Given data for strain energy and static dissipation
time_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
strain_energy = np.array([0, 1.41892E-05, 0.00065037, 0.00522198, 0.0206716, 0.0562287,
                          0.120796, 0.215267, 0.317232, 0.382143, 0.394874])
static_dissipation = np.array([0, 1.08506E-08, 2.34754E-07, 1.15341E-06, 3.09984E-06,
                               5.878E-06, 1.37485E-05, 0.0010075, 0.00148408, 0.0015955, 0.00160101])

# Smooth step function
def smooth_step(t):
    return t**3 * (10 - 15*t + 6*t**2)

# Calculate strain based on smooth step function
t_values = np.linspace(0.0, 1.0, 11)
strain = np.array([smooth_step(time) for time in t_values]) * 0.5

# Calculate Dissipation Fraction (STATIC DISSIPATION / STRAIN ENERGY)
dissipation_fraction = static_dissipation / strain_energy

plt.figure(figsize=(10, 6))
plt.plot(strain, dissipation_fraction, marker='o', linestyle='-', color='darkblue', label="Dissipation Fraction")
plt.axhline(y=0.05, color='darkred', linestyle='--', label="Threshold (0.05)")
plt.xlabel('Strain', fontsize=16)
plt.ylabel('Dissipation Fraction', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Given time points and dissipation data for 5 samples
time_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

samples = {
    "Sample 1": np.array([0, 0.000662637, 0.000315068, 0.00019367, 0.000126843, 8.32027E-05,
                          5.37188E-05, 3.49997E-05, 0.000100384, 9.40557E-05, 9.12575E-05]),
    "Sample 2": np.array([0, 0.000664785, 0.000317895, 0.000193546, 0.000124691, 7.84188E-05,
                          0.000479305, 0.000490018, 0.000388079, 0.000328811, 0.000317859]),
    "Sample 3": np.array([0, 0.000673664, 0.000315386, 0.000183843, 0.000112305, 6.75602E-05,
                          0.000833749, 0.000440486, 0.000282124, 0.000226949, 0.000218406]),
    "Sample 4": np.array([0, 0.000674422, 0.00031203, 0.000179089, 0.0001089, 6.68485E-05,
                          4.0751E-05, 2.52397E-05, 1.69895E-05, 1.37237E-05, 1.31923E-05]),
    "Sample 5": np.array([0, 0.000704069, 0.000331038, 0.000195966, 0.000124842, 8.00939E-05,
                          5.13075E-05, 3.36018E-05, 2.38139E-05, 1.97456E-05, 1.90661E-05])
}

# Plotting the dissipation for each sample
plt.figure(figsize=(10, 6))

for sample_name, dissipation_data in samples.items():
    plt.plot(time_points, dissipation_data, marker='o', linestyle='-', label=sample_name)

plt.axhline(y=0.05, color='darkred', linestyle='--', label="Abaqus Threshold")
plt.xlabel('Time', fontsize=16)
plt.ylabel('Dissipation', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0)
plt.show()
