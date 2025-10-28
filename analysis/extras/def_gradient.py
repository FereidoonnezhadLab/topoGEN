import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from TopoGEN.utils.setup import setup_output_directory
from matplotlib import rcParams, font_manager
font_path_regular = 'D:/FONT/SourceSansPro-Regular.otf'
font_path_bold = 'D:/FONT/SourceSansPro-Bold.otf'
font_manager.fontManager.addfont(font_path_regular)
font_manager.fontManager.addfont(font_path_bold)
rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'regular'
job_description = "def_map"
output_directory = setup_output_directory(job_description)

# Coarse Grid Setup
coarse_grid_size = 10  # use fewer polygons
x_min, x_max = -20, 20
y_min, y_max = -20, 20

# Define coordinates of coarse grid centers
x = np.linspace(x_min + (x_max - x_min)/coarse_grid_size/2, x_max - (x_max - x_min)/coarse_grid_size/2, coarse_grid_size)
y = np.linspace(y_min + (y_max - y_min)/coarse_grid_size/2, y_max - (y_max - y_min)/coarse_grid_size/2, coarse_grid_size)
X, Y = np.meshgrid(x, y)

# Displacement field: u_x = y, u_y = 0
u_x = Y
u_y = np.zeros_like(Y)

# Set up the color map
cmap = plt.colormaps["PuBuGn"]
norm = plt.Normalize(vmin=-20, vmax=20)

# Create figure and draw deformed cells
fig, ax = plt.subplots(figsize=(8, 8))

cell_width = (x_max - x_min) / coarse_grid_size
cell_height = (y_max - y_min) / coarse_grid_size

for i in range(coarse_grid_size):
    for j in range(coarse_grid_size):
        cx, cy = X[i, j], Y[i, j]
        # Define corners of each coarse cell
        corners_x = np.array([cx - cell_width/2, cx + cell_width/2, cx + cell_width/2, cx - cell_width/2])
        corners_y = np.array([cy - cell_height/2, cy - cell_height/2, cy + cell_height/2, cy + cell_height/2])
        
        # Apply displacement: u_x = y → shift corners_x by corners_y
        displaced_corners_x = corners_x + corners_y
        displaced_corners_y = corners_y

        color = cmap(norm(u_x[i, j]))
        ax.fill(displaced_corners_x, displaced_corners_y, color=color, edgecolor=None)

# Adjust plot settings
ax.set_xlim(x_min - 20, x_max + 20)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.axis('off')

# Save main figure
output_path = os.path.join(output_directory, "def_map.svg")
fig.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
plt.close(fig)

# === Generate Separate Legend ===
fig_legend, ax_legend = plt.subplots(figsize=(0.6, 6))
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig_legend.colorbar(sm, cax=ax_legend, orientation='vertical')
cbar.set_label("$u_x$", rotation=90, fontsize=14)
cbar.ax.tick_params(labelsize=12)

legend_path = os.path.join(output_directory, "def_map_legend.svg")
fig_legend.savefig(legend_path, format='svg', bbox_inches='tight', pad_inches=0)
plt.close(fig_legend)