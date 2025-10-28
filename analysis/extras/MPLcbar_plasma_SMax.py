"""
Color Scale Bar Generator

Copyright (c) 2016 by Mathias Peirlinck

All rights reserved. No part of this code may be reproduced, distributed, or transmitted in any form or by any means, without the prior written permission of the author.
If code is/was used for scientific research, you are obliged to contact the author for co-authorship and/or acknowledgements.
For permission or publishing results (partly) based on this code, write to the authors at the address below.

Mathias Peirlinck
mathias.peirlinck@ugent.be
Ghent University / Biofluid, Tissue and Solid Mechanics for Medical Applications Lab
Campus UZ
De Pintelaan 185
Ingang 36 (BLOK B)
B-9000, Gent, Belgium


Possible color values are:
Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r,
CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r,
OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r,
Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd,
PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r,
RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral,
Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r,
YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r,
brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r,
cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r,
gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern,
gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r,
hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral,
nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r,
rainbow, rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r,
terrain, terrain_r, viridis, viridis_r, winter, winter_r
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import os, sys
import numpy as np
import matplotlib as mpl
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
font_path = 'D:/FONT/SourceSansPro-Regular.otf'
font_manager.fontManager.addfont(font_path)

# Set the font globally for matplotlib
rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] =  ['Source Sans Pro']
import os
import datetime
import time
output_directory = None

def setup_output_directory(job_description):
    today = datetime.datetime.now()
    date_folder = today.strftime("%Y%m%d")
    #base_dir = r'U:\MAIN\AbaqusRepo\2024SanityCheck'
    base_dir =r'D:\CollaGEN\AbaqusFiles'
    output_directory = os.path.join(base_dir, date_folder, job_description)
    os.makedirs(output_directory, exist_ok=True)
    return output_directory


job_description = "ResultsPlots"
output_directory = setup_output_directory(job_description)

# cmap = plt.get_cmap('Accent',100)
# cmap = plt.get_cmap('Accent_r',100)
# cmap = plt.get_cmap('viridis')
# cmap = plt.get_cmap('viridis_r')

colorname='magma_r'
cmap = plt.get_cmap(colorname)

cm_data=[]
for i in range(cmap.N):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    cm_data.append(matplotlib.colors.rgb2hex(rgb))

hexlist=''
for i in cm_data:
    hexlist+="'%s', "%(i)


'''
Make a colorbar as a separate figure.
'''

from matplotlib import pyplot


# Make a figure and axes with dimensions as desired.
# fig = pyplot.figure(figsize=(0.5,1.5))
# ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
# ax1 = fig.add_axes([0.1, 0.05, 0.15, 0.90])#[left, bottom, width, height]
# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
# cmap = mpl.cm.cool
#~ norm = mpl.colors.LogNorm(vmin=1.0, vmax=1.8)
norm = mpl.colors.Normalize(vmin=0.,vmax=1.)

# lognorm = mpl.colors.LogNorm(norm)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.

#~ lvls=[1.0,1.2,1.4,1.6,1.8]
# lvlslabels=['1.00','1.25','1.5','1.75','2.00']
#~ lvlslabels=['$\\mathdefault{1.0}$','$\\mathdefault{1.2}$','$\\mathdefault{1.4}$','$\\mathdefault{1.6}$','$\\mathdefault{1.8}$']

# VERTICAL COLORBAR
# cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical', ticklocation='right', ticks = lvls, spacing='proportional')
# cb1.ax.tick_params(direction='inout',length=8,width=1,labelsize=20)
# cb1.set_ticklabels(lvlslabels)

# ticks = cb1.ax.get_yticks()
# tickStrs = [label.get_text() for label in cb1.ax.get_yticklabels()]
# ticksnew=np.append(ticks,[np.float64(1.0)])
# tickStrsnew=np.append(tickStrs,['']).tolist()
# print(ticks)
# print(tickStrs)
# cb1.set_ticks(ticksnew, update_ticks=True)

# cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical', ticklocation='left', spacing='proportional')
# cb1.ax.tick_params(direction='inout',length=8,width=1,labelsize=20)
# plt.show()
# exit()



# ax = cb1.ax
# ax.text(0.0,1.10,'Stress Mises (kPa)',fontsize=40,fontweight='bold')
# cb1.set_label('S')
# plt.savefig('%s_V.svg'%(__file__[3:-3]), format='svg', dpi=1200, transparent=True)
# plt.savefig('%s_V.png'%(__file__[:-3]), format='png', dpi=1200, transparent=True)
# print('Saved figure %s_V.png'%(__file__[:-3]))
# Define the color map and normalization
cmap = mpl.cm.magma_r  # Adjust based on the color scale in your image
norm = mpl.colors.Normalize(vmin=-10, vmax=100)  # Range based on the provided scale

# Define the ticks and their corresponding labels
ticks = [-10, 0, 10, 23, 34, 45, 56, 67, 78, 89, 100]  # As per your legend
tick_labels = [
    "-10", "0", "10", "23", "34",
    "45", "56", "67", "78", "89", "100"
]

# Set up the figure and the horizontal colorbar
fig = plt.figure(figsize=(10, 0.5))  # Adjust width for better readability
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]

# Create the colorbar
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal', spacing='proportional')
cb1.set_ticks(ticks)
cb1.set_ticklabels(tick_labels)
cb1.outline.set_linewidth(0.5)

# Customize ticks
cb1.ax.tick_params(direction='inout', width=0.5, labelsize=8)
output_path = os.path.join(output_directory, "horizontal_colormap.svg")
plt.savefig(output_path, format="svg", dpi=300, transparent=True)
plt.show()

# VERTICAL COLORBAR
fig = pyplot.figure(figsize=(0.5, 2.0))
ax1 = fig.add_axes([0.9, 0.1, 0.1, 0.8])#[left, bottom, width, height]
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical',ticklocation='left',spacing='proportional') # , ticks = lvls
# cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal', ticklocation='top', spacing='proportional')
cb1.set_ticks([0.1,2.5,5 ,7.5, 10])
# cb1.ax.tick_params(direction='inout',width=0.25,labelsize='x-small')	#,length=8,width=0.5,labelsize=20
cb1.outline.set_linewidth(0.00)
#~ cb1.set_ticklabels(lvlslabels)
cb1.ax.minorticks_off()
# plt.savefig('%s_V.svg'%(__file__[:-3]))
# print('Saved figure %s_V.svg'%(__file__[:-3]))
plt.show()


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
dark_purple_rgb = (48 / 255, 0 / 255, 72 / 255)  # RGB for the purple color

# Create a custom colormap transitioning to purple
purple_custom_colormap = LinearSegmentedColormap.from_list(
    "purple_custom_colormap",
    ["#000000", "#300048", "#FFFFFF"],  # Black, Dark Purple, White
    N=256
)
purple_custom_colormap
reversed_purple_custom_colormap = purple_custom_colormap.reversed()
reversed_purple_custom_colormap
boundary_colormap = cm.get_cmap(reversed_purple_custom_colormap, 100)
# Set up the figure and axes
fig = plt.figure(figsize=(1, 4))
#ax1 = fig.add_axes([0.9, 0.1, 0.1, 0.8])  # [left, bottom, width, height]
ax1 = fig.add_axes([0.6, 0.1, 0.40, 0.8])
# Define the colormap and normalization
cmap = plt.get_cmap(boundary_colormap)  # Use any colormap
norm = mpl.colors.Normalize(vmin=-1, vmax=1)  # Set normalization for negative to positive values

# Create the colorbar
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical', ticklocation='left', spacing='proportional')

# Set the desired ticks with negative values
cb1.set_ticks([- 1, - 0.5, 0, 0.5, 1])

# Optional: Customize the ticks further if needed
#cb1.ax.tick_params(direction='inout', width=0.25, labelsize='x-small')  # Adjust tick appearance
cb1.ax.tick_params(direction='inout', width=0.2,  labelsize='x-large')
# Remove colorbar outline if desired
cb1.outline.set_linewidth(0.00)

# Disable minor ticks
cb1.ax.minorticks_off()
# plt.savefig('%s_V.svg'%(__file__[:-3]))
# print('Saved figure %s_V.svg'%(__file__[:-3]))
# Show the plot
output_path = os.path.join(output_directory, 'colorbar_plot.svg')
plt.savefig(output_path, format='svg')
plt.show()
