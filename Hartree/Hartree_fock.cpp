import numpy
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
# ============ for latex fonts ============
from matplotlib import rc #, font_manager
rc('text.latex', preamble=r'\usepackage{lmodern}')# this helps use the plots in tex files
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'xtick.labelsize': 14,
		  'ytick.labelsize': 14,
		  'xtick.major.pad': 6,
		  'ytick.major.pad': 6,
		  'font.serif': 'Computer Modern Roman',
		  'axes.formatter.use_mathtext': True,
		  'axes.labelpad': 6.0 }) 
# ==========================================

# plt.figure(figsize=(8, 6), dpi=80)
plt.figure()
Ha = 27.211 # 1 Hartree in eV
a0 = 0.05292 # Bohr radius in nm
m = 0.067 # effective mass for GaAs
eps = 12.5 # dielectric constant for GaAs