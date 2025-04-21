import numpy
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.linalg import eigh
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
Ha = 27.211 # 1 Hartree in eV
a0 = 0.05292 # Bohr radius in nm
m = 0.067 # effective mass for GaAs

a = 15/a0 # in nm
d1 = 5/a0
d2 = 10/a0
n = 100
dz = a/(n)
z_table = (np.arange(n)) * dz
V1 = 0.27/Ha
m1 = 0.063
m2 = 0.083*0.3
T = 77

V =  np.zeros(n, dtype = complex)
for i in range(n):
    x = i * dz
    if(x > (d1) and x < (d2)):
        V[i] = V1

M =  np.zeros(n, dtype = complex)
for i in range(n):
    x = i * dz
    if(x > (d1) and x < (d2)):
        M[i] = (m1+m2)
    else:
        M[i] = m1

E_table = np.linspace(0, 1/Ha, 2)
T_table = []
R_table = []
for E in E_table:

    k =  np.zeros(n, dtype = complex)
    for i in range(n):
      k[i] = np.sqrt(2*M[i]*(E - V[i]))


    macierz = np.identity(2, dtype=complex)
    Monodrom = np.zeros((2, 2), dtype=complex)

    for i in range(n-1):
        z = i * dz
        wsp = (k[i+1]*M[i]) / (k[i]/M[i+1])
        print(wsp)
        Monodrom[0][0] = 0.5 * (1 + wsp) * np.exp(1j*z*(k[i+1] - k[i]))
        Monodrom[0][1] = 0.5 * (1 - wsp) * np.exp(-1j*z*(k[i+1] + k[i]))
        Monodrom[1][0] = 0.5 * (1 - wsp) * np.exp(1j*z*(k[i+1] + k[i]))
        Monodrom[1][1] = 0.5 * (1 + wsp) * np.exp(-1j*z*(k[i+1] - k[i]))
        # print(Monodrom)
        macierz = macierz @ Monodrom

    T = (k[n-1]*M[0])/(k[0]*M[n-1]) * (1 / np.abs(macierz[0][0])**2)
    R = np.abs(macierz[1][0])**2/np.abs(macierz[0][0])
    T_table.append(T)
    R_table.append(R)

plt.plot(E_table, T_table)
plt.show()
plt.plot(E_table, R_table)
plt.show()
plt.plot(E_table, T_table + R_table)
plt.show()

    