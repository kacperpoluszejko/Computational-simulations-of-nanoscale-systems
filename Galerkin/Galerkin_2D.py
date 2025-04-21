from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
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

n = 9 # no Gaussian functions in a basispi
N = n*n #liczba węzłów
Ha = 27.211 # 1 Hartree in eV
a0 = 0.0529 # Bohr radius in nm
m = 0.24 #0.067 # effective mass for GaAs
omega_x = 0.08/Ha
omega_y = 0.4/Ha
alfax = 1/m / omega_x
alfay = 1/m / omega_y


dx = 1/a0
a = dx * (n-1)/2
xk = np.zeros(N)
yk = np.zeros(N)

#Tworzenie siatki węzłów
for k in range(N):
    i = k // n  # dzielenie całkowite
    j = k % n   # reszta z dzielenia
    xk[k] = -a + dx * i
    yk[k] = -a + dx * j



def funkcja_bazowa(k, x, y, omegax, omegay):
    alfax = 1/(m * omegax)
    alfay = 1/(m * omegay)
    x_k = xk[k]
    y_k = yk[k]
    return (np.exp( -0.5/alfax * (x - x_k)**2) * (1 / alfax / np.pi)**0.25)*(np.exp( -0.5/alfay * (y - y_k)**2) * (1 / alfay / np.pi)**0.25)

#Sij values
def O_value(i, j, omegax, omegay):
    alfax = 1/(m * omegax)
    alfay = 1/(m * omegay)
    xi = xk[i]
    yi = yk[i]
    xj = xk[j]
    yj = yk[j]
    return np.exp(-(xi - xj)**2 / 4 / alfax -(yi - yj)**2 / 4 / alfay)

# Kij integrals 
def Ek_value(i, j, omegax, omegay):
    alfax = 1/(m * omegax)
    alfay = 1/(m * omegay)
    xi = xk[i]
    yi = yk[i]
    xj = xk[j]
    yj = yk[j]
    return (( (xi - xj)**2 - 2*alfax  ) / ( alfax * 2 )**2 + ( (yi - yj)**2 - 2*alfay  ) / ( alfay * 2 )**2 ) * O_value(i, j, omegax, omegay)* (-0.5)/m

# Vij integrals 
def Ep_value(i, j, omegax, omegay):
    alfax = 1/(m * omegax)
    alfay = 1/(m * omegay)
    xi = xk[i]
    yi = yk[i]
    xj = xk[j]
    yj = yk[j]
    return ( (omegax**2) * ((xi + xj)**2 + 2*alfax)/4 + (omegay**2) * ((yi + yj)**2 + 2*alfay)/4) * O_value(i, j, omegax, omegay) * 0.5 * m

def make_H_S(omegax, omegay):
    H = np.zeros((N, N))
    S = np.zeros((N, N)) 
    for i in np.arange(0, N):
        for j in np.arange(0, N):
            S[i, j] = O_value(i, j, omegax, omegay)
            H[i, j] = Ek_value(i, j, omegax, omegay) + Ep_value(i, j, omegax, omegay) 
    return (H, S)


# # Rysowanie Gaussjanów dla k = 0, 8, 9
# grid_points = 100
# x = np.linspace(-a, a, grid_points)
# y = np.linspace(-a, a, grid_points)
# X, Y = np.meshgrid(x, y)
# ks = [0, 8, 9]
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# for idx, k in enumerate(ks):
#     Z = funkcja_bazowa(k, X, Y, omega_x, omega_y)
#     ax = axs[idx]
#     c = ax.pcolor(X*a0, Y*a0, Z, cmap='viridis', shading='auto')
#     fig.colorbar(c, ax=ax)
#     ax.set_title(f'k={k}')
#     ax.set_xlabel('x [nm]')
#     ax.set_ylabel('y [nm]')
# plt.tight_layout()
# plt.savefig("Funkcja_bazowa.jpg")
# plt.show()



H, S = make_H_S(omega_x, omega_y)
eigvals, eigvecs = eigh(H, S, eigvals_only=False)
npoint = 300
x = np.linspace(-a, a, npoint)
y = np.linspace(-a, a, npoint)
X, Y = np.meshgrid(x, y)
psi = np.zeros((npoint, npoint)) * 1j
for k in np.arange(N):
    for i in np.arange(npoint):
            for j in np.arange(npoint):
                psi[i, j] = psi[i, j] + funkcja_bazowa(k, X[i ,j], Y[i ,j], omega_x, omega_y) * eigvecs[k, 5]
        # gausses[j, i] = funkcja_bazowa(i, xs[j], alfax, dx) * eigvecs[k, 1]



plt.pcolor(X*a0, Y*a0, (abs(psi)**2), cmap='viridis', shading='auto')
plt.xlabel(r"$x\ [\mathrm{nm}]$", usetex=True)
plt.ylabel(r"$y\ [\mathrm{nm}]$", usetex=True)
plt.colorbar()
plt.savefig("Stan5_2.jpg")
plt.show()

# OMEGAX = np.linspace(0.01/Ha, 0.5/Ha, 30)
# energies_all = []
# e_analytic_all = []
# for j in range(10):
#     energies = []
#     e_analytic = []
#     for i in range(len(OMEGAX)):
#         H, S = make_H_S(OMEGAX[i], omega_y)
#         eigvals, eigvecs = eigh(H, S, eigvals_only=False)
#         energies.append(eigvals[j]*Ha)
#         analytic = OMEGAX[i]*(0.5+j) + omega_y*(0.5)
#         e_analytic.append(analytic*Ha)

#     energies_all.append(energies)
#     e_analytic_all.append(e_analytic)
# #E_n = h*omegax*(0.5+n) + h*omegay*(0.5+n)

# plt.figure(figsize=(10, 6))
# for i, y in enumerate(energies_all):
#     plt.plot(OMEGAX*Ha, y)
#     if (i<3):
#       plt.plot(OMEGAX*Ha, e_analytic_all[i],linestyle = '--')
# plt.title(r"$E\ (\omega_x)$")
# plt.xlabel(r"$\omega\ [\mathrm{eV}]$", usetex=True)
# plt.ylabel(r"$E\ [\mathrm{eV}]$", usetex=True)
# plt.savefig("E_omegax.jpg")
# plt.show()

