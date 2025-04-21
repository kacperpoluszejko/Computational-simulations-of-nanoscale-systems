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
# plt.figure()
Ha = 27.211 # 1 Hartree in eV
a0 = 0.05292 # Bohr radius in nm
m = 0.067 # effective mass for GaAs
eps = 12.5 # dielectric constant for GaAs
kappa = 1/eps



@nb.njit
def urojony(dlugosc):

    n = 41
    l = 10/a0
    a = dlugosc/a0
    dx = 2*a/(n-1)
    dt = m*dx*dx*0.4

    psi = np.zeros((n,n))

    # fill the wave function with random numbers
    psi = np.random.random((n, n)) * 2 - 1

    # boundary condition: 
    psi[0, :] = 0
    psi[n - 1, :] = 0
    psi[:, 0] = 0
    psi[:, n - 1] = 0


    V = numpy.zeros((n, n)) # potential energy 
    X1 = numpy.zeros((n, n))
    X2 = numpy.zeros((n, n))
    for i in range(0, n): 
        for j in range(0, n):
            x1 = (i - n//2) * dx
            x2 = (j - n//2) * dx
            X1[i,j] = x1
            X2[i,j] = x2
            V[i, j] = kappa/(np.sqrt((x1-x2)**2+l*l))
    Ek_n = 1
    Ek_old = 2
    iteration = 0
    # energia = []
    # iteration_table = []

    while ( np.abs((Ek_n - Ek_old)/Ek_old) > 1e-9):
        if (iteration % 100 == 0 and iteration != 0):
            print(f"iteracja: ", iteration, np.abs((Ek_n - Ek_old)/Ek_old))

            # iteration_table.append(iteration)
            # energia.append(Ek_n*Ha)
        iteration = iteration + 1
        Ek_old = Ek_n

        # calculate F = H \psi 
        F = np.zeros((n, n))
        for i in np.arange(1, n-1):
            for j in np.arange(1, n-1):
                F[i, j] += -(1/2/m)*(psi[i + 1, j] + psi[i - 1, j] + psi[i, j-1] + psi[i, j+1] - 4*psi[i, j])/dx/dx
                F[i, j] += psi[i, j] * V[i,j]

        psi = psi - dt * F

        norm = np.sum(np.abs(psi**2)) *dx*dx
        psi = psi / np.sqrt(norm)

        Ek_n = np.sum( np.conjugate(psi)*F )*dx*dx
    # f.close()
    return X1, X2, np.abs(psi)**2

#Zad 3
# X1, X2, PSI = urojony(30)
# plt.pcolor(X1*a0, X2*a0, PSI, cmap='viridis', shading='auto')
# plt.xlabel(r"$x1\ [\mathrm{nm}]$", usetex=True)
# plt.ylabel(r"$x2\ [\mathrm{nm}]$", usetex=True)
# plt.colorbar()
# plt.savefig("Mezo2_4.jpg", bbox_inches='tight')
# plt.show()

#Zad 2
# lenght = np.arange(30, 65, 5)
# f = open("Mezo2_2.txt", 'w')
# for i in range (len(lenght)):
#     ene = urojony(lenght[i])
#     f.write(str(lenght[i]) + " " + str(ene) + "\n")
# f.close()

#Zad 1
# f = open("E_HF_L.txt", 'w')
# for i, en in enumerate(ene):
#     f.write(str(it[i]) + " " +  str(en) + "\n")
# f.close()