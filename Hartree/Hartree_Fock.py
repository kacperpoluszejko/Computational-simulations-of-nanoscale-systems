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
def H_T(dlugosc):

    n = 41
    l = 10/a0
    a = dlugosc/a0
    dx = 2*a/(n-1)
    dt = m*dx*dx*0.4

    psi_1 = np.zeros(n)
    psi_1 = np.random.random(n) * 2 - 1
    psi_1[0] = 0
    psi_1[n-1] = 0

    V = numpy.zeros((n, n)) # potential energy 
    for i in range(0, n): 
        for j in range(0, n):
            x1 = (i - n//2) * dx
            x2 = (j - n//2) * dx
            V[i, j] = kappa/(np.sqrt((x1-x2)**2+l*l))

    HF_iteration = 0
    HF_Ek_old = 2
    HF_Ek_n = 1
    while(np.abs((HF_Ek_n - HF_Ek_old)/HF_Ek_n) > 1e-9):
        if (HF_iteration % 10 == 0 and HF_iteration != 0):
            print(f"iteracja: ", HF_iteration, np.abs((HF_Ek_n - HF_Ek_old)/HF_Ek_n))
        HF_iteration += 1
        HF_Ek_old = HF_Ek_n

        J_1 = np.zeros(n)
        for i in range(1, n-1):
            for j in range(0, n):
                J_1[i] += V[i, j] * np.abs(psi_1[j])**2 * dx

        Ek_n = 1
        Ek_old = 2
        iteration = 0
        #Metoda czasu urojonego
        while ( np.abs((Ek_n - Ek_old)/Ek_n) > 1e-9):

            iteration = iteration + 1
            Ek_old = Ek_n

            # calculate F = H \psi 
            F = np.zeros(n)
            for i in np.arange(1, n-1):
                F[i] += (-1/2/m)*(psi_1[i+1] + psi_1[i-1] - 2 * psi_1[i])/dx/dx
                F[i] += J_1[i]*psi_1[i]

            psi_1 = psi_1 - dt * F

            norm = np.sum(np.abs(psi_1)**2) *dx
            psi_1 = psi_1 / np.sqrt(norm)

            Ek_n = np.sum( np.conjugate(psi_1)*F )*dx
        
        E_int = np.sum((psi_1)**2 * J_1 * dx)
        HF_Ek_n = 2*Ek_n - E_int
    
    return HF_Ek_n*Ha

#Zad 2
lenght = np.arange(30, 65, 5)
f = open("Mezo2_3.txt", 'w')
for i in range (len(lenght)):
    ene = H_T(lenght[i])
    f.write(str(lenght[i]) + " " + str(ene) + "\n")
f.close()
