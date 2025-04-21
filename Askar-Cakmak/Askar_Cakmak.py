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

n=99
a = 25/a0
d1 = 12/a0
d2 = 4/a0
V2 = 0.2/Ha
V1 = 0.25/Ha
dx = 2*a/(n+1)
dt = 1
alpha = 1/(2*m*dx*dx)
omega = 2.801816564*10**(-5) #Obliczone w zadaniu 1



def make_H(F, t):

    F = F*a0/(Ha * 10000)
    V_w = np.zeros(n)
    for i in range(n):
        x = (i - n//2) * dx
        if(x < (-d1) or x > d1):
            V_w[i] = V1
        if(x > (-d2) and x < d2):
            V_w[i] = V2
        V_w[i] += F*x*np.sin(omega*t)

        
    H = np.zeros((n,n))
    for i in range(n):
        H[i][i] = 2*alpha + V_w[i]
        if (i<(n-1)):
            H[i+1][i] = -alpha
            H[i][i+1] = -alpha
    
    return H

def E_F(F, t):

    H = make_H(F, t)
    eigvals, eigvecs = eigh(H)
    # return eigvals[0], eigvals[1], eigvals[2], eigvals[3]
    return eigvecs[:, 0], eigvecs[:, 1]

# E1, E2, E3, E4 = E_F(0, 0)
# print((E2 - E1))

x_table = (np.arange(n) - n//2) * dx
psi_0 = np.zeros(n, dtype=np.complex128)
psi_1 = np.zeros(n, dtype=np.complex128)

psi_0, psi_1 = E_F(0, 0)

norm_0 = np.sqrt(np.sum(np.abs(psi_0)**2) * dx)
psi_0 = psi_0 / norm_0

norm_1 = np.sqrt(np.sum(np.abs(psi_1)**2) * dx)
psi_1 = psi_1 / norm_1




#Cranck-Nicolson
F = 0.08*a0/(Ha * 10000)

psi_k = np.zeros(n, dtype=np.complex128)
psi_prim = np.zeros(n, dtype=np.complex128)
psi_k[:] = psi_0
for k in range(10):
    psi_prim[:] =  make_H(F, 0)@psi_0 + make_H(F, dt)@psi_k
    psi_k[:] = psi_0 + dt/(2j)*psi_prim


#Askar-Cakmak
@nb.njit
def Askar():  
    V_w = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        x = (i - n//2) * dx
        if(x < (-d1) or x > d1):
            V_w[i] = V1
        if(x > (-d2) and x < d2):
            V_w[i] = V2

    time = 3*10**6
    k = 10000
    psi_old = np.zeros(n, dtype=np.complex128)
    psi = np.zeros(n, dtype=np.complex128)
    psi_prim = np.zeros(n, dtype=np.complex128)
    psi_new = np.zeros(n, dtype=np.complex128)
    psi_old[:] = psi_0
    psi[:] = psi_k
    psi_prim[:]= psi
    P_0_table = []
    P_1_table = []
    t_table=[]

    for t in range(time):
        for i in range(1, n-1):
            psi_prim[i] = -alpha*(psi[i+1] + psi[i-1] - 2*psi[i]) + V_w[i]*psi[i] + F*x_table[i]*np.sin(omega*t)*psi[i]
        psi_prim[0] = -alpha*(psi[1] - 2*psi[0]) + V_w[0]*psi[0] + F*x_table[0]*np.sin(omega*t)*psi[0]
        psi_prim[n-1] = -alpha*(psi[n-2] - 2*psi[n-1]) + V_w[n-1]*psi[n-1] + F*x_table[n-1]*np.sin(omega*t)*psi[n-1]
        psi_new[:] = psi_old + 2*dt/(1j)*psi_prim
        psi_old[:] = psi
        psi[:] = psi_new
        if(t % k == 0):
            P_0 = np.complex128(0) # rzut na stan 0
            P_1 = np.complex128(0) # rzut na stan 1
            for i in range(n):
                P_0 += np.conj(psi[i])*psi_0[i]*dx
                P_1 += np.conj(psi[i])*psi_1[i]*dx
            P_0 = np.abs(P_0)**2
            P_1 = np.abs(P_1)**2
            P_0_table.append(P_0)
            P_1_table.append(P_1)
            t_table.append(t)
            print(P_0 + P_1, t)
    return P_0_table, P_1_table, t_table

P_0_table, P_1_table, t_table = Askar()
P_all_table = [x + y for x, y in zip(P_0_table, P_1_table)]
plt.plot(np.array(t_table) * 2.418884e-8, P_0_table, "black", label = r"$|<\Psi|0>|^2$")
plt.plot(np.array(t_table) * 2.418884e-8, P_1_table, "red", label = r"$|<\Psi|1>|^2$")
plt.plot(np.array(t_table) * 2.418884e-8, P_all_table, "blue", label = r"$|<\Psi|0>|^2 + |<\Psi|1>|^2$")
plt.ylabel("prob.", fontsize = 9)
plt.xlabel("t [ns]", fontsize = 9)
plt.legend(fontsize=6, markerscale = 0.5)
plt.savefig("Zad_3.jpg", bbox_inches='tight')
plt.show()




# Zad 2
# V_w = np.zeros(n)
# for i in range(n):
#     x = (i - n//2) * dx
#     if(x < (-d1) or x > d1):
#         V_w[i] = V1
#     if(x > (-d2) and x < d2):
#         V_w[i] = V2
# plt.plot(x_table * a0, psi_0, "black")
# plt.plot(x_table * a0, psi_1, "red")
# plt.xlabel("x [nm]", fontsize = 9)
# plt.ylabel(r"$\Psi [1/nm]$", fontsize = 9)
# plt.savefig("Zad_2.jpg", bbox_inches='tight')
# plt.show()


# Zad 1
# F_table = np.linspace(-2, 2, 100)
# E_table_0 = np.zeros(100)
# E_table_1 = np.zeros(100)
# E_table_2 = np.zeros(100)
# E_table_3 = np.zeros(100)
# for i in range(len(F_table)):
#     E_table_0[i], E_table_1[i], E_table_2[i], E_table_3[i] = E_F(F_table[i], 0)

# plt.plot(F_table, E_table_0*Ha, "black")
# plt.plot(F_table, E_table_1*Ha, "red")
# plt.xlabel("F (kV/cm)", fontsize = 9)
# plt.ylabel("E (eV)", fontsize = 9)
# plt.savefig("Zad_1_1.jpg", bbox_inches='tight')
# plt.show()

# plt.plot(F_table, E_table_0*Ha, "black")
# plt.plot(F_table, E_table_1*Ha, "red")
# plt.plot(F_table, E_table_2*Ha, "blue")
# plt.plot(F_table, E_table_3*Ha, "green")
# plt.xlabel("F (kV/cm)", fontsize = 9)
# plt.ylabel("E (eV)", fontsize = 9)
# plt.savefig("Zad_1_2.jpg", bbox_inches='tight')
# plt.show()