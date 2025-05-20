import numpy
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.linalg import eigh
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt

Ha = 27.211
a0 = 0.05292
m = 0.017

Nx = 19
Ny = 19
y_max = 10 /a0
y_min = -10/a0
Ly = y_max - y_min
dy = Ly / (Ny + 1) 
dx = dy
Lx = dx * (Nx + 1)
Vsg = -1
alfa = 1/(2*m*dx*dx)


# V = numpy.zeros((Nx, Ny)) # here potential energy is zero, but it can be changed easily
# for i in range(0, Nx): #diagonal elements
#     for j in range(0, Ny):
#         x = (i+1) * dx - Lx/2
#         y = (j+1) * dx - Ly/2
        
#         V[i, j] = -0.035/Ha * Vsg * np.exp(-(x**2/(300/a0)**2+(y+Ly/2)**2/(300/a0)**2)**2.) 
#         V[i, j] +=-0.035/Ha * Vsg * np.exp(-(x**2/(300/a0)**2+(y-Ly/2)**2/(300/a0)**2)**2.)


# x_table = np.linspace(-Lx/2, Lx/2, Nx)
# y_table = np.linspace(-Ly/2, Ly/2, Ny)
# X, Y = np.meshgrid(x_table, y_table, indexing='ij')
# plt.pcolormesh(X * a0, Y * a0, V * Ha, shading='auto')
# plt.colorbar(label='Potencjał [eV]')
# plt.xlabel('x [nm]')
# plt.ylabel('y [nm]')
# plt.title('Potencjał w QPC')
# plt.savefig("potential.png")
# plt.show()


#Zad 1

def make_H(kx):
    H = np.zeros((Nx,Ny), dtype = complex)
    for i in range(Ny):
        H[i][i] = 4*alfa - alfa*(np.exp(1j*kx*dx) + np.exp(-1j*kx*dx))
        if (i<(Nx-1)):
            H[i+1][i] = -alfa
            H[i][i+1] = -alfa
    return H


#Zad 2

def make_H_1(E):
    H = np.zeros((2*Nx,2*Nx), dtype=complex)
    for i in range(Nx):
        H[Nx +i][i] = alfa
        H[i][Nx+i] = 1
        H[Nx+i][Nx+i] = E - 4*alfa
        if (i<(Nx-1)):
            H[Nx+i+1][Nx+i] = alfa
            H[Nx+i][Nx+i+1] = alfa
    return H

def make_H_2():
    H = np.zeros((2*Nx,2*Nx), dtype=complex)
    for i in range(Nx):
        H[i][i] = 1
        H[Nx+i][Nx+i] = -alfa
    return H

H1 = make_H_1(0.4/Ha)
H2 = make_H_2()

eigvals, eigvecs = eig(H1,H2)
print(np.abs(eigvals))

def check_if_propagation(eigvecs):
    indexes = []
    for i in range(Ny):
        if(np.abs(eigvals[i]) > 0.99 and np.abs(eigvals[i]) < 1.01):
            indexes.append(i)
            print(i)
    return indexes
# norm = np.linalg.norm(eigvecs[17])
# eigvecs[17] = eigvecs[17]/norm

indexes = check_if_propagation(eigvals)
y_table = np.linspace(y_min, y_max, Ny)

# Plotowanie zadania 2
for i in indexes:
    norm = np.linalg.norm(eigvecs[i])
    eigvecs[i] = eigvecs[i]/norm
    plt.plot(y_table*a0, eigvecs[0:Ny,i].real, label = "$Re(u_{-,0})$")
    plt.plot(y_table*a0, -eigvecs[0:Ny,i].imag,  label = "$Im(u_{-,0})$")

plt.xlabel("y [nm]")
plt.ylabel(r"$u_{-,n} (a.u.)$")
plt.savefig("plot6.png")
plt.show()


#Zad 1
# K1 = np.log(eigvals[indexes[0]]).imag
# K2 = np.log(eigvals[indexes[1]]).imag
# # K3 = np.log(eigvals[indexes[2]]).imag
# # K4 = np.log(eigvals[indexes[3]]).imag
# K_table = [K1, K2]
# E_table = [0.2]


# k_xmax = np.pi/dx
# k_xmin = -np.pi/dx
# Nk = 100
# dk = (k_xmax - k_xmin)/Nk
# k_table = np.arange(k_xmin, k_xmax, dk)


# eigvals_table = np.zeros((Nx, Nk))
# for i, kx in enumerate(k_table):
#     H = make_H(kx)
#     eigvals, eigvecs = eigh(H)
#     for j in range(Nx):
#         eigvals_table[j][i] = eigvals[j]

# for i in range(Nx):
#     plt.plot(k_table/a0, eigvals_table[i]*Ha)
#     plt.scatter(K_table, E_table*2, color = "red", s = 20)
#     plt.plot(k_table/a0, E_table*100, linestyle = "--")
#     plt.xlabel("kx [1/nm]")
#     plt.ylabel("E (eV)")
#     plt.savefig("plot1.png")

    
# for i in range(Nx):
#     plt.plot(k_table[25:75]/a0, eigvals_table[i][25:75]*Ha)
#     plt.scatter(K_table, E_table*2, color = "red", s = 20)
#     plt.plot(k_table[25:75]/a0, E_table*50, linestyle = "--")
#     plt.xlabel("kx [1/nm]")
#     plt.ylabel("E (eV)")
#     plt.savefig("plot2.png")

plt.show()

