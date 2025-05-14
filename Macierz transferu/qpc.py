import numpy
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

Ha = 27.211
a0 = 0.05292

N = 100
d = 3 / a0
Vg = 4 / Ha
W = 50 / a0
L = 100 / a0
epsilon = 13.6
m = 0.063
dx = L / N
dy = W / N

def f(u, v, Vg):
    return Vg / (2 * np.pi * epsilon) * np.arctan(u * v / (d * np.sqrt(d*d + u*u + v*v)))

def potential(x, y, Vg):
    gate_begin = 0.3
    gate_end = 0.7
    gate_spacing = 0.6

    # Położenie bramki dolnej
    l = L * gate_begin
    r = L * gate_end
    t = -gate_spacing * W / 2
    b = -W * 2

    f1 = f(x - l, y - b, Vg)
    f2 = f(x - l, t - y, Vg)
    f3 = f(r - x, y - b, Vg)
    f4 = f(r - x, t - y, Vg)

    g1 = f1 + f2 + f3 + f4

    # Położenie bramki górnej
    t = gate_spacing * W / 2
    b = W * 2

    f1 = f(x - l, y - b, Vg)
    f2 = f(x - l, t - y, Vg)
    f3 = f(r - x, y - b, Vg)
    f4 = f(r - x, t - y, Vg)

    g2 = f1 + f2 + f3 + f4

    return g1 + g2


def make_H(x_index, V): 
    alpha = 1/(dy*dy*m)
    H = np.zeros((N,N))
    for i in range(N):
        H[i][i] = 2*alpha + V[x_index, i]
        if (i<(N-1)):
            H[i+1][i] = -alpha
            H[i][i+1] = -alpha

    return H

x_table = np.linspace(0, L, N)
y_table = np.linspace(-W/2, W/2, N)


E_table = np.zeros((5, N))
E_table2 = np.linspace(0.01 / Ha, 0.2 / Ha, N) 

# for i in range(N):
#     H = make_H(i)
#     eigvals, eigvecs = eigh(H)
#     E_table[0, i] = eigvals[0]
#     E_table[1, i] = eigvals[1]
#     E_table[2, i] = eigvals[2]
#     E_table[3, i] = eigvals[3]
#     E_table[4, i] = eigvals[4]


def calc_T(E_table, n):

    k = np.zeros(N, dtype=complex)
    for i in range(N):
        k[i] = np.sqrt(2 * m * (E_f- E_table[n, i] + 0j))


    M_total = np.identity(2, dtype=complex)

    for i in range(N-1):
        x = i * dx

        k1 = k[i]
        k2 = k[i+1]
        m1_local = m
        m2_local = m

        wsp = (k2 * m1_local) / (k1 * m2_local)

        M_step = np.zeros((2,2), dtype=complex)
        M_step[0,0] = 0.5 * (1 + wsp) * np.exp(1j * x * (k2 - k1))
        M_step[0,1] = 0.5 * (1 - wsp) * np.exp(-1j * x * (k2 + k1))
        M_step[1,0] = 0.5 * (1 - wsp) * np.exp(1j * x * (k2 + k1))
        M_step[1,1] = 0.5 * (1 + wsp) * np.exp(-1j * x * (k2 - k1))

        M_total = M_total @ M_step

    kL = k[0]   
    kR = k[-1]   
    mL = m
    mR = m

    denominator = np.abs(M_total[0,0])**2
    T = (kR * mL) / (kL * mR) * (1 / denominator)
    return T


def calc_T_all(Vg):

    V = np.zeros((N, N))
    for i, x in enumerate(x_table):
        for j, y in enumerate(y_table):
            V[i, j] = np.abs(potential(x, y, Vg))

    for i in range(N):
        H = make_H(i, V)
        eigvals, eigvecs = eigh(H)
        E_table[0, i] = eigvals[0]
        E_table[1, i] = eigvals[1]
        E_table[2, i] = eigvals[2]
        E_table[3, i] = eigvals[3]
        E_table[4, i] = eigvals[4]

    T_all = 0
    for j in range(5):
      T_all += calc_T(E_table, j)

    return T_all

Vg_table = np.linspace(0.01/Ha, 35/Ha, 150)

E_f = 0.05/Ha
T_table = []
for Vg in Vg_table:
    T_table.append(calc_T_all(Vg))
plt.plot(Vg_table*Ha, T_table)

E_f = 0.1/Ha
T_table2 = []
for Vg in Vg_table:
    T_table2.append(calc_T_all(Vg))
plt.plot(Vg_table*Ha, T_table2)
plt.xlabel("Vg [eV]")
plt.ylabel("G")
plt.savefig("Zad4_4.png")
plt.show()



# T_0 = calc_T(0)
# T_1 = calc_T(1)
# T_2 = calc_T(2)
# T_3 = calc_T(3)
# T_4 = calc_T(4)

# T_all = np.zeros(N)
# for i in range(N):
#     T_all[i] = T_0[i] + T_1[i] + T_2[i] + T_3[i] + T_4[i]

# plt.plot(E_table2*Ha, T_all)
# plt.xlabel(r"$E_n [eV]$")
# plt.ylabel(r"$2e^2/h$")
# plt.savefig("konduktancja.png")
# plt.show()

# plt.plot(x_table*a0, E_table[0, :]*Ha)
# plt.plot(x_table*a0, E_table[1, :]*Ha)
# plt.plot(x_table*a0, E_table[2, :]*Ha)
# plt.plot(x_table*a0, E_table[3, :]*Ha)
# plt.plot(x_table*a0, E_table[4, :]*Ha)
# plt.xlabel("x [nm]")
# plt.ylabel(r"$E_n [eV]$")
# plt.savefig("E_n(x).png")
# plt.show()


# X, Y = np.meshgrid(x_table, y_table, indexing='ij')
# plt.pcolormesh(X * a0, Y * a0, V * Ha, shading='auto')
# plt.colorbar(label='Potencjał [eV]')
# plt.xlabel('x [nm]')
# plt.ylabel('y [nm]')
# plt.title('Potencjał w QPC')
# plt.savefig("potential.png")
# plt.show()