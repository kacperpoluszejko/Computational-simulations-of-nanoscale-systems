import numpy
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

Ha = 27.211   
a0 = 0.05292  

a = 23 / a0     
d1 = 5 / a0
d2 = 10 / a0
d3 = 13 / a0
d4 = 18 / a0
n = 100
dz = a / n
z_table = np.arange(n) * dz

V1 = 0.27 / Ha  
m1 = 0.063      
m2 = 0.083 * 0.3


def calculate_T_table(V_bias):

    V = np.zeros(n, dtype=complex)
    M = np.zeros(n, dtype=complex)

    for i in range(n):
        x = i * dz
        if d1 < x < d2 or d3 < x < d4:
            V[i] = V1 - V_bias*x/a
            M[i] = m1 + m2
        else:
            V[i] = - V_bias*x/a
            M[i] = m1

    E_table = np.concatenate([
        np.linspace(0.01 / Ha, 0.2 / Ha, 300),
        np.linspace(0.2 / Ha, 0.5 / Ha, 100)
    ])
    T_table = []


    for E in E_table:
        k = np.zeros(n, dtype=complex)
        for i in range(n):
            k[i] = np.sqrt(2 * M[i] * (E - V[i]))

        M_total = np.identity(2, dtype=complex)

        for i in range(n-1):
            z = i * dz

            k1 = k[i]
            k2 = k[i+1]
            m1_local = M[i]
            m2_local = M[i+1]

            wsp = (k2 * m1_local) / (k1 * m2_local)

            M_step = np.zeros((2,2), dtype=complex)
            M_step[0,0] = 0.5 * (1 + wsp) * np.exp(1j * z * (k2 - k1))
            M_step[0,1] = 0.5 * (1 - wsp) * np.exp(-1j * z * (k2 + k1))
            M_step[1,0] = 0.5 * (1 - wsp) * np.exp(1j * z * (k2 + k1))
            M_step[1,1] = 0.5 * (1 + wsp) * np.exp(-1j * z * (k2 - k1))

            M_total = M_total @ M_step

        kL = k[0]   
        kR = k[-1]   
        mL = M[0]
        mR = M[-1]

        denominator = np.abs(M_total[0,0])**2

        T = (kR * mL) / (kL * mR) * (1 / denominator)
        # R = np.abs(M_total[1,0])**2 / denominator

        T_table.append(T.real)

    return E_table, T_table



#Liczenie całki
# Stałe
T_K = 77  
kB_Ha = 3.1668e-6  
Temp = kB_Ha * T_K  
mu_s = 0.087/Ha
mu_d = 0.087/Ha
m = 0.067
Ha_current = 6.623e-3

V_bias_table = np.linspace(0.001/Ha, 0.5/Ha, 200)
j_table = []

for V_bias in V_bias_table:

    Ez, T_Ez = calculate_T_table(V_bias)
    # Czynnik logarytmiczny
    log_term = np.log((1 + np.exp((mu_s - Ez) / (Temp))) /(1 + np.exp((mu_d - V_bias - Ez) / (Temp))))

    # Całkowana funkcja: T(Ez) * log(...)
    integrand = T_Ez * log_term

    # Całkowanie metodą trapezów
    integral = np.trapz(integrand, Ez)

    # Prefactor z wzoru
    prefactor = (m*Temp) / (2 * np.pi**2)


    # Prąd
    j = prefactor * integral
    print(j)
    j_table.append(j*Ha_current)

plt.plot(V_bias_table*Ha, j_table)
plt.xlabel("V_bias [V]")
plt.ylabel("j")
plt.savefig("j(V_bias).png")
plt.show()

# plt.plot(E_table * Ha, T_table, color = "black")
# plt.xlabel('E (eV)')
# plt.ylabel('T')
# plt.plot(E_table * Ha, R_table, color = "red")
# plt.xlabel('E (eV)')
# plt.savefig("zad2_1.png", bbox_inches='tight')
# plt.show()

# plt.plot(E_table, T_table + R_table)
# plt.show()

    