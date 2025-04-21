import numpy
import numpy as np
import matplotlib.pyplot as plt

# X, Y = np.loadtxt("E_HF_L.txt", delimiter=" ", unpack=True, usecols=(0, 1))
                  
# plt.plot(X, Y, "o-", markersize = 4, color = "black")
# plt.ylabel("Enegia [eV]")
# plt.xlabel("Iteracja")
# plt.savefig("Mezo_2_1.png")
# plt.show()

X, Y = np.loadtxt("Mezo2_2.txt", delimiter=" ", unpack=True, usecols=(0, 1))
X1, Y1 = np.loadtxt("Mezo2_3.txt", delimiter=" ", unpack=True, usecols=(0, 1))
                  
plt.plot(X, Y, "o-", markersize = 4, color = "black", label = "imgainary time")
plt.plot(X1, Y1, "o-", markersize = 4, color = "red", label = "Hartree-Fock")
plt.ylabel("Enegia [eV]")
plt.xlabel("a [nm]")
plt.legend()
plt.savefig("Mezo_2_2.jpg", bbox_inches='tight')
plt.show()