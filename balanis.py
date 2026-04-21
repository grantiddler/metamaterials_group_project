import numpy as np
from scipy.constants import epsilon_0, mu_0

f = 1e9
omega = 2.0 * np.pi * f
d = np.array([0, 72.07e-3, 0])
thetas = np.array([0, 0, 0], dtype=np.complex128)
eps = np.array([1, 2, 1], dtype=np.complex128) * epsilon_0
mu = np.array([1, 1, 1], dtype=np.complex128) * mu_0
gamma = 1j * omega * np.sqrt(eps * mu)
psi = d * gamma * np.cos(thetas)
print(psi)

A = np.array([0, 0, 1], dtype=np.complex128)
B = np.array([0, 0, 0], dtype=np.complex128)
C = np.array([0, 0, 1], dtype=np.complex128)
D = np.array([0, 0, 0], dtype=np.complex128)

for j in reversed(range(len(d) - 1)):
    cos_factor = np.cos(thetas[j + 1]) / np.cos(thetas[j])
    Y_jp1 = cos_factor * np.sqrt((eps[j + 1] * mu[j]) / (eps[j] * mu[j + 1]))
    Z_jp1 = cos_factor * np.sqrt((eps[j] * mu[j + 1]) / (eps[j + 1] * mu[j]))
    A[j] = 0.5 * np.exp(+psi[j]) * (A[j + 1] * (1.0 + Y_jp1) + B[j + 1] * (1 - Y_jp1))
    B[j] = 0.5 * np.exp(-psi[j]) * (A[j + 1] * (1.0 - Y_jp1) + B[j + 1] * (1 + Y_jp1))
    C[j] = 0.5 * np.exp(+psi[j]) * (C[j + 1] * (1.0 + Z_jp1) + D[j + 1] * (1 - Z_jp1))
    D[j] = 0.5 * np.exp(-psi[j]) * (C[j + 1] * (1.0 - Z_jp1) + D[j + 1] * (1 + Z_jp1))

print(f"S11 = {np.real(B[0] / A[0])} + j {np.imag(B[0] / A[0])}")
T1 = np.sqrt(1.0 - np.square(B[0] / A[0]))
print(f"S12 = {np.real(T1)} + j {np.imag(T1)}")
print(20 * np.log10(np.abs(T1)))
