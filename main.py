import numpy as np


# Get the ABCD matrix of a transmission line
def tline_abcd(k, eta, d):
    c = np.cos(k * d)
    s = 1j * np.sin(k * d)
    return np.array([[c, -eta * s], [-(1.0 / eta) * s, c]])


# Get the reflection coefficient of an ABCD matrix system
def abcd2s11(abcd, z, k_L, eta_L, eta_R):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    expl = np.exp(-2j * k_L * z, dtype=np.complex128)
    numerator = -A - B / eta_L + eta_R * C + (eta_R / eta_L) * D
    denominator = A - B / eta_L - eta_R * C + (eta_R / eta_L) * D
    return expl * numerator / denominator


# First version of transmission coefficient of an ABCD matrix system
def abcd2s12_1(abcd, z, k_L, k_R, eta_L, eta_R, Gamma, d):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    expl = np.exp(1j * k_R * z) * np.exp(1j * k_R * d)
    t1 = (A + B / eta_L) * np.exp(-1j * k_L * z)
    t2 = Gamma * (A - B / eta_L) * np.exp(1j * k_L * z)
    return expl * (t1 + t2)


# Second version of transmission coefficient of an ABCD matrix system
def abcd2s12_2(abcd, z, k_L, k_R, eta_L, eta_R, Gamma, d):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    expl = eta_R * np.exp(1j * k_R * z) * np.exp(1j * k_R * d)
    t1 = (C + D / eta_L) * np.exp(-1j * k_L * z)
    t2 = Gamma * (C - D / eta_L) * np.exp(1j * k_L * z)
    return expl * (t1 + t2)
