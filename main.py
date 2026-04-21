import numpy as np
from scipy.constants import epsilon_0, mu_0


def tline_abcd(f, eps, mu, d):
    k = 2.0 * np.pi * f * np.sqrt(eps * mu)
    eta = np.sqrt(mu / eps)
    c = np.cos(k * d)
    s = 1j * np.sin(k * d)
    return np.array([[c, eta * s], [(1 / eta) * s, c]])


def abcd2s(abcd, Z):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    G = A + B / Z + C * Z + D
    S11 = (A + B / Z - C * Z - D) / G
    S12 = 2 * (A * D - B * C) / G
    S21 = 2 / G
    S22 = (-A + B / Z - C * Z + D) / G
    return np.array([[S11, S12], [S21, S22]])


def s2t(s):
    s11 = s[0, 0]
    s12 = s[0, 1]
    s21 = s[1, 0]
    s22 = s[1, 1]
    G = s21
    t11 = -(s11 * s22 - s12 * s21) / G
    t12 = s11 / G
    t21 = -s22 / G
    t22 = 1.0 / G
    return np.array([[t11, t12], [t21, t22]])


def t2s(t):
    t11 = t[0, 0]
    t12 = t[0, 1]
    t21 = t[1, 0]
    t22 = t[1, 1]
    G = t22
    s11 = t12 / G
    s12 = (t11 * t22 - t12 * t21) / G
    s21 = 1 / G
    s22 = -t21 / G
    return np.array([[s11, s12], [s21, s22]])

def my_tline_abcd(k, eta, d):
    c = np.cos(k * d)
    s = 1j * np.sin(k * d)
    return np.array([[c, -eta * s], [-(1.0 / eta) * s, c]])

def abcd2s11(abcd, z, k_L, eta_L, eta_R):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    expl = np.exp(-2j * k_L * z, dtype=np.complex128)
    numerator = -A - B/eta_L + eta_R * C + (eta_R/eta_L) * D
    denominator = A - B/eta_L - eta_R * C + (eta_R/eta_L) * D
    return expl * numerator / denominator

def abcd2s12_1(abcd, z, k_L, k_R, eta_L, eta_R, Gamma, d):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    expl = np.exp(1j * k_R * z)*np.exp(1j * k_R * d)
    t1 = (A + B/eta_L) * np.exp(-1j * k_L * z)
    t2 = Gamma * (A - B/eta_L) * np.exp(1j * k_L * z)
    return expl * (t1 + t2)

def abcd2s12_2(abcd, z, k_L, k_R, eta_L, eta_R, Gamma, d):
    A = abcd[0, 0]
    B = abcd[0, 1]
    C = abcd[1, 0]
    D = abcd[1, 1]
    expl = eta_R * np.exp(1j * k_R * z)*np.exp(1j * k_R * d)
    t1 = (C + D/eta_L) * np.exp(-1j * k_L * z)
    t2 = Gamma * (C - D/eta_L) * np.exp(1j * k_L * z)
    return expl * (t1 + t2)

f = 1e9
eta_0 = np.sqrt(mu_0/epsilon_0)
k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
eta_1 = np.sqrt(mu_0/(2 * epsilon_0))
k_1 = 2.0 * np.pi * f * np.sqrt(2 * epsilon_0 * mu_0)
eta_2 = np.sqrt(mu_0/(3 * epsilon_0))
k_2 = 2.0 * np.pi * f * np.sqrt(3 * epsilon_0 * mu_0)

d1 = 40e-3
d2 = 30e-3
M1 = my_tline_abcd(k_1, eta_1, d1)
M2 = my_tline_abcd(k_2, eta_2, d2)
M = M1 @ M2

print(M)
Gamma = abcd2s11(M, 0, k_0, eta_0, eta_0)
T1 = abcd2s12_1(M, 0, k_0, k_0, eta_0, eta_0, Gamma, d1+d2)
T2 = abcd2s12_2(M, 0, k_0, k_0, eta_0, eta_0, Gamma, d1+d2)

db = lambda x: 20 * np.log10(np.abs(x))
print(Gamma, db(Gamma))
print(T1, db(T1))
print(T2, db(T2))
print(np.abs(T1)**2 + np.abs(Gamma)**2)
print(np.abs(T2)**2 + np.abs(Gamma)**2)