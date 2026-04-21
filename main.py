import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, mu_0

db = lambda x: 20 * np.log10(np.abs(x))


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


f = 1e9
num = 100
Gammas = np.zeros(num, dtype=np.complex128)
Ts = np.zeros(num, dtype=np.complex128)
incident_angles = np.linspace(0, np.pi / 2, num)

lambda_0 = 1 / (np.sqrt(epsilon_0 * mu_0) * f)
print(lambda_0)

epsrs = np.ones(12) * epsilon_0
epsrs[1] = 2
epsrs[-2] = 2
epss =  epsrs * epsilon_0
mus = np.ones(12) * mu_0
ds = np.array(
    [
        0,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        lambda_0 / 9,
        0,
    ]
)
ks = 2.0 * np.pi * f * np.sqrt(mus * epss)
kyz = 2.0 * np.pi * f * np.sqrt(mus[0] * epss[0]) * np.sin(incident_angles)
for j, ang in enumerate(incident_angles):
    thetas = np.zeros_like(ds, dtype=np.complex128)
    thetas[0] = incident_angles[j]
    for i in range(1, len(thetas)):
        thetas[i] = np.arcsin(ks[i - 1] / ks[i] * np.sin(thetas[i - 1]))
    etas_TM = np.sqrt(mus / epss) * np.cos(thetas)
    kzs = ks * np.cos(thetas)

    abcd_mats = []
    for i in range(1, len(epss) - 1):
        abcd_mats.append(tline_abcd(kzs[i], etas_TM[i], ds[i]))
    abcd_mat = abcd_mats[0]
    for i in range(1, len(abcd_mats)):
        abcd_mat = abcd_mat @ abcd_mats[i]

    Gamma = abcd2s11(abcd_mat, 0, kzs[0], etas_TM[0], etas_TM[-1])
    T = abcd2s12_1(
        abcd_mat, 0, kzs[0], kzs[-1], etas_TM[0], etas_TM[-1], Gamma, np.sum(ds)
    )
    Gammas[j] = Gamma
    Ts[j] = T

fig, ax = plt.subplots()
print(Gammas[0], db(Gammas[0]))
print(Ts[0], db(Ts[0]))
# ax.plot(np.rad2deg(incident_angles), np.abs(Gammas))
ax.plot(np.rad2deg(incident_angles), np.abs(Ts))

greens_func = 1.0 + -(kyz**2) / (ks[0] ** 2)
ax.plot(np.rad2deg(incident_angles), greens_func)
plt.show()
