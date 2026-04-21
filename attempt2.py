import numpy as np
from scipy.constants import epsilon_0, mu_0

f = 1e9
omega = 2.0 * np.pi * f
eps = np.array([1, 2, 8, 1], dtype=np.complex128) * epsilon_0
mu = np.ones_like(eps) * mu_0
k = omega * np.sqrt(eps * mu)
thetas = np.zeros_like(eps)
thetas[0] = np.pi/6

for i in range(1,len(thetas)):
    thetas[i] = np.arcsin(np.sin(thetas[i-1]) * k[i-1]/k[i])

for theta in thetas:
    print(np.rad2deg(np.real(theta)))