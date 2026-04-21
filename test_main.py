import numpy as np
from scipy.constants import epsilon_0, mu_0
from main import tline_abcd, abcd2s11, abcd2s12_1, abcd2s12_2

db = lambda x: 20 * np.log10(np.abs(x))


def test_multiple_tline_system_air_L_and_R():
    f = 1e9
    eta_0 = np.sqrt(mu_0 / epsilon_0)
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    eta_1 = np.sqrt(mu_0 / (2 * epsilon_0))
    k_1 = 2.0 * np.pi * f * np.sqrt(2 * epsilon_0 * mu_0)
    eta_2 = np.sqrt(mu_0 / (3 * epsilon_0))
    k_2 = 2.0 * np.pi * f * np.sqrt(3 * epsilon_0 * mu_0)
    eta_3 = np.sqrt(mu_0 / (4 * epsilon_0))
    k_3 = 2.0 * np.pi * f * np.sqrt(4 * epsilon_0 * mu_0)

    d1 = 40e-3
    d2 = 30e-3
    M1 = tline_abcd(k_1, eta_1, d1)
    M2 = tline_abcd(k_2, eta_2, d2)
    M = M1 @ M2

    Gamma = abcd2s11(M, 0, k_0, eta_0, eta_0)
    T1 = abcd2s12_1(M, 0, k_0, k_0, eta_0, eta_0, Gamma, d1 + d2)
    T2 = abcd2s12_2(M, 0, k_0, k_0, eta_0, eta_0, Gamma, d1 + d2)

    assert np.isclose(T1, T2)
    assert np.isclose(np.abs(T1) ** 2 + np.abs(Gamma) ** 2, 1.0)
    assert np.isclose(np.real(Gamma), -0.349, 0.001)
    assert np.isclose(np.imag(Gamma), 0.076, 0.01)
    assert np.isclose(db(Gamma), -8.936, 0.001)
    assert np.isclose(db(T1), -0.594, 0.001)

    if __name__ in "__main__":
        print(Gamma, db(Gamma))
        print(T1, db(T1))
        print(T2, db(T2))
        print(np.abs(T1) ** 2 + np.abs(Gamma) ** 2)
        print(np.abs(T2) ** 2 + np.abs(Gamma) ** 2)


def test_multiple_tline_system_other_L_and_R():
    f = 1e9
    eta_0 = np.sqrt(mu_0 / epsilon_0)
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    eta_1 = np.sqrt(mu_0 / (2 * epsilon_0))
    k_1 = 2.0 * np.pi * f * np.sqrt(2 * epsilon_0 * mu_0)
    eta_2 = np.sqrt(mu_0 / (3 * epsilon_0))
    k_2 = 2.0 * np.pi * f * np.sqrt(3 * epsilon_0 * mu_0)
    eta_3 = np.sqrt(mu_0 / (1.5 * epsilon_0))
    k_3 = 2.0 * np.pi * f * np.sqrt(1.5 * epsilon_0 * mu_0)

    d1 = 40e-3
    d2 = 30e-3
    M1 = tline_abcd(k_1, eta_1, d1)
    M2 = tline_abcd(k_2, eta_2, d2)
    M = M1 @ M2

    Gamma = abcd2s11(M, 0, k_3, eta_3, eta_3)
    T1 = abcd2s12_1(M, 0, k_3, k_3, eta_3, eta_3, Gamma, d1 + d2)
    T2 = abcd2s12_2(M, 0, k_3, k_3, eta_3, eta_3, Gamma, d1 + d2)

    assert np.isclose(T1, T2)
    assert np.isclose(np.abs(T1) ** 2 + np.abs(Gamma) ** 2, 1.0)
    assert np.isclose(np.real(Gamma), -0.238, 0.01)
    assert np.isclose(np.imag(Gamma), -0.012, 0.02)
    assert np.isclose(db(Gamma), -12.445, 0.001)
    assert np.isclose(db(T1), -0.255, 0.01)

    if __name__ in "__main__":
        print(Gamma, db(Gamma))
        print(T1, db(T1))
        print(np.abs(T1) ** 2 + np.abs(Gamma) ** 2)


def test_tline_system_other_L_air_R():
    f = 1e9

    eta_0 = np.sqrt(mu_0 / epsilon_0)
    eta_1 = np.sqrt(mu_0 / (2 * epsilon_0))
    eta_2 = np.sqrt(mu_0 / (3 * epsilon_0))
    k_0 = 2.0 * np.pi * f * np.sqrt(epsilon_0 * mu_0)
    k_1 = 2.0 * np.pi * f * np.sqrt(2 * epsilon_0 * mu_0)
    k_2 = 2.0 * np.pi * f * np.sqrt(3 * epsilon_0 * mu_0)

    d1 = 40e-3
    M = tline_abcd(k_1, eta_1, d1)

    Gamma = abcd2s11(M, 0, k_2, eta_2, eta_0)
    T1 = abcd2s12_1(M, 0, k_2, k_0, eta_2, eta_0, Gamma, d1)
    T2 = abcd2s12_2(M, 0, k_2, k_0, eta_2, eta_0, Gamma, d1)

    # assert np.isclose(T1, T2)
    # assert np.isclose(np.abs(T1)**2 + np.abs(Gamma)**2, 1.0)
    # assert np.isclose(np.real(Gamma), -0.101, 0.01)
    # assert np.isclose(np.imag(Gamma), 0.070, 0.01)
    # assert np.isclose(db(Gamma), -18.199, 0.001)
    # assert np.isclose(db(T1), -0.255, 0.01)

    if __name__ in "__main__":
        print(Gamma, db(Gamma))
        print(T1, db(T1))
        print(np.abs(T1) ** 2 + np.abs(Gamma) ** 2)


if __name__ in "__main__":
    test_tline_system_other_L_air_R()
    pass
