import numpy as np
import matplotlib.pyplot as plt

def get_Cs(E: float, nu: float, rho: float) -> float:
    G = E / (2 * (1 + nu))
    Cs = np.sqrt(G / rho)
    return Cs

def get_Cd(E: float, nu: float, rho: float) -> float:
    if nu >= 0.5 or nu <= -1.0:
        raise ValueError("Poisson's ratio must be between -1.0 and 0.5 (non-inclusive)")

    Cd = np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))
    return Cd


def alpha_s(C_f, C_s):
    return np.sqrt(1 - (C_f / C_s) ** 2)

def alpha_d(C_f, C_d):
    return np.sqrt(1 - (C_f / C_d) ** 2)

def D(alpha_s, alpha_d):
    return 4 * alpha_s * alpha_d - (1 + alpha_s ** 2) ** 2

def M_of_z(tau_p, X_c, z):
    return (2 / np.pi) * tau_p * ((1 + z / X_c) * np.arctan(1 / np.sqrt(z / X_c)) - np.sqrt(z / X_c))

def compute_A2(C_f, C_s, nu, D_value):
    alpha_s_value = alpha_s(C_f, C_s)
    psfactor = 1 / (1 - nu)
    return (C_f ** 2 * alpha_s_value * psfactor) / (C_s ** 2 * D_value)

def compute_K2(Gamma, E, nu, A2):
    return np.sqrt((Gamma * E) / ((1 - nu ** 2) * A2))

def compute_tau_p(K2, X_c):
    return K2 * np.sqrt(9 * np.pi / (32 * X_c))

def compute_stress_components(M_z_d, M_z_s, alpha_s_value, alpha_d_value):
    Sxx_tmp = (1 + 2 * alpha_d_value ** 2 - alpha_s_value ** 2) * M_z_d - (1 + alpha_s_value ** 2) * M_z_s
    Syy_tmp = M_z_d - M_z_s
    Sxy_tmp = 4 * alpha_s_value * alpha_d_value * M_z_d - (1 + alpha_s_value ** 2) ** 2 * M_z_s
    return Sxx_tmp, Syy_tmp, Sxy_tmp

def compute_stresses(Sxx_tmp, Syy_tmp, Sxy_tmp, alpha_s_value, D_value):
    Sxx = 2 * alpha_s_value / D_value * Sxx_tmp.imag
    Syy = -2 * alpha_s_value * (1 + alpha_s_value ** 2) / D_value * Syy_tmp.imag
    Sxy = Sxy_tmp.real / D_value
    return Sxx, Syy, Sxy

def delta_sigma_xy(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E):
    alpha_s_value = alpha_s(C_f, C_s)
    alpha_d_value = alpha_d(C_f, C_d)
    D_value = D(alpha_s_value, alpha_d_value)
    A2 = compute_A2(C_f, C_s, nu, D_value)
    K2 = compute_K2(Gamma, E, nu, A2)
    tau_p = compute_tau_p(K2, X_c)

    z_d_value = x + 1j * alpha_d_value * y
    z_s_value = x + 1j * alpha_s_value * y

    M_z_d = M_of_z(tau_p, X_c, z_d_value)
    M_z_s = M_of_z(tau_p, X_c, z_s_value)

    Sxx_tmp, Syy_tmp, Sxy_tmp = compute_stress_components(M_z_d, M_z_s, alpha_s_value, alpha_d_value)

    Sxx, Syy, Sxy = compute_stresses(Sxx_tmp, Syy_tmp, Sxy_tmp, alpha_s_value, D_value)

    delta_sigma = Sxy

    return delta_sigma

def delta_sigma_xx(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E):
    alpha_s_value = alpha_s(C_f, C_s)
    alpha_d_value = alpha_d(C_f, C_d)
    D_value = D(alpha_s_value, alpha_d_value)
    A2 = compute_A2(C_f, C_s, nu, D_value)
    K2 = compute_K2(Gamma, E, nu, A2)
    tau_p = compute_tau_p(K2, X_c)

    z_d_value = x + 1j * alpha_d_value * y
    z_s_value = x + 1j * alpha_s_value * y

    M_z_d = M_of_z(tau_p, X_c, z_d_value)
    M_z_s = M_of_z(tau_p, X_c, z_s_value)

    Sxx_tmp, Syy_tmp, Sxy_tmp = compute_stress_components(M_z_d, M_z_s, alpha_s_value, alpha_d_value)

    Sxx, Syy, Sxy = compute_stresses(Sxx_tmp, Syy_tmp, Sxy_tmp, alpha_s_value, D_value)

    delta_sigma = Sxx

    return delta_sigma

def convert_modeII_to_akantu(tau_c, G_IIc, beta):
    """
    Convert Mode II cohesive properties (tau_c, G_IIc)
    into Akantu parameters (sigma_c, fracture energy, beta).

    Inputs:
        tau_c : Mode II critical shear strength (MPa)
        G_IIc : Mode II fracture energy (N/mm)
        beta  : Akantu mode-mixity parameter (given by user)

    Output:
        sigma_c : Akantu's critical cohesive strength (MPa)
        G_c     : fracture energy = G_IIc (N/mm)
        gamma_c : computed shear displacement at failure (mm) for checking
    """

    # Akantu defines effective stress: sigma_eff = sqrt( σ_n^2 + τ^2/β^2 )
    # For pure Mode II: τ_c = β * σ_c
    sigma_c = tau_c / beta

    # Akantu's fracture energy is the Mode II fracture energy
    G_c = G_IIc

    # For verification: G_IIc = 0.5 * τ_c * γ_c
    gamma_c = 2 * G_IIc / tau_c

    return sigma_c, G_c, beta, gamma_c


def compute_Xc_modeII_SI(E, nu, tau_c, Gc):
    """
    Compute Mode II cohesive zone size in meters (ALL INPUTS IN SI UNITS)

    Parameters
    ----------
    E : float
        Young's modulus in Pa
    nu : float
        Poisson's ratio
    tau_c : float
        Cohesive shear strength in Pa
    Gc : float
        Fracture energy in J/m^2

    Returns
    -------
    Xc : float
        Cohesive zone size in meters
    """

    # Plane strain modulus
    E_prime = E / (1 - nu**2)

    # Mode II cohesive zone size formula:
    # Xc = (9π/32) * (E' / tau_c^2) * Gc
    coef = 9 * np.pi / 32
    Xc = coef * (E_prime / tau_c**2) * Gc

    return Xc

def compute_E_nu_from_VpVsRho(Vp, Vs, rho):
    """
    Compute Young's modulus (E) and Poisson's ratio (nu)
    from P-wave velocity (Vp), S-wave velocity (Vs), and density rho.

    Parameters
    ----------
    Vp : float
        P-wave velocity (m/s)
    Vs : float
        S-wave velocity (m/s)
    rho : float
        Density (kg/m³)

    Returns
    -------
    E : float
        Young's modulus in Pa
    nu : float
        Poisson's ratio
    """

    # Shear modulus
    G = rho * Vs**2

    # Bulk modulus
    K = rho * Vp**2 - (4.0/3.0) * G

    # Young's modulus
    E = 9 * K * G / (3 * K + G)

    # Poisson's ratio
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))

    return E, nu
