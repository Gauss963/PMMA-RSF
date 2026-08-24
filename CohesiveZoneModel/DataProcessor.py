import numpy as np
import scipy.signal

import CohesiveModel

def voltage_to_strain(raw_voltage: float|np.ndarray[float]) -> float|np.ndarray[float]:
    '''
    Convert raw voltage readings from a strain gauge circuit to strain.

    The function assumes a Wheatstone bridge circuit with a strain gauge
    and uses the following relationship:

    strain = (raw_voltage / Vex / Gain) * (2 / GF)

    Args:
        raw_voltage (float): The measured voltage output from the Wheatstone bridge circuit (in volts).

    Constants:
        Vex  (float): Excitation voltage applied to the Wheatstone bridge (in volts). Default is 4.98 V.
        GF   (float): Gauge factor of the strain gauge (dimensionless). Default is 2.12.
        Rg   (float): Resistance of the strain gauge (in ohms). Default is 350 Ohm.
        Gain (float): Amplification factor of the signal (dimensionless). Default is 1000.

    Returns:
        float: Calculated strain (dimensionless).

    Example:
        >>> voltage_to_strain(0.002)
        1.9178082191780822e-06
    '''
    Vex = 4.98    # Excitation voltage in volts
    GF = 2.12     # Gauge factor
    Rg = 350      # Resistance of strain gauge in ohms (not directly used)
    Gain = 1000   # Amplification factor

    strain = raw_voltage / Vex / Gain * 2 / GF

    return strain


def shear_strain_to_stress(E: float, poisson_ratio: float, strain: float|np.ndarray[float]) -> float|np.ndarray[float]:
    '''
    Calculate stress from shear strain by converting Young's modulus (E) to shear modulus (G).

    Args:
        E (float): Young's modulus (elastic modulus) in units of stress (e.g., Pa).
        strain (float): Shear strain (dimensionless).
        poisson_ratio (float): Poisson's ratio (dimensionless).

    Returns:
        float: Shear stress in the same units as Young's modulus.
    '''

    G = E / (2 * (1 + poisson_ratio))
    stress = G * strain

    return stress

def highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype = 'high', analog=False)
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data

def apply_taper(signal:np.ndarray, taper_ratio: float = 0.05) -> np.ndarray:
    """
    Apply a Tukey taper to a signal.

    Parameters:
        signal (numpy array): The input signal.
        taper_ratio (float): The proportion of the signal to taper (default: 5%).

    Returns:
        numpy array: The tapered signal.
    """
    N = len(signal)
    taper = np.hanning(int(2 * taper_ratio * N))  # Create a Hann window for the taper
    window = np.ones(N)

    # Apply the taper at the beginning
    window[: len(taper) // 2] = taper[: len(taper) // 2]
    # Apply the taper at the end
    window[-len(taper) // 2 :] = taper[-len(taper) // 2 :]

    return signal * window

def fitting_function(X_c: float, C_f: float, Gamma: float, x: float|np.ndarray, y:float) -> float:

    # Gamma = 0.21  # Fracture energy (J/m^2)
    E = 51e9      # Young's modulus (Pa)
    nu = 0.25     # Poisson's ratio
    C_s = 2760    # Shear wave speed (m/s)
    C_d = 4790    # Longitudinal wave speed (m/s)

    return CohesiveModel.delta_sigma_xy(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E)

def chi_square(X_c: float, Gamma: float, C_f: float, X: np.ndarray, Y: np.ndarray):
    '''
    X_c and Gamma are the parameters to be fitted
    '''
    MODEL = fitting_function(X_c, C_f, Gamma, X, Y) / 10**6

    # chi2 = sum( (data_i - model_i)^2 / sigma_i^2 )
    chi2 = np.sum((Y - MODEL)**2)
    return chi2

def do_deconvolution(SIGNAL_1: np.ndarray,
                     SIGNAL_2: np.ndarray,
                     water_level: float =  0.05,
                     damp_ratio: float = 0.05,
                     stabilizing_type: str = 'W') -> np.ndarray:

    NUMERATOR = np.fft.fft(SIGNAL_1)
    DENUMERATOR = np.fft.fft(SIGNAL_2)


    max_denominator = np.max(np.abs(DENUMERATOR))
    water_level_value = water_level * max_denominator


    # Do water leveling
    STABILIZED_DENOMINATOR = DENUMERATOR
    for i in range(len(DENUMERATOR)):
        if np.abs(DENUMERATOR[i]) < water_level_value:
            STABILIZED_DENOMINATOR[i] = ( DENUMERATOR[i] / np.abs(DENUMERATOR[i]) ) * water_level_value
    DECONV_STABILIZED_WTER = NUMERATOR / STABILIZED_DENOMINATOR
    DECONV_TIME_DOMAIN_WTER = np.fft.ifft(DECONV_STABILIZED_WTER)


    # Do damped
    damp_value = np.average(np.abs(DENUMERATOR)**2) * damp_ratio
    DECONV_STABILIZED_DAMP = (NUMERATOR * np.conjugate(DENUMERATOR)) / (DENUMERATOR * np.conjugate(DENUMERATOR) + damp_value)
    DECONV_TIME_DOMAIN_DAMP = np.fft.ifft(DECONV_STABILIZED_DAMP)


    # Choose type
    if stabilizing_type == 'W':
        DECONV_TIME_DOMAIN = DECONV_TIME_DOMAIN_WTER
    if stabilizing_type == 'D':
        DECONV_TIME_DOMAIN = DECONV_TIME_DOMAIN_DAMP

    # Do shifting
    total_len = int(len(DECONV_TIME_DOMAIN))
    midpoint = int(len(DECONV_TIME_DOMAIN) / 2)
    DECONV_TIME_DOMAIN_SHIFTED = np.append(DECONV_TIME_DOMAIN[midpoint: total_len-1], DECONV_TIME_DOMAIN[0:midpoint-1])

    return np.real(DECONV_TIME_DOMAIN_SHIFTED)


def get_ccf_full(DATA_FLOOR_1: np.ndarray, DATA_FLOOR_2: np.ndarray) -> np.ndarray:
    DATA_1 = DATA_FLOOR_1 - np.mean(DATA_FLOOR_1)
    DATA_2 = DATA_FLOOR_2 - np.mean(DATA_FLOOR_2)

    DATA_1 = scipy.signal.detrend(DATA_1)
    DATA_2 = scipy.signal.detrend(DATA_2)

    DATA_1 = DATA_1 - np.mean(DATA_1)
    DATA_2 = DATA_2 - np.mean(DATA_2)

    cross_correlation = scipy.signal.correlate(DATA_1, DATA_2, mode='full')
    return cross_correlation