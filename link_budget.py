"""
Burger FSR Simulation -- Link Budget
=====================================
Shared radar link equation and SNR computation.
Uses the CORRECTED (4pi)^3 denominator per Gashinova 2013 [R7].

See BURGER_SIM_PRD.txt §5.3 and calculation_validation.md §7.
"""

import numpy as np
from . import config as cfg
from .utils import dbi_to_linear


def received_power(P_t: float,
                   G_t_dBi: float,
                   G_r_dBi: float,
                   wavelength: float,
                   sigma_fs: float,
                   R_tx: float,
                   R_rx: float) -> float:
    """
    Bistatic radar received power (standard form).

    P_r = P_t · G_t · G_r · lam^2 · sigma_fs  /  ((4pi)^3 · R_tx^2 · R_rx^2)

    NOTE: The PRD writes (4pi)^2 which is INCORRECT.
    The correct form uses (4pi)^3 = 64pi^3  (confirmed by [R7] Eq. 13).
    """
    G_t = dbi_to_linear(G_t_dBi)
    G_r = dbi_to_linear(G_r_dBi)
    numerator   = P_t * G_t * G_r * wavelength ** 2 * sigma_fs
    denominator = (4 * np.pi) ** 3 * R_tx ** 2 * R_rx ** 2
    return numerator / denominator


def noise_power(bandwidth: float = None,
                temperature: float = None) -> float:
    """
    Thermal noise floor  N = k · T · B.
    """
    B = bandwidth   or cfg.BANDWIDTH
    T = temperature or cfg.NOISE_TEMP
    return cfg.K_BOLTZMANN * T * B


def snr(P_r: float, bandwidth: float = None, temperature: float = None) -> float:
    """
    Signal-to-noise ratio (linear).
    """
    N = noise_power(bandwidth, temperature)
    return P_r / N


def snr_db(P_r: float, **kwargs) -> float:
    """SNR in dB."""
    return 10.0 * np.log10(np.maximum(snr(P_r, **kwargs), 1e-300))
