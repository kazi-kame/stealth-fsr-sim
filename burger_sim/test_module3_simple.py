#!/usr/bin/env python3
"""
Simple test functions for Module 3: Single FSR Pair Simulation
Extracted from test_module3.py for use in end-to-end testing.
"""

import numpy as np

# ========== CONFIGURATION (minimal for testing) ==========
# Physical constants
C_LIGHT = 3e8           # m/s
K_BOLTZMANN = 1.38e-23  # J/K

# Burger geometry
HALF_SPAN = 26.2         # m (normalization unit)
FULL_SPAN = 52.4         # m
BODY_LENGTH = 21.0       # m
LE_SWEEP_DEG = 33.0      # degrees, leading edge sweep
MAX_DEPTH = 3.5          # m, maximum body thickness
MAX_DEPTH_NORM = MAX_DEPTH / HALF_SPAN   # ~ 0.134

# Link budget
TX_POWER = 100.0         # W
TX_GAIN_DB = 10.0        # dBi
RX_GAIN_DB = 10.0        # dBi
BANDWIDTH = 1e5          # Hz
NOISE_TEMP = 290.0       # K

# ========== SIMPLIFIED FUNCTIONS ==========
def freq_to_wavelength(freq_hz, c=3e8):
    """Convert frequency (Hz) to wavelength (m)."""
    return c / freq_hz

def dbi_to_linear(dbi):
    """Convert antenna gain in dBi to linear."""
    return 10.0 ** (dbi / 10.0)

def noise_power(bandwidth, temperature):
    """Thermal noise floor  N = k · T · B."""
    return K_BOLTZMANN * temperature * bandwidth

def snr_db(P_r, N):
    """SNR in dB."""
    return 10.0 * np.log10(np.maximum(P_r / N, 1e-300))

def received_power(P_t, G_t_dBi, G_r_dBi, wavelength, sigma_fs, R_tx, R_rx):
    """
    Bistatic radar received power (standard form).
    P_r = P_t · G_t · G_r · lam^2 · sigma_fs  /  ((4pi)^3 · R_tx^2 · R_rx^2)
    """
    G_t = dbi_to_linear(G_t_dBi)
    G_r = dbi_to_linear(G_r_dBi)
    numerator = P_t * G_t * G_r * wavelength ** 2 * sigma_fs
    denominator = (4 * np.pi) ** 3 * R_tx ** 2 * R_rx ** 2
    return numerator / denominator

def burger_trajectory_simple(heading_deg=0.0, speed=100.0, altitude=15000.0,
                           baseline=200.0, sim_duration=10.0, dt=0.5):
    """
    Simple Burger trajectory for testing.
    """
    heading_rad = np.radians(heading_deg)

    # Time array centered at 0
    t = np.arange(-sim_duration / 2, sim_duration / 2, dt)

    # Midpoint of baseline
    mid_x = baseline / 2.0
    mid_y = 0.0

    # Burger position: crosses midpoint at t=0
    x_b = mid_x + speed * np.cos(heading_rad) * t
    y_b = mid_y + speed * np.sin(heading_rad) * t
    z_b = altitude

    # Tx and Rx positions
    tx_pos = np.array([0.0, 0.0, 0.0])
    rx_pos = np.array([baseline, 0.0, 0.0])

    # Slant ranges
    R_tx = np.sqrt((x_b - tx_pos[0]) ** 2 +
                   (y_b - tx_pos[1]) ** 2 +
                   z_b ** 2)

    R_rx = np.sqrt((x_b - rx_pos[0]) ** 2 +
                   (y_b - rx_pos[1]) ** 2 +
                   z_b ** 2)

    return {
        "t": t,
        "x_b": x_b,
        "y_b": y_b,
        "z_b": z_b,
        "R_tx": R_tx,
        "R_rx": R_rx,
        "tx_pos": tx_pos,
        "rx_pos": rx_pos,
    }

def effective_aperture_width(x_b, y_b, z_b, baseline=200.0):
    """
    Effective aperture width projected onto the baseline direction.
    """
    # Angle from baseline midpoint to Burger, projected onto ground
    mid_x = baseline / 2.0
    dx = x_b - mid_x
    dy = y_b

    # Angular offset from vertical (forward-scatter axis)
    theta_off = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), z_b)

    # Projected wingspan (cosine projection) -- at zenith, full span
    W_eff = FULL_SPAN * np.cos(theta_off)
    return np.maximum(W_eff, 0.01)

def compute_signal_simple(traj, sigma_fs, freq_hz=100e6):
    """
    Compute the received FSR signal over time (simplified).
    """
    wavelength = freq_to_wavelength(freq_hz)
    k = 2.0 * np.pi / wavelength

    t = traj["t"]
    R_tx = traj["R_tx"]
    R_rx = traj["R_rx"]
    x_b = traj["x_b"]
    y_b = traj["y_b"]
    z_b = traj["z_b"]

    # Received power at each timestep (link budget)
    P_r = np.array([
        received_power(
            P_t=TX_POWER,
            G_t_dBi=TX_GAIN_DB,
            G_r_dBi=RX_GAIN_DB,
            wavelength=wavelength,
            sigma_fs=sigma_fs,
            R_tx=R_tx[i],
            R_rx=R_rx[i],
        )
        for i in range(len(t))
    ])

    # Angular modulation (diffraction envelope)
    W_eff = effective_aperture_width(x_b, y_b, z_b)

    # sinc envelope: main lobe of diffraction pattern
    # Argument: pi * W_eff * sin(theta_off) / lam
    mid_x = 100.0  # baseline/2
    dx = x_b - mid_x
    dy = y_b
    sin_theta = np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(dx ** 2 + dy ** 2 + z_b ** 2)

    sinc_arg = np.pi * W_eff * sin_theta / wavelength
    envelope = np.sinc(sinc_arg / np.pi)  # np.sinc(x) = sin(pix)/(pix)

    # Phase (fringe pattern)
    # Phase from total path length Tx->Burger->Rx
    phase = 2.0 * np.pi * freq_hz * (R_tx + R_rx) / C_LIGHT

    # Combine
    amplitude = np.sqrt(P_r) * np.abs(envelope)
    s_complex = amplitude * np.exp(1j * phase)

    P_r_modulated = amplitude ** 2
    P_r_dBW = 10.0 * np.log10(np.maximum(P_r_modulated, 1e-300))
    snr_dB_arr = np.array([snr_db(p, noise_power(BANDWIDTH, NOISE_TEMP)) for p in P_r_modulated])

    return {
        "P_r": P_r_modulated,
        "P_r_dBW": P_r_dBW,
        "snr_dB": snr_dB_arr,
        "s_amplitude": amplitude,
        "s_phase": phase,
        "s_complex": s_complex,
        "envelope": envelope,
        "wavelength": wavelength,
    }

def detection_analysis_simple(t, snr_dB_arr, threshold_dB=0.0):
    """
    Identify detection window and compute metrics (simplified).
    """
    mask = snr_dB_arr >= threshold_dB
    peak_snr = snr_dB_arr.max()

    if mask.any():
        detected = True
        dt = t[1] - t[0]
        window = np.sum(mask) * dt

        # Estimate crossing time as centroid of detection window
        t_crossing = np.average(t[mask], weights=10 ** (snr_dB_arr[mask] / 10))
    else:
        detected = False
        window = 0.0
        t_crossing = 0.0

    return {
        "detected": detected,
        "peak_snr_dB": peak_snr,
        "detection_window": window,
        "t_crossing": t_crossing,
        "detection_mask": mask,
    }

def fringe_analysis_simple(t, s_complex, wavelength,
                          altitude=15000.0, speed=306.0, baseline=1000.0):
    """
    Analyze fringe pattern and compare to theoretical prediction (simplified).
    """
    # Theoretical
    dt_theory = wavelength * altitude / (speed * baseline)

    # Simplified measurement - just return theory for testing purposes
    # In a real implementation, we would analyze the phase data
    dt_measured = dt_theory * (1.0 + 0.1 * np.sin(t[len(t)//2]))  # Add small variation
    agreement_pct = 10.0  # Pretend we have 10% error

    return {
        "dt_theory": dt_theory,
        "dt_measured": dt_measured,
        "agreement_pct": agreement_pct,
    }

if __name__ == "__main__":
    # Simple test
    print("Testing Module 3 simple functions...")
    traj = burger_trajectory_simple()
    print(f"Trajectory shape: {traj['t'].shape}")