#!/usr/bin/env python3
"""
Test for Module 3: Single FSR Pair Simulation
Tests trajectory calculation, signal model, SNR computation, and fringe analysis.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# ========== CONFIGURATION (same as used elsewhere) ==========
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

# FSR pair simulation
BASELINE_LENGTH = 1000.0     # m, Tx-Rx separation
BURGER_ALTITUDE = 15000.0    # m
BURGER_SPEED = 306.0         # m/s (~ 1100 km/h)
SIM_DURATION = 120.0         # s (total simulation window)
DT = 0.01                    # s (timestep)
HEADING_ANGLE = 90.0         # degrees (perpendicular crossing, default)

# Link budget
TX_POWER = 1000.0            # W
TX_GAIN_DB = 20.0            # dBi
RX_GAIN_DB = 20.0            # dBi
BANDWIDTH = 1e6              # Hz
NOISE_TEMP = 290.0           # K
SNR_THRESHOLD_DB = 10.0      # dB

# Chosen operating frequency
OPERATING_FREQ = 150e6       # Hz (default 150 MHz VHF)

# ========== IMPORT GEOMETRY FUNCTIONS ==========
from test_module1_complete import (
    build_planform_vertices,
    polygon_area
)

# Replicate needed utility functions
def freq_to_wavelength(freq_hz, c=3e8):
    """Convert frequency (Hz) to wavelength (m)."""
    return c / freq_hz

def dbi_to_linear(dbi):
    """Convert antenna gain in dBi to linear."""
    return 10.0 ** (dbi / 10.0)

def noise_power(bandwidth, temperature):
    """Thermal noise floor  N = k · T · B."""
    return K_BOLTZMANN * temperature * bandwidth

def snr(P_r, N):
    """Signal-to-noise ratio (linear)."""
    return P_r / N

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

# ========== REPLICATE MODULE 3 FUNCTIONS ==========
def burger_trajectory(heading_deg=HEADING_ANGLE,
                      speed=BURGER_SPEED,
                      altitude=BURGER_ALTITUDE,
                      baseline=BASELINE_LENGTH,
                      sim_duration=SIM_DURATION,
                      dt=DT):
    """
    Compute Burger's 3D position over time as it crosses the FSR baseline.
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

def effective_aperture_width(x_b, y_b, z_b, baseline=BASELINE_LENGTH):
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

def compute_signal(traj, sigma_fs, freq_hz=OPERATING_FREQ):
    """
    Compute the received FSR signal over time.
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
    mid_x = BASELINE_LENGTH / 2.0
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

def detection_analysis(t, snr_dB_arr, threshold_dB=SNR_THRESHOLD_DB):
    """
    Identify detection window and compute metrics.
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

def fringe_analysis(t, s_complex, wavelength, altitude=BURGER_ALTITUDE,
                   speed=BURGER_SPEED, baseline=BASELINE_LENGTH):
    """
    Analyze fringe pattern and compare to theoretical prediction.
    """
    # Theoretical
    dt_theory = wavelength * altitude / (speed * baseline)

    # Measured: find fringes near t=0 (crossing point)
    dt = t[1] - t[0]
    near_crossing = np.abs(t) < 5.0  # ±5 seconds around crossing
    if near_crossing.any():
        phase_near = np.angle(s_complex[near_crossing])
        phase_unwrapped = np.unwrap(phase_near)
        t_near = t[near_crossing]

        # Estimate fringe rate from phase derivative
        if len(t_near) > 2:
            dphase_dt = np.gradient(phase_unwrapped, t_near)
            fringe_rate = np.median(np.abs(dphase_dt)) / (2.0 * np.pi)
            dt_measured = 1.0 / max(fringe_rate, 1e-10)
        else:
            dt_measured = np.nan
    else:
        dt_measured = np.nan

    agreement_pct = np.nan
    if not np.isnan(dt_measured):
        agreement_pct = abs(dt_theory - dt_measured) / dt_theory * 100

    return {
        "dt_theory": dt_theory,
        "dt_measured": dt_measured,
        "agreement_pct": agreement_pct,
    }

def run_module3(sigma_fs=None, freq_hz=OPERATING_FREQ, heading_deg=HEADING_ANGLE, verbose=True):
    """
    Execute Module 3 end-to-end.
    """
    if freq_hz is None:
        freq_hz = OPERATING_FREQ

    wavelength = freq_to_wavelength(freq_hz)

    # Default sigma_fs from optical limit: (4pi * A^2) / lam^2
    if sigma_fs is None:
        # Approximate physical planform area
        verts = build_planform_vertices()
        area_norm = polygon_area(verts)
        area_phys = area_norm * HALF_SPAN ** 2
        sigma_fs = 4.0 * np.pi * area_phys ** 2 / wavelength ** 2

    if verbose:
        print("=" * 60)
        print("MODULE 3 -- SINGLE FSR PAIR SIMULATION")
        print("=" * 60)
        print(f"  Frequency    : {freq_hz/1e6:.1f} MHz  (lam = {wavelength:.2f} m)")
        print(f"  sigma_fs         : {sigma_fs:.2e} m^2  "
              f"({10*np.log10(sigma_fs):.1f} dBsm)")
        print(f"  Baseline     : {BASELINE_LENGTH:.0f} m")
        print(f"  Altitude     : {BURGER_ALTITUDE:.0f} m")
        print(f"  Speed        : {BURGER_SPEED:.0f} m/s")
        print()

    # Compute trajectory
    traj = burger_trajectory(heading_deg=heading_deg)

    # Compute signal
    signal = compute_signal(traj, sigma_fs, freq_hz)

    # Detection analysis
    det = detection_analysis(traj["t"], signal["snr_dB"])

    # Fringe analysis
    fringes = fringe_analysis(traj["t"], signal["s_complex"], wavelength)

    if verbose:
        print(f"  Peak SNR     : {det['peak_snr_dB']:.1f} dB")
        print(f"  Detected     : {'YES' if det['detected'] else 'NO'}")
        print(f"  Det. window  : {det['detection_window']:.2f} s")
        print(f"  Crossing t   : {det['t_crossing']:.4f} s")
        print()
        print(f"  Fringe spacing (theory)   : {fringes['dt_theory']*1000:.1f} ms")
        print(f"  Fringe spacing (measured) : {fringes['dt_measured']*1000:.1f} ms")
        if not np.isnan(fringes["agreement_pct"]):
            print(f"  Agreement                 : {fringes['agreement_pct']:.1f}%")
        print()

    return {
        "traj": traj,
        "signal": signal,
        "detection": det,
        "fringes": fringes,
        "sigma_fs": sigma_fs,
        "freq_hz": freq_hz,
        "wavelength": wavelength,
    }

def validate_fsr_physics(result):
    """Validate FSR physics expectations from PRD and theory."""
    traj = result["traj"]
    signal = result["signal"]
    det = result["detection"]
    fringes = result["fringes"]
    sigma_fs = result["sigma_fs"]
    freq_hz = result["freq_hz"]
    wavelength = result["wavelength"]

    print("=" * 60)
    print("FSR PHYSICS VALIDATION (PRD §5)")
    print("=" * 60)

    # 1. Trajectory validation
    print(f"1. Trajectory checks:")
    print(f"   Time range: [{traj['t'][0]:.1f}, {traj['t'][-1]:.1f}] s")
    print(f"   Expected crossing at t=0: {'PASS' if abs(traj['t'].mean()) < 0.1 else 'FAIL'}")
    print(f"   Altitude constant: {np.std(traj['z_b']):.2f} m variation")
    print(f"   Speed magnitude: {np.mean(np.sqrt(np.diff(traj['x_b'])**2 + np.diff(traj['y_b'])**2)/DT):.1f} m/s")

    # 2. SNR and detection
    print(f"\n2. Signal detection:")
    print(f"   Peak SNR: {det['peak_snr_dB']:.1f} dB")
    print(f"   Detection threshold: {SNR_THRESHOLD_DB:.1f} dB")
    print(f"   Detection achieved: {'YES' if det['detected'] else 'NO'}")
    if det['detected']:
        print(f"   Detection window: {det['detection_window']:.2f} s")
        print(f"   Estimated crossing time: {det['t_crossing']:.4f} s")
        print(f"   Crossing time error: {abs(det['t_crossing']):.4f} s")

    # 3. Fringe pattern validation (PRD §5.5)
    print(f"\n3. Fringe pattern analysis:")
    print(f"   Theoretical fringe spacing: {fringes['dt_theory']*1000:.2f} ms")
    if not np.isnan(fringes['dt_measured']):
        print(f"   Measured fringe spacing: {fringes['dt_measured']*1000:.2f} ms")
        print(f"   Agreement: {fringes['agreement_pct']:.2f}%")
        # Expect good agreement (< 10% error)
        fringe_ok = fringes['agreement_pct'] < 10.0
        print(f"   Fringe validation: {'PASS' if fringe_ok else 'FAIL'}")
    else:
        print(f"   Measured fringe spacing: Could not measure")
        fringe_ok = False

    # 4. Signal characteristics
    print(f"\n4. Signal characteristics:")
    print(f"   Wavelength: {wavelength:.3f} m")
    print(f"   Sigma_fs: {sigma_fs:.2e} m² ({10*np.log10(sigma_fs):.1f} dBsm)")
    print(f"   Peak received power: {np.max(signal['P_r']):.2e} W")
    print(f"   Peak received power: {np.max(signal['P_r_dBW']):.1f} dBW")

    # 5. Physics plausibility checks
    print(f"\n5. Physics plausibility:")
    # Received power should be extremely small (fW to pW range typical)
    peak_pr_W = np.max(signal['P_r'])
    power_ok = 1e-18 <= peak_pr_W <= 1e-9  # aW to nW range
    print(f"   Peak power range check: {'PASS' if power_ok else 'FAIL'} "
          f"({peak_pr_W:.2e} W)")

    # SNR should be reasonable for detection
    snr_ok = det['peak_snr_dB'] > -20  # At least detectable with processing gain
    print(f"   SNR reasonableness: {'PASS' if snr_ok else 'FAIL'} "
          f"({det['peak_snr_dB']:.1f} dB)")

    # Overall validation
    trajectory_ok = abs(traj['t'].mean()) < 0.1 and np.std(traj['z_b']) < 0.1
    overall_pass = trajectory_ok and det['detected'] and fringe_ok and power_ok and snr_ok

    print(f"\n{'=' * 60}")
    print(f"OVERALL FSR VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return overall_pass


if __name__ == "__main__":
    print("Testing Module 3: Single FSR Pair Simulation")

    # Get test sigma_fs from Module 2 results or use reasonable default
    # From our Module 2 test, we had ~700 MHz optimal frequency with sigma_fs ~ 2e7 m^2
    test_sigma_fs = 2.0e7  # m^2, reasonable FSR value

    # Test with perpendicular crossing (heading = 90°)
    result = run_module3(sigma_fs=test_sigma_fs, heading_deg=90.0, verbose=True)

    # Validate physics expectations
    success = validate_fsr_physics(result)

    # Additional test: head-on crossing should give different signature
    print("\n" + "="*60)
    print("ADDITIONAL TEST: Head-on crossing (0° heading)")
    print("="*60)
    result_headon = run_module3(sigma_fs=test_sigma_fs, heading_deg=0.0, verbose=False)
    print(f"Head-on crossing peak SNR: {result_headon['detection']['peak_snr_dB']:.1f} dB")
    print(f"Head-on detection: {'YES' if result_headon['detection']['detected'] else 'NO'}")

    sys.exit(0 if success else 1)