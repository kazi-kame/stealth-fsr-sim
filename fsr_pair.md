# 💻 Code Node: fsr_pair.py

## 🔗 Workspace Architecture Connections
[[geometry]], [[array_tracking]], [[diffraction]], [[utils]], [[link_budget]], [[main]], [[config]], [[array_optimizer]], [[__init__]], [[plots]], [[burger_geometry_data]]

## 📜 Code Source
```python
"""
Burger FSR Simulation -- Module 3: Single FSR Pair Simulation
===============================================================
Simulate one transmitter-receiver pair with Burger crossing at altitude.
Compute received signal strength, SNR, and fringe pattern over time.

See BURGER_SIM_PRD.txt §5.
"""

import numpy as np

from . import config as cfg
from .utils import freq_to_wavelength
from .link_budget import received_power, snr_db


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GEOMETRY & TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def burger_trajectory(heading_deg: float = None,
                      speed: float = None,
                      altitude: float = None,
                      baseline: float = None,
                      sim_duration: float = None,
                      dt: float = None) -> dict:
    """
    Compute Burger's 3D position over time as it crosses the FSR baseline.

    The Tx is at (0, 0, 0), the Rx is at (baseline, 0, 0).
    Burger crosses the midpoint at t = 0.

    Parameters
    ----------
    heading_deg  : heading angle (degrees from x-axis)
    speed        : Burger speed (m/s)
    altitude     : Burger altitude (m)
    baseline     : Tx-Rx separation (m)
    sim_duration : total simulation window (s)
    dt           : timestep (s)

    Returns
    -------
    traj : dict with keys
        't'       : (N,) time array
        'x_b'     : (N,) Burger x-position
        'y_b'     : (N,) Burger y-position
        'z_b'     : scalar altitude
        'R_tx'    : (N,) slant range Burger -> Tx
        'R_rx'    : (N,) slant range Burger -> Rx
        'tx_pos'  : (3,) Tx position
        'rx_pos'  : (3,) Rx position
    """
    if heading_deg is None:
        heading_deg = cfg.HEADING_ANGLE
    if speed is None:
        speed = cfg.BURGER_SPEED
    if altitude is None:
        altitude = cfg.BURGER_ALTITUDE
    if baseline is None:
        baseline = cfg.BASELINE_LENGTH
    if sim_duration is None:
        sim_duration = cfg.SIM_DURATION
    if dt is None:
        dt = cfg.DT

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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIGNAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def effective_aperture_width(x_b: np.ndarray,
                             y_b: np.ndarray,
                             z_b: float,
                             baseline: float = None) -> np.ndarray:
    """
    Effective aperture width projected onto the baseline direction.

    This is a simplified model; the full model uses the 2D diffraction
    pattern from Module 2.  Here, we use a sinc-envelope approximation
    based on the angular position of Burger relative to the baseline.

    Returns W_eff in meters (physical wingspan projected onto baseline).
    """
    if baseline is None:
        baseline = cfg.BASELINE_LENGTH

    # Angle from baseline midpoint to Burger, projected onto ground
    mid_x = baseline / 2.0
    dx = x_b - mid_x
    dy = y_b

    # Angular offset from vertical (forward-scatter axis)
    theta_off = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), z_b)

    # Projected wingspan (cosine projection) -- at zenith, full span
    W_eff = cfg.FULL_SPAN * np.cos(theta_off)
    return np.maximum(W_eff, 0.01)


def compute_signal(traj: dict,
                   sigma_fs: float,
                   freq_hz: float = None) -> dict:
    """
    Compute the received FSR signal over time.

    Parameters
    ----------
    traj     : trajectory dict from burger_trajectory()
    sigma_fs : forward scatter RCS in m^2 (from Module 2)
    freq_hz  : operating frequency (Hz)

    Returns
    -------
    signal : dict with keys
        'P_r'        : (N,) received power (W)
        'P_r_dBW'    : (N,) received power (dBW)
        'snr_dB'     : (N,) SNR in dB
        's_amplitude': (N,) signal amplitude envelope
        's_phase'    : (N,) signal phase (fringes)
        's_complex'  : (N,) complex signal
    """
    if freq_hz is None:
        freq_hz = cfg.OPERATING_FREQ

    wavelength = freq_to_wavelength(freq_hz)
    k = 2.0 * np.pi / wavelength

    t = traj["t"]
    R_tx = traj["R_tx"]
    R_rx = traj["R_rx"]
    x_b = traj["x_b"]
    y_b = traj["y_b"]
    z_b = traj["z_b"]

    # ── Received power at each timestep (link budget) ──────────────────
    P_r = np.array([
        received_power(
            P_t=cfg.TX_POWER,
            G_t_dBi=cfg.TX_GAIN_DB,
            G_r_dBi=cfg.RX_GAIN_DB,
            wavelength=wavelength,
            sigma_fs=sigma_fs,
            R_tx=R_tx[i],
            R_rx=R_rx[i],
        )
        for i in range(len(t))
    ])

    # ── Angular modulation (diffraction envelope) ──────────────────────
    W_eff = effective_aperture_width(x_b, y_b, z_b)

    # sinc envelope: main lobe of diffraction pattern
    # Argument: pi * W_eff * sin(theta_off) / lam
    mid_x = cfg.BASELINE_LENGTH / 2.0
    dx = x_b - mid_x
    dy = y_b
    sin_theta = np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(dx ** 2 + dy ** 2 + z_b ** 2)

    sinc_arg = np.pi * W_eff * sin_theta / wavelength
    envelope = np.sinc(sinc_arg / np.pi)  # np.sinc(x) = sin(pix)/(pix)

    # ── Phase (fringe pattern) ─────────────────────────────────────────
    # Phase from total path length Tx->Burger->Rx
    phase = 2.0 * np.pi * freq_hz * (R_tx + R_rx) / cfg.C_LIGHT

    # ── Combine ────────────────────────────────────────────────────────
    amplitude = np.sqrt(P_r) * np.abs(envelope)
    s_complex = amplitude * np.exp(1j * phase)

    P_r_modulated = amplitude ** 2
    P_r_dBW = 10.0 * np.log10(np.maximum(P_r_modulated, 1e-300))
    snr_dB_arr = np.array([snr_db(p) for p in P_r_modulated])

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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DETECTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def detection_analysis(t: np.ndarray,
                       snr_dB_arr: np.ndarray,
                       threshold_dB: float = None) -> dict:
    """
    Identify detection window and compute metrics.

    Returns
    -------
    analysis : dict
        'detected'          : bool
        'peak_snr_dB'       : float
        'detection_window'  : float (seconds)
        't_crossing'        : estimated crossing time
        'detection_mask'    : bool array
    """
    if threshold_dB is None:
        threshold_dB = cfg.SNR_THRESHOLD_DB

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


def fringe_analysis(t: np.ndarray,
                    s_complex: np.ndarray,
                    wavelength: float,
                    altitude: float = None,
                    speed: float = None,
                    baseline: float = None) -> dict:
    """
    Analyze fringe pattern and compare to theoretical prediction.

    Theoretical fringe spacing: Δt = lam * H / (v * baseline)
    """
    if altitude is None:
        altitude = cfg.BURGER_ALTITUDE
    if speed is None:
        speed = cfg.BURGER_SPEED
    if baseline is None:
        baseline = cfg.BASELINE_LENGTH

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

    return {
        "dt_theory": dt_theory,
        "dt_measured": dt_measured,
        "agreement_pct": abs(dt_theory - dt_measured) / dt_theory * 100
                         if not np.isnan(dt_measured) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL MODULE 3 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_module3(sigma_fs: float = None,
                freq_hz: float = None,
                heading_deg: float = None,
                verbose: bool = True) -> dict:
    """
    Execute Module 3 end-to-end.

    Parameters
    ----------
    sigma_fs    : forward scatter RCS (m^2). If None, uses optical-limit estimate.
    freq_hz     : operating frequency (Hz)
    heading_deg : Burger heading (degrees)

    Returns
    -------
    result : dict
    """
    if freq_hz is None:
        freq_hz = cfg.OPERATING_FREQ

    wavelength = freq_to_wavelength(freq_hz)

    # Default sigma_fs from optical limit: (4pi * A^2) / lam^2
    if sigma_fs is None:
        # Approximate physical planform area
        from .geometry import build_planform_vertices
        from .utils import polygon_area
        verts = build_planform_vertices()
        area_norm = polygon_area(verts)
        area_phys = area_norm * cfg.HALF_SPAN ** 2
        sigma_fs = 4.0 * np.pi * area_phys ** 2 / wavelength ** 2

    if verbose:
        print("=" * 60)
        print("MODULE 3 -- SINGLE FSR PAIR SIMULATION")
        print("=" * 60)
        print(f"  Frequency    : {freq_hz/1e6:.1f} MHz  (lam = {wavelength:.2f} m)")
        print(f"  sigma_fs         : {sigma_fs:.2e} m^2  "
              f"({10*np.log10(sigma_fs):.1f} dBsm)")
        print(f"  Baseline     : {cfg.BASELINE_LENGTH:.0f} m")
        print(f"  Altitude     : {cfg.BURGER_ALTITUDE:.0f} m")
        print(f"  Speed        : {cfg.BURGER_SPEED:.0f} m/s")
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

```
