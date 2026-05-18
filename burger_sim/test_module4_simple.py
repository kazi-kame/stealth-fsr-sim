#!/usr/bin/env python3
"""
Simple test functions for Module 4: Array Tracking & TDOA Triangulation
Extracted for use in end-to-end testing.
"""

import numpy as np
from scipy.optimize import least_squares

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

def burger_trajectory_simple(entry=(-200, 0, 5000),
                           heading_deg=45.0,
                           speed=50.0,
                           sim_duration=10.0,
                           dt=0.5):
    """
    Simple Burger trajectory for testing.
    """
    heading_rad = np.radians(heading_deg)
    t = np.arange(0, sim_duration, dt)

    x = entry[0] + speed * np.cos(heading_rad) * t
    y = entry[1] + speed * np.sin(heading_rad) * t
    z = entry[2]

    return {"t": t, "x": x, "y": y, "z": z}

def build_simple_array(spacing_x, spacing_y, n_pairs=4):
    """Build a simple small array for testing."""
    pairs = []
    half_bl = 50.0  # half baseline length

    # Create a small grid
    positions = []
    for i in range(3):
        for j in range(3):
            if len(positions) >= n_pairs:
                break
            x = -spacing_x + i * spacing_x
            y = -spacing_y + j * spacing_y
            positions.append((x, y))

    for (px, py) in positions[:n_pairs]:
        # Alternate baseline orientation
        angle = 0.0 if (len(pairs) % 2 == 0) else np.pi/2
        dx = half_bl * np.cos(angle)
        dy = half_bl * np.sin(angle)
        tx = np.array([px - dx, py - dy, 0.0])
        rx = np.array([px + dx, py + dy, 0.0])
        mid = np.array([px, py, 0.0])
        bl_dir = np.array([np.cos(angle), np.sin(angle)])
        pairs.append({
            "tx": tx, "rx": rx, "midpoint": mid,
            "baseline_dir": bl_dir,
            "fence": "test",
        })

    return pairs

def _signed_distance_to_baseline(px, py, mid, bl_dir):
    """Signed perpendicular distance from point (px, py) to the baseline
    line passing through `mid` with direction `bl_dir`."""
    # Normal to baseline direction
    normal = np.array([-bl_dir[1], bl_dir[0]])
    dx = px - mid[0]
    dy = py - mid[1]
    return dx * normal[0] + dy * normal[1]

def detect_crossing_events_simple(pairs, traj, freq_hz=50e6, snr_threshold_dB=-50.0):
    """Detect baseline crossing events for all FSR pairs (simplified)."""
    wavelength = freq_to_wavelength(freq_hz)
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    z = traj["z"]

    # Approximate sigma_fs for SNR estimate (optical limit)
    area_phys = 478.0  # approximate B-2 planform area in m^2
    sigma_fs = 4.0 * np.pi * area_phys ** 2 / wavelength ** 2

    events = []

    for idx, pair in enumerate(pairs):
        mid = pair["midpoint"][:2]  # ground position
        bl_dir = pair["baseline_dir"]

        # Compute signed distance at each timestep
        dist = np.array([_signed_distance_to_baseline(x[i], y[i], mid, bl_dir)
                         for i in range(len(t))])

        # Find zero crossings (sign changes)
        sign_changes = np.where(np.diff(np.sign(dist)) != 0)[0]

        for sc in sign_changes:
            # Linear interpolation for precise crossing time
            d0, d1 = dist[sc], dist[sc + 1]
            if abs(d1 - d0) < 1e-12:
                continue
            frac = -d0 / (d1 - d0)
            t_cross = t[sc] + frac * (t[sc + 1] - t[sc])
            x_cross = x[sc] + frac * (x[sc + 1] - x[sc])
            y_cross = y[sc] + frac * (y[sc + 1] - y[sc])

            # Compute SNR at crossing
            R_tx = np.sqrt((x_cross - pair["tx"][0]) ** 2 +
                           (y_cross - pair["tx"][1]) ** 2 +
                           z ** 2)
            R_rx = np.sqrt((x_cross - pair["rx"][0]) ** 2 +
                           (y_cross - pair["rx"][1]) ** 2 +
                           z ** 2)

            P_r = received_power(
                P_t=TX_POWER,
                G_t_dBi=TX_GAIN_DB,
                G_r_dBi=RX_GAIN_DB,
                wavelength=wavelength,
                sigma_fs=sigma_fs,
                R_tx=R_tx,
                R_rx=R_rx,
            )
            snr_val = snr_db(P_r, noise_power(BANDWIDTH, NOISE_TEMP))

            if snr_val >= snr_threshold_dB:
                events.append({
                    "pair_idx": idx,
                    "time": t_cross,
                    "snr_dB": snr_val,
                    "position": np.array([x_cross, y_cross]),
                    "midpoint": mid.copy(),
                    "baseline_dir": bl_dir.copy(),
                })

    # Sort by time
    events.sort(key=lambda e: e["time"])
    return events

def tdoa_triangulate_simple(events):
    """Reconstruct Burger's trajectory from crossing events using TDOA (simplified)."""
    if len(events) < 3:
        return {
            "positions": np.array([]),
            "heading_deg": np.nan,
            "speed_est": np.nan,
            "track_fit": None,
        }

    # Extract crossing positions and times
    times = np.array([e["time"] for e in events])
    midpoints = np.array([e["midpoint"] for e in events])
    bl_dirs = np.array([e["baseline_dir"] for e in events])

    # Simple method: use first three events to solve for position and velocity
    if len(times) >= 3:
        # Use linear least squares like in the full version
        normals = np.column_stack([-bl_dirs[:, 1], bl_dirs[:, 0]])  # (N, 2)

        N = len(events)
        A = np.zeros((N, 4))
        b = np.zeros(N)

        for i in range(N):
            nx, ny = normals[i]
            ti = times[i]
            mx, my = midpoints[i]

            A[i, 0] = nx
            A[i, 1] = ny
            A[i, 2] = nx * ti
            A[i, 3] = ny * ti
            b[i] = nx * mx + ny * my

        # Solve via least squares
        try:
            result_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            x0, y0, vx, vy = result_ls

            # Estimated heading
            heading_rad = np.arctan2(vy, vx)
            heading_deg = np.degrees(heading_rad) % 360

            # Estimated speed
            speed_est = np.sqrt(vx ** 2 + vy ** 2)

            # Reconstructed positions at each crossing time
            positions = np.column_stack([
                x0 + vx * times,
                y0 + vy * times,
            ])

            return {
                "positions": positions,
                "heading_deg": heading_deg,
                "speed_est": speed_est,
                "track_fit": {
                    "x0": x0, "y0": y0, "vx": vx, "vy": vy,
                    "residuals": residuals if len(residuals) > 0 else np.array([0.0]),
                },
            }
        except:
            # Fallback if linear algebra fails
            return {
                "positions": np.array([]),
                "heading_deg": np.nan,
                "speed_est": np.nan,
                "track_fit": None,
            }
    else:
        return {
            "positions": np.array([]),
            "heading_deg": np.nan,
            "speed_est": np.nan,
            "track_fit": None,
        }

if __name__ == "__main__":
    # Simple test
    print("Testing Module 4 simple functions...")
    pairs = build_simple_array()
    print(f"Built {len(pairs)} pairs")