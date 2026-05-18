#!/usr/bin/env python3
"""
Test for Module 4: Array Tracking & TDOA Triangulation
Tests array layout, crossing event detection, TDOA triangulation, and Monte Carlo error analysis.
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

# Array configuration
ARRAY_AREA_X = 5000.0    # m
ARRAY_AREA_Y = 5000.0    # m
ELEMENT_SPACING = 500.0  # m
PAIR_BASELINE = 200.0    # m (within each FSR pair)
BURGER_ENTRY = (-3000, 1000, 15000)  # m
BURGER_HEADING = 25.0    # degrees

# Monte Carlo
N_TRIALS = 500
TIMING_NOISE_STD = 0.001 # s (1 ms, GPS sync precision)

# Link budget
TX_POWER = 1000.0        # W
TX_GAIN_DB = 20.0        # dBi
RX_GAIN_DB = 20.0        # dBi
BANDWIDTH = 1e6          # Hz
NOISE_TEMP = 290.0       # K
SNR_THRESHOLD_DB = 10.0  # dB

# Chosen operating frequency
OPERATING_FREQ = 150e6   # Hz (default 150 MHz VHF)

# Burger motion
BURGER_SPEED = 306.0     # m/s
SIM_DURATION = 120.0     # s
DT = 0.01                # s

# ========== IMPORT GEOMETRY/UTILS FUNCTIONS ==========
from test_module1_complete import build_planform_vertices, polygon_area

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

def signed_polygon_area(vertices):
    """Signed area of a 2D polygon (shoelace formula)."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def polygon_area(vertices):
    """Unsigned area of a 2D polygon."""
    return abs(signed_polygon_area(vertices))

from scipy.spatial import Delaunay
from matplotlib.path import Path

def point_in_polygon(points, vertices):
    """Test which points lie inside a polygon."""
    path = Path(vertices)
    return path.contains_points(points)

def triangle_area(v0, v1, v2):
    """Area of a triangle from three 2D vertices."""
    return 0.5 * abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) -
        (v2[0] - v0[0]) * (v1[1] - v0[1])
    )

def ensure_ccw(vertices):
    """Ensure polygon vertices are in counter-clockwise order."""
    if signed_polygon_area(vertices) < 0:
        return vertices[::-1].copy()
    return vertices.copy()

# Replicate needed geometry functions
def _build_right_side_vertices():
    """Derive right-side (y > 0) vertices from public B-2 specs."""
    tan_le = np.tan(np.radians(LE_SWEEP_DEG))

    V0 = (0.000, 0.000)   # Nose tip

    y1 = 0.270
    V1 = (y1 * tan_le, y1)                  # (0.175, 0.270)

    y2 = 0.580
    V2 = (y1 * tan_le + (y2 - y1) * 0.64, y2)   # (~0.373, 0.580)

    y3 = 1.000
    V3 = (V2[0] + (y3 - y2) * 0.44, y3)          # (~0.558, 1.000)

    V4 = (V3[0] + 0.06, 0.935)                    # (~0.618, 0.935)

    V5 = (0.740, 0.670)

    V6 = (0.640, 0.420)

    x_max = BODY_LENGTH / HALF_SPAN       # 0.802
    V7 = (x_max, 0.000)

    return np.array([V0, V1, V2, V3, V4, V5, V6, V7], dtype=np.float64)

def build_planform_vertices():
    """Build the full planform polygon by mirroring the right side."""
    right = _build_right_side_vertices()

    inner = right[1:-1]                       # V1 … V6
    mirrored = inner[::-1].copy()             # V6…V1 reversed
    mirrored[:, 1] *= -1                      # flip y

    vertices = np.vstack([right, mirrored])   # V0..V7..m(V6)..m(V1)

    # Ensure CCW
    if signed_polygon_area(vertices) < 0:
        vertices = vertices[::-1].copy()

    return vertices

# ========== REPLICATE MODULE 4 FUNCTIONS ==========
def build_array_layout(area_x=None, area_y=None, spacing=None, pair_baseline=None):
    """Build a 2D array of FSR pairs with two orthogonal fences."""
    if area_x is None:
        area_x = ARRAY_AREA_X
    if area_y is None:
        area_y = ARRAY_AREA_Y
    if spacing is None:
        spacing = ELEMENT_SPACING
    if pair_baseline is None:
        pair_baseline = PAIR_BASELINE

    pairs = []
    half_bl = pair_baseline / 2.0

    # X-fence: pairs placed along x-axis, baselines oriented along y
    x_positions = np.arange(-area_x / 2, area_x / 2 + spacing / 2, spacing)
    y_center = 0.0
    for x in x_positions:
        tx = np.array([x, y_center - half_bl, 0.0])
        rx = np.array([x, y_center + half_bl, 0.0])
        mid = (tx + rx) / 2.0
        pairs.append({
            "tx": tx, "rx": rx, "midpoint": mid,
            "baseline_dir": np.array([0.0, 1.0]),
            "fence": "x",
        })

    # Y-fence: pairs placed along y-axis, baselines oriented along x
    y_positions = np.arange(-area_y / 2, area_y / 2 + spacing / 2, spacing)
    x_center = 0.0
    for y in y_positions:
        tx = np.array([x_center - half_bl, y, 0.0])
        rx = np.array([x_center + half_bl, y, 0.0])
        mid = (tx + rx) / 2.0
        pairs.append({
            "tx": tx, "rx": rx, "midpoint": mid,
            "baseline_dir": np.array([1.0, 0.0]),
            "fence": "y",
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

def array_burger_trajectory(entry=None, heading_deg=None, speed=None, duration=None, dt=None):
    """Compute Burger's trajectory through the array area."""
    if entry is None:
        entry = BURGER_ENTRY
    if heading_deg is None:
        heading_deg = BURGER_HEADING
    if speed is None:
        speed = BURGER_SPEED
    if duration is None:
        duration = SIM_DURATION
    if dt is None:
        dt = DT

    heading_rad = np.radians(heading_deg)
    t = np.arange(0, duration, dt)

    x = entry[0] + speed * np.cos(heading_rad) * t
    y = entry[1] + speed * np.sin(heading_rad) * t
    z = entry[2] if len(entry) > 2 else 15000.0

    return {"t": t, "x": x, "y": y, "z": z}

def detect_crossing_events(pairs, traj, freq_hz=None, snr_threshold_dB=None):
    """Detect baseline crossing events for all FSR pairs."""
    if freq_hz is None:
        freq_hz = OPERATING_FREQ
    if snr_threshold_dB is None:
        snr_threshold_dB = SNR_THRESHOLD_DB

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

def tdoa_triangulate(events, speed=None):
    """Reconstruct Burger's trajectory from crossing events using TDOA."""
    if speed is None:
        speed = BURGER_SPEED

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

    # Method: Crossing-time based position estimation
    normals = np.column_stack([-bl_dirs[:, 1], bl_dirs[:, 0]])  # (N, 2)

    # Build linear system: A * [x0, y0, vx, vy]^T = b
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

def monte_carlo_errors(events, true_heading, true_speed, n_trials=None, timing_noise_std=None, verbose=True):
    """Run Monte Carlo analysis with Gaussian timing noise."""
    if n_trials is None:
        n_trials = N_TRIALS
    if timing_noise_std is None:
        timing_noise_std = TIMING_NOISE_STD

    if len(events) < 3:
        if verbose:
            print("  WARNING: Fewer than 3 events -- cannot triangulate.")
        return {
            "heading_errors": np.array([]),
            "speed_errors": np.array([]),
            "position_errors": np.array([]),
            "heading_rms": np.inf,
            "speed_rms": np.inf,
            "position_rms": np.inf,
        }

    # True positions at crossing times
    true_times = np.array([e["time"] for e in events])
    true_positions = np.array([e["position"] for e in events])

    heading_errors = np.zeros(n_trials)
    speed_errors = np.zeros(n_trials)
    position_errors = np.zeros(n_trials)

    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        # Add timing noise to events
        noisy_events = []
        noise = rng.normal(0, timing_noise_std, len(events))
        for i, e in enumerate(events):
            noisy = e.copy()
            noisy["time"] = e["time"] + noise[i]
            noisy_events.append(noisy)

        # Triangulate with noisy events
        result = tdoa_triangulate(noisy_events)

        if np.isnan(result["heading_deg"]):
            heading_errors[trial] = np.nan
            speed_errors[trial] = np.nan
            position_errors[trial] = np.nan
            continue

        # Heading error (handle wrap-around)
        h_err = result["heading_deg"] - true_heading
        h_err = (h_err + 180) % 360 - 180
        heading_errors[trial] = h_err

        # Speed error
        speed_errors[trial] = result["speed_est"] - true_speed

        # Position error (RMS over crossing points)
        if len(result["positions"]) == len(true_positions):
            pos_err = np.sqrt(np.mean(
                np.sum((result["positions"] - true_positions) ** 2, axis=1)
            ))
            position_errors[trial] = pos_err
        else:
            position_errors[trial] = np.nan

    # Remove NaN trials
    valid = ~np.isnan(heading_errors)
    heading_errors = heading_errors[valid]
    speed_errors = speed_errors[valid]
    position_errors = position_errors[valid]

    heading_rms = np.sqrt(np.mean(heading_errors ** 2)) if len(heading_errors) > 0 else np.inf
    speed_rms = np.sqrt(np.mean(speed_errors ** 2)) if len(speed_errors) > 0 else np.inf
    position_rms = np.sqrt(np.mean(position_errors ** 2)) if len(position_errors) > 0 else np.inf

    if verbose:
        print(f"  Monte Carlo ({len(heading_errors)}/{n_trials} valid trials):")
        print(f"    Heading RMS error  : {heading_rms:.4f}°")
        print(f"    Speed RMS error    : {speed_rms:.3f} m/s")
        print(f"    Position RMS error : {position_rms:.2f} m")

    return {
        "heading_errors": heading_errors,
        "speed_errors": speed_errors,
        "position_errors": position_errors,
        "heading_rms": heading_rms,
        "speed_rms": speed_rms,
        "position_rms": position_rms,
    }

def run_module4(freq_hz=None, verbose=True):
    """Execute Module 4 end-to-end."""
    if freq_hz is None:
        freq_hz = OPERATING_FREQ

    if verbose:
        print("=" * 60)
        print("MODULE 4 -- ARRAY TRACKING & TDOA TRIANGULATION")
        print("=" * 60)

    # Build array
    pairs = build_array_layout()
    if verbose:
        print(f"  Array pairs  : {len(pairs)}")
        print(f"  Area         : {ARRAY_AREA_X:.0f} × {ARRAY_AREA_Y:.0f} m")
        print(f"  Spacing      : {ELEMENT_SPACING:.0f} m")
        print()

    # Compute trajectory
    traj = array_burger_trajectory()
    true_heading = BURGER_HEADING
    true_speed = BURGER_SPEED

    if verbose:
        print(f"  Entry point  : ({BURGER_ENTRY[0]:.0f}, {BURGER_ENTRY[1]:.0f}) m")
        print(f"  Heading      : {true_heading:.1f}°")
        print(f"  Speed        : {true_speed:.0f} m/s")
        print()

    # Detect crossing events
    events = detect_crossing_events(pairs, traj, freq_hz)
    if verbose:
        print(f"  Crossing events : {len(events)}")
        for e in events[:10]:
            print(f"    pair {e['pair_idx']:3d}  t={e['time']:.3f}s  "
                  f"SNR={e['snr_dB']:.1f}dB  "
                  f"pos=({e['position'][0]:.0f}, {e['position'][1]:.0f})")
        if len(events) > 10:
            print(f"    ... ({len(events) - 10} more)")
        print()

    # Triangulate (noise-free)
    tri_result = tdoa_triangulate(events)
    if verbose:
        print(f"  Estimated heading : {tri_result['heading_deg']:.2f}° "
              f"(true: {true_heading:.1f}°)")
        print(f"  Estimated speed   : {tri_result['speed_est']:.1f} m/s "
              f"(true: {true_speed:.0f})")
        print()

    # Monte Carlo
    mc = monte_carlo_errors(events, true_heading, true_speed, verbose=verbose)

    return {
        "pairs": pairs,
        "traj": traj,
        "events": events,
        "triangulation": tri_result,
        "monte_carlo": mc,
    }

def validate_array_tracking(result):
    """Validate array tracking results against expected physics."""
    pairs = result["pairs"]
    traj = result["traj"]
    events = result["events"]
    triang = result["triangulation"]
    mc = result["monte_carlo"]

    print("=" * 60)
    print("ARRAY TRACKING VALIDATION (PRD §6)")
    print("=" * 60)

    # 1. Array layout validation
    print(f"1. Array layout:")
    print(f"   Number of pairs: {len(pairs)}")
    expected_pairs = 2 * (int(ARRAY_AREA_X / ELEMENT_SPACING) + 1)  # Approximate
    print(f"   Expected ~{expected_pairs} pairs: {'REASONABLE' if 20 <= len(pairs) <= 100 else 'CHECK'}")

    # Check fence structure
    x_fence = [p for p in pairs if p["fence"] == "x"]
    y_fence = [p for p in pairs if p["fence"] == "y"]
    print(f"   X-fence pairs: {len(x_fence)} (baselines along y)")
    print(f"   Y-fence pairs: {len(y_fence)} (baselines along x)")

    # 2. Trajectory validation
    print(f"\n2. Burger trajectory through array:")
    print(f"   Time range: [{traj['t'][0]:.1f}, {traj['t'][-1]:.1f}] s")
    print(f"   X range: [{traj['x'].min():.1f}, {traj['x'].max():.1f}] m")
    print(f"   Y range: [{traj['y'].min():.1f}, {traj['y'].max():.1f}] m")
    print(f"   Array bounds: ±{ARRAY_AREA_X/2:.0f} m in x, ±{ARRAY_AREA_Y/2:.0f} m in y")

    # Check if trajectory passes through array
    x_in_bounds = np.logical_and(traj['x'] >= -ARRAY_AREA_X/2, traj['x'] <= ARRAY_AREA_X/2)
    y_in_bounds = np.logical_and(traj['y'] >= -ARRAY_AREA_Y/2, traj['y'] <= ARRAY_AREA_Y/2)
    in_array = np.logical_and(x_in_bounds, y_in_bounds)
    frac_in_array = np.mean(in_array)
    print(f"   Fraction of trajectory in array: {frac_in_array*100:.1f}%")
    traj_ok = frac_in_array > 0.1  # At least 10% should be in array
    print(f"   Trajectory validation: {'PASS' if traj_ok else 'FAIL'}")

    # 3. Crossing events
    print(f"\n3. Crossing events:")
    print(f"   Total events detected: {len(events)}")
    if len(events) > 0:
        snrs = [e['snr_dB'] for e in events]
        times = [e['time'] for e in events]
        print(f"   SNR range: [{np.min(snrs):.1f}, {np.max(snrs):.1f}] dB")
        print(f"   Time range: [{np.min(times):.1f}, {np.max(times):.1f}] s")
        events_ok = len(events) >= 3  # Need at least 3 for triangulation
        print(f"   Sufficient for triangulation: {'PASS' if events_ok else 'FAIL'} (≥3 events)")
    else:
        events_ok = False
        print(f"   Sufficient for triangulation: FAIL (0 events)")

    # 4. TDOA triangulation (noise-free)
    print(f"\n4. TDOA triangulation (noise-free):")
    if not np.isnan(triang["heading_deg"]):
        heading_err = abs(triang["heading_deg"] - BURGER_HEADING)
        # Handle wraparound
        heading_err = min(heading_err, 360 - heading_err)
        speed_err = abs(triang["speed_est"] - BURGER_SPEED)
        print(f"   Estimated heading: {triang['heading_deg']:.2f}° (true: {BURGER_HEADING:.1f}°)")
        print(f"   Heading error: {heading_err:.2f}°")
        print(f"   Estimated speed: {triang['speed_est']:.1f} m/s (true: {BURGER_SPEED:.0f} m/s)")
        print(f"   Speed error: {speed_err:.1f} m/s")

        # Expect good accuracy with zero noise
        triang_ok = heading_err < 1.0 and speed_err < 1.0  # <1° and <1 m/s error
        print(f"   Zero-noise accuracy: {'PASS' if triang_ok else 'FAIL'}")
    else:
        print(f"   Triangulation failed: Not enough events or singular geometry")
        triang_ok = False

    # 5. Monte Carlo error analysis
    print(f"\n5. Monte Carlo error analysis:")
    if not np.isinf(mc["position_rms"]):
        print(f"   Heading RMS error: {mc['heading_rms']:.4f}°")
        print(f"   Speed RMS error: {mc['speed_rms']:.3f} m/s")
        print(f"   Position RMS error: {mc['position_rms']:.2f} m")
        print(f"   Valid trials: {len(mc['heading_errors'])}/{N_TRIALS}")

        # Expect reasonable errors with 1ms timing noise
        # For 5km array, 1ms timing error → ~0.3m position error (c * Δt / 2)
        # Heading error depends on geometry
        mc_reasonable = mc['position_rms'] < 10.0 and mc['heading_rms'] < 10.0
        print(f"   Reasonable errors: {'PASS' if mc_reasonable else 'FAIL'} (<10° and <10m)")
    else:
        print(f"   Monte Carlo failed: Not enough valid trials")
        mc_reasonable = False

    # Overall validation
    layout_ok = len(pairs) > 0
    overall_pass = layout_ok and traj_ok and events_ok and triang_ok and mc_reasonable

    print(f"\n{'=' * 60}")
    print(f"OVERALL ARRAY TRACKING VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return overall_pass


if __name__ == "__main__":
    print("Testing Module 4: Array Tracking & TDOA Triangulation")

    # Run with verbose output to see what's happening
    result = run_module4(verbose=True)

    # Validate results
    success = validate_array_tracking(result)

    sys.exit(0 if success else 1)