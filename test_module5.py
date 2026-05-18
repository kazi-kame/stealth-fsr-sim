#!/usr/bin/env python3
"""
Test for Module 5: Array Configuration Optimization (GDOP)
Tests GDOP computation, grid search, local refinement, and random layout evaluation.
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

# Module 5 optimization parameters
N_PAIRS = 20             # Total FSR pairs (fixed by cost)
OPT_SPACING_X = [200, 300, 400, 500, 700, 1000]  # m, grid search values
OPT_SPACING_Y = [200, 300, 400, 500, 700, 1000]  # m, grid search values
OPT_ROTATIONS = [0, 15, 30, 45]       # degrees, grid search values
OPT_LAYOUTS = ['grid', 'hexagonal']   # layout types for grid search
N_RANDOM_LAYOUTS = 10    # Reduced for testing
HEADING_SWEEP_STEP = 5.0   # degrees, reduced for testing
GDOP_INF_SUBSTITUTE = 1e6

# ========== IMPORT GEOMETRY/UTILS/ARRAY_TRACKING FUNCTIONS ==========
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

# Replicate needed array_tracking functions
def build_custom_array(spacing_x, spacing_y, rotation_deg, n_pairs,
                      layout_type, area_x=None, area_y=None,
                      pair_baseline=None, random_seed=None):
    """Build an array with custom spacing, rotation, and layout type."""
    if area_x is None:
        area_x = ARRAY_AREA_X
    if area_y is None:
        area_y = ARRAY_AREA_Y
    if pair_baseline is None:
        pair_baseline = PAIR_BASELINE

    half_bl = pair_baseline / 2.0
    rot_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)

    def rotate(x, y):
        return (cos_r * x - sin_r * y,
                sin_r * x + cos_r * y)

    positions = []

    if layout_type == 'random':
        rng = np.random.RandomState(random_seed if random_seed is not None else 42)
        for _ in range(n_pairs):
            x = rng.uniform(-area_x / 2, area_x / 2)
            y = rng.uniform(-area_y / 2, area_y / 2)
            angle = rng.uniform(0, np.pi)
            positions.append((x, y, angle))
    else:
        # Generate grid positions
        nx = max(1, int(area_x / spacing_x))
        ny = max(1, int(area_y / spacing_y))
        xs = np.linspace(-area_x / 2, area_x / 2, nx)
        ys = np.linspace(-area_y / 2, area_y / 2, ny)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                px, py = x, y
                if layout_type == 'hexagonal' and j % 2 == 1:
                    px += spacing_x / 2.0

                # Apply rotation
                rx, ry = rotate(px, py)
                # Alternate baseline orientation for coverage
                angle = rot_rad + (np.pi / 2 if (i + j) % 2 == 0 else 0)
                positions.append((rx, ry, angle))

        # Limit to n_pairs (take a well-distributed subset)
        if len(positions) > n_pairs:
            indices = np.linspace(0, len(positions) - 1, n_pairs, dtype=int)
            positions = [positions[i] for i in indices]

    # Build pairs from positions
    pairs = []
    for (px, py, angle) in positions:
        dx = half_bl * np.cos(angle)
        dy = half_bl * np.sin(angle)
        tx = np.array([px - dx, py - dy, 0.0])
        rx = np.array([px + dx, py + dy, 0.0])
        mid = np.array([px, py, 0.0])
        bl_dir = np.array([np.cos(angle), np.sin(angle)])
        pairs.append({
            "tx": tx, "rx": rx, "midpoint": mid,
            "baseline_dir": bl_dir,
            "fence": "custom",
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

# ========== REPLICATE MODULE 5 FUNCTIONS ==========
def compute_gdop(events, target_pos=None):
    """Compute Geometric Dilution of Precision from crossing events."""
    if len(events) < 3:
        return GDOP_INF_SUBSTITUTE

    # Use centroid of event midpoints as approximate target position
    if target_pos is None:
        midpoints = np.array([e["midpoint"][:2] if len(e["midpoint"]) > 2
                              else e["midpoint"] for e in events])
        target_pos = midpoints.mean(axis=0)

    # Reference event (first)
    ref = events[0]
    ref_mid = ref["midpoint"][:2] if len(ref["midpoint"]) > 2 else ref["midpoint"]
    r0 = np.linalg.norm(target_pos - ref_mid)

    if r0 < 1e-6:
        r0 = 1.0  # avoid division by zero

    # Build H matrix: (N-1, 2)
    N = len(events)
    H = np.zeros((N - 1, 2))

    for i in range(1, N):
        mid_i = events[i]["midpoint"][:2] if len(events[i]["midpoint"]) > 2 \
                else events[i]["midpoint"]
        r_i = np.linalg.norm(target_pos - mid_i)
        if r_i < 1e-6:
            r_i = 1.0

        dx_i = (target_pos[0] - mid_i[0]) / r_i
        dy_i = (target_pos[1] - mid_i[1]) / r_i
        dx_0 = (target_pos[0] - ref_mid[0]) / r0
        dy_0 = (target_pos[1] - ref_mid[1]) / r0

        H[i - 1, 0] = dx_i - dx_0
        H[i - 1, 1] = dy_i - dy_0

    # Compute GDOP
    try:
        HtH = H.T @ H
        Q = np.linalg.inv(HtH)
        gdop = np.sqrt(np.trace(Q))
    except np.linalg.LinAlgError:
        gdop = GDOP_INF_SUBSTITUTE

    # Cap at substitute if unreasonably large
    if not np.isfinite(gdop) or gdop > GDOP_INF_SUBSTITUTE:
        gdop = GDOP_INF_SUBSTITUTE

    return gdop

def gdop_heading_sweep(pairs, heading_step=None, freq_hz=None,
                      speed=None, altitude=None, area_x=None, area_y=None):
    """Sweep over all heading angles and compute GDOP for each."""
    if heading_step is None:
        heading_step = HEADING_SWEEP_STEP
    if freq_hz is None:
        freq_hz = OPERATING_FREQ
    if speed is None:
        speed = BURGER_SPEED
    if altitude is None:
        altitude = 15000.0
    if area_x is None:
        area_x = ARRAY_AREA_X
    if area_y is None:
        area_y = ARRAY_AREA_Y

    headings = np.arange(0, 180, heading_step)
    gdops = np.zeros(len(headings))
    n_events = np.zeros(len(headings), dtype=int)

    # Entry position: outside array, moving through center
    entry_offset = max(area_x, area_y)

    for i, heading in enumerate(headings):
        heading_rad = np.radians(heading)

        # Entry point: outside array, heading toward center
        entry_x = -entry_offset * np.cos(heading_rad)
        entry_y = -entry_offset * np.sin(heading_rad)

        traj = array_burger_trajectory(
            entry=(entry_x, entry_y, altitude),
            heading_deg=heading,
            speed=speed,
            duration=2 * entry_offset / speed + 10,
        )

        events = detect_crossing_events(pairs, traj, freq_hz,
                                        snr_threshold_dB=-np.inf)  # keep all crossings for GDOP

        n_events[i] = len(events)
        gdops[i] = compute_gdop(events)

    # Summary statistics
    finite_mask = gdops < GDOP_INF_SUBSTITUTE
    if finite_mask.any():
        gdop_mean = np.mean(gdops[finite_mask])
        gdop_p95 = np.percentile(gdops[finite_mask], 95) if np.sum(finite_mask) > 1 else gdops[finite_mask][0]
    else:
        gdop_mean = GDOP_INF_SUBSTITUTE
        gdop_p95 = GDOP_INF_SUBSTITUTE

    worst_idx = np.argmax(gdops)

    return {
        "headings": headings,
        "gdops": gdops,
        "n_events": n_events,
        "gdop_worst": gdops[worst_idx],
        "gdop_mean": gdop_mean,
        "gdop_p95": gdop_p95,
        "worst_heading": headings[worst_idx],
    }

def grid_search(n_pairs=None, verbose=True):
    """Stage 1: Exhaustive search over discrete configuration space."""
    if n_pairs is None:
        n_pairs = N_PAIRS

    spacings_x = OPT_SPACING_X
    spacings_y = OPT_SPACING_Y
    rotations = OPT_ROTATIONS
    layouts = OPT_LAYOUTS

    total = len(spacings_x) * len(spacings_y) * len(rotations) * len(layouts)

    if verbose:
        print(f"  Stage 1: Grid search over {total} configurations...")

    records = []
    count = 0

    for layout in layouts:
        for rot in rotations:
            for sx in spacings_x:
                for sy in spacings_y:
                    count += 1

                    pairs = build_custom_array(
                        spacing_x=sx,
                        spacing_y=sy,
                        rotation_deg=rot,
                        n_pairs=n_pairs,
                        layout_type=layout,
                    )

                    sweep = gdop_heading_sweep(pairs)

                    records.append({
                        "spacing_x": sx,
                        "spacing_y": sy,
                        "rotation_deg": rot,
                        "layout_type": layout,
                        "n_actual_pairs": len(pairs),
                        "gdop_worst": sweep["gdop_worst"],
                        "gdop_mean": sweep["gdop_mean"],
                        "gdop_p95": sweep["gdop_p95"],
                        "worst_heading": sweep["worst_heading"],
                    })

                    if verbose and count % 50 == 0:
                        print(f"    [{count}/{total}] ...")

    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df.sort_values("gdop_worst", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if verbose:
            print(f"  Stage 1 complete. Best GDOP_worst = {df.iloc[0]['gdop_worst']:.2f}")
            print(f"  Top 5 configurations:")
            print(df.head(5).to_string(index=False))
            print()
    except ImportError:
        # Fallback if pandas not available - just return records
        df = records
        df.sort(key=lambda x: x['gdop_worst'])
        if verbose:
            print(f"  Stage 1 complete. Best GDOP_worst = {df[0]['gdop_worst']:.2f}")
            print(f"  Top 5 configurations:")
            for i in range(min(5, len(df))):
                print(f"    {df[i]}")
            print()

    return df

def _objective(params, n_pairs, layout_type):
    """Objective function for optimization: GDOP_worst."""
    sx, sy, rot = params

    # Bounds enforcement
    sx = max(100, min(sx, 2000))
    sy = max(100, min(sy, 2000))
    rot = rot % 90  # symmetry

    pairs = build_custom_array(
        spacing_x=sx,
        spacing_y=sy,
        rotation_deg=rot,
        n_pairs=n_pairs,
        layout_type=layout_type,
    )

    sweep = gdop_heading_sweep(pairs)
    return sweep["gdop_worst"]

def local_refinement(top_configs, n_top=5, n_pairs=None, verbose=True):
    """Stage 2: Local refinement of top configurations from grid search."""
    if n_pairs is None:
        n_pairs = N_PAIRS

    if verbose:
        print(f"  Stage 2: Refining top {n_top} configurations...")

    best_gdop = GDOP_INF_SUBSTITUTE
    best_config = None

    # Handle both pandas DataFrame and list of dicts
    if hasattr(top_configs, 'iloc'):  # pandas DataFrame
        iterable = [(i, top_configs.iloc[i]) for i in range(min(n_top, len(top_configs)))]
    else:  # list of dicts
        iterable = [(i, top_configs[i]) for i in range(min(n_top, len(top_configs)))]

    for i, row in iterable:
        if hasattr(row, 'spacing_x'):  # pandas Series
            x0 = [row["spacing_x"], row["spacing_y"], row["rotation_deg"]]
            layout = row["layout_type"]
        else:  # dict
            x0 = [row["spacing_x"], row["spacing_y"], row["rotation_deg"]]
            layout = row["layout_type"]

        try:
            from scipy.optimize import minimize
            result = minimize(
                _objective,
                x0=x0,
                args=(n_pairs, layout),
                method="Nelder-Mead",
                options={"maxiter": 50, "xatol": 5, "fatol": 0.1},
            )

            if result.fun < best_gdop:
                best_gdop = result.fun
                best_config = {
                    "spacing_x": max(100, min(result.x[0], 2000)),
                    "spacing_y": max(100, min(result.x[1], 2000)),
                    "rotation_deg": result.x[2] % 90,
                    "layout_type": layout,
                    "gdop_worst": result.fun,
                    "n_pairs": n_pairs,
                }

                if verbose:
                    print(f"    Config {i}: GDOP_worst = {result.fun:.2f} "
                          f"(sx={best_config['spacing_x']:.0f}, "
                          f"sy={best_config['spacing_y']:.0f}, "
                          f"rot={best_config['rotation_deg']:.1f}°, "
                          f"layout={layout})")

        except Exception as e:
            if verbose:
                print(f"    Config {i}: optimization failed -- {e}")

    if verbose and best_config:
        print(f"  Stage 2 complete. Best GDOP_worst = {best_gdop:.2f}")
        print()

    return best_config

def evaluate_random_layouts(n_random=None, n_pairs=None, verbose=True):
    """Evaluate N random array layouts for comparison."""
    if n_random is None:
        n_random = N_RANDOM_LAYOUTS
    if n_pairs is None:
        n_pairs = N_PAIRS

    if verbose:
        print(f"  Evaluating {n_random} random layouts...")

    gdop_worsts = np.zeros(n_random)
    best_gdop = GDOP_INF_SUBSTITUTE
    best_seed = 0
    best_pairs = None

    for seed in range(n_random):
        pairs = build_custom_array(
            spacing_x=500,   # not used for random
            spacing_y=500,   # not used for random
            rotation_deg=0,  # not used for random
            n_pairs=n_pairs,
            layout_type='random',
            random_seed=seed,
        )

        sweep = gdop_heading_sweep(pairs)
        gdop_worsts[seed] = sweep["gdop_worst"]

        if sweep["gdop_worst"] < best_gdop:
            best_gdop = sweep["gdop_worst"]
            best_seed = seed
            best_pairs = pairs

    if verbose:
        print(f"  Best random layout: seed={best_seed}, GDOP_worst={best_gdop:.2f}")
        print(f"  Mean random GDOP_worst: {np.mean(gdop_worsts):.2f}")
        print()

    return {
        "gdop_worsts": gdop_worsts,
        "best_gdop": best_gdop,
        "best_seed": best_seed,
        "best_pairs": best_pairs,
    }

def run_module5(verbose=True):
    """Execute Module 5 end-to-end."""
    if verbose:
        print("=" * 60)
        print("MODULE 5 -- ARRAY CONFIGURATION OPTIMIZATION")
        print("=" * 60)
        print()

    # Stage 1: Grid search
    df = grid_search(verbose=verbose)

    # Stage 2: Local refinement
    best_config = local_refinement(df, verbose=verbose)

    # Random layout comparison
    random_result = evaluate_random_layouts(verbose=verbose)

    # Generate the optimal array and do a final GDOP sweep
    optimal_pairs = None
    optimal_sweep = None

    if best_config:
        optimal_pairs = build_custom_array(
            spacing_x=best_config["spacing_x"],
            spacing_y=best_config["spacing_y"],
            rotation_deg=best_config["rotation_deg"],
            n_pairs=best_config["n_pairs"],
            layout_type=best_config["layout_type"],
        )
        optimal_sweep = gdop_heading_sweep(optimal_pairs)

    # Also sweep the naive uniform grid for comparison
    naive_pairs = build_custom_array(
        spacing_x=500, spacing_y=500, rotation_deg=0,
        n_pairs=N_PAIRS, layout_type='grid',
    )
    naive_sweep = gdop_heading_sweep(naive_pairs)

    # Top 3 configs for polar plot
    top3_sweeps = []
    df_top = df.head(3) if hasattr(df, 'head') else sorted(df, key=lambda x: x['gdop_worst'])[:3]
    for row in df_top:
        if hasattr(row, 'spacing_x'):  # pandas Series
            p = build_custom_array(
                spacing_x=row["spacing_x"],
                spacing_y=row["spacing_y"],
                rotation_deg=row["rotation_deg"],
                n_pairs=N_PAIRS,
                layout_type=row["layout_type"],
            )
        else:  # dict
            p = build_custom_array(
                spacing_x=row["spacing_x"],
                spacing_y=row["spacing_y"],
                rotation_deg=row["rotation_deg"],
                n_pairs=N_PAIRS,
                layout_type=row["layout_type"],
            )
        s = gdop_heading_sweep(p)
        s["label"] = (f"{row['layout_type'] if hasattr(row, 'layout_type') else row.get('layout_type', 'grid')} "
                      f"sx={row['spacing_x']:.0f} sy={row['spacing_y']:.0f} "
                      f"rot={row['rotation_deg']:.0f}°")
        top3_sweeps.append(s)

    # Comparison summary
    comparison = {
        "optimal_gdop_worst": best_config["gdop_worst"] if best_config else np.inf,
        "naive_gdop_worst": naive_sweep["gdop_worst"],
        "random_best_gdop_worst": random_result["best_gdop"],
        "improvement_vs_naive": (naive_sweep["gdop_worst"] -
                                 (best_config["gdop_worst"] if best_config else np.inf)),
    }

    if verbose:
        print("=" * 60)
        print("  OPTIMIZATION SUMMARY")
        print("=" * 60)
        if best_config:
            print(f"  Optimal configuration:")
            print(f"    Layout       : {best_config['layout_type']}")
            print(f"    Spacing X    : {best_config['spacing_x']:.0f} m")
            print(f"    Spacing Y    : {best_config['spacing_y']:.0f} m")
            print(f"    Rotation     : {best_config['rotation_deg']:.1f}°")
            print(f"    GDOP worst   : {best_config['gdop_worst']:.2f}")
            print()
        print(f"  Naive grid GDOP worst   : {naive_sweep['gdop_worst']:.2f}")
        print(f"  Best random GDOP worst  : {random_result['best_gdop']:.2f}")
        print(f"  Improvement vs naive    : {comparison['improvement_vs_naive']:.2f}")
        print()

    return {
        "grid_search": df,
        "best_optimized": best_config,
        "random_evaluation": random_result,
        "optimal_sweep": optimal_sweep,
        "optimal_pairs": optimal_pairs,
        "naive_sweep": naive_sweep,
        "top3_sweeps": top3_sweeps,
        "comparison": comparison,
    }

def validate_gdop_physics(result):
    """Validate GDOP optimization results against expected physics."""
    grid_search = result["grid_search"]
    best_config = result["best_optimized"]
    random_result = result["random_evaluation"]
    optimal_sweep = result["optimal_sweep"]
    comparison = result["comparison"]

    print("=" * 60)
    print("GDOP OPTIMIZATION VALIDATION (PRD §7)")
    print("=" * 60)

    # 1. Grid search validation
    print(f"1. Grid search results:")
    if hasattr(grid_search, 'iloc'):  # pandas DataFrame
        print(f"   Configurations evaluated: {len(grid_search)}")
        print(f"   Best GDOP_worst: {grid_search.iloc[0]['gdop_worst']:.2f}")
        print(f"   Worst GDOP_worst: {grid_search.iloc[-1]['gdop_worst']:.2f}")
        gdop_range = grid_search.iloc[-1]['gdop_worst'] - grid_search.iloc[0]['gdop_worst']
        print(f"   GDOP range: {gdop_range:.2f}")
    else:  # list of dicts
        print(f"   Configurations evaluated: {len(grid_search)}")
        print(f"   Best GDOP_worst: {grid_search[0]['gdop_worst']:.2f}")
        print(f"   Worst GDOP_worst: {grid_search[-1]['gdop_worst']:.2f}")
        gdop_range = grid_search[-1]['gdop_worst'] - grid_search[0]['gdop_worst']
        print(f"   GDOP range: {gdop_range:.2f}")

    search_ok = len(grid_search) > 0 and gdop_range > 0
    print(f"   Search validity: {'PASS' if search_ok else 'FAIL'}")

    # 2. Best configuration validation
    print(f"\n2. Best configuration:")
    if best_config:
        print(f"   Layout: {best_config['layout_type']}")
        print(f"   Spacing: {best_config['spacing_x']:.0f} × {best_config['spacing_y']:.0f} m")
        print(f"   Rotation: {best_config['rotation_deg']:.1f}°")
        print(f"   GDOP worst: {best_config['gdop_worst']:.2f}")
        print(f"   GDOP mean: {best_config['gdop_mean']:.2f}" if 'gdop_mean' in best_config else "   GDOP mean: N/A")
        print(f"   Worst heading: {best_config['worst_heading']:.1f}°" if 'worst_heading' in best_config else "   Worst heading: N/A")

        # Check that parameters are within bounds
        spacing_ok = (100 <= best_config['spacing_x'] <= 2000 and
                     100 <= best_config['spacing_y'] <= 2000)
        rotation_ok = 0 <= best_config['rotation_deg'] < 90
        layout_ok = best_config['layout_type'] in ['grid', 'hexagonal', 'random']
        config_ok = spacing_ok and rotation_ok and layout_ok
        print(f"   Parameter bounds: {'PASS' if config_ok else 'FAIL'}")
    else:
        print(f"   No valid configuration found")
        config_ok = False

    # 3. Optimal sweep validation
    print(f"\n3. Optimal configuration sweep:")
    if optimal_sweep:
        print(f"   Headings evaluated: {len(optimal_sweep['headings'])}")
        print(f"   GDOP worst: {optimal_sweep['gdop_worst']:.2f}")
        print(f"   GDOP mean: {optimal_sweep['gdop_mean']:.2f}")
        print(f"   GDOP p95: {optimal_sweep['gdop_p95']:.2f}")
        print(f"   Worst heading: {optimal_sweep['worst_heading']:.1f}°")

        # Check that we have valid GDOP values (not all INF)
        finite_gdops = np.sum(np.array(optimal_sweep['gdops']) < GDOP_INF_SUBSTITUTE)
        sweep_ok = finite_gdops > 0
        print(f"   Valid GDOP measurements: {'PASS' if sweep_ok else 'FAIL'} ({finite_gdops}/{len(optimal_sweep['headings'])})")
    else:
        print(f"   No optimal sweep available")
        sweep_ok = False

    # 4. Comparison with naive and random
    print(f"\n4. Configuration comparison:")
    print(f"   Naive grid GDOP worst: {comparison['naive_gdop_worst']:.2f}")
    print(f"   Best random GDOP worst: {comparison['random_best_gdop_worst']:.2f}")
    if best_config:
        print(f"   Optimized GDOP worst: {comparison['optimal_gdop_worst']:.2f}")
        print(f"   Improvement vs naive: {comparison['improvement_vs_naive']:.2f}")

        # Expect improvement over naive grid (negative means better)
        improvement_ok = comparison['improvement_vs_naive'] > 0  # Positive improvement means lower GDOP
        print(f"   Improvement over naive: {'PASS' if improvement_ok else 'FAIL'}")
    else:
        improvement_ok = False
        print(f"   Optimized GDOP worst: N/A")
        print(f"   Improvement vs naive: N/A")

    # 5. Random layout evaluation
    print(f"\n5. Random layout evaluation:")
    print(f"   Layouts evaluated: {len(random_result['gdop_worsts'])}")
    print(f"   Best random GDOP worst: {random_result['best_gdop']:.2f}")
    print(f"   Mean random GDOP worst: {np.mean(random_result['gdop_worsts']):.2f}")
    print(f"   Std random GDOP worst: {np.std(random_result['gdop_worsts']):.2f}")

    random_ok = len(random_result['gdop_worsts']) > 0
    print(f"   Random evaluation: {'PASS' if random_ok else 'FAIL'}")

    # Overall validation
    overall_pass = search_ok and config_ok and sweep_ok and improvement_ok and random_ok

    print(f"\n{'=' * 60}")
    print(f"OVERALL GDOP OPTIMIZATION VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return overall_pass


if __name__ == "__main__":
    print("Testing Module 5: Array Configuration Optimization (GDOP)")

    # Run with verbose output
    result = run_module5(verbose=True)

    # Validate results
    success = validate_gdop_physics(result)

    sys.exit(0 if success else 1)