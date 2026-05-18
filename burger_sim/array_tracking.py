"""
Burger FSR Simulation -- Module 4: Array Tracking & TDOA Triangulation
========================================================================
Simulate a 2D array of FSR pairs, detect crossing events, and reconstruct
Burger's trajectory using TDOA-based triangulation.

See BURGER_SIM_PRD.txt §6.
"""

import numpy as np
from scipy.optimize import least_squares

from . import config as cfg
from .utils import freq_to_wavelength
from .link_budget import received_power, snr_db


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARRAY LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_array_layout(area_x: float = None,
                       area_y: float = None,
                       spacing: float = None,
                       pair_baseline: float = None) -> list:
    """
    Build a 2D array of FSR pairs with two orthogonal fences.

    Default: x-fence (baselines along y) and y-fence (baselines along x).

    Returns
    -------
    pairs : list of dicts, each with:
        'tx'        : (3,) Tx position [x, y, z]
        'rx'        : (3,) Rx position [x, y, z]
        'midpoint'  : (3,) baseline midpoint
        'baseline_dir' : (2,) unit direction of baseline (ground plane)
        'fence'     : 'x' or 'y'
    """
    if area_x is None:
        area_x = cfg.ARRAY_AREA_X
    if area_y is None:
        area_y = cfg.ARRAY_AREA_Y
    if spacing is None:
        spacing = cfg.ELEMENT_SPACING
    if pair_baseline is None:
        pair_baseline = cfg.PAIR_BASELINE

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


def build_custom_array(spacing_x: float,
                       spacing_y: float,
                       rotation_deg: float,
                       n_pairs: int,
                       layout_type: str = 'grid',
                       area_x: float = None,
                       area_y: float = None,
                       pair_baseline: float = None,
                       random_seed: int = None) -> list:
    """
    Build an array with custom spacing, rotation, and layout type.
    Used by Module 5 optimizer.

    Parameters
    ----------
    spacing_x, spacing_y : element spacing (m)
    rotation_deg : rotation of the grid (degrees)
    n_pairs      : total number of FSR pairs
    layout_type  : 'grid', 'hexagonal', or 'random'
    """
    if area_x is None:
        area_x = cfg.ARRAY_AREA_X
    if area_y is None:
        area_y = cfg.ARRAY_AREA_Y
    if pair_baseline is None:
        pair_baseline = cfg.PAIR_BASELINE

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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BURGER TRAJECTORY THROUGH ARRAY
# ─────────────────────────────────────────────────────────────────────────────

def array_burger_trajectory(entry: tuple = None,
                            heading_deg: float = None,
                            speed: float = None,
                            duration: float = None,
                            dt: float = None) -> dict:
    """
    Compute Burger's trajectory through the array area.

    Returns
    -------
    traj : dict with keys
        't'   : time array
        'x'   : ground x-position array
        'y'   : ground y-position array
        'z'   : altitude (scalar)
    """
    if entry is None:
        entry = cfg.BURGER_ENTRY
    if heading_deg is None:
        heading_deg = cfg.BURGER_HEADING
    if speed is None:
        speed = cfg.BURGER_SPEED
    if duration is None:
        duration = cfg.SIM_DURATION
    if dt is None:
        dt = cfg.DT

    heading_rad = np.radians(heading_deg)
    t = np.arange(0, duration, dt)

    x = entry[0] + speed * np.cos(heading_rad) * t
    y = entry[1] + speed * np.sin(heading_rad) * t
    z = entry[2] if len(entry) > 2 else cfg.BURGER_ALTITUDE

    return {"t": t, "x": x, "y": y, "z": z}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CROSSING EVENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _signed_distance_to_baseline(px, py, mid, bl_dir):
    """
    Signed perpendicular distance from point (px, py) to the baseline
    line passing through `mid` with direction `bl_dir`.
    """
    # Normal to baseline direction
    normal = np.array([-bl_dir[1], bl_dir[0]])
    dx = px - mid[0]
    dy = py - mid[1]
    return dx * normal[0] + dy * normal[1]


def detect_crossing_events(pairs: list,
                           traj: dict,
                           freq_hz: float = None,
                           snr_threshold_dB: float = None) -> list:
    """
    Detect baseline crossing events for all FSR pairs.

    A crossing occurs when Burger's ground track crosses the extended
    baseline of an FSR pair (sign change in perpendicular distance).

    Parameters
    ----------
    pairs          : list of pair dicts from build_array_layout()
    traj           : trajectory dict from array_burger_trajectory()
    freq_hz        : operating frequency (Hz)
    snr_threshold_dB : minimum SNR for valid detection (dB)

    Returns
    -------
    events : list of dicts, each with:
        'pair_idx'     : index into pairs list
        'time'         : crossing timestamp (s)
        'snr_dB'       : SNR at crossing (dB)
        'position'     : (x, y) ground position at crossing
        'midpoint'     : pair midpoint position
        'baseline_dir' : pair baseline direction
    """
    if freq_hz is None:
        freq_hz = cfg.OPERATING_FREQ
    if snr_threshold_dB is None:
        snr_threshold_dB = cfg.SNR_THRESHOLD_DB

    wavelength = freq_to_wavelength(freq_hz)
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    z = traj["z"]

    # Approximate sigma_fs for SNR estimate (optical limit)
    # This is a simplified estimate; Module 2 provides the actual value
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
                P_t=cfg.TX_POWER,
                G_t_dBi=cfg.TX_GAIN_DB,
                G_r_dBi=cfg.RX_GAIN_DB,
                wavelength=wavelength,
                sigma_fs=sigma_fs,
                R_tx=R_tx,
                R_rx=R_rx,
            )
            snr_val = snr_db(P_r)

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


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TDOA TRIANGULATION
# ─────────────────────────────────────────────────────────────────────────────

def tdoa_triangulate(events: list,
                     speed: float = None) -> dict:
    """
    Reconstruct Burger's trajectory from crossing events using TDOA.

    Method:
    1. For each pair of crossing events, the TDOA constrains the target
       to a locus of points (hyperbola-like, but using crossing-time TDOA).
    2. Use least-squares to find the best-fit position and heading.

    Parameters
    ----------
    events : list of crossing event dicts
    speed  : Burger speed (m/s) -- assumed known for track reconstruction

    Returns
    -------
    result : dict with keys
        'positions'   : (K, 2) estimated position at each crossing
        'heading_deg' : estimated heading (degrees)
        'speed_est'   : estimated speed (m/s)
        'track_fit'   : dict with line-fit parameters
    """
    if speed is None:
        speed = cfg.BURGER_SPEED

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

    # ── Method: Crossing-time based position estimation ────────────────
    # Each crossing event gives: the target was on baseline i at time t_i.
    # The baseline is a line through midpoint_i with direction bl_dir_i.
    #
    # If target moves at constant velocity (vx, vy):
    #   position at t_i = (x0 + vx*t_i,  y0 + vy*t_i)
    #   This point lies on baseline i:
    #   normal_i · (pos(t_i) - midpoint_i) = 0
    #
    # This gives a linear system: solve for (x0, y0, vx, vy).

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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MONTE CARLO ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_errors(events: list,
                       true_heading: float,
                       true_speed: float,
                       n_trials: int = None,
                       timing_noise_std: float = None,
                       verbose: bool = True) -> dict:
    """
    Run Monte Carlo analysis with Gaussian timing noise.

    Parameters
    ----------
    events           : list of crossing events (noise-free)
    true_heading     : ground truth heading (degrees)
    true_speed       : ground truth speed (m/s)
    n_trials         : number of noise realizations
    timing_noise_std : standard deviation of timing noise (s)

    Returns
    -------
    mc : dict with keys
        'heading_errors'  : (n_trials,) heading error (degrees)
        'speed_errors'    : (n_trials,) speed error (m/s)
        'position_errors' : (n_trials,) RMS position error (m)
        'heading_rms'     : float
        'speed_rms'       : float
        'position_rms'    : float
    """
    if n_trials is None:
        n_trials = cfg.N_TRIALS
    if timing_noise_std is None:
        timing_noise_std = cfg.TIMING_NOISE_STD

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


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL MODULE 4 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_module4(freq_hz: float = None,
                verbose: bool = True) -> dict:
    """
    Execute Module 4 end-to-end.

    Returns
    -------
    result : dict with keys
        'pairs'      : FSR pair layout
        'traj'       : Burger trajectory through array
        'events'     : crossing events list
        'triangulation' : TDOA result dict
        'monte_carlo'   : MC error analysis dict
    """
    if freq_hz is None:
        freq_hz = cfg.OPERATING_FREQ

    if verbose:
        print("=" * 60)
        print("MODULE 4 -- ARRAY TRACKING & TDOA TRIANGULATION")
        print("=" * 60)

    # Build array
    pairs = build_array_layout()
    if verbose:
        print(f"  Array pairs  : {len(pairs)}")
        print(f"  Area         : {cfg.ARRAY_AREA_X:.0f} × {cfg.ARRAY_AREA_Y:.0f} m")
        print(f"  Spacing      : {cfg.ELEMENT_SPACING:.0f} m")
        print()

    # Compute trajectory
    traj = array_burger_trajectory()
    true_heading = cfg.BURGER_HEADING
    true_speed = cfg.BURGER_SPEED

    if verbose:
        print(f"  Entry point  : ({cfg.BURGER_ENTRY[0]:.0f}, {cfg.BURGER_ENTRY[1]:.0f}) m")
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
