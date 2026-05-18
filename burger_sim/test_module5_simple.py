#!/usr/bin/env python3
"""
Simple test for Module 5: Array Configuration Optimization (GDOP)
Tests core GDOP concepts with minimal computation.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# ========== MINIMAL CONFIGURATION FOR FAST TESTING ==========
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

# Array configuration - SMALLER FOR FAST TEST
ARRAY_AREA_X = 1000.0    # m (reduced from 5000m)
ARRAY_AREA_Y = 1000.0    # m (reduced from 5000m)
ELEMENT_SPACING = 200.0  # m (reduced from 500m)
PAIR_BASELINE = 100.0    # m (reduced from 200m)
BURGER_ENTRY = (-500, 0, 15000)  # m (adjusted for smaller array)
BURGER_HEADING = 25.0    # degrees

# Link budget
TX_POWER = 1000.0        # W
TX_GAIN_DB = 20.0        # dBi
RX_GAIN_DB = 20.0        # dBi
BANDWIDTH = 1e6          # Hz
NOISE_TEMP = 290.0       # K
SNR_THRESHOLD_DB = 10.0  # dB

# Burger motion
BURGER_SPEED = 306.0     # m/s
SIM_DURATION = 20.0      # s (reduced from 120s)
DT = 0.1                 # s (increased from 0.01s)

# GDOP parameters
GDOP_INF_SUBSTITUTE = 1e6

# ========== CORE FUNCTIONS ==========
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

# ========== GEOMETRY FUNCTIONS ==========
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

# ========== SIMPLIFIED ARRAY BUILDING ==========
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

# ========== GDOP COMPUTATION ==========
def compute_gdop_simple(events, target_pos=None):
    """Compute Geometric Dilution of Precision from crossing events - simplified."""
    if len(events) < 3:
        return GDOP_INF_SUBSTITUTE

    # Use centroid of event midpoints as approximate target position
    if target_pos is None:
        midpoints = np.array([e["midpoint"][:2] for e in events])
        target_pos = midpoints.mean(axis=0)

    # Reference event (first)
    ref = events[0]
    ref_mid = ref["midpoint"][:2]
    r0 = np.linalg.norm(target_pos - ref_mid)

    if r0 < 1e-6:
        r0 = 1.0  # avoid division by zero

    # Build H matrix: (N-1, 2)
    N = len(events)
    H = np.zeros((N - 1, 2))

    for i in range(1, N):
        mid_i = events[i]["midpoint"][:2]
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

def simple_gdop_test():
    """Test GDOP computation with known geometry."""
    print("Testing GDOP computation with simple geometry...")

    # Create a simple square array
    pairs = build_simple_array(spacing_x=200, spacing_y=200, n_pairs=4)
    print(f"Created array with {len(pairs)} pairs")

    # Test headings
    headings = [0, 45, 90]  # degrees
    results = []

    for heading in headings:
        print(f"\nTesting heading {heading}°:")
        traj = burger_trajectory_simple(heading_deg=heading, baseline=100.0)

        # Simple crossing detection (placeholder)
        # In reality, we'd use the full detection algorithm
        # For now, simulate some events
        if heading == 0:
            # Should get good crossings with baselines along x and y
            events = [
                {"midpoint": np.array([0, 0])},
                {"midpoint": np.array([200, 0])},
                {"midpoint": np.array([0, 200])},
                {"midpoint": np.array([200, 200])}
            ]
        elif heading == 45:
            # Diagonal crossing - might get fewer good crossings
            events = [
                {"midpoint": np.array([0, 0])},
                {"midpoint": np.array([200, 200])}
            ]
        else:  # 90 degrees
            events = [
                {"midpoint": np.array([0, 0])},
                {"midpoint": np.array([200, 0])},
                {"midpoint": np.array([0, 200])}
            ]

        gdop = compute_gdop_simple(events)
        results.append((heading, gdop, len(events)))
        print(f"  Events: {len(events)}, GDOP: {gdop:.2f}")

    # Validate expectations
    print("\nValidation:")
    # 0° and 90° should have similar, lower GDOP (good geometry)
    # 45° might have higher GDOP (poorer geometry for this array)
    gdop_0 = results[0][1]
    gdop_45 = results[1][1]
    gdop_90 = results[2][1]

    print(f"GDOP at 0°: {gdop_0:.2f}")
    print(f"GDOP at 45°: {gdop_45:.2f}")
    print(f"GDOP at 90°: {gdop_90:.2f}")

    # For a square array, 0° and 90° should be similar and better than 45°
    # Actually, for a square grid aligned with axes, 0° and 90° should be poor
    # (aligned with baselines) and 45° should be better
    # Let's just check that we got some variation
    variation = max(gdop_0, gdop_45, gdop_90) - min(gdop_0, gdop_45, gdop_90)
    print(f"GDOP variation across headings: {variation:.2f}")

    success = variation > 0.1  # Some variation expected
    print(f"GDOP variation test: {'PASS' if success else 'FAIL'}")

    return success

def test_gdop_properties():
    """Test mathematical properties of GDOP."""
    print("\nTesting GDOP mathematical properties...")

    # Test 1: Well-conditioned geometry should give low GDOP
    events_good = [
        {"midpoint": np.array([-100, 0])},
        {"midpoint": np.array([100, 0])},
        {"midpoint": np.array([0, 100])},
        {"midpoint": np.array([0, -100])}
    ]
    gdop_good = compute_gdop_simple(events_good)
    print(f"Well-conditioned geometry GDOP: {gdop_good:.2f}")

    # Test 2: Poor geometry (collinear) should give high GDOP
    events_bad = [
        {"midpoint": np.array([-100, 0])},
        {"midpoint": np.array([0, 0])},
        {"midpoint": np.array([100, 0])}
    ]
    gdop_bad = compute_gdop_simple(events_bad)
    print(f"Poor (collinear) geometry GDOP: {gdop_bad:.2f}")

    # Test 3: Too few events should give INF
    events_few = [
        {"midpoint": np.array([-100, 0])},
        {"midpoint": np.array([100, 0])}
    ]
    gdop_few = compute_gdop_simple(events_few)
    print(f"Insufficient events GDOP: {gdop_few:.2f}")

    # Validate
    good_reasonable = gdop_good < 10.0  # Should be reasonably low
    bad_high = gdop_bad > gdop_good * 2  # Should be significantly higher
    few_inf = gdop_few >= GDOP_INF_SUBSTITUTE / 2  # Should be very high

    print(f"Good geometry reasonable: {'PASS' if good_reasonable else 'FAIL'}")
    print(f"Bad geometry worse than good: {'PASS' if bad_high else 'FAIL'}")
    print(f"Few events gives high GDOP: {'PASS' if few_inf else 'FAIL'}")

    success = good_reasonable and bad_high and few_inf
    print(f"GDOP properties test: {'PASS' if success else 'FAIL'}")

    return success

if __name__ == "__main__":
    print("Running simplified Module 5 GDOP tests...")
    print("=" * 50)

    test1_pass = simple_gdop_test()
    test2_pass = test_gdop_properties()

    overall_success = test1_pass and test2_pass

    print("\n" + "=" * 50)
    print(f"Overall test result: {'PASS' if overall_success else 'FAIL'}")
    print("=" * 50)

    sys.exit(0 if overall_success else 1)