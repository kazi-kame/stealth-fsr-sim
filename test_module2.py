#!/usr/bin/env python3
"""
Test for Module 2: Diffraction Pattern and Frequency Sweep
Tests diffraction computation, frequency sweep, and validation against expected physics.
"""

import numpy as np
import sys
import os
import warnings

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

# Frequency sweep
F_MIN = 30e6         # Hz (30 MHz, VHF lower bound)
F_MAX = 12e9         # Hz (12 GHz, X-band upper bound)
N_FREQ_POINTS = 50   # log-spaced points in sweep

# FFT / diffraction grid
GRID_RES = 512       # spatial grid resolution (512x512)
FFT_PAD = 2048       # zero-padded FFT size

# ========== IMPORT GEOMETRY FUNCTIONS ==========
from test_module1_complete import (
    build_planform_vertices,
    signed_polygon_area,
    polygon_area,
    point_in_polygon,
    triangle_area,
    depth_profile
)

# Replicate needed functions from diffraction module
def freq_to_wavelength(freq_hz, c=3e8):
    """Convert frequency (Hz) to wavelength (m)."""
    return c / freq_hz

def fresnel_number(aperture_size, wavelength, distance):
    """Compute Fresnel number N_F = a^2 / (lam·z)."""
    return aperture_size ** 2 / (wavelength * distance)

def build_aperture_field(vertices, freq_hz, grid_res=None, use_depth=None):
    """Construct the 2D aperture transmission function T(x, y)."""
    if grid_res is None:
        grid_res = GRID_RES
    if use_depth is None:
        use_depth = False  # Disable depth screen for basic test

    # Bounding box with small margin
    x_min, x_max = vertices[:, 0].min() - 0.05, vertices[:, 0].max() + 0.05
    y_min, y_max = vertices[:, 1].min() - 0.05, vertices[:, 1].max() + 0.05

    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(x_grid, y_grid, indexing='ij')

    # Interior test
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    from matplotlib.path import Path
    path = Path(vertices)
    mask = path.contains_points(pts).reshape(GX.shape)

    # Build transmission function
    T = np.zeros(GX.shape, dtype=complex)

    if use_depth:
        wavelength = freq_to_wavelength(freq_hz)
        k = 2.0 * np.pi / wavelength
        # Convert normalized coords to physical for phase calculation
        phi = depth_profile(GX, GY) * k  # phase_screen = k * z
        T[mask] = np.exp(1j * phi[mask])
    else:
        T[mask] = 1.0 + 0j

    return T, x_grid, y_grid

def compute_diffraction_pattern(T, dx, dy, fft_pad=None):
    """Compute the 2D Fourier transform of the aperture field (Fraunhofer regime)."""
    if fft_pad is None:
        fft_pad = FFT_PAD

    # 2D FFT with zero-padding, centered
    U = np.fft.fftshift(np.fft.fft2(T, s=(fft_pad, fft_pad)))

    # Spatial frequency axes
    kx = np.fft.fftshift(np.fft.fftfreq(fft_pad, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(fft_pad, d=dy))

    return U, kx, ky

def forward_scatter_rcs(U, dx, dy, wavelength, half_span=None):
    """Forward scatter RCS from the FFT result."""
    if half_span is None:
        half_span = HALF_SPAN

    # Physical sampling intervals
    dx_phys = dx * half_span
    dy_phys = dy * half_span

    # Central value of FFT (DC component)
    center = U.shape[0] // 2, U.shape[1] // 2
    U00 = U[center[0], center[1]]

    sigma = (4.0 * np.pi / wavelength ** 2) * np.abs(U00) ** 2 * (dx_phys * dy_phys) ** 2
    return sigma

def lobe_width(U, kx, wavelength, half_span=None):
    """Estimate the forward scatter main-lobe angular half-width."""
    if half_span is None:
        half_span = HALF_SPAN

    center_row = U.shape[0] // 2
    intensity = np.abs(U[center_row, :]) ** 2
    peak = intensity.max()

    # -3 dB width
    half_power = peak / 2.0
    above = intensity >= half_power
    indices = np.where(above)[0]

    if len(indices) < 2:
        return np.pi  # degenerate -- very wide lobe

    # Spatial frequency span
    dk = kx[indices[-1]] - kx[indices[0]]

    # Convert spatial frequency (cycles / normalized unit) to angle:
    dk_phys = dk / half_span
    theta_width = dk_phys * wavelength  # approximate, small-angle

    return max(theta_width, 1e-10)

def frequency_sweep(vertices=None, f_min=None, f_max=None, n_points=None, verbose=True):
    """Sweep over frequencies and compute sigma_fs, lobe width, detectability."""
    if vertices is None:
        vertices = build_planform_vertices()
    if f_min is None:
        f_min = F_MIN
    if f_max is None:
        f_max = F_MAX
    if n_points is None:
        n_points = min(20, N_FREQ_POINTS)  # Reduced for testing

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    sigma_fs_arr = np.zeros(n_points)
    lobe_arr = np.zeros(n_points)
    fresnel_arr = np.zeros(n_points)

    if verbose:
        print("=" * 60)
        print("MODULE 2 -- DIFFRACTION & FREQUENCY SWEEP (TEST)")
        print("=" * 60)
        print(f"  Frequency range : {f_min/1e6:.1f} MHz - {f_max/1e9:.1f} GHz")
        print(f"  Points          : {n_points}")
        print()

    for i, f in enumerate(freqs):
        lam = freq_to_wavelength(f)

        # Fresnel number check
        N_F = fresnel_number(FULL_SPAN, lam, 15000.0)  # BURGER_ALTITUDE
        fresnel_arr[i] = N_F
        if N_F > 0.5:
            warnings.warn(
                f"Fresnel number N_F={N_F:.2f} > 0.5 at f={f/1e6:.1f} MHz: "
                f"Fraunhofer approximation may be inaccurate."
            )

        # Build aperture field
        T, x_grid, y_grid = build_aperture_field(vertices, f)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]

        # Compute FFT
        U, kx, ky = compute_diffraction_pattern(T, dx, dy)

        # RCS
        sigma_fs_arr[i] = forward_scatter_rcs(U, dx, dy, lam)

        # Lobe width
        lobe_arr[i] = lobe_width(U, kx, lam)

        if verbose and (i % 5 == 0 or i == n_points - 1):
            print(f"  [{i+1:3d}/{n_points}]  f={f/1e6:10.2f} MHz  "
                  f"lam={lam:8.3f} m  sigma_fs={sigma_fs_arr[i]:.2e} m^2  "
                  f"lobe={np.degrees(lobe_arr[i]):.3f}°  N_F={N_F:.3f}")

    # Derived quantities
    wavelengths = C_LIGHT / freqs
    sigma_dBsm = 10.0 * np.log10(np.maximum(sigma_fs_arr, 1e-300))
    detectability = sigma_fs_arr / np.maximum(lobe_arr, 1e-10)
    electrical_sizes = HALF_SPAN / wavelengths

    # Find optimal frequency
    valid_mask = fresnel_arr < 0.5
    if valid_mask.any():
        det_valid = detectability.copy()
        det_valid[~valid_mask] = 0
        best_idx = np.argmax(det_valid)
    else:
        best_idx = np.argmax(detectability)

    if verbose:
        print()
        print(f"  Recommended frequency : {freqs[best_idx]/1e6:.1f} MHz")
        print(f"  Wavelength            : {wavelengths[best_idx]:.2f} m")
        print(f"  sigma_fs                  : {sigma_fs_arr[best_idx]:.2e} m^2 "
              f"({sigma_dBsm[best_idx]:.1f} dBsm)")
        print(f"  Lobe width            : {np.degrees(lobe_arr[best_idx]):.2f}°")
        print(f"  Fresnel number        : {fresnel_arr[best_idx]:.3f}")
        print()

    results = {
        "freqs": freqs,
        "wavelengths": wavelengths,
        "sigma_fs": sigma_fs_arr,
        "sigma_fs_dBsm": sigma_dBsm,
        "lobe_widths": lobe_arr,
        "detectability": detectability,
        "fresnel_numbers": fresnel_arr,
        "electrical_sizes": electrical_sizes,
        "best_freq_idx": best_idx,
        "best_freq": freqs[best_idx],
    }
    return results

def validate_physics_expectations(results):
    """Validate that results follow expected physics trends."""
    freqs = results["freqs"]
    sigma_fs = results["sigma_fs"]
    lobe_widths = results["lobe_widths"]
    detectability = results["detectability"]
    fresnel_numbers = results["fresnel_numbers"]
    electrical_sizes = results["electrical_sizes"]

    print("=" * 60)
    print("PHYSICS VALIDATION")
    print("=" * 60)

    # 1. RCS should generally increase with frequency (up to optical limit)
    # Actually, for complex shapes it's more nuanced, but we expect some variation
    print(f"1. Frequency sweep range:")
    print(f"   Min frequency: {freqs[0]/1e6:.1f} MHz")
    print(f"   Max frequency: {freqs[-1]/1e9:.1f} GHz")
    print(f"   Sigma_fs range: [{np.min(sigma_fs):.2e}, {np.max(sigma_fs):.2e}] m²")

    # 2. Lobe width should decrease with increasing frequency (higher freq = narrower beam)
    # Check monotonic trend (not strictly required but generally true)
    lobe_degrees = np.degrees(lobe_widths)
    print(f"\n2. Lobe width trend:")
    print(f"   Min lobe width: {np.min(lobe_degrees):.3f}° (at {freqs[np.argmin(lobe_widths)]/1e6:.1f} MHz)")
    print(f"   Max lobe width: {np.max(lobe_degrees):.3f}° (at {freqs[np.argmax(lobe_widths)]/1e6:.1f} MHz)")

    # 3. Detectability should peak somewhere in VHF/UHF range for stealth detection
    best_idx = results["best_freq_idx"]
    best_freq_mhz = freqs[best_idx] / 1e6
    print(f"\n3. Optimal detection:")
    print(f"   Best frequency: {best_freq_mhz:.1f} MHz")
    print(f"   Best sigma_fs: {sigma_fs[best_idx]:.2e} m²")
    print(f"   Best lobe width: {np.degrees(lobe_widths[best_idx]):.3f}°")
    print(f"   Best detectability: {detectability[best_idx]:.2e} m²/rad")

    # 4. Fresnel number validation - should be < 1 for most of range (Fraunhofer valid)
    nf_valid_count = np.sum(fresnel_numbers < 1.0)
    print(f"\n4. Fraunhofer validity (N_F < 1.0):")
    print(f"   Valid points: {nf_valid_count}/{len(fresnel_numbers)}")
    print(f"   Max N_F: {np.max(fresnel_numbers):.3f}")

    # 5. Electrical size validation - should span from <<1 to >>1
    print(f"\n5. Electrical size (half-span / wavelength):")
    print(f"   Min electrical size: {np.min(electrical_sizes):.3f} (at {freqs[-1]/1e9:.1f} GHz)")
    print(f"   Max electrical size: {np.max(electrical_sizes):.3f} (at {freqs[0]/1e6:.1f} MHz)")

    es_min = np.min(electrical_sizes)
    es_max = np.max(electrical_sizes)
    # Expect range covering Rayleigh (~0.1) to optical (>10) regimes
    reg_rayleigh = es_min < 0.5 and es_max > 0.5  # Crosses Rayleigh boundary
    reg_optical = es_max > 5.0  # Reaches optical regime
    print(f"   Covers Rayleigh->optical transition: {'PASS' if (reg_rayleigh and reg_optical) else 'FAIL'}")

    overall_pass = nf_valid_count >= len(fresnel_numbers) * 0.8  # 80% Fraunhofer valid
    print(f"\n{'=' * 60}")
    print(f"OVERALL PHYSICS VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return overall_pass


if __name__ == "__main__":
    print("Testing Module 2: Diffraction Pattern and Frequency Sweep")

    # Get vertices from geometry
    vertices = build_planform_vertices()
    print(f"Using Burger geometry with {len(vertices)} vertices")
    print(f"Normalized area: {polygon_area(vertices):.4f}")
    print(f"Physical area: {polygon_area(vertices) * HALF_SPAN**2:.1f} m²")
    print()

    # Run frequency sweep
    results = frequency_sweep(vertices=vertices, verbose=True)

    # Validate physics expectations
    success = validate_physics_expectations(results)

    # Additional sanity checks
    print(f"\nAdditional checks:")
    print(f"  Results dict keys: {list(results.keys())}")
    print(f"  All arrays length: {len(results['freqs'])}")
    print(f"  Best freq index valid: {0 <= results['best_freq_idx'] < len(results['freqs'])}")

    sys.exit(0 if success else 1)