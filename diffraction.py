"""
Burger FSR Simulation -- Module 2: Diffraction Pattern & Frequency Sweep
=========================================================================
Compute the far-field diffraction pattern of Burger's planform via FFT.
Sweep over frequencies to find the optimal detection band.

See BURGER_SIM_PRD.txt §4.
"""

import warnings
import numpy as np
from matplotlib.path import Path

from . import config as cfg
from .utils import freq_to_wavelength, fresnel_number
from .geometry import build_planform_vertices, depth_profile, phase_screen


# ─────────────────────────────────────────────────────────────────────────────
# 1.  APERTURE FIELD CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_aperture_field(vertices: np.ndarray,
                         freq_hz: float,
                         grid_res: int = None,
                         use_depth: bool = None) -> np.ndarray:
    """
    Construct the 2D aperture transmission function T(x, y).

    T = 1 inside planform, 0 outside.
    If depth screen is enabled: T = exp(i * phi(x,y)) inside.

    Parameters
    ----------
    vertices : (N, 2) planform polygon
    freq_hz  : frequency in Hz
    grid_res : spatial grid resolution (default from config)
    use_depth : whether to apply depth phase screen (default from config)

    Returns
    -------
    T       : complex ndarray (grid_res, grid_res) -- aperture field
    x_grid  : 1D x coordinates
    y_grid  : 1D y coordinates
    """
    if grid_res is None:
        grid_res = cfg.GRID_RES
    if use_depth is None:
        use_depth = cfg.USE_DEPTH_SCREEN

    # Bounding box with small margin
    x_min, x_max = vertices[:, 0].min() - 0.05, vertices[:, 0].max() + 0.05
    y_min, y_max = vertices[:, 1].min() - 0.05, vertices[:, 1].max() + 0.05

    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(x_grid, y_grid, indexing='ij')

    # Interior test
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    path = Path(vertices)
    mask = path.contains_points(pts).reshape(GX.shape)

    # Build transmission function
    T = np.zeros(GX.shape, dtype=complex)

    if use_depth:
        wavelength = freq_to_wavelength(freq_hz)
        k = 2.0 * np.pi / wavelength
        # Convert normalized coords to physical for phase calculation
        phi = phase_screen(GX, GY, k)
        T[mask] = np.exp(1j * phi[mask])
    else:
        T[mask] = 1.0 + 0j

    return T, x_grid, y_grid


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FAR-FIELD DIFFRACTION (FFT)
# ─────────────────────────────────────────────────────────────────────────────

def compute_diffraction_pattern(T: np.ndarray,
                                dx: float,
                                dy: float,
                                fft_pad: int = None) -> tuple:
    """
    Compute the 2D Fourier transform of the aperture field (Fraunhofer regime).

    U(kx, ky) = FFT2{ T(x, y) }

    Parameters
    ----------
    T       : (Nx, Ny) complex aperture field
    dx, dy  : spatial sampling intervals (normalized units)
    fft_pad : zero-padded FFT size (default from config)

    Returns
    -------
    U       : complex (fft_pad, fft_pad) -- far-field amplitude
    kx      : 1D spatial frequency array
    ky      : 1D spatial frequency array
    """
    if fft_pad is None:
        fft_pad = cfg.FFT_PAD

    # 2D FFT with zero-padding, centered
    U = np.fft.fftshift(np.fft.fft2(T, s=(fft_pad, fft_pad)))

    # Spatial frequency axes
    kx = np.fft.fftshift(np.fft.fftfreq(fft_pad, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(fft_pad, d=dy))

    return U, kx, ky


def forward_scatter_rcs(U: np.ndarray,
                        dx: float,
                        dy: float,
                        wavelength: float,
                        half_span: float = None) -> float:
    """
    Forward scatter RCS from the FFT result.

    sigma_fs = (4pi/lam^2) * |U(0,0)|^2 * (dx*dy)^2

    All dimensions must be in the SAME units.  The aperture field T is in
    normalized coordinates, so we convert to physical before computing RCS.

    Parameters
    ----------
    wavelength  : in metres
    half_span   : normalization constant (default from config)
    """
    if half_span is None:
        half_span = cfg.HALF_SPAN

    # Physical sampling intervals
    dx_phys = dx * half_span
    dy_phys = dy * half_span

    # Central value of FFT (DC component)
    center = U.shape[0] // 2, U.shape[1] // 2
    U00 = U[center[0], center[1]]

    sigma = (4.0 * np.pi / wavelength ** 2) * np.abs(U00) ** 2 * (dx_phys * dy_phys) ** 2
    return sigma


def lobe_width(U: np.ndarray,
               kx: np.ndarray,
               wavelength: float,
               half_span: float = None) -> float:
    """
    Estimate the forward scatter main-lobe angular half-width.

    Takes a 1D slice through the central row of |U|^2 and finds the
    first null (or -3 dB point).

    Returns angular width in radians.
    """
    if half_span is None:
        half_span = cfg.HALF_SPAN

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
    # sin(theta) = kx_spatial * lam / (2pi) ... but our kx is in cycles/normalized-unit
    # Physical spatial frequency: kx_phys = kx / half_span
    # sin(theta) ~ kx_phys * lam  (for fftfreq convention without 2pi)
    dk_phys = dk / half_span
    theta_width = dk_phys * wavelength  # approximate, small-angle

    return max(theta_width, 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FREQUENCY SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def frequency_sweep(vertices: np.ndarray = None,
                    f_min: float = None,
                    f_max: float = None,
                    n_points: int = None,
                    verbose: bool = True) -> dict:
    """
    Sweep over frequencies and compute sigma_fs, lobe width, detectability.

    Returns
    -------
    results : dict with arrays
        'freqs'          : Hz
        'wavelengths'    : m
        'sigma_fs'       : m^2
        'sigma_fs_dBsm'  : dBsm
        'lobe_widths'    : radians
        'detectability'  : m^2/rad
        'fresnel_numbers': dimensionless
        'electrical_sizes': half-span in wavelengths
    """
    if vertices is None:
        vertices = build_planform_vertices()
    if f_min is None:
        f_min = cfg.F_MIN
    if f_max is None:
        f_max = cfg.F_MAX
    if n_points is None:
        n_points = cfg.N_FREQ_POINTS

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    sigma_fs_arr = np.zeros(n_points)
    lobe_arr = np.zeros(n_points)
    fresnel_arr = np.zeros(n_points)

    if verbose:
        print("=" * 60)
        print("MODULE 2 -- DIFFRACTION & FREQUENCY SWEEP")
        print("=" * 60)
        print(f"  Frequency range : {f_min/1e6:.1f} MHz - {f_max/1e9:.1f} GHz")
        print(f"  Points          : {n_points}")
        print()

    for i, f in enumerate(freqs):
        lam = freq_to_wavelength(f)

        # Fresnel number check
        N_F = fresnel_number(cfg.FULL_SPAN, lam, cfg.BURGER_ALTITUDE)
        fresnel_arr[i] = N_F
        if N_F > 0.5:
            warnings.warn(
                f"Fresnel number N_F={N_F:.2f} > 0.5 at f={f/1e6:.1f} MHz: "
                f"Fraunhofer approximation may be inaccurate."
            )

        # Build aperture field (use depth screen if configured)
        T, x_grid, y_grid = build_aperture_field(vertices, f)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]

        # Compute FFT
        U, kx, ky = compute_diffraction_pattern(T, dx, dy)

        # RCS
        sigma_fs_arr[i] = forward_scatter_rcs(U, dx, dy, lam)

        # Lobe width
        lobe_arr[i] = lobe_width(U, kx, lam)

        if verbose and (i % 10 == 0 or i == n_points - 1):
            print(f"  [{i+1:3d}/{n_points}]  f={f/1e6:10.2f} MHz  "
                  f"lam={lam:8.3f} m  sigma_fs={sigma_fs_arr[i]:.2e} m^2  "
                  f"lobe={np.degrees(lobe_arr[i]):.3f}°  N_F={N_F:.3f}")

    # Derived quantities
    wavelengths = cfg.C_LIGHT / freqs
    sigma_dBsm = 10.0 * np.log10(np.maximum(sigma_fs_arr, 1e-300))
    detectability = sigma_fs_arr / np.maximum(lobe_arr, 1e-10)
    electrical_sizes = cfg.HALF_SPAN / wavelengths

    # Find optimal frequency
    # Use detectability weighted by Fresnel validity
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


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REPRESENTATIVE DIFFRACTION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def compute_representative_patterns(vertices: np.ndarray = None,
                                    freqs_hz: list = None) -> list:
    """
    Compute diffraction patterns at representative frequencies.

    Default: one VHF (100 MHz), one UHF (500 MHz), one L-band (1.5 GHz).

    Returns list of dicts with keys: freq, wavelength, U, kx, ky, sigma_fs
    """
    if vertices is None:
        vertices = build_planform_vertices()
    if freqs_hz is None:
        freqs_hz = [100e6, 500e6, 1.5e9]

    patterns = []
    for f in freqs_hz:
        lam = freq_to_wavelength(f)
        T, x_grid, y_grid = build_aperture_field(vertices, f)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        U, kx, ky = compute_diffraction_pattern(T, dx, dy)
        sigma = forward_scatter_rcs(U, dx, dy, lam)

        patterns.append({
            "freq": f,
            "wavelength": lam,
            "U": U,
            "kx": kx,
            "ky": ky,
            "sigma_fs": sigma,
        })
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL MODULE 2 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_module2(vertices: np.ndarray = None, verbose: bool = True) -> dict:
    """
    Execute Module 2 end-to-end.

    Returns
    -------
    result : dict with keys
        'sweep'     : frequency sweep results dict
        'patterns'  : list of representative pattern dicts
    """
    if vertices is None:
        from .geometry import build_planform_vertices
        vertices = build_planform_vertices()

    sweep = frequency_sweep(vertices, verbose=verbose)
    patterns = compute_representative_patterns(vertices)

    return {
        "sweep": sweep,
        "patterns": patterns,
    }
