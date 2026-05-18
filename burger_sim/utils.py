"""
Burger FSR Simulation -- Shared Utilities
=========================================
Coordinate transforms, polygon helpers, unit conversions.
"""

import numpy as np
from matplotlib.path import Path


# ── Unit conversions ────────────────────────────────────────────────────────

def db_to_linear(db: float) -> float:
    """Convert dB (power) to linear scale."""
    return 10.0 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """Convert linear power to dB."""
    return 10.0 * np.log10(np.maximum(linear, 1e-300))


def dbi_to_linear(dbi: float) -> float:
    """Convert antenna gain in dBi to linear."""
    return db_to_linear(dbi)


def freq_to_wavelength(freq_hz: float, c: float = 3e8) -> float:
    """Convert frequency (Hz) to wavelength (m)."""
    return c / freq_hz


# ── Polygon utilities ──────────────────────────────────────────────────────

def signed_polygon_area(vertices: np.ndarray) -> float:
    """
    Signed area of a 2D polygon (shoelace formula).
    Positive = counter-clockwise, negative = clockwise.
    vertices: (N, 2) array of (x, y) pairs.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def polygon_area(vertices: np.ndarray) -> float:
    """Unsigned area of a 2D polygon."""
    return abs(signed_polygon_area(vertices))


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Ensure polygon vertices are in counter-clockwise order."""
    if signed_polygon_area(vertices) < 0:
        return vertices[::-1].copy()
    return vertices.copy()


def point_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Test which points lie inside a polygon.
    points:   (M, 2) array of (x, y) query points
    vertices: (N, 2) array of polygon vertices
    Returns:  (M,) boolean array
    """
    path = Path(vertices)
    return path.contains_points(points)


def triangle_area(v0, v1, v2):
    """Area of a triangle from three 2D vertices."""
    return 0.5 * abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) -
        (v2[0] - v0[0]) * (v1[1] - v0[1])
    )


# ── Coordinate helpers ─────────────────────────────────────────────────────

def normalize_coords(physical_coords: np.ndarray, half_span: float) -> np.ndarray:
    """Convert physical (m) coordinates to normalized (by half-span)."""
    return physical_coords / half_span


def physical_coords(normalized: np.ndarray, half_span: float) -> np.ndarray:
    """Convert normalized coordinates back to physical (m)."""
    return normalized * half_span


# ── Fresnel validation ─────────────────────────────────────────────────────

def fresnel_number(aperture_size: float, wavelength: float, distance: float) -> float:
    """
    Compute Fresnel number N_F = a^2 / (lam·z).
    Fraunhofer approximation valid when N_F << 1.
    """
    return aperture_size ** 2 / (wavelength * distance)
