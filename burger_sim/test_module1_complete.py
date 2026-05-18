#!/usr/bin/env python3
"""
Complete test for Module 1: Burger Geometry Implementation
Tests geometry, triangulation, and validation against PRD requirements
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# ========== CONFIGURATION (extracted from config.md) ==========
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
USE_DEPTH_SCREEN = False # Toggle depth phase screen

# ========== GEOMETRY FUNCTIONS ==========

def _build_right_side_vertices():
    """
    Derive right-side (y > 0) vertices from public B-2 specs.
    """
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

def triangulate_planform(vertices):
    """Triangulate the interior of the planform polygon."""
    tri = Delaunay(vertices)
    simplices = tri.simplices

    centroids = vertices[simplices].mean(axis=1)
    inside = point_in_polygon(centroids, vertices)
    interior_simplices = simplices[inside]

    area_sum = 0.0
    for s in interior_simplices:
        area_sum += triangle_area(vertices[s[0]], vertices[s[1]], vertices[s[2]])

    return interior_simplices, area_sum

def depth_profile(x, y):
    """Smooth depth profile z(x, y) over the planform."""
    sigma_x, sigma_y = 0.2, 0.3
    cx, cy = 0.4, 0.0
    z = MAX_DEPTH_NORM * np.exp(
        -((x - cx) ** 2 / (2 * sigma_x ** 2) +
          (y - cy) ** 2 / (2 * sigma_y ** 2))
    )
    return z

def phase_screen(x, y, wavenumber):
    """Phase contribution from the depth profile for FSR."""
    return wavenumber * depth_profile(x, y)

def run_module1(save=False, verbose=True):
    """Execute Module 1 end-to-end."""
    vertices = build_planform_vertices()
    poly_area = polygon_area(vertices)

    if verbose:
        print("=" * 60)
        print("MODULE 1 -- BURGER GEOMETRY")
        print("=" * 60)
        print(f"  Vertices       : {len(vertices)}")
        print(f"  Polygon area   : {poly_area:.4f}  (normalized)")
        phys = poly_area * HALF_SPAN ** 2
        print(f"  Physical area  : {phys:.1f} m^2")
        print()
        print("  Vertex table (normalized):")
        for i, (vx, vy) in enumerate(vertices):
            print(f"    V{i:2d}  ({vx:+.4f}, {vy:+.4f})")
        print()

    triangles, tri_area = triangulate_planform(vertices)

    if verbose:
        print(f"  Triangles      : {len(triangles)}")
        print(f"  Triangle area  : {tri_area:.4f}")
        print(f"  Area match     : {abs(tri_area - poly_area) / poly_area * 100:.2f}% error")
        print()

    depth_data = None
    if USE_DEPTH_SCREEN:
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        gx = np.linspace(x_min, x_max, 200)
        gy = np.linspace(y_min, y_max, 200)
        GX, GY = np.meshgrid(gx, gy)
        GZ = depth_profile(GX, GY)

        pts = np.column_stack([GX.ravel(), GY.ravel()])
        mask = point_in_polygon(pts, vertices).reshape(GX.shape)
        GZ[~mask] = np.nan

        depth_data = {"x": gx, "y": gy, "z": GZ}

        if verbose:
            print(f"  Depth screen   : enabled")
            print(f"  Peak depth     : {np.nanmax(GZ):.4f}  (expected ~{MAX_DEPTH_NORM:.4f})")
            print()

    if save:
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "burger_geometry.npy")
        np.save(path, vertices)
        if verbose:
            print(f"  Saved vertices -> {path}")

    result = {
        "vertices":   vertices,
        "triangles":  triangles,
        "poly_area":  poly_area,
        "tri_area":   tri_area,
        "depth_data": depth_data,
    }
    return result

def validate_geometry_invariants(result):
    """Validate geometric invariants from PRD §3"""
    vertices = result["vertices"]
    triangles = result["triangles"]
    poly_area = result["poly_area"]
    tri_area = result["tri_area"]

    print("\n" + "=" * 60)
    print("GEOMETRY VALIDATION (PRD §3)")
    print("=" * 60)

    # Check 1: Normalized coordinates bounds
    x_norm = vertices[:, 0]
    y_norm = vertices[:, 1]
    x_max_expected = BODY_LENGTH / HALF_SPAN  # 0.802
    y_max_expected = 1.0

    print(f"1. Coordinate bounds:")
    print(f"   x range: [{x_norm.min():.4f}, {x_norm.max():.4f}] (expected: [0, {x_max_expected:.4f}])")
    print(f"   y range: [{y_norm.min():.4f}, {y_norm.max():.4f}] (expected: [-1, {y_max_expected:.4f}])")

    x_ok = abs(x_norm.max() - x_max_expected) < 0.01 and x_norm.min() >= -0.01
    y_ok = abs(y_norm.max() - y_max_expected) < 0.01 and abs(y_norm.min() - (-1.0)) < 0.01
    print(f"   Bounds check: {'PASS' if x_ok and y_ok else 'FAIL'}")

    # Check 2: Symmetry about x-axis (y=0)
    print(f"\n2. Symmetry check:")
    y_values = vertices[:, 1]
    # For each positive y, there should be a corresponding negative y with same x
    pos_vertices = vertices[y_values > 0]
    neg_vertices = vertices[y_values < 0]

    # Sort by |y| for comparison
    pos_sorted = pos_vertices[np.argsort(pos_vertices[:, 1])]
    neg_sorted = neg_vertices[np.argsort(-neg_vertices[:, 1])]  # reverse for matching

    if len(pos_sorted) == len(neg_sorted):
        y_matches = np.allclose(pos_sorted[:, 1], -neg_sorted[:, 1], atol=0.01)
        x_matches = np.allclose(pos_sorted[:, 0], neg_sorted[:, 0], atol=0.01)
        symmetry_ok = y_matches and x_matches
        print(f"   Mirror symmetry: {'PASS' if symmetry_ok else 'FAIL'}")
        if not symmetry_ok:
            print(f"   Y match: {y_matches}, X match: {x_matches}")
    else:
        print(f"   Vertex count mismatch: {len(pos_sorted)} positive vs {len(neg_sorted)} negative")
        symmetry_ok = False

    # Check 3: Triangle area conservation
    print(f"\n3. Area conservation:")
    area_error = abs(tri_area - poly_area) / poly_area * 100
    print(f"   Polygon area: {poly_area:.6f}")
    print(f"   Triangle sum: {tri_area:.6f}")
    print(f"   Error: {area_error:.4f}%")
    area_ok = area_error < 0.1  # Less than 0.1% error
    print(f"   Area conservation: {'PASS' if area_ok else 'FAIL'}")

    # Check 4: Physical dimensions
    print(f"\n4. Physical dimensions:")
    phys_area = poly_area * HALF_SPAN ** 2
    print(f"   Normalized area: {poly_area:.4f}")
    print(f"   Physical area: {phys_area:.2f} m²")
    # Expected B-2 planform area is approximately 450-500 m²
    area_reasonable = 400 <= phys_area <= 600
    print(f"   Reasonable B-2 area (400-600 m²): {'PASS' if area_reasonable else 'FAIL'}")

    # Check 5: Vertex ordering (should be CCW)
    print(f"\n5. Vertex ordering:")
    signed_area = signed_polygon_area(vertices)
    ccw_ok = signed_area > 0
    print(f"   Signed area: {signed_area:.6f} ({'CCW' if ccw_ok else 'CW'})")
    print(f"   Ordering check: {'PASS' if ccw_ok else 'FAIL'}")

    overall_pass = x_ok and y_ok and symmetry_ok and area_ok and area_reasonable and ccw_ok
    print(f"\n{'=' * 60}")
    print(f"OVERALL VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    return overall_pass

if __name__ == "__main__":
    print("Testing Module 1: Burger Geometry Implementation")
    result = run_module1(save=False, verbose=True)
    success = validate_geometry_invariants(result)
    sys.exit(0 if success else 1)