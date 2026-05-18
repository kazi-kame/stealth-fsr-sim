#!/usr/bin/env python3
"""
Test script for Module 1: Burger Geometry
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Import config
import config as cfg

# Define the geometry functions directly from the markdown
def _build_right_side_vertices():
    """
    Derive right-side (y > 0) vertices from public B-2 specs.
    """
    tan_le = np.tan(np.radians(cfg.LE_SWEEP_DEG))

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

    x_max = cfg.BODY_LENGTH / cfg.HALF_SPAN       # 0.802
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
    z = cfg.MAX_DEPTH_NORM * np.exp(
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
        phys = poly_area * cfg.HALF_SPAN ** 2
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
    if cfg.USE_DEPTH_SCREEN:
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
            print(f"  Peak depth     : {np.nanmax(GZ):.4f}  (expected ~{cfg.MAX_DEPTH_NORM:.4f})")
            print()

    if save:
        path = os.path.join(cfg.DATA_DIR, "burger_geometry.npy")
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

if __name__ == "__main__":
    result = run_module1(save=False, verbose=True)
    print('Module 1 completed successfully')
    print(f'Vertices shape: {result["vertices"].shape}')
    print(f'Triangles shape: {result["triangles"].shape}')
    print(f'Polygon area: {result["poly_area"]:.4f} normalized')
    print(f'Physical area: {result["poly_area"] * cfg.HALF_SPAN**2:.1f} m^2')