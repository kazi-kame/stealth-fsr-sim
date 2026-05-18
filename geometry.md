# 💻 Code Node: geometry.py

## 🔗 Workspace Architecture Connections
[[array_tracking]], [[diffraction]], [[utils]], [[fsr_pair]], [[link_budget]], [[main]], [[config]], [[array_optimizer]], [[__init__]], [[plots]], [[burger_geometry_data]]

## 📜 Code Source
```python
"""
Burger FSR Simulation -- Module 1: Burger Geometry
===================================================
Define Burger's planform as a normalized 2D polygon from public B-2 specs.
Triangulate the polygon.  Optionally add a depth phase screen.

See BURGER_SIM_PRD.txt §3.
"""

import numpy as np
from scipy.spatial import Delaunay

from . import config as cfg
from .utils import (
    ensure_ccw,
    polygon_area,
    point_in_polygon,
    triangle_area,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PLANFORM VERTICES
# ─────────────────────────────────────────────────────────────────────────────

def _build_right_side_vertices() -> np.ndarray:
    """
    Derive right-side (y > 0) vertices from public B-2 specs.

    Coordinate system (PRD §3.1):
      - Origin at nose tip
      - x positive rearward (tail)
      - y positive to starboard (right)
      - All normalized by half-span (26.2 m)

    Key constraints:
      - x_max  = 21.0 / 26.2 ~ 0.802  (body length)
      - y_max  = 1.0                    (wingtip, by definition)
      - LE sweep ~ 33 deg
      - Trailing edge has W-shape
    """
    # Leading edge -- cranked delta with 33° primary sweep
    # tan(33°) ~ 0.6494
    tan_le = np.tan(np.radians(cfg.LE_SWEEP_DEG))

    # ── Right-side vertices (nose -> LE -> wingtip -> TE -> centerline) ──
    V0 = (0.000, 0.000)   # Nose tip

    # LE inner kink: fuselage-wing blend, y ~ 0.27
    y1 = 0.270
    V1 = (y1 * tan_le, y1)                  # (0.175, 0.270)

    # LE outer kink: cranked sweep reduces outboard
    y2 = 0.580
    V2 = (y1 * tan_le + (y2 - y1) * 0.64, y2)   # (~0.373, 0.580)

    # Wingtip leading edge
    y3 = 1.000
    V3 = (V2[0] + (y3 - y2) * 0.44, y3)          # (~0.558, 1.000)

    # Wingtip trailing edge (slight back-sweep)
    V4 = (V3[0] + 0.06, 0.935)                    # (~0.618, 0.935)

    # W-trailing-edge outer notch (aft-most point on outer wing)
    V5 = (0.740, 0.670)

    # W-trailing-edge inner valley
    V6 = (0.640, 0.420)

    # Trailing edge centerline (rearmost point)
    x_max = cfg.BODY_LENGTH / cfg.HALF_SPAN       # 0.802
    V7 = (x_max, 0.000)

    return np.array([V0, V1, V2, V3, V4, V5, V6, V7], dtype=np.float64)


def build_planform_vertices() -> np.ndarray:
    """
    Build the full planform polygon by mirroring the right side.

    Returns
    -------
    vertices : ndarray, shape (N, 2)
        (x, y) pairs in counter-clockwise order forming a closed polygon.
    """
    right = _build_right_side_vertices()

    # V0 … V7 is the right side.
    # Mirror V1..V6 -> left side (y < 0), traverse in reverse so polygon
    # winds continuously: V0, V1…V7, mirror(V6)…mirror(V1).
    inner = right[1:-1]                       # V1 … V6
    mirrored = inner[::-1].copy()             # V6…V1 reversed
    mirrored[:, 1] *= -1                      # flip y

    # Assemble: nose -> right LE -> wingtip -> right TE -> center TE -> left TE -> left LE
    vertices = np.vstack([right, mirrored])   # V0..V7..m(V6)..m(V1)

    vertices = ensure_ccw(vertices)
    return vertices


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRIANGULATION
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_planform(vertices: np.ndarray):
    """
    Triangulate the interior of the planform polygon using Delaunay
    constrained by interior-point filtering.

    Returns
    -------
    triangles : ndarray, shape (M, 3)
        Indices into `vertices` for each triangle.
    tri_area_sum : float
        Sum of triangle areas (should match polygon area).
    """
    # Delaunay over convex hull of the vertices
    tri = Delaunay(vertices)
    simplices = tri.simplices          # (K, 3) index array

    # Keep only triangles whose centroid lies inside the polygon
    centroids = vertices[simplices].mean(axis=1)  # (K, 2)
    inside = point_in_polygon(centroids, vertices)
    interior_simplices = simplices[inside]

    # Compute total area
    area_sum = 0.0
    for s in interior_simplices:
        area_sum += triangle_area(vertices[s[0]], vertices[s[1]], vertices[s[2]])

    return interior_simplices, area_sum


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEPTH PHASE SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def depth_profile(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Smooth depth profile z(x, y) over the planform.
    Gaussian bump centered at (0.4, 0) with sigma_x=0.2, sigma_y=0.3.

    Parameters
    ----------
    x, y : ndarray (same shape)
        Normalized coordinates.

    Returns
    -------
    z : ndarray
        Normalized depth (peak ~ 0.134).
    """
    sigma_x, sigma_y = 0.2, 0.3
    cx, cy = 0.4, 0.0
    z = cfg.MAX_DEPTH_NORM * np.exp(
        -((x - cx) ** 2 / (2 * sigma_x ** 2) +
          (y - cy) ** 2 / (2 * sigma_y ** 2))
    )
    return z


def phase_screen(x: np.ndarray, y: np.ndarray, wavenumber: float) -> np.ndarray:
    """
    Phase contribution from the depth profile for FSR.
      phi(x,y) = k · z(x,y)     [one-way, correct for forward scatter]

    Parameters
    ----------
    wavenumber : float
        k = 2pi / lam
    """
    return wavenumber * depth_profile(x, y)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL MODULE 1 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_module1(save: bool = True, verbose: bool = True):
    """
    Execute Module 1 end-to-end.

    Returns
    -------
    result : dict with keys
        'vertices'    : (N, 2) planform polygon
        'triangles'   : (M, 3) triangle indices
        'poly_area'   : polygon area (shoelace)
        'tri_area'    : sum of triangle areas
        'depth_grid'  : dict with x, y, z arrays  (if USE_DEPTH_SCREEN)
    """
    import os

    # ── Build planform ──────────────────────────────────────────────────
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

    # ── Triangulate ─────────────────────────────────────────────────────
    triangles, tri_area = triangulate_planform(vertices)

    if verbose:
        print(f"  Triangles      : {len(triangles)}")
        print(f"  Triangle area  : {tri_area:.4f}")
        print(f"  Area match     : {abs(tri_area - poly_area) / poly_area * 100:.2f}% error")
        print()

    # ── Depth screen (optional) ─────────────────────────────────────────
    depth_data = None
    if cfg.USE_DEPTH_SCREEN:
        # Build grid over bounding box
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        gx = np.linspace(x_min, x_max, 200)
        gy = np.linspace(y_min, y_max, 200)
        GX, GY = np.meshgrid(gx, gy)
        GZ = depth_profile(GX, GY)

        # Mask outside planform
        pts = np.column_stack([GX.ravel(), GY.ravel()])
        mask = point_in_polygon(pts, vertices).reshape(GX.shape)
        GZ[~mask] = np.nan

        depth_data = {"x": gx, "y": gy, "z": GZ}

        if verbose:
            print(f"  Depth screen   : enabled")
            print(f"  Peak depth     : {np.nanmax(GZ):.4f}  (expected ~{cfg.MAX_DEPTH_NORM:.4f})")
            print()

    # ── Save ────────────────────────────────────────────────────────────
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

```
