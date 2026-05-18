# 💻 Code Node: array_optimizer.py

## 🔗 Workspace Architecture Connections
[[geometry]], [[array_tracking]], [[diffraction]], [[utils]], [[fsr_pair]], [[link_budget]], [[main]], [[config]], [[__init__]], [[plots]], [[burger_geometry_data]]

## 📜 Code Source
```python
"""
Burger FSR Simulation -- Module 5: Array Configuration Optimization
=====================================================================
GDOP computation, configuration search, minimax optimization.
Find the array layout that minimizes worst-case GDOP across all headings.

See BURGER_SIM_PRD.txt §7.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as cfg
from .utils import freq_to_wavelength
from .array_tracking import (
    build_custom_array,
    array_burger_trajectory,
    detect_crossing_events,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GDOP COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gdop(events: list, target_pos: np.ndarray = None) -> float:
    """
    Compute Geometric Dilution of Precision from crossing events.

    GDOP = sqrt(trace((H^T H)^{-1}))

    where H is the measurement Jacobian relating crossing-time
    differences to 2D ground position.

    Parameters
    ----------
    events     : list of crossing event dicts (need at least 3)
    target_pos : (2,) estimated target ground position (uses centroid if None)

    Returns
    -------
    gdop : float (GDOP_INF_SUBSTITUTE if degenerate)
    """
    if len(events) < 3:
        return cfg.GDOP_INF_SUBSTITUTE

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
        gdop = cfg.GDOP_INF_SUBSTITUTE

    # Cap at substitute if unreasonably large
    if not np.isfinite(gdop) or gdop > cfg.GDOP_INF_SUBSTITUTE:
        gdop = cfg.GDOP_INF_SUBSTITUTE

    return gdop


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GDOP SWEEP OVER HEADINGS
# ─────────────────────────────────────────────────────────────────────────────

def gdop_heading_sweep(pairs: list,
                       heading_step: float = None,
                       freq_hz: float = None,
                       speed: float = None,
                       altitude: float = None,
                       area_x: float = None,
                       area_y: float = None) -> dict:
    """
    Sweep over all heading angles and compute GDOP for each.

    Parameters
    ----------
    pairs        : list of FSR pair dicts
    heading_step : angular resolution (degrees)

    Returns
    -------
    sweep : dict with keys
        'headings'    : (M,) heading angles (degrees)
        'gdops'       : (M,) GDOP values
        'n_events'    : (M,) number of crossing events per heading
        'gdop_worst'  : max GDOP
        'gdop_mean'   : mean GDOP
        'gdop_p95'    : 95th percentile GDOP
        'worst_heading' : heading with worst GDOP
    """
    if heading_step is None:
        heading_step = cfg.HEADING_SWEEP_STEP
    if freq_hz is None:
        freq_hz = cfg.OPERATING_FREQ
    if speed is None:
        speed = cfg.BURGER_SPEED
    if altitude is None:
        altitude = cfg.BURGER_ALTITUDE
    if area_x is None:
        area_x = cfg.ARRAY_AREA_X
    if area_y is None:
        area_y = cfg.ARRAY_AREA_Y

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
    finite_mask = gdops < cfg.GDOP_INF_SUBSTITUTE
    if finite_mask.any():
        gdop_mean = np.mean(gdops[finite_mask])
        gdop_p95 = np.percentile(gdops[finite_mask], 95) if np.sum(finite_mask) > 1 else gdops[finite_mask][0]
    else:
        gdop_mean = cfg.GDOP_INF_SUBSTITUTE
        gdop_p95 = cfg.GDOP_INF_SUBSTITUTE

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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRID SEARCH (STAGE 1)
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(n_pairs: int = None,
                verbose: bool = True) -> pd.DataFrame:
    """
    Stage 1: Exhaustive search over discrete configuration space.

    Sweeps spacing_x × spacing_y × rotation × layout_type.

    Returns DataFrame sorted by gdop_worst ascending.
    """
    if n_pairs is None:
        n_pairs = cfg.N_PAIRS

    spacings_x = cfg.OPT_SPACING_X
    spacings_y = cfg.OPT_SPACING_Y
    rotations = cfg.OPT_ROTATIONS
    layouts = cfg.OPT_LAYOUTS

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

    df = pd.DataFrame(records)
    df.sort_values("gdop_worst", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"  Stage 1 complete. Best GDOP_worst = {df.iloc[0]['gdop_worst']:.2f}")
        print(f"  Top 5 configurations:")
        print(df.head(5).to_string(index=False))
        print()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOCAL REFINEMENT (STAGE 2)
# ─────────────────────────────────────────────────────────────────────────────

def _objective(params, n_pairs, layout_type):
    """
    Objective function for optimization: GDOP_worst.

    params = [spacing_x, spacing_y, rotation_deg]
    """
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


def local_refinement(top_configs: pd.DataFrame,
                     n_top: int = 10,
                     n_pairs: int = None,
                     verbose: bool = True) -> dict:
    """
    Stage 2: Local refinement of top configurations from grid search.

    Uses Nelder-Mead optimization to refine spacing and rotation.

    Returns
    -------
    best : dict with optimized configuration parameters
    """
    if n_pairs is None:
        n_pairs = cfg.N_PAIRS

    if verbose:
        print(f"  Stage 2: Refining top {n_top} configurations...")

    best_gdop = cfg.GDOP_INF_SUBSTITUTE
    best_config = None

    for i in range(min(n_top, len(top_configs))):
        row = top_configs.iloc[i]
        x0 = [row["spacing_x"], row["spacing_y"], row["rotation_deg"]]
        layout = row["layout_type"]

        try:
            result = minimize(
                _objective,
                x0=x0,
                args=(n_pairs, layout),
                method="Nelder-Mead",
                options={"maxiter": 100, "xatol": 10, "fatol": 0.1},
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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RANDOM LAYOUT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_random_layouts(n_random: int = None,
                            n_pairs: int = None,
                            verbose: bool = True) -> dict:
    """
    Evaluate N random array layouts for comparison.

    Returns
    -------
    result : dict with keys
        'gdop_worsts'  : (n_random,) GDOP_worst per layout
        'best_gdop'    : best GDOP_worst found
        'best_seed'    : seed of best layout
        'best_pairs'   : pair list for best layout
    """
    if n_random is None:
        n_random = cfg.N_RANDOM_LAYOUTS
    if n_pairs is None:
        n_pairs = cfg.N_PAIRS

    if verbose:
        print(f"  Evaluating {n_random} random layouts...")

    gdop_worsts = np.zeros(n_random)
    best_gdop = cfg.GDOP_INF_SUBSTITUTE
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


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL MODULE 5 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_module5(verbose: bool = True) -> dict:
    """
    Execute Module 5 end-to-end.

    Returns
    -------
    result : dict with keys
        'grid_search'       : DataFrame of grid search results
        'best_optimized'    : dict of optimal config
        'random_evaluation' : dict of random layout results
        'optimal_sweep'     : GDOP heading sweep for optimal config
        'optimal_pairs'     : pair list for optimal config
        'comparison'        : dict comparing layouts
    """
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
        n_pairs=cfg.N_PAIRS, layout_type='grid',
    )
    naive_sweep = gdop_heading_sweep(naive_pairs)

    # Top 3 configs for polar plot
    top3_sweeps = []
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        p = build_custom_array(
            spacing_x=row["spacing_x"],
            spacing_y=row["spacing_y"],
            rotation_deg=row["rotation_deg"],
            n_pairs=cfg.N_PAIRS,
            layout_type=row["layout_type"],
        )
        s = gdop_heading_sweep(p)
        s["label"] = (f"{row['layout_type']} "
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

```
