# 💻 Code Node: main.py

## 🔗 Workspace Architecture Connections
[[geometry]], [[array_tracking]], [[diffraction]], [[utils]], [[fsr_pair]], [[link_budget]], [[config]], [[array_optimizer]], [[__init__]], [[plots]], [[burger_geometry_data]]

## 📜 Code Source
```python
"""
Burger FSR Simulation -- Main Entry Point
==========================================
Runs all five modules in sequence.

Usage:
    python -m burger_sim.main              # Run all modules
    python -m burger_sim.main --module 2   # Run specific module
    python -m burger_sim.main --quick      # Skip Module 5 optimization (slow)
"""

import sys
import os
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Burger Forward Scatter Radar Simulation",
    )
    parser.add_argument("--module", type=int, default=0,
                        help="Run a specific module (1-5). 0 = run all.")
    parser.add_argument("--quick", action="store_true",
                        help="Skip Module 5 optimization (slow).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output.")
    return parser.parse_args()


def run_all(module: int = 0, quick: bool = False,
            no_plots: bool = False, quiet: bool = False):
    """
    Run the full simulation pipeline.

    Parameters
    ----------
    module   : 1-5 to run a specific module, 0 for all
    quick    : skip Module 5 if True
    no_plots : skip plot generation
    quiet    : suppress verbose output
    """
    verbose = not quiet
    t_start = time.time()

    if verbose:
        print("+" + "=" * 58 + "+")
        print("|    BURGER FORWARD SCATTER RADAR SIMULATION               |")
        print("+" + "=" * 58 + "+")
        print()

    results = {}

    # ═══════════════════════════════════════════════════════════════════
    #  MODULE 1 -- BURGER GEOMETRY
    # ═══════════════════════════════════════════════════════════════════
    if module in (0, 1):
        from .geometry import run_module1
        t1 = time.time()
        results["module1"] = run_module1(save=True, verbose=verbose)

        if not no_plots:
            from .plots import plot_planform
            plot_planform(
                results["module1"]["vertices"],
                results["module1"]["triangles"],
                results["module1"].get("depth_data"),
            )

        if verbose:
            print(f"  Module 1 complete in {time.time() - t1:.1f}s")
            print()

    # ═══════════════════════════════════════════════════════════════════
    #  MODULE 2 -- DIFFRACTION & FREQUENCY SWEEP
    # ═══════════════════════════════════════════════════════════════════
    if module in (0, 2):
        from .diffraction import run_module2
        from .geometry import build_planform_vertices

        t2 = time.time()
        vertices = (results.get("module1", {}).get("vertices")
                    if "module1" in results
                    else build_planform_vertices())

        results["module2"] = run_module2(vertices=vertices, verbose=verbose)

        if not no_plots:
            from .plots import plot_diffraction_patterns, plot_frequency_sweep
            plot_diffraction_patterns(results["module2"]["patterns"])
            plot_frequency_sweep(results["module2"]["sweep"])

        if verbose:
            print(f"  Module 2 complete in {time.time() - t2:.1f}s")
            print()

    # ═══════════════════════════════════════════════════════════════════
    #  MODULE 3 -- SINGLE FSR PAIR
    # ═══════════════════════════════════════════════════════════════════
    if module in (0, 3):
        from .fsr_pair import run_module3

        t3 = time.time()

        # Use sigma_fs from Module 2 if available
        sigma_fs = None
        freq_hz = None
        if "module2" in results:
            sweep = results["module2"]["sweep"]
            best_idx = sweep["best_freq_idx"]
            sigma_fs = sweep["sigma_fs"][best_idx]
            freq_hz = sweep["best_freq"]

        results["module3"] = run_module3(sigma_fs=sigma_fs,
                                         freq_hz=freq_hz,
                                         verbose=verbose)

        if not no_plots:
            from .plots import plot_fsr_signal, plot_snr
            m3 = results["module3"]
            plot_fsr_signal(m3["traj"], m3["signal"], m3["detection"])
            plot_snr(m3["traj"], m3["signal"], m3["detection"])

        if verbose:
            print(f"  Module 3 complete in {time.time() - t3:.1f}s")
            print()

    # ═══════════════════════════════════════════════════════════════════
    #  MODULE 4 -- ARRAY TRACKING
    # ═══════════════════════════════════════════════════════════════════
    if module in (0, 4):
        from .array_tracking import run_module4

        t4 = time.time()
        freq_hz = (results.get("module2", {}).get("sweep", {}).get("best_freq")
                   or None)

        results["module4"] = run_module4(freq_hz=freq_hz, verbose=verbose)

        if not no_plots:
            from .plots import plot_array_map, plot_track, plot_errors
            m4 = results["module4"]
            plot_array_map(m4["pairs"], m4["traj"], m4["events"])
            plot_track(m4["traj"], m4["triangulation"], m4["events"])
            plot_errors(m4["monte_carlo"])

        if verbose:
            print(f"  Module 4 complete in {time.time() - t4:.1f}s")
            print()

    # ═══════════════════════════════════════════════════════════════════
    #  MODULE 5 -- ARRAY OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    if module in (0, 5) and not quick:
        from .array_optimizer import run_module5

        t5 = time.time()
        results["module5"] = run_module5(verbose=verbose)

        if not no_plots:
            from .plots import (plot_gdop_heatmaps, plot_gdop_polar,
                                plot_optimal_layout, plot_layout_comparison)
            m5 = results["module5"]
            plot_gdop_heatmaps(m5["grid_search"])
            plot_gdop_polar(m5["top3_sweeps"])
            if m5["optimal_pairs"]:
                plot_optimal_layout(m5["optimal_pairs"],
                                    m5["optimal_sweep"],
                                    m5["best_optimized"])
            plot_layout_comparison(m5["comparison"])

        if verbose:
            print(f"  Module 5 complete in {time.time() - t5:.1f}s")
            print()

    elif module in (0, 5) and quick:
        if verbose:
            print("  Module 5 SKIPPED (--quick mode)")
            print()

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    total_time = time.time() - t_start
    if verbose:
        print("+" + "=" * 58 + "+")
        print(f"|  Simulation complete in {total_time:.1f}s" +
              " " * (33 - len(f"{total_time:.1f}")) + "|")
        print("+" + "=" * 58 + "+")

    return results


def main():
    args = parse_args()
    run_all(
        module=args.module,
        quick=args.quick,
        no_plots=args.no_plots,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

```
