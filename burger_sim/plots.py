"""
Burger FSR Simulation -- Centralized Plotting
=============================================
All matplotlib plotting functions live here.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from . import config as cfg


# ── Shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
})


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 1  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_planform(vertices: np.ndarray,
                  triangles: np.ndarray | None = None,
                  depth_data: dict | None = None,
                  save: bool = True):
    """
    Plot the Burger planform polygon, triangulation overlay,
    and optional depth heatmap.
    """
    n_panels = 2 if depth_data is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: Planform polygon with labeled vertices ─────────────────
    ax = axes[0]
    closed = np.vstack([vertices, vertices[0]])
    ax.fill(closed[:, 1], closed[:, 0], alpha=0.25, color="steelblue",
            label="planform")
    ax.plot(closed[:, 1], closed[:, 0], "k-", lw=1.2)

    # Label vertices
    for i, (vx, vy) in enumerate(vertices):
        ax.plot(vy, vx, "ko", ms=4)
        ax.annotate(f"V{i}", (vy, vx), fontsize=6,
                    textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("y  (span, normalized)")
    ax.set_ylabel("x  (length, normalized)")
    ax.set_title("Burger Planform")
    ax.set_aspect("equal")
    ax.invert_yaxis()            # nose at top
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Triangulation overlay ──────────────────────────────────
    ax = axes[1]
    ax.fill(closed[:, 1], closed[:, 0], alpha=0.12, color="steelblue")
    ax.plot(closed[:, 1], closed[:, 0], "k-", lw=1.0)

    if triangles is not None:
        triang = mtri.Triangulation(vertices[:, 1], vertices[:, 0],
                                    triangles=triangles)
        ax.triplot(triang, "b-", lw=0.5, alpha=0.7)

    ax.set_xlabel("y  (span)")
    ax.set_ylabel("x  (length)")
    ax.set_title(f"Triangulation  ({len(triangles) if triangles is not None else 0} triangles)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Depth heatmap (if enabled) ─────────────────────────────
    if depth_data is not None:
        ax = axes[2]
        x, y, z = depth_data["x"], depth_data["y"], depth_data["z"]
        Y, X = np.meshgrid(y, x)    # note: x is along rows
        im = ax.pcolormesh(Y, X, z.T, shading="auto", cmap="inferno")
        cb = fig.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label("depth (normalized)")

        # Overlay planform outline
        ax.plot(closed[:, 1], closed[:, 0], "w-", lw=1.0)

        ax.set_xlabel("y  (span)")
        ax.set_ylabel("x  (length)")
        ax.set_title("Depth Phase Screen")
        ax.set_aspect("equal")
        ax.invert_yaxis()

    fig.suptitle("Module 1 -- Burger Geometry", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module1_geometry.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")

    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 2  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_diffraction_patterns(patterns: list, save: bool = True):
    """Plot |U(kx,ky)|^2 at representative frequencies."""
    n = len(patterns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, pat in enumerate(patterns):
        ax = axes[i]
        intensity = np.abs(pat["U"]) ** 2
        # Show central region only
        c = intensity.shape[0] // 2
        w = min(100, c)
        sub = intensity[c - w:c + w, c - w:c + w]
        im = ax.imshow(np.log10(sub / sub.max() + 1e-12),
                       cmap="inferno", aspect="equal",
                       extent=[-w, w, -w, w])
        f_mhz = pat["freq"] / 1e6
        lam = pat["wavelength"]
        ax.set_title(f"{f_mhz:.0f} MHz (lam={lam:.2f}m)")
        ax.set_xlabel("ky bin")
        ax.set_ylabel("kx bin")
        fig.colorbar(im, ax=ax, shrink=0.7, label="log10 |U|^2")

    fig.suptitle("Module 2 -- Diffraction Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module2_diffraction.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_frequency_sweep(sweep: dict, save: bool = True):
    """Plot sigma_fs, lobe width, and detectability vs frequency."""
    freqs = sweep["freqs"]
    f_mhz = freqs / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # sigma_fs vs frequency
    ax = axes[0, 0]
    ax.loglog(f_mhz, sweep["sigma_fs"], "b-", lw=1.5)
    ax.axvline(sweep["best_freq"] / 1e6, color="r", ls="--", alpha=0.7,
               label=f"Best: {sweep['best_freq']/1e6:.0f} MHz")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("sigma_fs (m^2)")
    ax.set_title("Forward Scatter RCS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # sigma_fs in dBsm
    ax = axes[0, 1]
    ax.semilogx(f_mhz, sweep["sigma_fs_dBsm"], "g-", lw=1.5)
    ax.axvline(sweep["best_freq"] / 1e6, color="r", ls="--", alpha=0.7)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("sigma_fs (dBsm)")
    ax.set_title("Forward Scatter RCS (dBsm)")
    ax.grid(True, alpha=0.3)

    # Lobe width
    ax = axes[1, 0]
    ax.loglog(f_mhz, np.degrees(sweep["lobe_widths"]), "m-", lw=1.5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Lobe width (degrees)")
    ax.set_title("Forward Scatter Main Lobe Width")
    ax.grid(True, alpha=0.3)

    # Detectability
    ax = axes[1, 1]
    ax.loglog(f_mhz, sweep["detectability"], "r-", lw=1.5)
    ax.axvline(sweep["best_freq"] / 1e6, color="r", ls="--", alpha=0.7,
               label=f"Best: {sweep['best_freq']/1e6:.0f} MHz")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("D(f) = sigma_fs / lobe_width")
    ax.set_title("Detectability Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Module 2 -- Frequency Sweep", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module2_frequency_sweep.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 3  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_fsr_signal(traj: dict, signal: dict, detection: dict,
                    save: bool = True):
    """Plot received signal amplitude and trajectory."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = traj["t"]

    # Trajectory projection
    ax = axes[0]
    ax.plot(traj["x_b"] / 1000, traj["y_b"] / 1000, "b-", lw=1.2, label="Burger track")
    ax.plot(traj["tx_pos"][0] / 1000, traj["tx_pos"][1] / 1000, "r^", ms=10, label="Tx")
    ax.plot(traj["rx_pos"][0] / 1000, traj["rx_pos"][1] / 1000, "gs", ms=10, label="Rx")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Burger Trajectory (ground projection)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Signal amplitude
    ax = axes[1]
    ax.plot(t, signal["s_amplitude"], "b-", lw=0.5, alpha=0.8)
    if detection["detected"]:
        mask = detection["detection_mask"]
        ax.fill_between(t, 0, signal["s_amplitude"],
                        where=mask, alpha=0.2, color="green",
                        label=f"Detection window ({detection['detection_window']:.1f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal amplitude (sqrtW)")
    ax.set_title(f"Received Signal  |  Peak SNR = {detection['peak_snr_dB']:.1f} dB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Module 3 -- FSR Signal", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module3_signal.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_snr(traj: dict, signal: dict, detection: dict,
             save: bool = True):
    """Plot SNR vs time with detection threshold."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    t = traj["t"]
    ax.plot(t, signal["snr_dB"], "b-", lw=0.8)
    ax.axhline(cfg.SNR_THRESHOLD_DB, color="r", ls="--", lw=1.5,
               label=f"Threshold ({cfg.SNR_THRESHOLD_DB:.0f} dB)")

    if detection["detected"]:
        mask = detection["detection_mask"]
        ax.fill_between(t, cfg.SNR_THRESHOLD_DB, signal["snr_dB"],
                        where=mask, alpha=0.2, color="green")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(f"Module 3 -- SNR vs Time  |  "
                 f"Peak = {detection['peak_snr_dB']:.1f} dB  |  "
                 f"Window = {detection['detection_window']:.1f}s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module3_snr.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 4  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_array_map(pairs: list, traj: dict, events: list,
                   save: bool = True):
    """Plot array layout with Burger trajectory and crossing events."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Plot pairs
    for p in pairs:
        tx, rx = p["tx"], p["rx"]
        color = "blue" if p["fence"] == "x" else ("red" if p["fence"] == "y" else "gray")
        ax.plot([tx[0], rx[0]], [tx[1], rx[1]], "-", color=color, lw=0.8, alpha=0.5)
        ax.plot(tx[0], tx[1], "^", color=color, ms=3)
        ax.plot(rx[0], rx[1], "s", color=color, ms=3)

    # Trajectory
    ax.plot(traj["x"], traj["y"], "k-", lw=1.5, label="Burger track")
    ax.plot(traj["x"][0], traj["y"][0], "go", ms=8, label="Entry")

    # Crossing events
    if events:
        ex = [e["position"][0] for e in events]
        ey = [e["position"][1] for e in events]
        ax.scatter(ex, ey, c="red", s=30, zorder=5, label=f"Crossings ({len(events)})")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Module 4 -- Array Layout & Crossing Events")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module4_array_map.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_track(traj: dict, triangulation: dict, events: list,
               save: bool = True):
    """Plot reconstructed vs true trajectory."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # True track
    ax.plot(traj["x"], traj["y"], "k-", lw=2, label="True track", alpha=0.5)

    # Crossing event positions
    if events:
        ex = [e["position"][0] for e in events]
        ey = [e["position"][1] for e in events]
        ax.scatter(ex, ey, c="blue", s=20, alpha=0.5, label="Crossing positions")

    # Reconstructed
    if triangulation and len(triangulation.get("positions", [])) > 0:
        pos = triangulation["positions"]
        ax.plot(pos[:, 0], pos[:, 1], "r--", lw=2,
                label=f"Reconstructed (heading={triangulation['heading_deg']:.1f}°)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Module 4 -- Track Reconstruction")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module4_track.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_errors(mc: dict, save: bool = True):
    """Plot Monte Carlo error distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if len(mc.get("heading_errors", [])) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle("Module 4 -- Error Analysis (insufficient data)")
        fig.tight_layout()
        if save:
            path = os.path.join(cfg.OUTPUT_DIR, "module4_errors.png")
            fig.savefig(path, bbox_inches="tight")
        plt.show()
        return fig

    ax = axes[0]
    ax.hist(mc["heading_errors"], bins=40, color="steelblue", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("Heading error (°)")
    ax.set_ylabel("Count")
    ax.set_title(f"Heading  RMS={mc['heading_rms']:.4f}°")

    ax = axes[1]
    ax.hist(mc["speed_errors"], bins=40, color="coral", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("Speed error (m/s)")
    ax.set_title(f"Speed  RMS={mc['speed_rms']:.3f} m/s")

    ax = axes[2]
    ax.hist(mc["position_errors"], bins=40, color="mediumseagreen", edgecolor="k", alpha=0.7)
    ax.set_xlabel("Position error (m)")
    ax.set_title(f"Position  RMS={mc['position_rms']:.2f} m")

    fig.suptitle("Module 4 -- Monte Carlo Error Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module4_errors.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  MODULE 5  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_gdop_heatmaps(df, save: bool = True):
    """Plot GDOP_worst heatmaps: spacing_x vs spacing_y per layout & rotation."""
    layouts = df["layout_type"].unique()
    rotations = sorted(df["rotation_deg"].unique())
    n_rows = len(layouts)
    n_cols = len(rotations)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for i, layout in enumerate(layouts):
        for j, rot in enumerate(rotations):
            ax = axes[i, j]
            sub = df[(df["layout_type"] == layout) & (df["rotation_deg"] == rot)]
            if sub.empty:
                ax.set_visible(False)
                continue

            pivot = sub.pivot_table(index="spacing_y", columns="spacing_x",
                                    values="gdop_worst", aggfunc="first")
            im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto",
                           origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{int(c)}" for c in pivot.columns], fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{int(r)}" for r in pivot.index], fontsize=7)
            ax.set_title(f"{layout} rot={rot}°", fontsize=9)
            if j == 0:
                ax.set_ylabel("Spacing Y (m)")
            if i == n_rows - 1:
                ax.set_xlabel("Spacing X (m)")
            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Module 5 -- GDOP Worst-Case Heatmaps", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module5_gdop_heatmaps.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_gdop_polar(top3_sweeps: list, save: bool = True):
    """Polar plot of GDOP vs heading for top configurations."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                           subplot_kw={"projection": "polar"})

    colors = ["blue", "red", "green"]
    for i, sweep in enumerate(top3_sweeps):
        headings_rad = np.radians(sweep["headings"])
        # Mirror to full 360°
        full_h = np.concatenate([headings_rad, headings_rad + np.pi])
        full_g = np.concatenate([sweep["gdops"], sweep["gdops"]])
        order = np.argsort(full_h)
        color = colors[i % len(colors)]
        label = sweep.get("label", f"Config {i+1}")
        ax.plot(full_h[order], full_g[order], "-", color=color, lw=1.2,
                label=label, alpha=0.8)

    ax.set_title("GDOP vs Heading Angle", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    fig.suptitle("Module 5 -- GDOP Polar Plot", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module5_gdop_polar.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_optimal_layout(pairs: list, sweep: dict, config: dict,
                        save: bool = True):
    """Plot optimal array layout with worst-case trajectory."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    for p in pairs:
        tx, rx = p["tx"], p["rx"]
        ax.plot([tx[0], rx[0]], [tx[1], rx[1]], "b-", lw=1, alpha=0.6)
        ax.plot(tx[0], tx[1], "b^", ms=4)
        ax.plot(rx[0], rx[1], "bs", ms=4)

    # Worst-case trajectory
    if sweep:
        wh = sweep["worst_heading"]
        wh_rad = np.radians(wh)
        L = max(cfg.ARRAY_AREA_X, cfg.ARRAY_AREA_Y)
        x0 = -L * np.cos(wh_rad)
        y0 = -L * np.sin(wh_rad)
        x1 = L * np.cos(wh_rad)
        y1 = L * np.sin(wh_rad)
        ax.plot([x0, x1], [y0, y1], "r--", lw=2,
                label=f"Worst heading ({wh:.0f}°)")

    title = (f"Optimal: {config['layout_type']} "
             f"sx={config['spacing_x']:.0f} sy={config['spacing_y']:.0f} "
             f"rot={config['rotation_deg']:.1f}°  "
             f"GDOP_worst={config['gdop_worst']:.2f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Module 5 -- Optimal Array Layout", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module5_optimal_layout.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig


def plot_layout_comparison(comparison: dict, save: bool = True):
    """Bar chart comparing GDOP_worst across layout types."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    labels = ["Optimized", "Naive Grid", "Best Random"]
    values = [
        comparison["optimal_gdop_worst"],
        comparison["naive_gdop_worst"],
        comparison["random_best_gdop_worst"],
    ]
    # Cap display at a reasonable value
    cap = 100
    display = [min(v, cap) for v in values]
    colors = ["green", "steelblue", "coral"]

    bars = ax.bar(labels, display, color=colors, edgecolor="k", alpha=0.8)

    for bar, val in zip(bars, values):
        label = f"{val:.1f}" if val < cap else f">{cap}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("GDOP (worst-case)")
    ax.set_title("Module 5 -- Layout Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save:
        path = os.path.join(cfg.OUTPUT_DIR, "module5_layout_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved plot     -> {path}")
    plt.show()
    return fig
