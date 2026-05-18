# 💻 Code Node: config.py

## 🔗 Workspace Architecture Connections
[[geometry]], [[array_tracking]], [[diffraction]], [[utils]], [[fsr_pair]], [[link_budget]], [[main]], [[array_optimizer]], [[__init__]], [[plots]], [[burger_geometry_data]]

## 📜 Code Source
```python
"""
Burger Forward Scatter Radar Simulation -- Configuration
========================================================
All simulation parameters in one place.
See BURGER_SIM_PRD.txt §9 for full documentation.
"""

import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
C_LIGHT        = 3e8           # m/s
K_BOLTZMANN    = 1.38e-23      # J/K

# ── Burger geometry ─────────────────────────────────────────────────────────
HALF_SPAN      = 26.2          # m (normalization unit)
FULL_SPAN      = 52.4          # m
BODY_LENGTH    = 21.0          # m
LE_SWEEP_DEG   = 33.0          # degrees, leading edge sweep
MAX_DEPTH      = 3.5           # m, maximum body thickness
MAX_DEPTH_NORM = MAX_DEPTH / HALF_SPAN   # ~ 0.134
USE_DEPTH_SCREEN = False       # Toggle depth phase screen

# ── Frequency sweep ─────────────────────────────────────────────────────────
F_MIN          = 30e6          # Hz (30 MHz, VHF lower bound)
F_MAX          = 12e9          # Hz (12 GHz, X-band upper bound)
N_FREQ_POINTS  = 50            # log-spaced points in sweep

# ── FFT / diffraction grid ─────────────────────────────────────────────────
GRID_RES       = 512           # spatial grid resolution (512x512)
FFT_PAD        = 2048          # zero-padded FFT size

# ── FSR pair simulation ────────────────────────────────────────────────────
BASELINE_LENGTH  = 1000.0      # m, Tx-Rx separation
BURGER_ALTITUDE  = 15000.0     # m
BURGER_SPEED     = 306.0       # m/s (~ 1100 km/h)
SIM_DURATION     = 120.0       # s (total simulation window)
DT               = 0.01        # s (timestep)
HEADING_ANGLE    = 90.0        # degrees (perpendicular crossing, default)

# ── Link budget ─────────────────────────────────────────────────────────────
TX_POWER         = 1000.0      # W
TX_GAIN_DB       = 20.0        # dBi
RX_GAIN_DB       = 20.0        # dBi
BANDWIDTH        = 1e6         # Hz
NOISE_TEMP       = 290.0       # K
SNR_THRESHOLD_DB = 10.0        # dB

# ── Array configuration ────────────────────────────────────────────────────
ARRAY_AREA_X     = 5000.0      # m
ARRAY_AREA_Y     = 5000.0      # m
ELEMENT_SPACING  = 500.0       # m
PAIR_BASELINE    = 200.0       # m (within each FSR pair)
BURGER_ENTRY     = (-3000, 1000, 15000)  # m
BURGER_HEADING   = 25.0        # degrees

# ── Monte Carlo ─────────────────────────────────────────────────────────────
N_TRIALS         = 500
TIMING_NOISE_STD = 0.001       # s (1 ms, GPS sync precision)

# ── Chosen operating frequency (updated after Module 2 sweep) ──────────────
OPERATING_FREQ   = 150e6       # Hz (default 150 MHz VHF)

# ── Module 5 -- Array Optimization ──────────────────────────────────────────
N_PAIRS          = 20
OPT_SPACING_X    = [200, 300, 400, 500, 700, 1000]     # m
OPT_SPACING_Y    = [200, 300, 400, 500, 700, 1000]     # m
OPT_ROTATIONS    = [0, 15, 30, 45]                     # degrees
OPT_LAYOUTS      = ['grid', 'hexagonal']
N_RANDOM_LAYOUTS = 50
HEADING_SWEEP_STEP = 1.0       # degrees
GDOP_INF_SUBSTITUTE = 1e6

# ── Output paths ────────────────────────────────────────────────────────────
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_BASE, "data")
OUTPUT_DIR  = os.path.join(_BASE, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

```
