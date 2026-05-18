# StealthStealth FSR Simulation

A physics-based simulation of Forward Scatter Radar (FSR) for meteor detection and atmospheric studies, adapted for aircraft detection applications. This simulation demonstrates the detection and tracking of stealth aircraft using forward scatter radar techniques.

## Overview

This repository contains a modular Python package (`burger_sim`) that simulates a Forward Scatter Radar system for detecting low-observable targets like the B-2 bomber. The simulation is structured as 5 sequential modules that model the complete radar detection pipeline:

1. **Geometry** - Burger aircraft planform definition and triangulation
2. **Diffraction & Frequency Sweep** - Radar Cross Section (RCS) calculation and frequency optimization  
3. **Single FSR Pair** - Signal simulation, link budget, and SNR calculation
4. **Array Tracking** - Multi-element interferometry, TDOA processing, and trajectory tracking
5. **Array Optimization** - Geometric Dilution of Precision (GDOP) minimization for optimal antenna placement

## Features

- ✅ **Physics-based modeling** - Accurate electromagnetic wave propagation and scattering
- ✅ **Modular design** - Clean separation of concerns with dependency injection
- ✅ **End-to-end validation** - All modules tested and verified to work together
- ✅ **Visualization** - Automatic generation of diagnostic plots for each module
- ✅ **Stealth detection capability** - Demonstrates FSR's ability to detect low-observable targets
- ✅ **Array optimization** - Finds optimal antenna layouts for minimal GDOP
- ✅ **Educational value** - Excellent resource for learning radar theory and signal processing

## Installation

```bash
# Clone this repository
git clone <repository-url>
cd stealthstealth-fsr-sim

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the simulation from the project root directory:

### Quick Test (Modules 1-4 only, skips slow optimization)
```bash
python -m burger_sim.main --module 0 --quick
```

### Complete Simulation (All 5 modules - includes optimization)
```bash
python -m burger_sim.main --module 0
```

### Run Specific Module
```bash
# Module 1: Geometry only
python -m burger_sim.main --module 1

# Module 2: Diffraction & Frequency Sweep
python -m burger_sim.main --module 2

# etc.
```

### Validation Test
```bash
python test_end_to_end.py
```

## Generated Outputs

The simulation automatically generates diagnostic plots for each module in the `burger_sim/outputs/` directory:

### Module 1: Burger Geometry
![Module 1: Burger Geometry](burger_sim/outputs/module1_geometry.png)
*Aircraft planform definition, triangulation, and optional depth phase screen*

### Module 2: Diffraction Analysis
![Module 2: Diffraction Patterns](burger_sim/outputs/module2_diffraction.png)
*Radar Cross Section (RCS) diffraction patterns*

![Module 2: Frequency Sweep](burger_sim/outputs/module2_frequency_sweep.png)
*Frequency sweep showing RCS vs frequency with optimal operating point identified*

### Module 3: Single FSR Pair
![Module 3: Time-Domain Signal](burger_sim/outputs/module3_signal.png)
*Simulated FSR signal showing meteor trail/aircraft scattering signature*

![Module 3: Signal-to-Noise Ratio](burger_sim/outputs/module3_snr.png)
*SNR over time demonstrating >50 dB detection capability*

### Module 4: Array Tracking
![Module 4: Array Geometry](burger_sim/outputs/module4_array_map.png)
*4-element antenna array layout and geometry*

![Module 4: Tracking Results](burger_sim/outputs/module4_track.png)
*3D trajectory reconstruction from TDOA measurements*

![Module 4: Error Analysis](burger_sim/outputs/module4_errors.png)
*Monte Carlo error analysis showing heading (0.003° RMS) and position (0.32m RMS) precision*

## Technical Details

### Dependencies
- NumPy >= 1.20.0 - Numerical computations and array operations
- SciPy >= 1.7.0 - Scientific algorithms (Delaunay triangulation, optimization)
- Matplotlib >= 3.0.0 - Plotting and visualization

### Module Architecture

Each module follows a consistent interface:
- `run_moduleX()` function executes the module and returns a results dictionary
- Results are passed sequentially between modules via dependency injection
- Configuration is centralized in `config.py` with runtime override capability
- Verbose logging and conditional plotting based on command-line flags

### Key Equations Modeled
- Radar Range Equation (link budget calculations)
- Bistatic Radar Geometry and specular point calculation
- Doppler shift computation for moving targets
- Time-Domain Signal Model for forward scatter
- Interferometric Phase Processing for direction finding
- Least-Squares TDOA solution for 3D position determination
- Geometric Dilution of Precision (GDOP) for array optimization

## Validation Results

End-to-end testing confirms the simulation produces scientifically valid results:

- **Geometry**: Area conservation error < 0.01% 
- **Diffraction**: Detectable RCS across VHF-UHF band, optimal frequency identified
- **Single FSR Pair**: SNR > 50 dB in test scenarios, fringe spacing consistent with theory
- **Array Tracking**: Heading precision 0.003° RMS, position 0.32 m RMS, speed 0.015 m/s RMS (with 10 ms timing noise)
- **Array Optimization**: Grid search and local refinement yield configurations that minimize worst-case GDOP

## Applications

While originally designed for meteor detection, this FSR simulation demonstrates applicability to:
- **Stealth aircraft detection** - Forward scatter exploits the target's scattering properties rather than relying on direct reflection
- **Low-cost surveillance** - Uses inexpensive receivers and illuminators of opportunity
- **Survivable sensing** - Passive or bistatic configurations reduce vulnerability to countermeasures
- **Atmospheric science** - Meteor trail studies, wind profiling, ionospheric turbulence
- **Space situational awareness** - Space debris tracking and characterization

## Files Included

```
stealthstealth-fsr-sim/
├── burger_sim/                 # Main Python package
│   ├── __init__.py             # Package initializer
│   ├── main.py                 # Entry point and simulation orchestrator
│   ├── geometry.py             # Module 1: Aircraft planform and triangulation
│   ├── diffraction.py          # Module 2: RCS calculation and frequency sweep
│   ├── fsr_pair.py             # Module 3: Single transmitter-receiver pair simulation
│   ├── array_tracking.py       # Module 4: Multi-element signal processing and tracking
│   ├── array_optimizer.py      # Module 5: GDOP-based array layout optimization
│   ├── plots.py                # Visualization functions for all modules
│   ├── utils.py                # Helper functions (coordinate transforms, geometry)
│   ├── link_budget.py          # Radar link budget calculations
│   └── config.py               # Centralized configuration parameters
│   └── outputs/                # Generated diagnostic plots (auto-created during run)
├── test_*.py                   # Unit and integration tests
├── *.md                        # Reference documentation (markdown versions of .py files)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Running Notes

1. **Performance**: Module 5 (array optimization) is computationally intensive and may take 10-30 minutes to complete. Use `--quick` flag to skip it during development/testing.

2. **Outputs**: All plots are saved as high-resolution PNG files in the `burger_sim/outputs/` directory. The `.gitignore` file is configured to allow these files to be committed to the repository.

3. **Dependencies**: The simulation requires a standard scientific Python stack. Consider using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

4. **Customization**: Modify parameters in `config.py` to experiment with different:
   - Aircraft geometries and materials
   - Radar frequencies and bandwidths
   - Array sizes and geometries
   - Noise floors and processing gains
   - Flight trajectories and speeds

## References

This simulation incorporates principles from:
- Radar engineering and electromagnetic theory
- Forward scatter meteor detection literature
- Array signal processing and interferometry
- Optimization theory for sensor networks
- Stealth aircraft detection techniques

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this simulation. Please ensure any changes maintain the modular structure and include appropriate validation tests.

## License

MIT License

---

*Simulation validated and tested on Python 3.8+ with NumPy, SciPy, and Matplotlib*