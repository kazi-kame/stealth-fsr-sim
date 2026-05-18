Practical Feasibility and Hurdles for Burger FSR System

Transmitter/Receiver Hardware
- Transmit power: 1 kW CW or pulsed is achievable with solid state or tube amplifiers at VHF/UHF.
- Antenna gain: 20 dBi corresponds to modest sized Yagi or phased array elements.
- Receiver: software defined radio with 1 MHz bandwidth and coherent integration over observation window.
- Illuminator options: dedicated transmitter or signals of opportunity (FM broadcast, DAB, DVB-T, communications beacons).
- Dynamic range: must handle strong direct path leakage and weak forward-scatter returns (nW–µW levels).

Timing and Synchronization
- Sub‑meter position accuracy requires nanosecond level time synchronization between nodes.
- GPS‑disciplined oscillators or White Rabbit links provide <10 ns stability over kilometer baselines.
- For fully covert operation, two‑way fiber timing or coherent carrier phase transfer over the same RF link can be used.
- Internal delay calibration of each receiver chain is necessary.

Signal Processing Load
- Module 2 FFT sweep: a few million complex multiplies per frequency, easily done in real time on a CPU or GPU.
- Per‑pair processing (range, power, matched filter, phase): vectorized execution handles hundreds of pairs in <10 ms per update on modern hardware.
- TDOA solution: building and solving the H matrix is negligible for tens of events.
- Monte Carlo error assessment is optional for performance monitoring; online tracking can use a Kalman filter.

Array Deployment and Geometry
- Number of pairs: scalable; example 20 pairs over a 5 km × 5 km area gives ~4 pairs/km².
- GDOP optimization provides systematic placement to minimize worst‑case geometry across all headings.
- Layout choices: regular grid, hexagonal, or random; hexagonal offers more uniform coverage.
- Site requirements: power, line of sight to surveillance volume, terrain masking, access for maintenance.
- Mobile or vehicle mounted nodes can supplement fixed infrastructure for temporary defenses or gap filling.

Software Integration
- Existing code is modular Python with clear inputs/outputs (dicts, NumPy arrays).
- For deployment, wrap each module in a lightweight service (e.g., ZeroMQ, ROS2, DDS) and fuse TDOA fixes with a track‑while‑scan filter (e.g., IMM Kalman filter).
- Provide situational awareness display showing tracks, array health, GDOP contours.

Cost Estimate (indicative)
- TX/Rx node (SDR + amplifier + antenna + GPSDO + enclosure): roughly $2k–$5k each.
- For 20 pairs (40 RX + 20 TX): $120k–$280k.
- Comms and timing backhaul (fiber or microwave): $10k–$50k.
- Processing server(s): $5k–$10k.
- Installation and survey (labor, GNSS rentals): $30k–$80k.
- Total indicative range: $200k–$400k for a 20‑pair, 5 km × 5 km system.
- Compared to conventional 3‑D surveillance radars covering the same area (often several million dollars), the FSR concept can be one to two orders of magnitude cheaper, especially when using illuminators of opportunity.

Environmental and Operational Limits
- Atmospheric refraction: correct baseline geometry with standard models (ITU‑R P.452‑16).
- Ionospheric scintillation (VHF): consider higher UHF/L‑band if problematic, or apply diversity combining.
- Ground clutter and multipath: mitigate with antenna elevation, sidelobe suppression, clutter maps, and multi‑baseline coincidence requirements.
- Weather (rain, fog): minimal direct effect on propagating wavelengths at VHF/UHF; forward‑scatter RCS depends primarily on target geometry and dielectric properties, not significantly affected by atmospheric water content at these frequencies.
- Emitter agility/EW resistance: passive reception of opportunity emissions makes the receiver hard to detect; active illuminators can be made frequency agile, spread spectrum, or low probability of intercept to reduce signature.

Summary
The Burger FSR simulation confirms the underlying physics and algorithms are correct. Transitioning to a fielded system requires attention to timing synchronization, hardware selection, site selection, and integration of the processing chain. No fundamental showstoppers appear; the remaining work is conventional systems engineering for a distributed, multistatic radar network.