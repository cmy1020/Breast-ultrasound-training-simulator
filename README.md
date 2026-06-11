# Breast Ultrasound Training Simulator

A real-time finite-element simulation of ultrasound probe–tissue interaction, built on the [SOFA Framework](https://www.sofa-framework.org/). The system supports both **haptic teleoperation** (Force Dimension Omega.6) and **predefined-path** modes, with force feedback and GAN-based ultrasound image generation.

This repository is the **SofaPython3** upgrade of the original SofaPython2 codebase used in the [IJCARS 2020 paper](#references).

---

## Features

- **Two control modes:**
  - **Omega.6 Haptic** — real-time probe control via Force Dimension haptic device with force feedback (proximity + deformation resistance)
  - **Predefined Path** — automated probe trajectories from ground-truth data for batch data generation
- **Finite-element soft tissue model** — tetrahedral mesh, multiple constitutive laws (Corotated, NeoHookean, St. Venant–Kirchhoff), configurable material parameters
- **Collision detection & response** — three methods: Penalty, Lagrange Multiplier (LM), and Prescribed Displacement
- **Real-time ultrasound simulation** — geometric ray-casting + Pix2Pix GAN generating realistic B-mode images from cross-section masks
- **Breast & liver models** — supports cross-organ transferability experiments
- **Performance monitoring** — per-frame profiling of FEM solver, cross-section, rendering, and GAN inference
- **Analysis toolkit** — scripts for force–displacement characterization, stiffness calibration, and comparison with experimental data

---

## Structure

```
├── app.py                  # Main PyQt5 application entry point
├── simulation.py           # SOFA scene definition (standalone mode)
├── objects/
│   ├── probe.py            # Ultrasound probe + Omega.6 device wrapper
│   ├── breast.py           # Deformable breast/liver FEM model + lesion
│   ├── UltrasoundSimulator.py  # Geometric ray-casting ultrasound simulator
│   ├── us_imager.py        # Ultrasound imager utilities
│   └── us_view.py          # Ultrasound image viewer
├── components/
│   ├── header.py           # SOFA scene header (collision pipeline, animation loop)
│   ├── solver.py           # FEM solver factory (CG / Sparse LDL)
│   ├── tetrahedral.py      # Mesh loading, topology, force field, collision models
│   └── utils.py            # YAML config loader, enums, math helpers
├── main_window.py / .ui    # Qt Designer UI definition
├── performance_monitor.py  # Real-time FPS / latency profiler
├── scripts/analysis/       # Offline analysis & visualization tools
├── inData/                 # Input meshes, tumour ground-truth, config variants
├── input_parameters.yml    # Main configuration file
└── requirements.txt        # Python dependencies
```

---

## Setup

### 1. Install SOFA + SofaPython3

- [SOFA](https://www.sofa-framework.org/download/) (tested with v21.12, v22.06)
- [SofaPython3](https://sofapython3.readthedocs.io/en/latest/menu/Compilation.html) plugin

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Additional runtime dependencies (not in requirements.txt — install as needed):

| Package | Purpose |
|---------|---------|
| `PyQt5` | GUI framework |
| `PyOpenGL` | 3D rendering |
| `torch`, `torchvision` | GAN inference |
| `matplotlib` | Ultrasound / analysis plots |
| `opencv-python` | Data preparation |
| `pandas` | CSV analysis |
| `vtk` | 3D analysis scripts |

### 3. Omega.6 Haptic (optional)

Requires Force Dimension SDK (`dhd64.dll`) and a connected Omega.6 device. Update the DLL path in `objects/probe.py` if needed.

---

## Usage

### GUI application (recommended)

```bash
python app.py
```

- Click **"Omega6 Haptic"** for real-time haptic control
- Click **"Predefined Path"** for automated trajectory replay
- Use the **parameter panel** (top-left) to adjust Youngʼs modulus, Poisson ratio, and density at runtime
- The right panel shows the generated ultrasound image

### Standalone (headless / batch)

```bash
python simulation.py
```

Prompts you to choose Omega.6 or predefined-path mode.

### Configuration

Edit [`input_parameters.yml`](input_parameters.yml) to change:

- `model_name` — `Breast` or `Liver`
- `type` — collision method: `Penalty`, `LM`, or `PrescrDispl`
- `E` / `nu` / `rho` — material properties
- `dt` — simulation time step
- `alarm_distance` / `contact_distance` — collision detection thresholds
- Mesh file paths, probe velocity, lesion configuration

---

## Analysis Scripts

Located in [`scripts/analysis/`](scripts/analysis/):

| Script | Description |
|--------|-------------|
| `plot_fx.py` | Plot force–displacement curves from simulation data |
| `final_compare.py` | Compare simulation vs. experimental F–x data |
| `stiffness_analysis.py` | Extract linear stiffness from experimental data |
| `analyze_waypoints.py` | Analyse force patterns from waypoint CSV |
| `forceback.py` | Interactive 3D/2D mesh slice viewer (VTK) |
| `cross_section_calculator.py` | High-performance mesh–plane intersection |

---

## References

Tagliabue, E., DallʼAlba, D., Magnabosco, E. et al.  
[Biomechanical modelling of probe to tissue interaction during ultrasound scanning](https://link.springer.com/article/10.1007/s11548-020-02183-2)  
*International Journal of Computer Assisted Radiology and Surgery* (2020).

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

Altair Robotics Lab — University of Verona  
eleonora \[dot\] tagliabue \[at\] univr \[dot\] it
