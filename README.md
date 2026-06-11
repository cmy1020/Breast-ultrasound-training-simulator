# Breast Ultrasound Training Simulator

A real-time training simulator that integrates finite-element (FEM) soft-tissue
deformation, neural B-mode image synthesis, and 6-DoF haptic force feedback
within a single interactive environment on commodity hardware. This is the
reference implementation for an associated manuscript currently under peer
review; the full citation will be added once the paper is accepted.

The system architecture, haptic force model, geometry-to-image pipeline,
conditional-GAN B-mode synthesiser, decoupled dual-loop runtime, the entire
quantitative and user-study evaluation reported in the manuscript, and the
ongoing extension to other anatomical sites are contributions of this work.
The SOFA scene scaffold and the predefined-path probe-trajectory utility are
adapted from a publicly available SOFA baseline
(see [Acknowledgements](#acknowledgements)).

---

## Highlights

- **Decoupled dual-loop architecture** — a deterministic synchronous
  physics–haptic loop coupled to an asynchronous GAN-inference loop through a
  bounded queue, so that stochastic neural-inference latency cannot perturb
  the force channel.
- **6-DoF haptic force rendering** — a two-term model (proximity force +
  deformation-coupled force) with saturation, dead-zone, and a geometric
  collision projection, driving the Force Dimension Omega.6.
- **Geometry-to-image pipeline** — pose-driven mesh–plane intersection that
  extracts instantaneous tissue cross-sections from the deforming FEM mesh
  and encodes them as one-hot semantic masks.
- **Conditional GAN B-mode synthesiser** — Pix2Pix (U-Net + PatchGAN)
  trained on BUSI; a seven-way encoding ablation is included for
  reproducibility.
- **Real-time profiling** — per-module timing instrumentation for the FEM
  solver, cross-section, rendering, and GAN inference, including the
  dual-loop coefficient-of-variation analysis used in the manuscript.
- **User-evaluation pipeline** — questionnaire, scoring, and statistical
  analysis scripts used in the manuscript's expert–novice study.
- **Cross-organ extension (in progress)** — the same architecture is being
  extended to additional soft-tissue targets (currently a liver mesh option
  is provided); this extension is part of ongoing work and is not evaluated
  in the manuscript.

---

## Scope of evaluation

The manuscript reports quantitative evaluation **only for the breast model
under the Corotated constitutive law** with the haptic + GAN pipeline. The
codebase additionally provides, for development and exploratory use:

- a liver mesh option (`model_name: Liver`) as part of the cross-organ
  extension work in progress (see Highlights);
- alternative constitutive laws (NeoHookean, St. Venant–Kirchhoff) and
  alternative collision schemes (Penalty, Lagrange Multiplier, Prescribed
  Displacement), retained as configurable variants from the SOFA baseline;
- a legacy geometric ray-casting ultrasound module
  (`objects/UltrasoundSimulator.py`).

These options are **not part of the evaluation reported in the manuscript**.

---

## Repository structure

```
├── app.py                       # PyQt5 application entry point
├── simulation.py                # SOFA scene definition (standalone mode)
├── objects/
│   ├── probe.py                 # Ultrasound probe + Omega.6 device wrapper
│   ├── breast.py                # Deformable breast/liver FEM model + lesion
│   ├── UltrasoundSimulator.py   # Legacy ray-casting US module (not used in manuscript)
│   ├── us_imager.py             # Ultrasound imager utilities
│   └── us_view.py               # Ultrasound image viewer
├── components/
│   ├── header.py                # SOFA scene header (collision pipeline, loop)
│   ├── solver.py                # FEM solver factory (CG / Sparse LDL)
│   ├── tetrahedral.py           # Mesh loading, topology, force field, collision
│   └── utils.py                 # YAML config loader, enums, math helpers
├── main_window.py / .ui         # Qt Designer UI definition
├── performance_monitor.py       # Real-time FPS / latency profiler
├── scripts/analysis/            # Offline analysis & visualisation tools
├── inData/                      # Input meshes, lesion ground-truth, configs
├── input_parameters.yml         # Main configuration file
└── requirements.txt             # Python dependencies
```

---

## Setup

### 1. SOFA + SofaPython3
- [SOFA](https://www.sofa-framework.org/download/) (tested with v21.12, v22.06)
- [SofaPython3](https://sofapython3.readthedocs.io/en/latest/menu/Compilation.html) plugin

### 2. Python dependencies
```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| PyQt5 | GUI framework |
| PyOpenGL | 3D rendering |
| torch, torchvision | GAN inference |
| matplotlib | Ultrasound / analysis plots |
| opencv-python | Data preparation |
| pandas | CSV analysis |
| vtk | 3D analysis scripts |

### 3. Omega.6 haptic device (optional)
The haptic mode requires the Force Dimension SDK (`dhd64.dll`) and a
connected Omega.6 device. The DLL path can be updated in `objects/probe.py`.

---

## Usage

### GUI application (recommended)
```bash
python app.py
```
- *Omega6 Haptic* — real-time haptic control with force feedback (the mode
  evaluated in the manuscript).
- *Predefined Path* — automated trajectory replay for batch data generation.
- The parameter panel (top-left) exposes Young's modulus, Poisson ratio, and
  density at runtime.
- The right-hand panel shows the GAN-synthesised B-mode image.

### Headless / batch
```bash
python simulation.py
```

### Configuration (`input_parameters.yml`)
- `model_name` — `Breast` or `Liver` (see Highlights)
- `type` — collision method: `Penalty`, `LM`, or `PrescrDispl`
- `E`, `nu`, `rho` — material properties
- `dt` — simulation time step
- `alarm_distance`, `contact_distance` — collision-detection thresholds
- Mesh file paths, probe velocity, lesion configuration

---

## Analysis scripts

Located in `scripts/analysis/`:

| Script | Description |
|---|---|
| `plot_fx.py` | Force–displacement curves from simulation data |
| `final_compare.py` | Simulation vs. experimental F–x comparison |
| `stiffness_analysis.py` | Linear-stiffness extraction from experimental data |
| `analyze_waypoints.py` | Force-pattern analysis from waypoint CSVs |
| `forceback.py` | Interactive 3D/2D mesh slice viewer (VTK) |
| `cross_section_calculator.py` | Mesh–plane intersection used for the GAN conditioning mask |

---

## Citation

If you use this code in academic work, please cite the manuscript above (full
reference will be added once the paper is accepted).

---

## Acknowledgements

The SOFA scene scaffold and the predefined-path probe-trajectory utility are
adapted from a publicly available SofaPython baseline (Tagliabue et al.,
IJCARS, 2020).

---

## License

Released under the [MIT License](LICENSE).

---

## Contact

For questions about this repository or the associated manuscript, please open
a GitHub issue or contact the corresponding author of the manuscript.
