# Model Transferability Demo — SOFA Run Package

This folder holds everything needed to run the **three geometric / lesion
variants** of the breast-ultrasound simulator. The goal is to capture
screenshots that demonstrate the pipeline (FEM deformation + mesh–plane
cross-section + haptic probing) transfers to different breast shapes and
lesion configurations without code changes.

## 1. What is inside

```
inData/variants/
├── README.md                            (this file)
├── input_parameters_baseline.yml        ready-to-use YAML for BASELINE run
├── input_parameters_flat.yml            ready-to-use YAML for FLAT run
├── input_parameters_upperouter.yml      ready-to-use YAML for UPPEROUTER run
│
├── flat/                                FLAT variant mesh pack
│   ├── breast_13k.msh                   tetrahedral mesh (same topology as original)
│   ├── breast_5001.stl                  surface mesh
│   ├── lesion_nodule.stl                Ø16 mm nodule, transformed with same affine
│   └── tumor1/Fiducial0.txt             new lesion centre (3 floats, metres)
│
└── upperouter/                          UPPEROUTER variant pack
    ├── lesion_nodule.stl                Ø12 mm nodule at the upper-outer quadrant
    └── tumor1/Fiducial0.txt             new lesion centre (3 floats, metres)
```

## 2. The three variants

| Variant      | Breast mesh                              | Lesion                                                                 |
|--------------|------------------------------------------|------------------------------------------------------------------------|
| `baseline`   | original `inData/breast_13k.msh`         | original Ø16 mm nodule, original location                              |
| `flat`       | **new** `variants/flat/breast_13k.msh`   | original Ø16 mm nodule, affine-transformed with the breast             |
| `upperouter` | original `inData/breast_13k.msh`         | **new** Ø12 mm nodule, relocated +25 mm lateral, +10 mm toward nipple  |

The `flat` tetrahedral mesh keeps the same vertex indexing as the original
`breast_13k.msh`, so the existing `inData/fixednodes_13k.txt` (fixed-boundary
indices for the chest-wall attachment) applies unchanged.

## 3. How to run a variant

From the project root (`/Users/camille/Desktop/分析`):

```bash
# 1) pick a variant and put its YAML into the canonical location
cp inData/variants/input_parameters_baseline.yml      input_parameters.yml
# or
cp inData/variants/input_parameters_flat.yml          input_parameters.yml
# or
cp inData/variants/input_parameters_upperouter.yml    input_parameters.yml

# 2) launch the simulator exactly as usual
python app.py        # (or whichever script you normally use)
```

When you are done, restore the original run if you want:

```bash
cp inData/variants/input_parameters_baseline.yml input_parameters.yml
```

(or keep your own edited copy; everything outside the `variants/` folder is
untouched.)

## 4. What to capture for the paper figure

For **each** of the three variants we want **4 screenshots**:

| Column | State                              | How to capture                                              |
|--------|------------------------------------|-------------------------------------------------------------|
| 1      | **Static 3D** (probe away)         | screenshot right after the GUI opens, before any deflection |
| 2      | **Small deformation**              | gently press the probe onto the skin, ≈ 2–3 mm indent       |
| 3      | **Large deformation**              | press further, ≈ 6–8 mm indent                              |
| 4      | **RGB cross-section mask**         | screenshot of the cross-section / mask viewport at state (3)|

Tips for clean screenshots (so the final compose step is easy):
- use the same window size for all three variants (otherwise the 3D camera
  angle will differ);
- use the same camera orientation (orbit + zoom) across variants so shape
  differences are the only visible change;
- for column 4, if the GUI shows the mask in its own viewport you can just
  crop that subwindow;
- save all 12 files as PNG into a new folder `inData/variants/captures/`
  using the naming convention below.

Suggested file names (12 total):

```
inData/variants/captures/
├── baseline_static.png
├── baseline_shallow.png
├── baseline_deep.png
├── baseline_mask.png
├── flat_static.png
├── flat_shallow.png
├── flat_deep.png
├── flat_mask.png
├── upperouter_static.png
├── upperouter_shallow.png
├── upperouter_deep.png
└── upperouter_mask.png
```

## 5. When screenshots are ready

Drop them into `inData/variants/captures/` (or anywhere convenient) and
tell me the folder path. I will:

1. background-remove / crop as needed;
2. align the 3 variants to a shared canvas and camera-frame;
3. compose the final `fig_model_transfer.png` (3 rows × 4 cols) with a
   single caption and panel labels `(a1–a4, b1–b4, c1–c4)`;
4. wire it into `sections/methods.tex` or `sections/discussion.tex`
   (whichever you prefer) as a new “extensibility” figure.

## 6. Troubleshooting

- **SOFA complains about `Fiducial0.txt not found`** → check that the YAML’s
  `lesion_basedir` ends with the literal prefix `tumor` (not `tumor1`); the
  simulator appends `tumorID=1` itself, forming `…/tumor1/Fiducial0.txt`.
- **Flat variant explodes / diverges** → the 0.55 y-scale is aggressive;
  bump `nu` down to 0.47 or reduce `dt` to 0.005 if the solver struggles.
  Also confirm `breast_fixed_file` still points to `./inData/fixednodes_13k.txt`
  (same topology, same indices).
- **Upperouter lesion floats outside the breast** → should not happen
  (centre was clamped inside the bbox with a 3 mm margin), but if you see
  it, move the centre in `variants/upperouter/tumor1/Fiducial0.txt` by a
  few mm toward the mesh centroid and regenerate the surface nodule with
  `python generate_lesion_from_fiducial.py` pointed at that file.
