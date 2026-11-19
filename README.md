# Geometry-derived phase references for S-bend waveguides

Reproducibility package for **Neural-operator prediction of complex electric fields in S-bend waveguides using geometry-derived phase references**, manuscript version 4.3.

[English manuscript (PDF)](paper/manuscript_v4_3_en.pdf) · [English Markdown](paper/manuscript_v4_3_en.md) · [中文完整指南](docs/GUIDE_zh.md) · [Data dictionary](docs/DATA.md) · [Scope and provenance](docs/PROVENANCE.md)

The four output references are Linear, ArcOnly, LocalTangent, and Path v2. They share the architecture, 12 real input features, training splits, and training budgets. The reference phase changes the learned output envelope. Path v1 remains a **common input feature**, although it is not one of the four reported output methods.

## Included in this repository

| Path | Contents |
| --- | --- |
| `paper/` | English Markdown and PDF, Chinese v4.3 review manuscript, aligned Chinese supplement, four main figures and seven supplement figures |
| `src/phase_reference/` | Portable geometry/features, original network bodies with compatibility imports, fresh training, geometry-only inference, field and local-phase evaluation |
| `assets/checkpoints/` | All 12 selected checkpoints: seeds 42, 2026, 2027 × four methods; SHA-256 hashes in `configs/checkpoints.json` |
| `data/manifests/` | Fixed geometries and split assignments |
| `data/fields/` | 288 compact primary-resolution field exports: original 180 Euler, independent 72 Euler, and 36 sinusoidal structures |
| `results/` | Archived per-geometry metrics, seed summaries, paired comparisons, grid diagnostics, timing, optical postprocessing, and exploratory Y-branch results |
| `historical_source/` | Original experiment scripts and protocols, including numerical diagnostics; historical paths/dependencies are retained |
| `third_party/` | Required PIC-Flow FDTD/material sources and NeurOLight model sources, with upstream MIT licenses |
| `validation/` | Publication checks and their stated scope |

Checkpoints occupy about 136 MiB and field exports about 330 MiB. All are ordinary Git files: **no Git LFS, separate model download, or external field archive is required**. A checkout is roughly 0.5 GiB before Git metadata. Raw full-domain/H-field diagnostic arrays, high-resolution field collections, full Y-branch raw data, optimizer/resume states, and redundant caches are not included. See [PROVENANCE.md](docs/PROVENANCE.md) for the limits of reproduction.

## Installation

Use Linux or WSL2 for Meep. The portable GPU path was checked with Python 3.12, NumPy 2.2.6, SciPy 1.18.1, PyTorch 2.11.0+cu128 and an NVIDIA RTX 5070 Laptop GPU. Material preparation was checked separately with Python 3.11 and Meep 1.35.1. Recorded environments are in `environments/`; these are observed versions, not a promise that every platform provides identical builds.

```bash
git clone https://github.com/Jacksonhy/geometry-phase-sbend.git
cd geometry-phase-sbend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a PyTorch build suitable for your GPU/driver first.
# To match the recorded GPU environment when this build is available:
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements-gpu.txt
python -m pip install --no-deps -e .
```

This package uses repository assets relative to the source tree; **use an editable installation and keep the checkout**. A standalone wheel without the repository assets is not supported. The commands below run from the repository root. CPU inference can be requested with `--device cpu`; publication numerical comparisons were performed on the recorded GPU setup.

Meep is only needed to prepare inputs from new geometry or to run FDTD. It is not needed for evaluating the supplied fields or retraining. A separate Conda environment avoids mixing Meep and GPU Python dependencies:

```bash
conda create -n phase-meep -c conda-forge python=3.11 pymeep=1.35.1 scipy
conda run -n phase-meep python -m pip install --no-deps -e .
conda run -n phase-meep python -m pip install --no-deps -e .
```

If that exact Meep build is unavailable for your platform, use a supported Linux build and record the resulting environment; different numerical libraries may change floating-point results. The supplied fields and checkpoints preserve the original calculation artifacts.

## Quick verification without GPU or Meep

```bash
python scripts/verify_assets.py
python scripts/verify_tables.py --tables results --output outputs/table_check.json
```

Asset verification checks all 12 checkpoint hashes and 288 field hashes and the manuscript image links. Table verification recomputes saved summary arithmetic; it does **not** execute FDTD, training, or inference.

## Predict a field using geometry only

Prepare inputs under Meep Python. This initializes the material grid at simulation time zero and explicitly forbids `Simulation.run` and `Simulation.step`:

```bash
conda run -n phase-meep python -m phase_reference.predict \
  --geometry-json examples/geometry.json --prepare-only --output outputs/inputs

# Run this second command in the GPU environment:
python -m phase_reference.predict \
  --geometry-json examples/geometry.json --prepared-inputs outputs/inputs \
  --method local_tangent --seed 2026 --output outputs/query
```

Alternatively, supply `--meep-python /absolute/path/to/meep/python` instead of `--prepared-inputs` to invoke the separate preparation process automatically. Both Python environments must have an editable installation of this checkout. Use a fresh preparation directory; existing inputs are not silently reused.

The JSON accepts one geometry, a list, or an object containing `records`. Euler queries use `wg_width_um`, `lateral_offset_um`, and `R_min_um` within the evaluated ranges and 0.025 μm lattice, plus `input_port` 1 or 2. Sinusoidal queries use `length_um` instead of `R_min_um`; use the supplied sinusoidal manifest for the tested cases. Predictions for arbitrary new shapes or out-of-distribution parameters are not validated. Output NPZ files contain the reconstructed physical complex `Ez` field and its grid; no ground-truth field is read by this command.

## Recompute model metrics

```bash
# All 12 checkpoints on the 72 independent Euler structures:
python -m phase_reference.evaluate --dataset euler_independent72 --output outputs/euler72

# Frozen Euler-trained models on the 36 sinusoidal structures:
python -m phase_reference.evaluate --dataset sinusoidal36 --output outputs/sine36

# Retrospective original test set (100 structures):
python -m phase_reference.evaluate --dataset euler_original180 --split test --output outputs/original100

# Short check instead of a complete evaluation:
python -m phase_reference.evaluate --limit 2 --output outputs/smoke
```

`--methods local_tangent arc_only`, `--seeds 2026`, `--batch-size 2`, and `--device cuda` select a smaller comparison. Generated `per_geometry.csv` and `summary.json` contain whole-field and bend-core RelL2, phase WMAE, normal-pair phase, and transverse cross-term RelL2. The default gate is 0.05 and the normal-pair displacement is ±w/4. Truth fields are used only for scoring. Aggregate **per-seed geometry means** before calculating a mean and sample standard deviation across seeds; the 216 seed/geometry combinations are not 216 independent geometries.

Expected three-seed means on the independent Euler set:

| Reference | Whole-field RelL2 | Bend-core phase WMAE (degrees) |
| --- | ---: | ---: |
| Linear | 0.3636 | 12.4333 |
| ArcOnly | 0.1610 | 5.5342 |
| LocalTangent | 0.1472 | 4.1876 |
| Path v2 | 0.1474 | 4.1556 |

Small floating-point differences may occur across hardware and software. The figures in the manuscript are the existing research figures, copied without redrawing; archived timing values describe the original hardware/workflows.

Publication validation recomputed all 12 checkpoints on both the 72-structure independent Euler set and the 36-structure sinusoidal set. The five primary metrics match the archived geometry-level results in 6,480 comparisons (maximum absolute difference below 4e-16). A fresh time-zero Meep input check, a clean-checkout inference check, and a two-update training execution check also passed. Detailed reports and recomputed tables are in `validation/`; this validation did not rerun all FDTD simulations or all 12 complete training runs.

## Train from the fixed training split

```bash
python -m phase_reference.train --method local_tangent --seed 2026 --output outputs/train-lt-2026
# A short execution check; this is not a paper-quality trained model:
python -m phase_reference.train --method local_tangent --seed 2026 --steps 2 --output outputs/train-smoke
```

Run all combinations of four methods and three seeds to repeat the main training comparison. Each full run uses 2400 AdamW updates, batch 2, learning rate 0.002, cosine decay to 0.0001, weight decay 0.0001, clipping at 5, and the same fixed 64/16 split. Validation is performed after update 1 and every 50 updates, selecting the lowest mean validation RelL2. Calibration is already frozen in `configs/calibration.json` and is never fitted to validation or test fields. Fresh runs require an empty output directory; this compact CLI does not resume interrupted optimizer states. Original resume-capable scripts are archived in `historical_source/`.

## Manuscript and figures

The English manuscript is a translation of the current Chinese v4.3 draft. Equations, scientific values, section structure, and the four retained main figures are preserved. The data-availability paragraph is updated for this repository. The Chinese draft retains its original historical availability statement. The supplement is provided in its current Chinese form; it has not been translated as part of the English manuscript export.

Main figures map to the original figures 1, 2, 3, and 6; original figure 6 is now main figure 4. Supplement S6 uses the existing reference-diagnostic figure and S7 uses the existing sinusoidal examples. No deleted main-text section or figure has been restored.

To regenerate the English PDF from Markdown, install Pandoc and Chrome/Chromium, then run:

```bash
python scripts/export_manuscript.py --pandoc /path/to/pandoc --chrome /path/to/chrome
```

The exporter embeds the existing images and MathML into a temporary local HTML document and prints to PDF with a separate headless browser profile. It does not use the user's browser session or fetch remote resources. This is a review manuscript, without author metadata or journal-specific submission declarations.

## Attribution and reuse

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSING.md](LICENSING.md). Upstream MIT notices are retained. No additional blanket license has been selected for the original manuscript, data, checkpoints, or project-specific contributions. Publication on GitHub alone should not be interpreted as a permissive license. `CITATION.cff` identifies the software repository and maintainer account; it does not assert the paper's author list or a DOI.
