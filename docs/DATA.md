# Data dictionary and evaluation conventions

## Fixed datasets

`data/manifests/{dataset}.json` provides a `records` list. Each record has `geometry_id`, `params`, `input_port`, and `split`. Euler parameters are `wg_width_um`, `lateral_offset_um`, and `R_min_um`. Sinusoidal parameters are `wg_width_um`, `lateral_offset_um`, and `length_um`; the offset is the actual matched endpoint displacement, not the nominal Euler constructor parameter. Lengths are in micrometers. Ports 1 and 2 denote left and right excitation.

| Dataset | Count | Splits |
| --- | ---: | --- |
| `euler_original180` | 180 | `train`: 64, `val`: 16, `test`: 100 |
| `euler_independent72` | 72 | `blind`: 72, geometry/angle stratified |
| `sinusoidal36` | 36 | `transfer`: 36, frozen Euler models |

Sample order is the manifest order. The original training order is retained before stable grouping of train and validation records. Independent samples were selected after all 12 checkpoints were fixed; original100 is retrospective. Higher-resolution subsets overlap these structures and must not be counted as additional independent samples.

## Primary field files

`data/fields/{dataset}/{geometry_id}.npz` is a ZIP of NumPy arrays, read with `np.load(path, allow_pickle=False)`. Every retained `.npy` ZIP member is copied byte-for-byte from the original sample; the ZIP compression is changed. `data/field_manifest.json` records the new container hash, original container hash, and retained member names. Array values are not downsampled, recomputed, normalized, or converted during export.

| Key | Shape / meaning |
| --- | --- |
| `Ez` | `(160,480)`, complex frequency-domain electric field, original amplitude and source/DFT phase convention |
| `eps` | `(160,480)`, relative permittivity on the material grid |
| `x_um`, `y_um` | 480 x coordinates and 160 y coordinates; rows correspond to y, columns to x |
| `src_mask` | `(160,480)`, fixed source-feature mask |
| `source_center_um` | 3-vector of source coordinates |
| `source_size_um` | Source extent, where present |
| `input_port` | Scalar port index 1 or 2 |
| `geometry_id`, `split` | Identity and split labels, where present |
| `params/...` | Scalar geometry parameters |
| `alpha_native`, `S_native` | Original native modal coefficients / scattering outputs |
| `port1/...`, `port2/...` | Retained port coordinates, weights, E/H samples, mode samples, and overlap coefficients |
| `port_masks`, `port_ids`, `S_EH_grid`, `S_Ez_proxy` | Archived port masks and diagnostic optical quantities, where present |

The non-PML region spans 24 × 8 micrometers with spacing 0.05 micrometers at resolution 20 pixels/micrometer. Use the stored coordinate arrays for interpolation; do not infer pixel centers from nominal region endpoints. The primary full-grid H fields and redundant full-domain arrays are omitted, but small port E/H diagnostics are retained. These primary exports do not suffice to regenerate every historical H-field or high-grid diagnostic without new FDTD simulations.

## Common inputs and physical reconstruction

`features.pack` maps 12 real features into seven complex channels (14 real slots). Channel 0 has only a real component; channels 1–5 carry pairs of features; channel 6 has only a real component. The two unused imaginary slots are zero. All four output-reference models receive the identical tensor.

The 12 features are normalized permittivity, source mask, x/y coordinates, source x/y coordinates, excitation sign, calibrated propagation constant per pixel, and the Path v1, Linear, LocalTangent, and Path v2 distances. The network's internal exponential wave prior is retained. ArcOnly changes only the output carrier; it does not remove any common input feature.

`configs/calibration.json` stores the frozen training-only beta fit and amplitude scale. The model output is multiplied by the selected unit-modulus carrier and then by `scale` to recover physical `Ez`. No test-field phase alignment or fitted amplitude rescaling is applied. No query-dependent mode solve is used to fit beta.

## Metrics

`evaluate.py` computes full-region RelL2 with no amplitude threshold; bend-core phase WMAE at a 5% true bend-core peak-amplitude gate; normal-pair relative phase at displacements ±w/4 using complex bilinear interpolation; and complex cross-term RelL2. Undefined near-zero predicted phases receive the original worst-case convention. Reported phase units are degrees.

The emitted evaluation CSV has one row per geometry, seed, and method. `summary.json` gives geometry means separately for each seed. The manuscript's uncertainty is the **sample standard deviation of the three seed means**, not a confidence interval over 216 independent structures. Archived 10,000-resample paired bootstrap intervals and stratified/grid/sensitivity calculations are retained in `results/`; the compact evaluator does not regenerate every archived diagnostic table.

## Newly prepared inputs and outputs

`predict --prepare-only` initializes Meep at time zero and exports `x`, all four `carrier_*` arrays, material/source metadata, and a preparation contract. It forbids `Simulation.run` and `step`; no field labels are read. Inference checks geometry, port, and frozen calibration identity. The field-free audit guard blocks published truth paths, `sample.npz`, and historical prediction folders. It is an accidental-dependency check, not a general-purpose security sandbox.

`predict` writes one NPZ per geometry containing predicted `Ez`, `x_um`, `y_um`, `method`, and `seed`, plus `inference.json`. Generated directories under `outputs/` are ignored by Git. Do not overwrite the provided field files with predictions.
