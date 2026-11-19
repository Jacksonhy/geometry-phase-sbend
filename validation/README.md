# Publication validation

- `table_arithmetic.json`: 232 arithmetic comparisons on archived summaries and cost tables; not model inference.
- `portable_checks.json`: actual portable-model primary-metric comparisons against archived geometry results, plus the two-update training smoke run. The report explicitly lists the number of geometries and models checked.
- `fresh_geometry_check.json`: one geometry prepared using Meep initialization at time zero, without time-domain propagation. Material/source arrays, common features, and all four carriers match the original inputs. Single-query batch-size-1 inference shows a small TF32/batch-dependent difference from archived batch-size-2 predictions; the magnitude is recorded.
- `pdf_checks.json`: English translation retains all 11 display-equation groups, decimal values, table row counts, and four main figures. All 13 PDF pages were rendered and visually reviewed.
- `publication_inventory.json`: file inventory, file-size check, and credential-pattern scan before upload.
- `clean_checkout_check.json`: asset verification and GPU inference from a clean Git checkout, using two geometries and seed-2026 LocalTangent.
- `recomputed_euler72_*` and `recomputed_sine36_*`: fresh portable evaluator outputs for all 12 checkpoints. The five primary metrics match archived per-geometry results in 6,480 comparisons, with maximum absolute difference below 4e-16. See `portable_checks.json` for individual metrics and counts.

These checks do not represent new full FDTD runs, higher-resolution convergence tests, or all 12 complete retrainings. Historical result tables are preserved separately in `results/`. No timing benchmark is replaced by a new performance claim.
