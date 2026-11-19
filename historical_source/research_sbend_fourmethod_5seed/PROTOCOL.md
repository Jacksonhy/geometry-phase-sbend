# Four phase references, five training seeds — frozen protocol
Date: 2026-11-13 (Asia/Shanghai).
Authorization: first execute seed2026 only; seeds2027–2030 are planned, not launched.
Methods: linear, arc_only, local_tangent, path_v2.
Primary question: separate accumulated centerline arc from transverse tangent correction.
ArcOnly: d = sigma * (sc(x) - source_s); c = exp(i beta d), cast complex64.
Use the exact historical Euler polyline, source projection, units and training calibration.
No additional input channels. All four methods use identical cached seven complex input channels.
Reuse frozen 64 train / 16 validation / 100 test geometry manifest. No new FDTD.
Architecture and optimization are exactly research_sbend_formal/model.py and train.py:
12 Fourier blocks, width64, modes40/70, head256; 2,944,802 real scalar degrees;
FP32/TF32, dropout0.1, AdamW lr0.002, wd0.0001, clip5; batch2; 2400 updates;
cosine lr ending0.0001; validation step1/every50; strict minimum validation mean RelL2;
best and last checkpoints. No augmentation or test-based phase alignment.
For each future seed2026–2030: same seed across methods, same initial parameters and
PCG64 batch schedule. Geometry/calibration remain fixed across training seeds.
Historical seed2026 linear/local_tangent/path_v2 may be reused only after source,
weights, cache, calibration and input hashes pass, and validation replay <=1e-7.
Historical reuse is not a new independent training repetition.
New seed2026 arc_only reuses the unchanged historical training loop via an adapter,
redirecting all writes to this directory. No historical scripts are executed as main.
Preflight includes unit-modulus, source-anchor, left/right sign, straight-lead equality,
independent scalar arc oracle, exact LocalTangent carrier reconstruction, finite
forward/backward, identical initialization/batches and historical validation replay.
No test Ez is read in preflight or training. Geometry-only test carrier is constructed
after training, with the same fixed formula. Test is an already-used benchmark.
Primary endpoints: per-geometry unaligned physical complex RelL2; best primary,
last sensitivity. Report all four means/median/p90 and both excitation directions.
All six pairs: mean paired difference, fraction improved, 10,000 geometry bootstrap
95% percentile CI (PCG64 seed2026), conditional on trained models; exploratory intervals,
no multiplicity correction. No formal equivalence claim or training-seed CI from one seed.
Secondary: unchanged core/bend_core/halo/far errors and non-input complex/amplitude/
squared-amplitude proxy errors. Proxy is not certified native S/power.
Save arc_only physical predictions best/last; reuse old saved metrics for other methods.
Convergence: retain historical logging, which records current-batch relative MSE and
validation RelL2 at step1/every50; best-checkpoint whole-train RelL2 at completion.
Do not relabel current-batch MSE as whole-train RelL2.
Freeze source/historical manifests before optimization. Stop on mismatch/nonfinite/OOM;
no method substitutions or hyperparameter search. Resume only exact saved state.
At finish verify protected historical hashes unchanged; save report, tables and curves.
