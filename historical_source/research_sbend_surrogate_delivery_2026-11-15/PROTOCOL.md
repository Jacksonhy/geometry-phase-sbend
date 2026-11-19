# Frozen delivery and timing protocol

Task: geometry-only inference, cost/break-even audit, 72-geometry field tables/figures. No training, no time-domain propagation, no new geometry truth, no phase/amplitude recalibration. All outputs in this folder; historical artifacts remain read-only.

Inference target is original resolution20, fixed1.55um, 2D effective-index Ez, original single-port Gaussian EigenModeSource. All common input channels and all four carriers retained for exact protocol replay; cost includes this work. Meep geometry initialization allowed, Simulation.run/step forbidden. Deployment guard forbids sample.npz and all data/cache/predictions directories. Verification/reporting separately may read saved truth/predictions.

First reconstruct 72 geometry-only inputs and compare eps/source/coordinates/all features/carriers to existing exports. Then replay 4 methods x3 seeds with original best checkpoints. Numerical equivalence tolerance declared: max prediction difference / max historical amplitude <=1e-5; same-device deterministic bitwise equality is checked separately. Preserve any failures. No best-seed selection.

Costs: new measurements separate geometry/material initialization, common features, input I/O, model loading, synchronized GPU inference, transfer, output export. Four methods benchmark on fixed seed2026 (specified before timing). Cold CLI 3 repeats per method on first low-angle input1 geometry. Resident-model single requests on six geometries: first input1/2 in each angle stratum. Batch1/2 timings over all72. Include process startup in cold timings; resident GPU still launches a fresh geometry worker per single request. Shared preprocessing can be amortized only in batched workflow, explicitly labelled.

Offline deployment budget: historical64 training+16 validation labels, original calibration and common feature preparation, a single frozen model training (including validation/checkpointing). Training/calibration/log costs with missing wall records remain explicitly missing; never inferred from timestamps as measured durations. Report measured subtotal, missing costs and sensitivity. Evaluation/testing/method/seed search costs separate. Reused existing training treated as sunk only in an explicitly separate incremental scenario.

Accuracy: six endpoints whole/bend complex RelL2, bend absolute phase WMAE, normal-pair relative phase WMAE, transverse complex cross term RelL2, bend amplitude RelL2. Original 5% phase gate, eta.5 pairs, no fitted truth alignment. Means per geometry per seed; geometry-paired CIs conditional on fixed models. Three seeds never treated as216 independent geometries. Descriptive accuracy/cost curves, not certification. Predeclared illustrative thresholds whole RelL2 <=0.10/0.15/0.20 and joint bend RelL2<=.15 with bend phase<=5deg; these are proposed reporting tolerances, not physical user requirements, not prospective tests.

Figures: seed2026 fixed; geometry-only representatives nearest angle-stratum median (tie ID), plus worst LT bend RelL2 and largest LT-ArcOnly bend RelL2 regression. All four methods + truth, shared scales, no phase fit. Selection manifest preserved. If overlap, do not replace with more favorable cases.

Stopping: no expansion to new simulation, optimization or training. Complete the three requested artifacts and audit the input hashes.
