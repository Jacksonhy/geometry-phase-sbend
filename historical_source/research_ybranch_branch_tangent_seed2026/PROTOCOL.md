# Y-branch BranchTangent seed2026 frozen protocol
Date: 2026-11-13 Asia/Shanghai. User authorized one new seed2026 BranchTangent comparison.
Historical methods: Linear, Branch Path; new method: BranchTangent.
Fixed historical 64 train, 8 validation, E2/M4 100+100 previously-used benchmark geometries.
No FDTD, new data, input channels, architecture, hyperparameter search or phase alignment.
Existing smooth upper/lower centerlines and midpoint branch assignment are retained exactly.
Let b(x,y) be the historical branch, y_b(x) its centerline in pixel coordinates.
Let t_by = y_b'(x)/sqrt(1+y_b'(x)^2), using np.gradient(center,edge_order=1).
Let q=+1 for left trunk excitation, q=-1 for right excitation except q=+1 on the other
branch strictly beyond the historical junction column. This is the derivative sign of
the existing route arc, NOT one globally constant sign for all right-excited branches.
Raw transverse correction u=q*t_by*(y-y_b(x)); subtract bilinear u at historical
source-mask centroid (same source interpolation convention used in M4), yielding u0.
d_BT = existing float32 d_Path promoted to float64 + u0. c_BT=exp(i*k*d_BT), complex64.
k is historical training-calibrated rad/pixel; do not multiply by physical grid size again.
At the source centroid interpolated u0=0, retaining historical Path source phase.
No test-dependent smoothing, soft branch mixtures, nearest-point projection or v2 kernel.
Branch/junction hard assignment is preserved and can create phase seams; diagnose its
distance/carrier jumps on train/validation, report rather than tune or silently smooth.
Discontinuity is not asserted absent; only bounded finite geometric carrier is required.
Independent synthetic oracle checks straight slanted branches, source anchoring and all
three excitation routes, including positive outgoing other-branch tangent sign.
Common inputs: exactly historical ten real features packed as six complex channels;
retains existing NeurOLight exp wave prior, real degrees 2,944,738.
Training loop reused unchanged from research_e3_2400/train.py via isolated adapter:
seed2026, initialization and PCG64 batch hashes identical; 2400 steps, batch2, FP32/TF32;
AdamW lr0.002 wd0.0001 clip5, cosine eta_min0.0001, dropout0.1, no augmentation.
Original 8 validation only, step1/every50; choose minimum validation mean unaligned RelL2.
Save best and last full restorable checkpoints. All writes isolated in this directory.
Preflight: source/weight hashes, exact calibration and input replay, four historical
best/last validation per-sample replays, unit modulus, routing/source oracle, finite
forward/backward without updates. Reset full model seed at new training start.
No E2/M4 truth field used in preflight/training. Retrospective benchmark status disclosed.
After training freeze weights then evaluate E2/M4 new model best/last. Reuse historical
E2 best Linear/Path metrics/predictions and M4 best/last; E2 old last predictions were not
saved, so generate E2 Linear/Path last once in this isolated evaluation for sensitivity.
All method predictions are physical complex Ez, no truth phase/amplitude alignment.
Primary comparison Path vs BranchTangent, best checkpoints. Also Linear vs each.
Report per-geometry complex RelL2; mean/median/p90, each of ports1,2,3 and equal-port macro.
Secondary amplitude RelL2, total intensity ratio and intensity-weighted phase MAE,
identical to historical metric_arrays; these are field metrics, not certified S/power.
10,000 paired geometry bootstrap PCG64(2026), overall and each port. Exploratory 95%
percentile CIs, conditional on trained models, no multiplicity correction, no training
seed CI or formal equivalence. Do not pool E2/M4 as independent model repetitions.
Retain adverse geometries and best/last disagreements. No post-test tuning.
At finish independent saved-array arithmetic verification and protected-file hash check.
