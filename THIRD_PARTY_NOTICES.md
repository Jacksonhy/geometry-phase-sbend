# Third-party notices

## PIC-Flow

`third_party/PIC-Flow/FDTD/` contains source and material tables copied from the research copy of PIC-Flow. Copyright (c) 2026 Joseph Quaratiello and Anthony Rizzo. MIT License: [`third_party/PIC-Flow/LICENSE`](third_party/PIC-Flow/LICENSE).

The Euler constructor, overlapping-block discretization, source and port definitions, and effective-index tables are retained. Publication wrappers initialize this material/source grid without advancing Meep time during inference preprocessing. Meep itself is an external dependency and is not vendored here.

## NeurOLight

`third_party/NeurOLight/core/models/` contains the four required model source files copied from the research copy of NeurOLight. Copyright (c) 2022 Jiaqi Gu. MIT License: [`third_party/NeurOLight/LICENSE`](third_party/NeurOLight/LICENSE).

The compatibility code in `src/phase_reference/compat/` loads original class/function bodies by AST and supplies local dependency names. The exercised network uses GELU and zero stochastic depth. The compatibility replacements are documented in the code; this is not a claim that the complete original legacy dependency stack is reproduced.

## Other dependencies

NumPy, SciPy, PyTorch, and Meep are installed separately under their own licenses. Pandoc and Chrome/Chromium are optional document-export dependencies. None of their executables is included in this repository.
