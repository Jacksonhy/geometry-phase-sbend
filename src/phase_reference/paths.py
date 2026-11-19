from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PICFLOW = ROOT / "third_party/PIC-Flow"
METHODS = ["linear", "arc_only", "local_tangent", "path_v2"]
SEEDS = [42, 2026, 2027]
