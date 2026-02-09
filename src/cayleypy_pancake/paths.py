from pathlib import Path
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
ROOT = project_root()
NOTEBOOKS = ROOT / "notebooks"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
