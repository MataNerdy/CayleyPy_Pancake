from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RunPaths:
    out_dir: Path

    @property
    def progress_csv(self) -> Path:
        return self.out_dir / "submission_progress.csv"

    @property
    def final_csv(self) -> Path:
        return self.out_dir / "submission.csv"

    @property
    def baseline_csv(self) -> Path:
        return self.out_dir / "baseline_submission.csv"

    @property
    def test_csv(self) -> Path:
        return self.out_dir / "test.csv"
