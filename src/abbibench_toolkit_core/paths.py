from __future__ import annotations

from pathlib import Path
from typing import Optional

from .datasets import resolve_dataset_id
from .contract import OUTPUT_SUFFIX


def gt_csv_path(root_dir: Path, dataset_id: str) -> Path:
    ds = resolve_dataset_id(dataset_id)
    return root_dir / "data" / "binding_affinity" / f"{ds}_benchmarking_data.csv"


def output_csv_path(
    root_dir: Path,
    dataset_id: str,
    model_id: str,
    *,
    outputs_dir: Optional[Path] = None,
) -> Path:
    """
    Standard output path expected by make_table.py / discovery:
      <root>/outputs/<DATASET>_benchmarking_data_<MODEL>_scores.csv
    """
    ds = resolve_dataset_id(dataset_id)
    outdir = outputs_dir if outputs_dir is not None else (root_dir / "outputs")
    return outdir / f"{ds}_benchmarking_data_{model_id}{OUTPUT_SUFFIX}"

