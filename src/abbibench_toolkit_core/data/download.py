from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_REPO_ID = "AbBibench/Antibody_Binding_Benchmark_Dataset"
DEFAULT_REVISION = "main"
DEFAULT_OUT_DIRNAME = "data"


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """
    Parameters to materialize the AbBiBench dataset repo into a local directory.

    This downloads the *dataset repository files* (like a git clone), not Arrow cache artifacts.
    """
    repo_id: str = DEFAULT_REPO_ID
    revision: str = DEFAULT_REVISION
    out_dirname: str = DEFAULT_OUT_DIRNAME
    force: bool = False
    local_dir_use_symlinks: bool = False  # prefer real files under ./data


class DownloadError(RuntimeError):
    pass


def download_hf_dataset(
    *,
    root_dir: str | Path,
    spec: Optional[DownloadSpec] = None,
) -> Path:
    """
    Download (snapshot) the AbBiBench Hugging Face dataset repository into:

        <root_dir>/<spec.out_dirname>   (default: <root_dir>/data)

    Returns:
        Path to the local directory containing the dataset repo files.

    Notes:
      - Requires: huggingface_hub (install via `abbibench-core[hf]` or add dependency yourself)
      - If the repo requires authentication:
          `huggingface-cli login`
    """
    spec = spec or DownloadSpec()
    root = Path(root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    out_dir = (root / spec.out_dirname).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise DownloadError(
            "huggingface_hub is required for dataset download. "
            "Install with `pip install huggingface_hub` or `abbibench-core[hf]`."
        ) from e

    try:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=str(out_dir),
            local_dir_use_symlinks=spec.local_dir_use_symlinks,
            force_download=spec.force,
        )
    except Exception as e:
        raise DownloadError(
            f"Failed to download dataset repo {spec.repo_id!r}@{spec.revision!r} into {out_dir}."
        ) from e

    return out_dir


