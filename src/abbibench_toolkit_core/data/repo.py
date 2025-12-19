from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------
# Dataset ID normalization (CSV-facing)
# ---------------------------------------------------------------------
#
# The AbBiBench repo may store some datasets under *_LC filenames (e.g., 1mhp_LC),
# while users/tooling may refer to them without the suffix (e.g., 1mhp).
#
# This mapping is the single source of truth for:
#   - GT CSV lookup under data/binding_affinity/
#   - output file naming under outputs/
#
# IMPORTANT:
# normalize_csv_dataset_id() lowercases the input for lookup, so keys here MUST be lowercase.
# Values are canonical on-disk dataset ids (case/suffix preserved).
#
DATASET_CSV_ALIASES: dict[str, str] = {
    # case / naming variants
    "aayl49_ml": "aayl49_ML",

    # leaderboard sometimes refers to LC sets without suffix
    "1mhp": "1mhp_LC",
    "1mhp_lc": "1mhp_LC",

    "aayl50": "aayl50_LC",
    "aayl50_lc": "aayl50_LC",

    "aayl51": "aayl51_LC",
    "aayl51_lc": "aayl51_LC",

    "aayl52": "aayl52_LC",
    "aayl52_lc": "aayl52_LC",
}


BENCHMARKING_SUFFIX = "_benchmarking_data.csv"


def normalize_csv_dataset_id(dataset_id: str) -> str:
    """
    Normalize a dataset id used for *CSV file lookup* under data/binding_affinity/.

    Examples:
      - "aayl49_ml" -> "aayl49_ML"
      - "aayl50"    -> "aayl50_LC"
      - "aayl52"    -> "aayl52_LC"
      - "1mhp"      -> "1mhp_LC"
      - "3gbn_h1"   -> "3gbn_h1"   (no change)

    Notes:
      - Lookup is case-insensitive (we lower() the key).
      - Values returned are canonical on-disk dataset ids.
    """
    ds = dataset_id.strip()
    if not ds:
        raise ValueError("dataset_id must be non-empty")
    key = ds.lower()
    return DATASET_CSV_ALIASES.get(key, ds)


def base_id_from_dataset_id(dataset_id: str) -> str:
    """
    Convert a CSV dataset id (e.g., "3gbn_h1") into its base id (e.g., "3gbn")
    for metadata/PDB lookup.

    We normalize CSV id first, then take the part before the first underscore.
    """
    ds = normalize_csv_dataset_id(dataset_id)
    return ds.split("_", 1)[0]


# ---------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ComplexMeta:
    key: str
    pdb: str
    pdb_path: Path
    heavy_chain: str
    light_chain: str
    antigen_chains: list[str]
    affinity_data: list[Path]
    receptor_chains: list[str]
    ligand_chains: list[str]
    chain_order: list[str]
    epitope_chain: str
    paratope_chain: str


class AbBiBenchRepo:
    """
    Core wrapper around a locally materialized AbBiBench dataset repo.

    Expected layout:
      <root_dir>/data/
        binding_affinity/*.csv
        complex_structure/*.pdb
        metadata.json
        ...

    This class intentionally does NOT require HF `datasets` or CLI frameworks.
    It focuses on:
      - path resolution
      - metadata access
      - GT CSV loading (pandas)
      - optional structure loading (biotite, if installed)
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.data_dir = self.root_dir / "data"
        self.binding_dir = self.data_dir / "binding_affinity"
        self.struct_dir = self.data_dir / "complex_structure"
        self.metadata_path = self.data_dir / "metadata.json"

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Expected data dir at: {self.data_dir}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json at: {self.metadata_path}")

    # -------------------------
    # metadata.json (raw + normalized)
    # -------------------------

    @cached_property
    def metadata_raw(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    @cached_property
    def metadata(self) -> dict[str, ComplexMeta]:
        out: dict[str, ComplexMeta] = {}
        for key, v in self.metadata_raw.items():
            out[key] = self._normalize_one(key, v)
        return out

    def keys(self) -> list[str]:
        return sorted(self.metadata.keys())

    def get_meta(self, key_or_base_id: str) -> ComplexMeta:
        """
        Lookup metadata by:
          1) exact key in metadata.json
          2) base-id derived from input (split at first underscore)

        This fixes the old bug where `pdb_path_for("3gbn_h1")` would fail.
        """
        k = key_or_base_id.strip()
        if not k:
            raise ValueError("key_or_base_id must be non-empty")

        if k in self.metadata:
            return self.metadata[k]

        base = k.split("_", 1)[0]
        if base in self.metadata:
            return self.metadata[base]

        raise KeyError(f"No metadata entry for: {key_or_base_id!r} (tried: {k!r}, base: {base!r})")

    # -------------------------
    # canonical paths
    # -------------------------

    def gt_csv_path(self, dataset_id: str) -> Path:
        """
        Path to <root>/data/binding_affinity/{dataset}_benchmarking_data.csv
        using CSV-facing dataset id normalization.
        """
        ds = normalize_csv_dataset_id(dataset_id)
        p = self.binding_dir / f"{ds}{BENCHMARKING_SUFFIX}"
        if not p.exists():
            raise FileNotFoundError(f"Missing GT CSV: {p}")
        return p

    def output_scores_path(self, *, dataset_id: str, model_id: str) -> Path:
        """
        Canonical output path:
          <root>/outputs/{dataset}_benchmarking_data_{model}_scores.csv

        Note: dataset_id is normalized (LC/case) before constructing filename.
        """
        ds = normalize_csv_dataset_id(dataset_id)
        return (self.root_dir / "outputs" / f"{ds}_benchmarking_data_{model_id}_scores.csv").resolve()

    def pdb_path_for(self, key_or_dataset_id: str) -> Path:
        """
        Return PDB path from metadata.json with robust base-id fallback.
        """
        return self.get_meta(key_or_dataset_id).pdb_path

    # -------------------------
    # GT loading
    # -------------------------

    def load_gt(self, dataset_id: str) -> pd.DataFrame:
        """
        Load GT CSV as pandas DataFrame.
        """
        return pd.read_csv(self.gt_csv_path(dataset_id))

    def available_gt_datasets(self) -> list[str]:
        """
        Enumerate dataset ids available under data/binding_affinity/*_benchmarking_data.csv.
        Returns the on-disk dataset ids (prefix before BENCHMARKING_SUFFIX).
        """
        if not self.binding_dir.exists():
            return []
        out: list[str] = []
        for p in sorted(self.binding_dir.glob(f"*{BENCHMARKING_SUFFIX}")):
            name = p.name
            out.append(name[: -len(BENCHMARKING_SUFFIX)])
        return out

    # -------------------------
    # optional structure loading (biotite)
    # -------------------------

    def load_structure(
        self,
        *,
        dataset_id: str,
        chains: Optional[Iterable[str]] = None,
    ):
        """
        Load structure for a dataset using biotite, selecting chains if provided.

        Requires:
          pip install biotite
        or:
          abbibench-core[structure]

        Notes:
          - Structure is resolved from metadata by base-id:
              base_id_from_dataset_id(dataset_id)
        """
        try:
            import biotite.structure.io as bsio  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "biotite is required for structure loading. "
                "Install with `pip install biotite` or `abbibench-core[structure]`."
            ) from e

        base = base_id_from_dataset_id(dataset_id)
        pdb_path = self.pdb_path_for(base)
        atom_array = bsio.load_structure(str(pdb_path))

        if chains is not None:
            chain_set = set(chains)
            mask = [c in chain_set for c in atom_array.chain_id]
            atom_array = atom_array[mask]

        return atom_array

    # -------------------------
    # internal: normalization
    # -------------------------

    def _normalize_one(self, key: str, v: dict[str, Any]) -> ComplexMeta:
        pdb_path = self._resolve_repo_relative_path(str(v["pdb_path"]))
        affinity_paths = [self._resolve_repo_relative_path(str(p)) for p in v.get("affinity_data", [])]

        return ComplexMeta(
            key=key,
            pdb=str(v["pdb"]),
            pdb_path=pdb_path,
            heavy_chain=str(v["heavy_chain"]),
            light_chain=str(v["light_chain"]),
            antigen_chains=list(v.get("antigen_chains", [])),
            affinity_data=affinity_paths,
            receptor_chains=list(v.get("receptor_chains", [])),
            ligand_chains=list(v.get("ligand_chains", [])),
            chain_order=list(v.get("chain_order", [])),
            epitope_chain=str(v.get("epitope_chain", "")),
            paratope_chain=str(v.get("paratope_chain", "")),
        )

    def _resolve_repo_relative_path(self, p: str) -> Path:
        """
        Resolve metadata paths robustly.

        Supported forms:
          - "./data/..."
          - "data/..."
          - absolute paths
          - other relative paths (treated relative to root_dir)
        """
        pp = Path(p)

        if pp.is_absolute():
            return pp.resolve()

        parts = list(pp.parts)
        # "./data/..." -> root_dir/data/...
        if len(parts) >= 2 and parts[0] == "." and parts[1] == "data":
            return (self.root_dir / Path(*parts[1:])).resolve()
        # "data/..." -> root_dir/data/...
        if len(parts) >= 1 and parts[0] == "data":
            return (self.root_dir / pp).resolve()
        # fallback: relative to root_dir
        return (self.root_dir / pp).resolve()


# ---------------------------------------------------------------------
# Lightweight dataset view (replaces old DatasetConfig)
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DatasetView:
    """
    Lightweight view for one CSV dataset id (e.g., "3gbn_h1", "2fjg", "aayl49_ML").

    - `csv_dataset_id`: user-facing dataset id
    - normalization happens via normalize_csv_dataset_id()
    - base_id is used for metadata/PDB lookup
    """
    repo: AbBiBenchRepo
    csv_dataset_id: str

    @cached_property
    def canonical_csv_id(self) -> str:
        return normalize_csv_dataset_id(self.csv_dataset_id)

    @cached_property
    def base_id(self) -> str:
        return base_id_from_dataset_id(self.csv_dataset_id)

    @cached_property
    def gt_path(self) -> Path:
        return self.repo.gt_csv_path(self.canonical_csv_id)

    def load_gt(self) -> pd.DataFrame:
        return self.repo.load_gt(self.canonical_csv_id)

    def meta(self) -> ComplexMeta:
        return self.repo.get_meta(self.base_id)

    def pdb_path(self) -> Path:
        return self.repo.pdb_path_for(self.base_id)

    def load_structure(self, *, chains: Optional[Sequence[str]] = None):
        return self.repo.load_structure(dataset_id=self.canonical_csv_id, chains=chains)

