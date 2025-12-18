from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------
# Canonical AbBiBench-style contract constants
# ---------------------------------------------------------------------

GT_COLUMN: Final[str] = "binding_score"

# Default score column name used by most model outputs
SCORE_COLUMN_DEFAULT: Final[str] = "log-likelihood"

# Toolkit discovery expects *_scores.csv
OUTPUT_SUFFIX: Final[str] = "_scores.csv"

# Columns that may serve as stable identifiers for merge.
# Not all datasets have all columns.
ID_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "heavy_chain_seq",
    "light_chain_seq",
    "mut_heavy_chain_seq",
    "mut_light_chain_seq",
    "antigen_seq",
    "mut_antigen_seq",
)

# Some “metric” models use different score column names (from the official script)
SPECIAL_SCORE_COLUMNS: Final[dict[str, str]] = {
    "FoldX": "dg",
    "epitopeSA": "EpitopeSASA (mut)",
}

# Models whose correlation should be computed with NEGATED values (official script behavior)
NEGATE_MODELS: Final[set[str]] = {"FoldX", "epitopeSA"}


@dataclass(frozen=True)
class Contract:
    """
    A lightweight container for contract defaults.
    """
    gt_column: str = GT_COLUMN
    score_column_default: str = SCORE_COLUMN_DEFAULT

