from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Defaults aligned with AbBiBench-style outputs
# ---------------------------------------------------------------------

DEFAULT_GT_COLUMN = "binding_score"
DEFAULT_SCORE_COLUMN = "log-likelihood"

# Common identifier columns across AbBiBench datasets / outputs
DEFAULT_ID_CANDIDATES: tuple[str, ...] = (
    "heavy_chain_seq",
    "light_chain_seq",
    "mut_heavy_chain_seq",
    "mut_light_chain_seq",
    "antigen_seq",
    "mut_antigen_seq",
)

EvalEngine = Literal["pandas", "scipy"]
EvalMethod = Literal["direct", "merge", "occ_merge", "row_order"]


# ---------------------------------------------------------------------
# Result / Spec
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvalSpec:
    """
    Evaluation specification.

    - engine="pandas" replicates the official script behavior:
        pd.concat([x, y], axis=1).corr(method="spearman").iloc[0, 1]
      (No p-value available from this engine.)
    - engine="scipy" uses scipy.stats.spearmanr and returns p-value as well.

    - If common ID columns exist in GT & PRED:
        merge on those columns.
      If duplicates exist:
        add occurrence index per key group (cumcount) and merge on keys+occ.
      Otherwise:
        fallback to row-order evaluation (if allowed).
    """
    gt_column: str = DEFAULT_GT_COLUMN
    score_column: str = DEFAULT_SCORE_COLUMN
    id_candidates: Sequence[str] = DEFAULT_ID_CANDIDATES

    # prefer official-repro behavior by default
    engine: EvalEngine = "pandas"

    # drop rows where either gt or pred score is NaN
    dropna_pairs: bool = True

    # if no keys in common, allow row-order evaluation
    allow_row_order_fallback: bool = True

    # Optional score transform (e.g., negate FoldX / epitopeSA outputs)
    score_transform: Optional[Callable[[pd.Series], pd.Series]] = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    method: EvalMethod
    rho: float
    p_value: Optional[float]
    n: int

    # bookkeeping for debugging
    key_cols: tuple[str, ...]
    n_gt: int
    n_pred: int
    n_merged: int

    # extra hints
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------

def _select_common_keys(gt: pd.DataFrame, pred: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [c for c in candidates if c in gt.columns and c in pred.columns]


def _add_occurrence(df: pd.DataFrame, keys: Sequence[str], occ_col: str = "__occ") -> pd.DataFrame:
    # Preserve original row order; sort=False keeps group order stable.
    out = df.copy()
    out[occ_col] = out.groupby(list(keys), sort=False).cumcount()
    return out


def _compute_spearman(
    x: pd.Series,
    y: pd.Series,
    *,
    engine: EvalEngine,
    dropna_pairs: bool,
) -> tuple[float, Optional[float], int]:
    if dropna_pairs:
        m = x.notna() & y.notna()
        x = x[m]
        y = y[m]

    n = int(len(x))
    if n == 0:
        return float("nan"), None, 0

    if engine == "pandas":
        rho = float(pd.concat([x, y], axis=1).corr(method="spearman").iloc[0, 1])
        return rho, None, n

    # scipy engine
    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scipy is required for engine='scipy'. Install scipy or switch engine='pandas'."
        ) from e

    rho, p = spearmanr(x.to_numpy(), y.to_numpy())
    return float(rho), float(p), n


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def evaluate_frames(
    *,
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    spec: EvalSpec = EvalSpec(),
) -> EvalResult:
    """
    Evaluate Spearman correlation between GT binding_score and predicted score_column.

    - Always uses GT's gt_column as the ground truth source.
    - Uses PRED's score_column as the predicted values source.
    - Merge strategy:
        1) If common keys exist: merge on keys.
        2) If duplicates exist on keys in either side: occ_merge on keys+occ.
        3) Else if no keys and allowed: row_order alignment.
    """
    notes: list[str] = []

    if spec.gt_column not in gt.columns:
        raise KeyError(f"GT is missing required column {spec.gt_column!r}. cols={list(gt.columns)}")
    if spec.score_column not in pred.columns:
        raise KeyError(f"PRED is missing required column {spec.score_column!r}. cols={list(pred.columns)}")

    x_gt = gt[spec.gt_column]
    y_pred_raw = pred[spec.score_column]

    if spec.score_transform is not None:
        y_pred = spec.score_transform(y_pred_raw)
        notes.append("score_transform_applied")
    else:
        y_pred = y_pred_raw

    key_cols = _select_common_keys(gt, pred, spec.id_candidates)

    # 0) Optional fast path: if pred already includes gt_column and no merge is desired.
    # We intentionally DO NOT do this by default because it can mask alignment issues.
    # (Keep method="direct" for explicit usage in the future.)

    # 1) Merge-based evaluation when keys exist
    if key_cols:
        gt_dup = int(gt.duplicated(key_cols).sum())
        pred_dup = int(pred.duplicated(key_cols).sum())

        if gt_dup == 0 and pred_dup == 0:
            merged = gt[key_cols + [spec.gt_column]].merge(
                pred[key_cols + [spec.score_column]],
                on=key_cols,
                how="inner",
            )
            if len(merged) != len(gt) or len(merged) != len(pred):
                notes.append("inner_merge_rowcount_mismatch")

            rho, p, n = _compute_spearman(
                merged[spec.gt_column],
                merged[spec.score_column] if spec.score_transform is None else spec.score_transform(merged[spec.score_column]),
                engine=spec.engine,
                dropna_pairs=spec.dropna_pairs,
            )
            return EvalResult(
                method="merge",
                rho=rho,
                p_value=p,
                n=n,
                key_cols=tuple(key_cols),
                n_gt=len(gt),
                n_pred=len(pred),
                n_merged=len(merged),
                notes=tuple(notes),
            )

        # 2) occurrence merge when duplicates exist
        notes.append(f"duplicate_keys_detected(gt={gt_dup},pred={pred_dup})")
        gt2 = _add_occurrence(gt[key_cols + [spec.gt_column]], key_cols)
        pred2 = _add_occurrence(pred[key_cols + [spec.score_column]], key_cols)

        merged = gt2.merge(pred2, on=list(key_cols) + ["__occ"], how="inner")
        if len(merged) != len(gt) or len(merged) != len(pred):
            notes.append("occ_merge_rowcount_mismatch")

        rho, p, n = _compute_spearman(
            merged[spec.gt_column],
            merged[spec.score_column] if spec.score_transform is None else spec.score_transform(merged[spec.score_column]),
            engine=spec.engine,
            dropna_pairs=spec.dropna_pairs,
        )
        return EvalResult(
            method="occ_merge",
            rho=rho,
            p_value=p,
            n=n,
            key_cols=tuple(key_cols),
            n_gt=len(gt),
            n_pred=len(pred),
            n_merged=len(merged),
            notes=tuple(notes),
        )

    # 3) Row-order fallback if no keys
    if not spec.allow_row_order_fallback:
        raise ValueError(
            "No common identifier columns between GT and PRED, and row-order fallback is disabled. "
            f"GT cols={list(gt.columns)} PRED cols={list(pred.columns)}"
        )

    if len(gt) != len(pred):
        raise ValueError(
            "Row-order evaluation requires equal lengths but got "
            f"len(gt)={len(gt)} len(pred)={len(pred)}"
        )

    rho, p, n = _compute_spearman(x_gt, y_pred, engine=spec.engine, dropna_pairs=spec.dropna_pairs)
    notes.append("row_order_alignment")
    return EvalResult(
        method="row_order",
        rho=rho,
        p_value=p,
        n=n,
        key_cols=tuple(),
        n_gt=len(gt),
        n_pred=len(pred),
        n_merged=len(gt),
        notes=tuple(notes),
    )


def evaluate_csvs(
    *,
    gt_csv: str,
    pred_csv: str,
    spec: EvalSpec = EvalSpec(),
    pandas_read_kwargs: Optional[dict] = None,
) -> EvalResult:
    """
    Convenience wrapper around evaluate_frames() for two CSV files.
    """
    pandas_read_kwargs = pandas_read_kwargs or {}
    gt = pd.read_csv(gt_csv, **pandas_read_kwargs)
    pred = pd.read_csv(pred_csv, **pandas_read_kwargs)
    return evaluate_frames(gt=gt, pred=pred, spec=spec)

