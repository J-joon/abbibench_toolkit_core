from __future__ import annotations

from typing import Final


# Dataset aliases as observed in your repo/tooling.
# If users pass "aayl50", on disk it's "aayl50_LC" (same for aayl52).
DATASET_ALIASES: Final[dict[str, str]] = {
    "aayl50": "aayl50_LC",
    "aayl52": "aayl52_LC",
}

# Reverse for display (optional)
DATASET_DISPLAY: Final[dict[str, str]] = {
    "aayl50_LC": "aayl50",
    "aayl52_LC": "aayl52",
}


def resolve_dataset_id(dataset_id: str) -> str:
    """
    Map user-facing dataset id to the on-disk dataset id used for GT filenames.
    """
    return DATASET_ALIASES.get(dataset_id, dataset_id)


def display_dataset_name(dataset_id: str) -> str:
    """
    Map on-disk dataset id to user-facing display name.
    """
    return DATASET_DISPLAY.get(dataset_id, dataset_id)

