from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence


RunnerKind = Literal["uvtool", "uvrun", "conda"]


@dataclass(frozen=True)
class RunnerSpec:
    kind: RunnerKind = "uvtool"
    # Optional fields for conda wrappers (if you later need them)
    env_name: Optional[str] = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    type: str
    package: str
    entrypoint: str
    runner: RunnerSpec = RunnerSpec()
    args: dict[str, Any] = None  # extra CLI args for the model package


@dataclass(frozen=True)
class RunSpec:
    root_dir: Path
    outputs_dir: Path
    datasets: Sequence[str]
    models: Sequence[ModelSpec]

    @staticmethod
    def load(path: Path) -> "RunSpec":
        obj = json.loads(path.read_text(encoding="utf-8"))

        def _req(k: str):
            if k not in obj:
                raise KeyError(f"Missing required field in spec: {k}")
            return obj[k]

        models: list[ModelSpec] = []
        for m in _req("models"):
            r = m.get("runner", {}) or {}
            runner = RunnerSpec(kind=r.get("kind", "uvtool"), env_name=r.get("env_name"))
            models.append(
                ModelSpec(
                    id=m["id"],
                    type=m.get("type", "Other"),
                    package=m["package"],
                    entrypoint=m["entrypoint"],
                    runner=runner,
                    args=m.get("args") or {},
                )
            )

        return RunSpec(
            root_dir=Path(_req("root_dir")),
            outputs_dir=Path(_req("outputs_dir")),
            datasets=list(_req("datasets")),
            models=models,
        )

