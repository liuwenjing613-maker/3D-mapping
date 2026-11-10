"""Separate runner for unmodified dataset-official evaluators.

No function here imports or reuses the order-invariant Replica AP matcher.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .io import sha256_file
from .schema import EvaluationError


def _read_metrics(path: Path, spec: dict) -> dict:
    if spec["format"] == "json":
        source = json.loads(path.read_text(encoding="utf-8"))
        for key in spec.get("object_path", []):
            source = source[key]
    elif spec["format"] == "csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selector = spec.get("aggregate_row", {})
        selected = [row for row in rows if all(row.get(key) == value for key, value in selector.items())]
        if len(selected) != 1:
            raise EvaluationError("Official CSV must have exactly one selected aggregate row")
        source = selected[0]
    else:
        raise EvaluationError("Official metric format must be json or csv")
    result = {}
    for destination, source_key in spec["metric_columns"].items():
        if destination not in ("AP_official_native", "AP50_official_native", "AP25_official_native"):
            raise EvaluationError("Official metric names must use the _official_native suffix")
        value = source[source_key]
        result[destination] = None if value is None or value == "" else float(value)
        if result[destination] is not None and not math.isfinite(result[destination]):
            raise EvaluationError("Official evaluator emitted a non-finite metric")
    return result


def run_official_native(recipe_file: str | Path, output_directory: str | Path) -> dict:
    """Run an explicitly pinned official script as a subprocess and log its files.

    A recipe supplies the official script, exact argv, input paths, result file,
    metric columns and source commit. This is not a replacement AP formula.
    """
    recipe_file = Path(recipe_file)
    output = Path(output_directory)
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    dataset = recipe.get("dataset")
    if dataset not in ("ScanNet200", "ScanNet++"):
        raise EvaluationError("Official/native runner supports ScanNet200 and ScanNet++; Replica has no official 3D instance benchmark")
    script = Path(recipe["official_evaluator_script"])
    if not script.is_absolute() or not script.is_file():
        raise EvaluationError("Official evaluator script must be an existing absolute path")
    if not recipe.get("official_source_commit"):
        raise EvaluationError("Official evaluator source commit must be pinned")
    inputs = [Path(item) for item in recipe["input_files"]]
    if not inputs or any(not item.is_absolute() or not item.is_file() for item in inputs):
        raise EvaluationError("All official prediction and GT inputs must be existing absolute paths")
    result_file = Path(recipe["metric_file"])
    if not result_file.is_absolute():
        raise EvaluationError("Official metric file must be an absolute path")
    if result_file.exists():
        raise EvaluationError("Official metric output file already exists; use a fresh run directory")
    argv = recipe["argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise EvaluationError("Official evaluator argv must be a list of strings")
    workdir = Path(recipe["working_directory"])
    if not workdir.is_absolute() or not workdir.is_dir():
        raise EvaluationError("Official evaluator working directory must be an existing absolute path")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(script), *argv]
    completed = subprocess.run(command, cwd=workdir, text=True, capture_output=True, check=False)
    (output / "official_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "official_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    manifest = {
        "mode": "AP_official/native", "dataset": dataset,
        "official_source_commit": recipe["official_source_commit"],
        "official_evaluator_script": str(script), "official_evaluator_sha256": sha256_file(script),
        "recipe": str(recipe_file), "recipe_sha256": sha256_file(recipe_file),
        "command_argv": command, "working_directory": str(workdir),
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
        "returncode": completed.returncode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compatibility_status": "UNVERIFIED: official-vs-wrapper fixture comparison still required",
    }
    if completed.returncode == 0 and result_file.is_file():
        metrics = _read_metrics(result_file, recipe["metric_spec"])
        manifest["metric_file"] = str(result_file)
        manifest["metric_file_sha256"] = sha256_file(result_file)
        manifest["status"] = "OFFICIAL_SCRIPT_EXECUTED"
    else:
        metrics = {}
        manifest["status"] = "FAILED"
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if manifest["status"] == "FAILED":
        raise EvaluationError(f"Official evaluator failed; see {output / 'official_stderr.txt'}")
    return metrics
