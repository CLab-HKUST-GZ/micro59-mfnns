#!/usr/bin/env python3
"""Select and optionally submit Figure 18 ANSMET/MFNNS YAMLs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROVENANCE = SCRIPT_DIR / "data/simulator_provenance.tsv"
RUNNER = REPO_ROOT / "simulator/memory/run_yaml_case.py"
VERIFIED_STATUS = "verified_original_yaml_stats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--dataset", choices=["deep1b", "t2i1b"])
    parser.add_argument("--recall-tag", choices=["r10", "r100"])
    parser.add_argument("--method", choices=["ansmet", "mfnns"])
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select all configurations allowed by the other filters.",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include reconstructed/cycle-mismatch rerun recipes.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit with the repository runner; otherwise only print commands.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        help="Required with --submit; must be memory/YYYYMMDD/NNN_name.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Prepared HNSW index; required with --submit.",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        help="Prepared query file; required with --submit.",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        help="Prepared ground-truth file; required with --submit.",
    )
    parser.add_argument("--partition", default="i96m3tu")
    parser.add_argument("--mem", default="1600G")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--time-limit", default="12:00:00")
    return parser.parse_args()


def load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 108:
        raise ValueError(f"Expected 108 simulator rows, found {len(rows)}")
    return rows


def select(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.all and not any((args.dataset, args.recall_tag, args.method)):
        raise ValueError(
            "Refusing an implicit full run: pass a filter or explicitly pass --all"
        )
    selected = []
    for row in rows:
        if args.dataset and row["dataset"] != args.dataset:
            continue
        if args.recall_tag and row["recall_tag"] != args.recall_tag:
            continue
        if args.method and row["method"] != args.method:
            continue
        if not args.include_unverified and row["config_status"] != VERIFIED_STATUS:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No Figure 18 configurations match the selection")
    return selected


def validate_result_root(path: Path) -> Path:
    resolved = path.resolve()
    memory_root = (REPO_ROOT / "memory").resolve()
    try:
        relative = resolved.relative_to(memory_root)
    except ValueError as exc:
        raise ValueError(f"Result root must be under {memory_root}: {resolved}") from exc
    parts = relative.parts
    if len(parts) != 2:
        raise ValueError("Result root must have form memory/YYYYMMDD/NNN_name")
    if not re.fullmatch(r"[0-9]{8}", parts[0]):
        raise ValueError("Result root date must use YYYYMMDD")
    if not re.fullmatch(r"[0-9]{3}_.+", parts[1]):
        raise ValueError("Result directory must start with a three-digit sequence")
    return resolved


def replace_path(text: str, key: str, value: Path | str) -> str:
    pattern = re.compile(
        rf"^(\s{{2}}{re.escape(key)}:)\s*.*$",
        flags=re.MULTILINE,
    )
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)} {json.dumps(str(value))}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"Expected one top-level key {key}")
    return updated


def validate_inputs(args: argparse.Namespace) -> None:
    for path, flag, label in (
        (args.model_path, "--model-path", "model"),
        (args.query_path, "--query-path", "query"),
        (args.gt_path, "--gt-path", "ground truth"),
    ):
        if path is None:
            raise ValueError(f"{flag} is required with --submit")
        if not path.resolve().is_file():
            raise FileNotFoundError(f"Missing {label} input: {path.resolve()}")


def prepare_configs(
    selected: list[dict[str, str]],
    args: argparse.Namespace,
    result_root: Path,
) -> list[Path]:
    runtime_dir = result_root / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=False)
    configs: list[Path] = []
    for row in selected:
        source = (REPO_ROOT / row["portable_config_ref"]).resolve()
        method_dir = runtime_dir / row["method"]
        method_dir.mkdir(parents=True, exist_ok=True)
        destination = method_dir / source.name
        text = source.read_text(encoding="utf-8")
        text = replace_path(text, "model_path", args.model_path.resolve())
        text = replace_path(text, "query_path", args.query_path.resolve())
        text = replace_path(text, "gt_path", args.gt_path.resolve())
        text = replace_path(
            text,
            "stat_path",
            f"stats/{row['method']}/{source.stem}_stats.yml",
        )
        destination.write_text(text, encoding="utf-8")
        configs.append(destination)
    return configs


def build_command(
    configs: list[Path],
    args: argparse.Namespace,
    result_root: Path | None,
) -> list[str]:
    command = [
        "python3",
        str(RUNNER),
        *[str(path) for path in configs],
        "--skip-build",
        "--partition",
        args.partition,
        "--mem",
        args.mem,
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--time-limit",
        args.time_limit,
        "--job-name-prefix",
        "fig18",
    ]
    if result_root is not None:
        command.extend(
            [
                "--result-root",
                str(result_root / "runs"),
                "--workdir-root",
                str(result_root / "workdirs"),
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    rows = load(args.provenance)
    selected = select(rows, args)
    result_root = None
    if args.submit:
        if args.result_root is None:
            raise ValueError("--result-root is required with --submit")
        if args.dataset is None or args.recall_tag is None:
            raise ValueError(
                "--dataset and --recall-tag are required with --submit so one "
                "model/query/ground-truth set cannot be applied across panels"
            )
        validate_inputs(args)
        result_root = validate_result_root(args.result_root)
        result_root.mkdir(parents=True, exist_ok=False)

    verified = sum(row["config_status"] == VERIFIED_STATUS for row in selected)
    print(
        f"Selected {len(selected)} configs "
        f"({verified} verified, {len(selected)-verified} rerun-only)"
    )
    if not args.submit:
        for row in selected[:10]:
            print(row["portable_config_ref"])
        if len(selected) > 10:
            print(f"... {len(selected) - 10} more")
        print(
            "Read-only selection only; pass --submit, the three input paths, "
            "and --result-root to create runtime YAMLs and submit."
        )
        return

    configs = prepare_configs(selected, args, result_root)
    command = build_command(configs, args, result_root)
    print(shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
