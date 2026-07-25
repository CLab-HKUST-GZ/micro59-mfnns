#!/usr/bin/env python3
"""Select and optionally submit Figure 21 simulator configurations."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROVENANCE = SCRIPT_DIR / "data/simulator_provenance.tsv"
RUNNER = REPO_ROOT / "simulator/memory/run_yaml_case.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["mfnns", "ansmet"])
    parser.add_argument("--ef", type=int, choices=[20, 30, 40])
    parser.add_argument("--queue", type=int)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select all configurations allowed by the other filters.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create runtime YAMLs and submit them; default is read-only.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        help="Required with --submit; must be memory/YYYYMMDD/NNN_name.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT
        / "mfnns_hnswlib/cpu_index/t2i1m/hnsw_index_M32_ef100.bin",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=SCRIPT_DIR / "inputs/query_vectors_n100_seed42.bin",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=SCRIPT_DIR / "inputs/gt_labels_topk32_n100_seed42.bin",
    )
    parser.add_argument("--partition", default="i64m512u")
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--exclude")
    return parser.parse_args()


def read_provenance() -> list[dict[str, str]]:
    with PROVENANCE.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 246:
        raise ValueError(f"Expected 246 provenance rows, found {len(rows)}")
    return rows


def select(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.all and not any((args.method, args.ef, args.queue is not None)):
        raise ValueError(
            "Refusing an implicit 246-case run: pass a filter or explicitly pass --all"
        )
    selected = []
    for row in rows:
        if args.method and row["method"] != args.method:
            continue
        if args.ef is not None and int(row["ef_search"]) != args.ef:
            continue
        if args.queue is not None and int(row["queue_size"]) != args.queue:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No Figure 21 configurations match the selection")
    return selected


def validate_result_root(path: Path) -> Path:
    resolved = path.resolve()
    memory_root = (REPO_ROOT / "memory").resolve()
    try:
        relative = resolved.relative_to(memory_root)
    except ValueError as exc:
        raise ValueError(f"Result root must be under {memory_root}: {resolved}") from exc
    if len(relative.parts) != 2:
        raise ValueError("Result root must have form memory/YYYYMMDD/NNN_name")
    if not re.fullmatch(r"[0-9]{8}", relative.parts[0]):
        raise ValueError("Result root date must use YYYYMMDD")
    if not re.fullmatch(r"[0-9]{3}_.+", relative.parts[1]):
        raise ValueError("Result directory must start with a three-digit sequence")
    return resolved


def replace_path(text: str, key: str, value: Path | str) -> str:
    pattern = re.compile(
        rf"^(\s{{2}}{re.escape(key)}:)\s*.*$",
        flags=re.MULTILINE,
    )
    updated, count = pattern.subn(rf"\1 {value}", text, count=1)
    if count != 1:
        raise ValueError(f"Expected one top-level key {key}")
    return updated


def validate_inputs(args: argparse.Namespace) -> None:
    for path, label in [
        (args.model_path, "model"),
        (args.query_path, "query"),
        (args.gt_path, "ground truth"),
    ]:
        if not path.resolve().is_file():
            raise FileNotFoundError(f"Missing {label} input: {path.resolve()}")


def prepare_configs(
    selected: list[dict[str, str]],
    args: argparse.Namespace,
    result_root: Path,
) -> list[Path]:
    runtime_dir = result_root / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for row in selected:
        source = REPO_ROOT / row["config_ref"]
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
        paths.append(destination)
    return paths


def build_command(configs: list[Path], args: argparse.Namespace, result_root: Path) -> list[str]:
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
        "fig21",
        "--result-root",
        str(result_root / "runs"),
        "--workdir-root",
        str(result_root / "workdirs"),
    ]
    if args.exclude:
        command.extend(["--exclude", args.exclude])
    return command


def main() -> None:
    args = parse_args()
    selected = select(read_provenance(), args)
    method_counts = {
        method: sum(row["method"] == method for row in selected)
        for method in ("mfnns", "ansmet")
    }
    print(
        f"Selected {len(selected)} configs "
        f"(MFNNS={method_counts['mfnns']}, ANSMET={method_counts['ansmet']})"
    )
    if not args.submit:
        for row in selected[:10]:
            print(row["config_ref"])
        if len(selected) > 10:
            print(f"... {len(selected) - 10} more")
        print("Read-only selection only; pass --submit and --result-root to submit.")
        return

    if args.result_root is None:
        raise ValueError("--result-root is required with --submit")
    result_root = validate_result_root(args.result_root)
    validate_inputs(args)
    result_root.mkdir(parents=True, exist_ok=False)
    configs = prepare_configs(selected, args, result_root)
    command = build_command(configs, args, result_root)
    print(shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
