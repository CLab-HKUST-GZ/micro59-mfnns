#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shlex
import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_SCRIPT = ROOT / "simulator/run-template/run.sh"
DEFAULT_BUILD_DIR = ROOT / "simulator/build"
DEFAULT_RAMULATOR_BIN = ROOT / "simulator/build/ramulator2"


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "case"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def parse_stat_path(yaml_path: Path) -> tuple[str | None, str | None]:
    text = yaml_path.read_text()
    match = re.search(r"^\s*stat_path:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None, "stat_path not found in yaml"
    raw = match.group(1).split("#", 1)[0].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]
    if not raw:
        return None, "stat_path is empty in yaml"
    return raw, None


def resolve_output_path(raw_path: str, workdir: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else workdir / path


def build_inner_command(args: argparse.Namespace, yaml_path: Path) -> str:
    build_dir = Path(args.build_dir).resolve() if args.build_dir else DEFAULT_BUILD_DIR.resolve()
    ramulator_bin = Path(args.ramulator_bin).resolve() if args.ramulator_bin else DEFAULT_RAMULATOR_BIN.resolve()
    parts = []
    if not args.no_module_load:
        parts.append("if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi")
        parts.append(f"module load {shlex.quote(args.gcc_module)}")
    parts.append(f"export BUILD_DIR={shlex.quote(str(build_dir))}")
    parts.append(f"export RAMULATOR_BIN={shlex.quote(str(ramulator_bin))}")
    if args.skip_build:
        parts.append("export SKIP_BUILD=1")
    parts.append(f"bash {shlex.quote(str(args.run_script.resolve()))} {shlex.quote(str(yaml_path.resolve()))}")
    return "; ".join(parts)


def build_slurm_args(args: argparse.Namespace, job_name: str) -> list[str]:
    cmd: list[str] = []
    if args.partition:
        cmd.extend(["--partition", args.partition])
    if args.time_limit:
        cmd.extend(["--time", args.time_limit])
    if args.nodes is not None:
        cmd.extend(["--nodes", str(args.nodes)])
    if args.ntasks is not None:
        cmd.extend(["--ntasks", str(args.ntasks)])
    if args.cpus_per_task is not None:
        cmd.extend(["--cpus-per-task", str(args.cpus_per_task)])
    if args.mem:
        cmd.extend(["--mem", args.mem])
    if args.nodelist:
        cmd.extend(["--nodelist", args.nodelist])
    if args.exclude:
        cmd.extend(["--exclude", args.exclude])
    if args.constraint:
        cmd.extend(["--constraint", args.constraint])
    if args.account:
        cmd.extend(["--account", args.account])
    if args.qos:
        cmd.extend(["--qos", args.qos])
    if args.exclusive:
        cmd.append("--exclusive")
    cmd.extend(["--job-name", job_name])
    return cmd


def parse_sbatch_job_id(text: str) -> str | None:
    first = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    if not first:
        return None
    if ";" in first:
        return first.split(";", 1)[0].strip()
    match = re.search(r"\b(\d+)\b", first)
    return match.group(1) if match else None


def write_case_record(path: Path, record: dict[str, str]) -> None:
    path.write_text("".join(f"{key}\t{value}\n" for key, value in record.items()))


def run_one_yaml(args: argparse.Namespace, yaml_path: Path, result_root: Path, workdir_root: Path, index: int) -> dict[str, str]:
    abs_yaml = yaml_path.resolve()
    case_name = sanitize_name(abs_yaml.stem)
    case_tag = f"{index:02d}_{case_name}"
    case_result_dir = result_root / case_tag
    case_result_dir.mkdir(parents=True, exist_ok=True)
    workdir = workdir_root / f"{case_tag}_{now_tag()}"
    workdir.mkdir(parents=True, exist_ok=True)

    wrapper_stdout_log = case_result_dir / "wrapper_stdout.log"
    case_record_path = case_result_dir / "run_record.tsv"
    stat_path_raw, parse_err = parse_stat_path(abs_yaml)
    stat_path_resolved = resolve_output_path(stat_path_raw, workdir) if stat_path_raw else None
    build_dir = Path(args.build_dir).resolve() if args.build_dir else DEFAULT_BUILD_DIR.resolve()
    ramulator_bin = Path(args.ramulator_bin).resolve() if args.ramulator_bin else DEFAULT_RAMULATOR_BIN.resolve()
    command = build_inner_command(args, abs_yaml)

    record = {
        "yaml_path": str(abs_yaml),
        "status": "PENDING",
        "return_code": "NA",
        "host": socket.gethostname(),
        "start_time": dt.datetime.now().isoformat(timespec="seconds"),
        "end_time": "NA",
        "gcc_module": args.gcc_module if not args.no_module_load else "DISABLED",
        "launcher": args.launcher,
        "run_script": str(args.run_script.resolve()),
        "build_dir": str(build_dir),
        "ramulator_bin": str(ramulator_bin),
        "workdir": str(workdir),
        "wrapper_stdout_log": str(wrapper_stdout_log),
        "inner_stdout_log": str(workdir / "stdout"),
        "launcher_command": "NA",
        "job_script": "NA",
        "slurm_job_id": "NA",
        "slurm_stdout_log": "NA",
        "slurm_stderr_log": "NA",
        "stat_path_raw": stat_path_raw or "NA",
        "stat_path_resolved": str(stat_path_resolved) if stat_path_resolved else "NA",
        "command": command,
        "note": "",
    }

    if parse_err:
        record["status"] = "YAML_PARSE_FAILED"
        record["note"] = parse_err
        record["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
        write_case_record(case_record_path, record)
        return record
    if not args.skip_build and not build_dir.exists():
        record["status"] = "MISSING_BUILD_DIR"
        record["note"] = f"BUILD_DIR not found: {build_dir}"
        record["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
        write_case_record(case_record_path, record)
        return record
    if not ramulator_bin.exists():
        record["status"] = "MISSING_RAMULATOR_BIN"
        record["note"] = f"RAMULATOR_BIN not found: {ramulator_bin}"
        record["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
        write_case_record(case_record_path, record)
        return record
    assert stat_path_resolved is not None
    stat_path_resolved.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        record["status"] = "DRY_RUN"
        record["note"] = "command not executed"
        record["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
        write_case_record(case_record_path, record)
        return record

    if args.launcher != "sbatch":
        raise ValueError("This restored runner currently supports --launcher sbatch, which is what the experiment commands use.")

    job_script = case_result_dir / "job.sbatch"
    slurm_stdout_template = case_result_dir / "slurm-%j.out"
    slurm_stderr_template = case_result_dir / "slurm-%j.err"
    job_script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(workdir))}",
                "echo \"[INFO] Host: $(hostname)\"",
                "echo \"[INFO] Start: $(date --iso-8601=seconds)\"",
                f"bash -lc {shlex.quote(command)}",
                "echo \"[INFO] End: $(date --iso-8601=seconds)\"",
                "",
            ]
        )
    )
    job_script.chmod(0o755)

    job_name = sanitize_name(f"{args.job_name_prefix}_{case_tag}")
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--output",
        str(slurm_stdout_template),
        "--error",
        str(slurm_stderr_template),
    ]
    if args.sbatch_wait:
        sbatch_cmd.append("--wait")
    sbatch_cmd.extend(build_slurm_args(args, job_name))
    sbatch_cmd.append(str(job_script))

    with wrapper_stdout_log.open("w") as fout:
        fout.write(f"[INFO] cwd={workdir}\n")
        fout.write(f"[INFO] inner_command={command}\n")
        fout.write(f"[INFO] launcher_command={shell_join(sbatch_cmd)}\n")
        fout.write(f"[INFO] job_script={job_script}\n")
        fout.flush()
        proc = subprocess.run(sbatch_cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.stdout:
            print(proc.stdout, end="")
            fout.write(proc.stdout)

    job_id = parse_sbatch_job_id(proc.stdout or "")
    record["launcher_command"] = shell_join(sbatch_cmd)
    record["job_script"] = str(job_script)
    record["slurm_job_id"] = job_id or "NA"
    record["slurm_stdout_log"] = str(slurm_stdout_template).replace("%j", job_id or "%j")
    record["slurm_stderr_log"] = str(slurm_stderr_template).replace("%j", job_id or "%j")
    record["return_code"] = str(proc.returncode)
    record["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
    if proc.returncode == 0:
        record["status"] = "SUBMITTED"
        record["note"] = "sbatch submitted; inspect slurm stdout/stderr and stat_path after completion"
    else:
        record["status"] = "RUN_FAILED"
        record["note"] = f"sbatch returned non-zero ({proc.returncode})"
    write_case_record(case_record_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simulator cases by yaml path and record stdout / stat yml locations.")
    parser.add_argument("yamls", nargs="+")
    parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--workdir-root", type=Path, default=None)
    parser.add_argument("--gcc-module", default="compilers/gcc-13.1.0")
    parser.add_argument("--ramulator-bin", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-module-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launcher", choices=["sbatch"], default="sbatch")
    parser.add_argument("--partition", default="i64m512u")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--nodelist", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--sbatch-wait", action="store_true")
    parser.add_argument("--job-name-prefix", default="runyaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.run_script.exists():
        raise FileNotFoundError(f"run script not found: {args.run_script}")
    result_root = args.result_root.resolve() if args.result_root else ROOT / "simulator/memory/run_yaml_case_results" / now_tag()
    workdir_root = args.workdir_root.resolve() if args.workdir_root else result_root / "workdirs"
    result_root.mkdir(parents=True, exist_ok=True)
    workdir_root.mkdir(parents=True, exist_ok=True)

    records = [run_one_yaml(args, Path(yaml_str), result_root, workdir_root, idx) for idx, yaml_str in enumerate(args.yamls, 1)]
    fieldnames = [
        "yaml_path",
        "status",
        "return_code",
        "host",
        "start_time",
        "end_time",
        "gcc_module",
        "launcher",
        "run_script",
        "build_dir",
        "ramulator_bin",
        "workdir",
        "wrapper_stdout_log",
        "inner_stdout_log",
        "launcher_command",
        "job_script",
        "slurm_job_id",
        "slurm_stdout_log",
        "slurm_stderr_log",
        "stat_path_raw",
        "stat_path_resolved",
        "command",
        "note",
    ]
    summary_path = result_root / "summary.tsv"
    with summary_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    print("\n=== Run Summary ===")
    print(f"result_root\t{result_root}")
    print(f"summary_tsv\t{summary_path}")
    for record in records:
        print("\t".join([record["status"], record["slurm_job_id"], record["yaml_path"], record["wrapper_stdout_log"], record["stat_path_resolved"], record["note"]]))


if __name__ == "__main__":
    main()
