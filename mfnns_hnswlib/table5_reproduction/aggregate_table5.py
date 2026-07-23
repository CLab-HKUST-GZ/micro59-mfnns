#!/usr/bin/env python3
"""Assemble per-dataset measurements into the layout used by paper Table 5."""

import argparse
import csv
from pathlib import Path


DATASETS = [
    ("T2I", "t2i1m", "normalized"),
    ("WK", "wiki1m", "normalized"),
    ("GV", "glove2m", "normalized"),
    ("DP", "deep10m", "normalized"),
    ("PM", "pubmed", "raw"),
    ("W2V", "w2v1m", "normalized"),
    ("SF", "sift1m", "normalized"),
]

ROWS = [
    ("Recall", "FP16", "fp16_recall", 2, True),
    ("Recall", "FPMA", "fpma_recall", 2, True),
    ("Recall", "FP16+ET", "fp16_et_recall", 2, True),
    ("Recall", "FPMA+ET", "fpma_et_recall", 2, True),
    ("ET Ratio", "FP16+ET", "fp16_et_ratio_pct", 2, False),
    ("ET Ratio", "FPMA+ET", "fpma_et_ratio_pct", 2, False),
    ("FPMA Influence", "Risk Update", "risk_update_pct", 2, False),
    ("FPMA Influence", "Flip Ratio", "flip_ratio_pct", 3, False),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--details-csv", required=True)
    return parser.parse_args()


def load_rows(input_dir):
    loaded = {}
    for _, dataset, variant in DATASETS:
        path = input_dir / f"{dataset}_{variant}.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"{path} must contain exactly one data row")
        row = rows[0]
        key = (row["dataset"], row["variant"])
        if key != (dataset, variant):
            raise ValueError(f"{path} identifies {key}, expected {(dataset, variant)}")
        loaded[key] = row

    configs = {
        (
            row["query_count"],
            row["k"],
            row["ef_search"],
            row["risk_ratio_upper"],
        )
        for row in loaded.values()
    }
    if len(configs) != 1:
        raise ValueError(f"Per-dataset configurations differ: {sorted(configs)}")
    return loaded


def format_percent(row, field, digits, fraction):
    value = float(row[field])
    if fraction:
        value *= 100.0
    return f"{value:.{digits}f}%"


def write_details(path, loaded):
    first = next(iter(loaded.values()))
    fieldnames = list(first)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for _, dataset, variant in DATASETS:
            writer.writerow(loaded[(dataset, variant)])


def table_rows(loaded):
    output = []
    for metric, variant_name, field, digits, fraction in ROWS:
        row = [metric, variant_name]
        for _, dataset, variant in DATASETS:
            row.append(
                format_percent(
                    loaded[(dataset, variant)], field, digits, fraction
                )
            )
        output.append(row)
    return output


def write_table_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Metric", "Variant"] + [item[0] for item in DATASETS])
        writer.writerows(rows)


def write_markdown(path, rows, loaded):
    sample = next(iter(loaded.values()))
    with path.open("w") as handle:
        handle.write("# Reproduced Table 5\n\n")
        handle.write(
            f"- queries: `{sample['query_count']}`\n"
            f"- k: `{sample['k']}`\n"
            f"- efSearch: `{sample['ef_search']}`\n"
            f"- risk condition: "
            f"`1 < D_b_exact / D_c_exact < {sample['risk_ratio_upper']}`\n\n"
        )
        handle.write("| Metric | Variant | " + " | ".join(item[0] for item in DATASETS) + " |\n")
        handle.write("| --- | --- | " + " | ".join("---:" for _ in DATASETS) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row) + " |\n")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    details_csv = Path(args.details_csv)
    for path in (output_csv, output_md, details_csv):
        path.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_rows(input_dir)
    rows = table_rows(loaded)
    write_details(details_csv, loaded)
    write_table_csv(output_csv, rows)
    write_markdown(output_md, rows, loaded)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")
    print(f"Wrote {details_csv}")


if __name__ == "__main__":
    main()
