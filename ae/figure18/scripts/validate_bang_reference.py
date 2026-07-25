#!/usr/bin/env python3
"""Validate the synchronized Deep1B BANG reference and its paper relationship."""

from __future__ import annotations

import csv
from collections import Counter
from decimal import Decimal
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
FIGURE_DATA = FIGURE_DIR / "data" / "figure18_recall_qps.csv"
REFERENCE = FIGURE_DIR / "data" / "expected_deep1b_bang_curve.tsv"


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    figure_rows = [
        row
        for row in read(FIGURE_DATA, ",")
        if row["dataset"] == "deep1b" and row["method"] == "bang"
    ]
    reference_rows = read(REFERENCE, "\t")
    if len(figure_rows) != 18 or len(reference_rows) != 18:
        raise ValueError(
            f"Expected 18 Deep1B BANG rows, got "
            f"figure={len(figure_rows)}, reference={len(reference_rows)}"
        )

    counts = Counter(int(row["k"]) for row in reference_rows)
    if counts != Counter({10: 10, 100: 8}):
        raise ValueError(f"Unexpected BANG reference counts: {dict(counts)}")

    expected_l = {
        10: [10, 15, 20, 30, 40, 60, 80, 160, 320, 512],
        100: [100, 130, 160, 200, 240, 280, 320, 512],
    }
    for k, expected in expected_l.items():
        actual_l = [int(row["L"]) for row in reference_rows if int(row["k"]) == k]
        if actual_l != expected:
            raise ValueError(
                f"Unexpected Deep1B BANG L grid for k={k}: {actual_l}"
            )

    exact_qps_matches = 0
    for figure, reference in zip(figure_rows, reference_rows):
        reference_tag = "r10" if int(reference["k"]) == 10 else "r100"
        if figure["recall_tag"] != reference_tag:
            raise ValueError("Deep1B BANG metric order mismatch")
        figure_recall = Decimal(figure["recall_raw"])
        reference_recall = Decimal(reference["recall_percent"]) / Decimal(100)
        if abs(figure_recall - reference_recall) > Decimal("0.0001"):
            raise ValueError(
                "Deep1B BANG recall grid differs by more than 0.0001: "
                f"figure={figure_recall}, reference={reference_recall}"
            )
        exact_qps_matches += (
            Decimal(figure["qps"]) == Decimal(reference["median_last3_qps"])
        )

    if exact_qps_matches != 3:
        raise ValueError(
            f"Expected 3 exact paper/reference QPS anchors, got {exact_qps_matches}"
        )
    print(
        "Validated Deep1B BANG reference: 18 compatible-recall rows, "
        "3 exact QPS anchors; reference does not replace frozen paper QPS."
    )


if __name__ == "__main__":
    main()
