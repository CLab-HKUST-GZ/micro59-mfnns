#!/usr/bin/env python3
import csv
import math
import os
import statistics
from collections import defaultdict

ROOT = "/hpc2hdd/home/rmeng603/workspace/CPU_HNSW"
MEMDIR = os.path.join(ROOT, "memory/20260615/004_hnswlib_cpu_fresh_qps_recall")
RESULTS = os.path.join(MEMDIR, "results")
PLOTS = os.path.join(MEMDIR, "plots")

SOURCES = [
    ("deep1b", "R@10", os.path.join(RESULTS, "deep1b_r10_fresh_n100.tsv")),
    ("deep1b", "R@100", os.path.join(RESULTS, "deep1b_r100_fresh_n100.tsv")),
    ("deep1b", "R@100", os.path.join(RESULTS, "deep1b_r100_fresh_n100_extra_ef1000.tsv")),
    ("t2i1b_norml2", "R@10", os.path.join(RESULTS, "t2i1b_norml2_r10_fresh_n100.tsv")),
    ("t2i1b_norml2", "R@100", os.path.join(RESULTS, "t2i1b_norml2_r100_fresh_n100.tsv")),
]

RAW_FIELDS = [
    "dataset", "metric", "k", "ef", "trial", "recall", "qps", "threads", "nq",
    "query_ops", "warmup_queries", "avg_results_per_query", "source_file",
]

SUMMARY_FIELDS = [
    "dataset", "metric", "k", "ef", "trials", "recall_median", "recall_mean",
    "qps_median", "qps_mean", "qps_std", "qps_min", "qps_max",
    "avg_results_per_query_median", "threads", "nq", "query_ops", "warmup_queries",
    "source_file",
]


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    val = as_float(value)
    if val is None:
        return None
    return int(val)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.9g}"
    return str(value)


def read_raw():
    rows = []
    for dataset, metric, path in SOURCES:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for rec in reader:
                ef = as_int(rec.get("ef"))
                recall = as_float(rec.get("recall"))
                qps = as_float(rec.get("qps"))
                if ef is None or recall is None or qps is None:
                    continue
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "k": as_int(rec.get("k")),
                    "ef": ef,
                    "trial": as_int(rec.get("trial")) or 1,
                    "recall": recall,
                    "qps": qps,
                    "threads": as_int(rec.get("threads")),
                    "nq": as_int(rec.get("nq")),
                    "query_ops": as_int(rec.get("query_ops")),
                    "warmup_queries": as_int(rec.get("warmup_queries")),
                    "avg_results_per_query": as_float(rec.get("avg_results_per_query")),
                    "source_file": path,
                })
    return sorted(rows, key=lambda r: (r["dataset"], r["metric"], r["ef"], r["trial"]))


def write_tsv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fields})


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["metric"], row["ef"])].append(row)
    out = []
    for _, group in sorted(groups.items()):
        recalls = [r["recall"] for r in group]
        qps = [r["qps"] for r in group]
        avg_results = [r["avg_results_per_query"] for r in group if r["avg_results_per_query"] is not None]
        first = group[0]
        out.append({
            "dataset": first["dataset"],
            "metric": first["metric"],
            "k": first["k"],
            "ef": first["ef"],
            "trials": len(group),
            "recall_median": statistics.median(recalls),
            "recall_mean": statistics.mean(recalls),
            "qps_median": statistics.median(qps),
            "qps_mean": statistics.mean(qps),
            "qps_std": statistics.stdev(qps) if len(qps) > 1 else 0.0,
            "qps_min": min(qps),
            "qps_max": max(qps),
            "avg_results_per_query_median": statistics.median(avg_results) if avg_results else None,
            "threads": first["threads"],
            "nq": first["nq"],
            "query_ops": first["query_ops"],
            "warmup_queries": first["warmup_queries"],
            "source_file": first["source_file"],
        })
    return sorted(out, key=lambda r: (r["dataset"], r["metric"], r["ef"]))


def threshold_rows(summary_rows):
    fields = ["dataset", "metric", "target_recall", "qps_median", "selected_recall", "ef", "trials", "qps_std"]
    out = []
    groups = defaultdict(list)
    for row in summary_rows:
        groups[(row["dataset"], row["metric"])].append(row)
    for (dataset, metric), group in sorted(groups.items()):
        for i in range(50, 97):
            target = i / 100.0
            candidates = [r for r in group if r["recall_median"] >= target]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: r["qps_median"])
            out.append({
                "dataset": dataset,
                "metric": metric,
                "target_recall": target,
                "qps_median": best["qps_median"],
                "selected_recall": best["recall_median"],
                "ef": best["ef"],
                "trials": best["trials"],
                "qps_std": best["qps_std"],
            })
    return fields, out


def make_svg(rows, metric, path):
    rows = [r for r in rows if r["metric"] == metric and r["qps_median"] > 0]
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = 980, 620
    left, right, top, bottom = 92, 28, 42, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin = max(0.0, math.floor((min(r["recall_median"] for r in rows) - 0.01) * 100) / 100)
    xmax = min(1.0, math.ceil((max(r["recall_median"] for r in rows) + 0.01) * 100) / 100)
    if xmin == xmax:
        xmin -= 0.01
        xmax += 0.01
    log_ymin = math.floor(math.log10(min(r["qps_median"] for r in rows)) * 10) / 10
    log_ymax = math.ceil(math.log10(max(r["qps_median"] for r in rows)) * 10) / 10

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        return top + (log_ymax - math.log10(y)) / (log_ymax - log_ymin) * plot_h

    colors = {"deep1b": "#2563eb", "t2i1b_norml2": "#dc2626"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#222}.title{font-size:18px;font-weight:700}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#222;stroke-width:1.2}.line{fill:none;stroke-width:2}</style>',
        f'<text class="title" x="{left}" y="26">Fresh hnswlib CPU {metric} QPS-Recall</text>',
    ]
    for i in range(6):
        x = xmin + i * (xmax - xmin) / 5
        px = sx(x)
        svg.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}"/>')
        svg.append(f'<text x="{px:.1f}" y="{top + plot_h + 24}" text-anchor="middle">{x:.2f}</text>')
    for e in range(math.floor(log_ymin), math.ceil(log_ymax) + 1):
        for m in (1, 2, 5):
            y = m * (10 ** e)
            if y < 10 ** log_ymin or y > 10 ** log_ymax:
                continue
            py = sy(y)
            label = str(int(y)) if y >= 10 else f"{y:g}"
            svg.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}"/>')
            svg.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end">{label}</text>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    svg.append(f'<text x="{left + plot_w/2:.1f}" y="{height - 22}" text-anchor="middle">{metric}</text>')
    svg.append(f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle">Median QPS (log scale)</text>')

    for idx, (dataset, group) in enumerate(sorted(grouped.items())):
        group = sorted(group, key=lambda r: r["recall_median"])
        color = colors.get(dataset, "#111827")
        points = " ".join(f"{sx(r['recall_median']):.1f},{sy(r['qps_median']):.1f}" for r in group)
        svg.append(f'<polyline class="line" points="{points}" stroke="{color}" opacity="0.85"/>')
        for row in group:
            x, y = sx(row["recall_median"]), sy(row["qps_median"])
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" opacity="0.85"/>')
        lx = left + plot_w - 230
        ly = top + 20 + idx * 22
        svg.append(f'<rect x="{lx:.1f}" y="{ly - 8:.1f}" width="12" height="12" fill="{color}" opacity="0.85"/>')
        svg.append(f'<text x="{lx + 18:.1f}" y="{ly + 3:.1f}">{dataset}</text>')
    svg.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(svg) + "\n")


def main():
    raw = read_raw()
    summary = summarize(raw)
    write_tsv(os.path.join(RESULTS, "fresh_raw_all_trials.tsv"), raw, RAW_FIELDS)
    write_tsv(os.path.join(RESULTS, "fresh_summary_by_ef.tsv"), summary, SUMMARY_FIELDS)
    fields, thresholds = threshold_rows(summary)
    write_tsv(os.path.join(RESULTS, "fresh_best_qps_by_recall_threshold_050_096.tsv"), thresholds, fields)
    make_svg(summary, "R@10", os.path.join(PLOTS, "fresh_cpu_r10.svg"))
    make_svg(summary, "R@100", os.path.join(PLOTS, "fresh_cpu_r100.svg"))
    print(f"raw_rows={len(raw)}")
    print(f"summary_rows={len(summary)}")
    print(os.path.join(RESULTS, "fresh_summary_by_ef.tsv"))


if __name__ == "__main__":
    main()
