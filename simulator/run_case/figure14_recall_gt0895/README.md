# Figure 14 portable run cases

This directory contains only the final runnable Figure 14 simulator
configurations and their compact manifests. Search-stage YAMLs and historical
scheduler snapshots are intentionally excluded.

## Configuration matrix

```text
configs/final/k<5|10|100>/<dataset>/<design>.yaml
```

The matrix contains:

- 3 result counts: k=5, k=10, and k=100;
- 7 datasets: Deep10M, GloVe2M, SIFT1M, T2I1M, W2V1M, Wiki1M, PubMed;
- 6 simulator designs: CPU, ANSMET, NMP-Base, NMP-FPMA, NMP-FPSA-ET,
  and MFNNS.

There are 126 YAMLs in total. Every runnable input/output path is repository
relative and uses the normalized, variant-free CPU-index layout.

## Validate

From the repository root:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
```

## Run one case

Build the simulator first, then run:

```bash
simulator/run_case/figure14_recall_gt0895/scripts/run_final_case.sh \
  simulator/run_case/figure14_recall_gt0895/configs/final/k100/glove2m/ndp_base.yaml
```

Generated stats are written below `results/` according to each YAML's
repository-relative `stat_path`.

`manifests/final_cases.tsv` records the full runnable matrix.
`manifests/k100_sources.tsv` records the k=100 Figure/source mapping. The seven
NMP-FPMA k=100 rows are derived from the documented NMP-Base timing model and
must not be interpreted as direct NMP-FPMA measurements.
