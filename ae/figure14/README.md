# Figure 14: throughput comparison

## Environment

- Python 3 with the packages in `ae/requirements.txt`.
- Run from the repository root.

## Reproduce and validate

```bash
python3 ae/figure14/plot_figure14.py --check-only
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
bash ae/figure14/reproduce_figure14.sh
```

## Expected output

The plot validation prints `DATA_OK rows=168 ...`. The YAML validation prints
`PASS: 126 parsed, normalized, repository-relative Figure 14 YAMLs ...`.
Reproduction writes:

```text
ae/figure14/output/figure14.pdf
ae/figure14/output/figure14.png
ae/figure14/output/figure14_summary.tsv
```

The 126 YAMLs cover `3 top-k values x 7 datasets x 6 simulator designs` and
are under
`simulator/run_case/figure14_recall_gt0895/configs/final/`. They are
repository-relative rerun recipes; executing them additionally requires the
prepared index, queries, ground truth, and existing `simulator/build/ramulator2`.
See `simulator/run_case/figure14_recall_gt0895/README.md` for one-case
execution.
