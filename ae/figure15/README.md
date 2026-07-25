# Figure 15 test

## Environment

- Python 3.
- Install `ae/figure15/requirements.txt`.
- Run from the repository root.

## Test

```bash
python3 ae/figure15/build_figure15_data.py --check-only
python3 ae/figure15/plot_figure15.py --check-only
bash ae/figure15/reproduce_figure15.sh
```

## Expected output

Both checks print `CHECK_OK`. The reproduction ends with `Figure 15 reproduced under ae/figure15/output/` and writes:

```text
ae/figure15/data/figure15_area_efficiency.csv
ae/figure15/output/figure15_summary.tsv
ae/figure15/output/figure15.pdf
ae/figure15/output/figure15.png
```

## Inputs and provenance

The builder reads `ae/figure14/data/figure14_results.csv` and
`ae/figure15/data/area_specs.csv`. It selects Recall@10 rows for seven
datasets and six methods. Figure 15 inherits QPS, recall, and final-YAML
provenance from Figure 14; it does not require an independent simulator run.
