# Figure 15 data test

## Environment

- Python 3 with the packages in `ae/figure15/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure15/build_figure15_data.py --check-only
(cd ae/figure15/data && sha256sum -c SHA256SUMS)
```

## Expected output

The builder prints `CHECK_OK rows=42 ...`; `sha256sum` reports every listed file as `OK`. The generated data file is `ae/figure15/data/figure15_area_efficiency.csv`, and the summary is `ae/figure15/output/figure15_summary.tsv`.
