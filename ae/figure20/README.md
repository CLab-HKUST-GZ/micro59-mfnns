# Figure 20 test

## Environment

- Python 3 with the packages in `ae/figure20/requirements.txt`.
- Run from the repository root.

## Test

```bash
python3 ae/figure20/plot_figure20.py --check-only
bash ae/figure20/reproduce_figure20.sh
```

## Expected output

The check prints `Validated Figure 20: 2 metrics x 3 designs x 4 components ...`. The reproduction prints `Figure 20 reproduced under ae/figure20/output/` and writes:

```text
ae/figure20/output/figure20.pdf
ae/figure20/output/figure20.png
ae/figure20/output/figure20_normalized.csv
ae/figure20/output/figure20_summary.tsv
```

The figure is derived from
`ae/figure20/data/figure20_area_power_breakdown.tsv`; it has no simulator YAML
or runtime hardware dependency.
