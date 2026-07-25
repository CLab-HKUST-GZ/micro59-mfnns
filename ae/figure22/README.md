# Figure 22 test

## Environment

- Python 3 with the packages in `ae/figure22/requirements.txt`.
- Run from the repository root.

## Test

```bash
python3 ae/figure22/plot_figure22.py --check-only
bash ae/figure22/reproduce_figure22.sh
```

## Expected output

The check prints `CHECK_OK rows=14 ...`. The reproduction prints `Figure 22 reproduced under ae/figure22/output/` and writes:

```text
ae/figure22/output/figure22.pdf
ae/figure22/output/figure22.png
ae/figure22/output/figure22_summary.tsv
```
