# Figure 23 test

## Environment

- Python 3 with the packages in `ae/figure23/requirements.txt`.
- Run from the repository root.

## Test

```bash
python3 ae/figure23/plot_figure23.py --check-only
bash ae/figure23/reproduce_figure23.sh
```

## Expected output

The check prints `CHECK_OK rows=21 yamls=21 ...`. The reproduction prints the paths of:

```text
ae/figure23/output/figure23.pdf
ae/figure23/output/figure23.png
ae/figure23/output/figure23_summary.tsv
```
