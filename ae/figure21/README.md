# Figure 21 test

## Environment

- Linux, Bash, Python 3, GNU `sha256sum`, and the packages in `ae/figure21/requirements.txt`.
- Run from the repository root.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure21/validate_figure21.py
bash ae/figure21/reproduce_figure21.sh
```

## Expected output

Validation prints `CHECK_OK sweep=243 ansmet=3 configs=246 ...`. The reproduction ends with `Figure 21 reproduced under ae/figure21/output/` and writes:

```text
ae/figure21/output/figure21.pdf
ae/figure21/output/figure21.png
ae/figure21/output/figure21_plot_data.tsv
```
