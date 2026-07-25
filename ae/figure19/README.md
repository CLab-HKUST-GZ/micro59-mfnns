# Figure 19 test

## Environment

- Python 3 with the packages in `ae/figure19/requirements.txt`.
- Run from the repository root.

## Test

```bash
python3 ae/figure19/plot_figure19.py --check-only
bash ae/figure19/reproduce_figure19.sh
```

## Expected output

The read-only check exits with status 0 after validating the frozen data and checksums. The reproduction prints `Figure 19 reproduced under ae/figure19/output/` and writes:

```text
ae/figure19/output/figure19.pdf
ae/figure19/output/figure19.png
```
