# Figure 23 configuration test

## Environment

- Python 3 with the packages in `ae/figure23/requirements.txt`.
- Run from the repository root.

## Test

```bash
python3 ae/figure23/plot_figure23.py --check-only
```

## Expected output

The command verifies all 21 archived YAML digests and prints `CHECK_OK rows=21 yamls=21 ...`. It does not execute the historical absolute paths. Generated figure outputs are under `ae/figure23/output/`.
