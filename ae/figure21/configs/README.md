# Figure 21 configuration test

## Environment

- Python 3, PyYAML, and GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure21/validate_figure21.py
(cd ae/figure21/configs && sha256sum -c SHA256SUMS)
```

## Expected output

The validator reports 246 configurations and exits with status 0; every YAML checksum reports `OK`. Submitted reruns write records and statistics below the explicit `--result-root` passed to `ae/figure21/run_figure21_sweep.py`.
