# Figure 17 data test

## Environment

- Python 3 with the packages in `ae/figure17/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure17/build_figure17_data.py --check-only
(cd ae/figure17/data && sha256sum -c SHA256SUMS)
```

## Expected output

Both commands exit with status 0, and `sha256sum` reports every listed file as `OK`. The generated table is `ae/figure17/data/figure17_energy_breakdown.csv`; figure files and the summary are under `ae/figure17/output/`.
