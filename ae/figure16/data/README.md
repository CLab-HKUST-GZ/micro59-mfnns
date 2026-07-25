# Figure 16 data test

## Environment

- Python 3 with the packages in `ae/figure16/requirements.txt`.
- GNU `sha256sum`.
- Run from the repository root.

## Test

```bash
python3 ae/figure16/build_figure16_data.py --check-only
(cd ae/figure16/data && sha256sum -c SHA256SUMS)
```

## Expected output

Both commands exit with status 0, and `sha256sum` reports every listed file as `OK`. The generated table is `ae/figure16/data/figure16_energy_efficiency.csv`; figure files and the summary are under `ae/figure16/output/`.
