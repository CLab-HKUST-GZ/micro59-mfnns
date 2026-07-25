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

## Reproducibility boundary

Portable reproduction uses only the frozen tables in `ae/figure19/data/`.
`figure19_mfnns_provenance.csv` records point-level author-workspace YAML and
stats references and SHA-256 digests, but the 189 full YAML/stats pairs are
not duplicated. The optional provenance rebuild therefore requires the
original author workspace:

```bash
python3 ae/figure19/build_figure19_data.py --check-only \
  --source-root /path/to/MFANNS
```

This limitation does not affect regeneration or validation of the archived
paper figure.
