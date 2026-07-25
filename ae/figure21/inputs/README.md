# Figure 21 input test

## Environment

- Python 3.
- Run from the repository root.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure21/validate_figure21.py
```

## Expected output

The command prints `CHECK_OK ...` after verifying the included query and ground-truth sizes and SHA-256 digests. The external HNSW index must be placed at `mfnns_hnswlib/cpu_index/t2i1m/hnsw_index_M32_ef100.bin` for simulator reruns; generated results go under the runner's explicit `--result-root`.
