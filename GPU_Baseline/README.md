# GPU baseline tests

## Environment

- Linux, Bash, Python 3, and GNU core utilities.
- BANG builds require an executable PipeANN/DiskANN `build_disk_index`.
- CAGRA builds require an NVIDIA GPU and a Python environment with NumPy, CuPy, and cuVS.
- Run from the repository root.

## Test

```bash
GPU_Baseline/build_index.sh --list
bash script/test_bang_index_build.sh
GPU_Baseline/configure.sh --check
```

Run `configure.sh --check` after configuring both machine-local dependencies.

## Expected output

The list command prints seven dataset configurations, the BANG regression prints `TEST_BANG_INDEX_BUILD=PASS`, and the configured environment check prints `GPU_BASELINE_CONFIGURATION=PASS`. Real indexes are written below `GPU_Baseline/index/<dataset>/BANG/` or `GPU_Baseline/index/<dataset>/CAGRA/`, unless `GPU_BASELINE_INDEX_ROOT` overrides the root.
