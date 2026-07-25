# BANG index-builder test

## Environment

- Linux, Bash, Python 3, `flock`, and `/usr/bin/time`.
- A real build requires a compatible PipeANN/DiskANN `build_disk_index`.
- Run from the repository root.

## Test

```bash
GPU_Baseline/configure.sh \
  --bang-builder /path/to/build_disk_index
bash script/test_bang_index_build.sh
GPU_Baseline/build_index.sh bang wiki1M \
  --base /path/to/wiki1M_base.bin --dry-run
```

## Expected output

The regression ends with `TEST_BANG_INDEX_BUILD=PASS`. The dry run validates the FBIN input and prints the resolved builder command without creating an index. A real build writes index files, logs, and a build manifest under `GPU_Baseline/index/wiki1M/BANG/`.
