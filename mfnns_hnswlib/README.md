# MFNNS HNSW tool test

## Environment

- Linux on x86-64.
- GCC 9+ or Clang 10+ with C++17, OpenMP, and pthread support.
- GNU Make; Python 3 is required only for Table 5 aggregation.

## Test

From `mfnns_hnswlib/`:

```bash
make
build/mfnns_hnsw_tool --help
```

For a dataset-backed smoke test:

```bash
build/mfnns_hnsw_tool build \
  --base data/base.fbin --base-format fbin \
  --index cpu_index/example/hnsw_index_M16_ef100.bin \
  --normalize 1 --m 16 --ef-construction 100 \
  --threads 1 --limit 1000
build/mfnns_hnsw_tool inspect \
  --index cpu_index/example/hnsw_index_M16_ef100.bin
```

## Expected output

Compilation creates `mfnns_hnswlib/build/mfnns_hnsw_tool` and `mfnns_hnswlib/build/table5_dataset_runner`. The help command prints the available subcommands. The smoke test writes the requested index path, and `inspect` prints its element count, dimension, `M`, `ef_construction`, graph level, and entry point.
