# Table 5 test

## Environment

- Linux, Bash, Python 3, GNU Make, and a C++17 compiler with OpenMP.
- Seven prepared HNSW indexes plus cached query and ground-truth files.
- Set repository-relative `TABLE5_CACHE_ROOT` and, if needed, `TABLE5_INDEX_ROOT`.
- Run from `mfnns_hnswlib/`.

## Test

```bash
TABLE5_CACHE_ROOT=data/table5_cache \
TABLE5_INDEX_ROOT=cpu_index \
TABLE5_JOBS=7 \
TABLE5_THREADS=8 \
./run_table5.sh
```

`TABLE5_QUERY_LIMIT` may be reduced for a pipeline smoke test; use the default 1000 for the paper configuration.

## Expected output

The command ends with `[table5] complete` and writes:

```text
mfnns_hnswlib/artifacts/table5_reproduction/table5.csv
mfnns_hnswlib/artifacts/table5_reproduction/table5.md
mfnns_hnswlib/artifacts/table5_reproduction/table5_details.csv
mfnns_hnswlib/artifacts/table5_reproduction/datasets/*.csv
mfnns_hnswlib/artifacts/table5_reproduction/logs/*.log
```
