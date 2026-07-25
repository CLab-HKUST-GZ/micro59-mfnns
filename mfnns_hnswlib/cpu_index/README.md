# CPU index test

## Environment

- Linux, Bash, GNU Make, and a C++17 compiler with OpenMP.
- Real builds require the datasets selected by `CPU_DATA_ROOT`.
- Run from the repository root.

## Test

```bash
script/cpu_index_build.sh --list
script/cpu_index_build.sh --dry-run deep10m/normalized
```

To build the selected index, rerun the second command without `--dry-run`.

## Expected output

`--list` prints the supported dataset/variant matrix. `--dry-run` prints the exact `mfnns_hnsw_tool build` command without writing files. The real build writes:

```text
mfnns_hnswlib/cpu_index/<dataset>/<variant>/hnsw_index_M<M>_ef<ef_construction>.bin
```
