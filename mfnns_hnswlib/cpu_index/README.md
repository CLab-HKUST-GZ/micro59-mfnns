# CPU index cache

This tracked directory is the canonical output root for
[`../../script/cpu_index_build.sh`](../../script/cpu_index_build.sh), which is
run from the top-level repository root. The directory layout is:

```text
cpu_index/<dataset>/<variant>/hnsw_index_M<M>_ef<ef_construction>.bin
```

The leaf directories are committed so the expected cache structure exists
after cloning. Serialized indexes are deliberately ignored because the 1B
indexes are very large and can be regenerated from the source datasets.

The 15 CPU-scale raw/normalized cache variants use `M=32` and
`ef_construction=100`. Deep1B/normalized and T2I1B/normalized use `M=16` and
`ef_construction=500`.
