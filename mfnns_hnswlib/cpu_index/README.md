# CPU index cache

This tracked directory is the canonical output root for
[`../../script/cpu_index_build.sh`](../../script/cpu_index_build.sh), which is
run from the top-level repository root. All indexes use the normalized policy
by default. The historical `raw/normalized` layer has been removed:

```text
cpu_index/<dataset>/hnsw_index_M<M>_ef<ef_construction>.bin
```

The leaf directories are committed so the expected cache structure exists
after cloning. Serialized indexes are deliberately ignored because the 1B
indexes are very large and can be regenerated from the source datasets.

The eight CPU-scale indexes use `M=32` and `ef_construction=100`. Deep1B and
T2I1B use `M=16` and `ef_construction=500`. PubMed's source vectors are already
L2-normalized, so the builder avoids a redundant second normalization pass.
