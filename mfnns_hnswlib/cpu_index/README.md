# Normalized CPU index/query/GT cache

This tracked directory is the canonical output root for
[`../../script/cpu_index_build.sh`](../../script/cpu_index_build.sh), which is
run from the top-level repository root. The builder enforces one invariant:
the HNSW index, persisted queries, and exact ground truth all use the same
L2-normalized vector space.

```text
cpu_index/<dataset>/hnsw_index_M<M>_ef<ef_construction>.bin
cpu_index/<dataset>/hnsw_index_M<M>_ef<ef_construction>.bin.metadata.tsv
cpu_index/<dataset>/query_vectors_n1000_seed42.bin
cpu_index/<dataset>/query_indices_n1000_seed42.bin
cpu_index/<dataset>/gt_labels_topk5_n1000_seed42.bin
cpu_index/<dataset>/gt_labels_topk10_n1000_seed42.bin
cpu_index/<dataset>/gt_labels_topk100_n1000_seed42.bin
cpu_index/<dataset>/bundle_metadata_n1000_seed42.tsv
```

`query_vectors_*.bin` stores the normalized query values, not the raw source
rows. Ground truth is recomputed by an exact streaming L2 scan over normalized
base vectors; official raw-metric GT is never reused. Top-5 and top-10 are
verified prefixes of the same exact top-100 result.

The eight CPU-scale indexes use `M=32` and `ef_construction=100`. Deep1B and
T2I1B use `M=16` and `ef_construction=500`. PubMed's source vectors are already
normalized, but the builder deliberately applies the same normalization pass
again so every target follows one enforceable code path.

## Build

Prepare raw datasets first, then select explicit targets:

```bash
script/cpu_index_build.sh deep10m t2i1m wiki1m w2v1m glove2m sift1m pubmed
```

The tool rejects zero-norm/non-finite base or query rows. An existing index is
accepted only when its sidecar proves `normalization=l2` and its serialized
HNSW parameters match the requested values. A legacy index without that
provenance is not silently reused:

```bash
CPU_INDEX_FORCE=1 script/cpu_index_build.sh t2i1m
```

Use `--dry-run` to inspect both the index and query/GT commands without
building. `CPU_QUERY_COUNT`, `CPU_QUERY_SEED`, and `CPU_GT_TOPKS` can change
the cache shape, but the defaults must be retained for Figure 14 YAMLs.

Run the small CPU-only end-to-end invariant test without downloading a
dataset:

```bash
script/test_cpu_index_bundle.sh
```

It independently reads the serialized HNSW vectors, normalized query file,
and exact GT; it also verifies that unproven indexes, corrupted queries, and
zero-norm source vectors are rejected.

The leaf directories are committed so the expected structure exists after
cloning. Generated indexes, queries, GT, and metadata are ignored because
the 1B artifacts are very large and are reproducible from source vectors.
