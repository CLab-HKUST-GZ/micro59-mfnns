# Table 5 reproduction

This directory contains the standalone runner and aggregator for the paper's
Table 5. It reproduces:

- FP16 and FPMA recall;
- FP16+ET and FPMA+ET recall;
- dual-queue early-termination ratios;
- directional risky-update ratios;
- FPMA-induced boundary-decision flip ratios.

The implementation uses the bundled MFANNS HNSW headers. It does not require
the historical Python extension, NumPy, or Faiss.

## Required cache and index layout

Set `TABLE5_CACHE_ROOT` to a repository-relative directory containing cached
queries and ground truth:

```text
<cache-root>/<dataset>/<variant>/
  query_vectors_n1000_seed42.bin
  gt_labels_topk10_n1000_seed42.bin
```

Indexes are read separately from `TABLE5_INDEX_ROOT`, which defaults to the
tracked `cpu_index` directory:

```text
cpu_index/<dataset>/<variant>/
  hnsw_index_M32_ef100.bin
```

Create all seven required CPU indexes with
`script/cpu_index_build.sh deep10m/normalized glove2m/normalized pubmed/raw
sift1m/normalized t2i1m/normalized w2v1m/normalized wiki1m/normalized`.
The same script also supports all raw CPU-cache variants and the two 1B
datasets.

The query and ground-truth files use a little-endian matrix format:

```text
int32 rows
int32 columns
element[rows][columns]
```

Queries contain `float32` elements and ground truth contains `int32` labels.
The HNSW files use the native serialized format of the bundled MFANNS fork.

## Run

From the repository root:

```bash
./run_table5.sh
```

The paper configuration is fixed by default to 1,000 cached queries,
`k=10`, `efSearch=500`, and the directional risk condition
`1 < D_b_exact / D_c_exact < 1.0073`.

Useful overrides:

```bash
TABLE5_CACHE_ROOT=data/table5_cache \
TABLE5_INDEX_ROOT=cpu_index \
TABLE5_OUTPUT_DIR=artifacts/table5_output \
TABLE5_BUILD_DIR=build \
TABLE5_JOBS=7 \
TABLE5_THREADS=8 \
./run_table5.sh
```

`TABLE5_JOBS` controls dataset-level process parallelism.
`TABLE5_THREADS` controls query parallelism inside each recall run. Boundary
logging is intentionally serial, matching the original experiment.

For a non-paper smoke test, `TABLE5_QUERY_LIMIT` may be reduced. Such a run
checks the pipeline but cannot reproduce the paper values.

Outputs are written below `artifacts/table5_reproduction` by default and are
excluded from version control:

- `table5.csv`: paper-layout values;
- `table5.md`: Markdown rendering;
- `table5_details.csv`: unrounded counters and measurements;
- `datasets/*.csv`: one detailed row per dataset;
- `logs/*.log`: one execution log per dataset.

## Provenance

The standalone workflow preserves the definitions used by:

- `memory/20260613/02_fp16_dual_queue_accuracy_7datasets`;
- `memory/20260614/002_fpma_error_ganns_7datasets`;
- `memory/20260614/004_directional_candidate_top_ratio`.

It combines them into one dependency-minimal executable and one standard
library-only aggregation script.
