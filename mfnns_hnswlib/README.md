# MFNNS HNSW index and recall-loss tool

This directory is a self-contained C++17 snapshot of the HNSW and
dynamic-precision distance code used by MFANNS recall analysis. It provides one
command-line tool and reproducible scripts for:

- streaming an FBIN or FVECS dataset into a serialized HNSW index;
- inspecting the metadata of a serialized index;
- computing exact L2 ground truth and measuring recall loss for FP32, FP16,
  FPMA, FP8, INT16, and INT8 distance modes.

The repository contains source and documentation only. Datasets, indexes,
binaries, and generated artifacts are intentionally excluded.

## Requirements

- Linux on x86-64;
- GCC 9+ or Clang 10+ with C++17 support;
- OpenMP and pthreads.

No Python, CMake, Faiss, or external hnswlib installation is required.

Build with the repository-relative default:

```bash
make
build/mfnns_hnsw_tool --help
```

## Input format

The build command accepts little-endian FBIN:

```text
int32 number_of_vectors
int32 dimension
float32[number_of_vectors][dimension]
```

It rejects truncated files and files with trailing bytes. The serialized HNSW
format is the native x86-64 format used by the bundled MFANNS hnswlib fork; it
is not promised to be portable across endianness or `size_t` width.

FVECS input is also supported. Each row consists of an `int32` dimension
followed by that many `float32` values; every row must have the same dimension.

## Build all CPU indexes

The standard entry point builds all 15 historical CPU-scale cache variants
plus the normalized Deep1B and T2I1B indexes. With no target arguments it
builds all 17 sequentially:

```bash
script/cpu_index_build.sh
```

To inspect the complete configuration without building:

```bash
script/cpu_index_build.sh --list
script/cpu_index_build.sh --dry-run
```

The default dataset root is the relative path `../../../vectordb`. Override it
with another repository-relative path when the datasets are elsewhere:

```bash
CPU_DATA_ROOT=data/vectordb script/cpu_index_build.sh \
  deep10m/normalized glove2m/normalized
```

Generated indexes are written below `cpu_index/`. The expected directory tree
is committed, while all serialized index files are ignored and must not be
committed. See [`cpu_index/README.md`](cpu_index/README.md).

## Build one index directly

```bash
TOOL=build/mfnns_hnsw_tool

"$TOOL" build \
  --base data/base.fbin \
  --base-format fbin \
  --index cpu_index/example/normalized/hnsw_index_M16_ef500.bin \
  --normalize 1 \
  --m 16 \
  --ef-construction 500 \
  --threads 16 \
  --batch-size 100000 \
  --seed 100
```

`--threads` parallelizes L2 normalization. `--insertion-threads` controls HNSW
construction and defaults to the same value. The bundled fork's lock-order fix
acquires the HNSW global lock before the new-node lock,
removing the inherited concurrent lock-order cycle while keeping partially
initialized nodes private. Set
`--insertion-threads 1` when byte-for-byte deterministic output is required.

Use `--limit N` for a subset smoke test. Existing output is protected unless
`--force 1` is explicitly supplied.

## Inspect an index

```bash
"$TOOL" inspect --index cpu_index/example/normalized/hnsw_index_M16_ef500.bin
```

The command checks the header and prints the element count, inferred FP32
dimension, M, `ef_construction`, graph level, and entry point.

## Evaluate recall loss

```bash
"$TOOL" evaluate \
  --base data/base.fbin \
  --queries data/query.fbin \
  --index cpu_index/example/normalized/hnsw_index_M16_ef500.bin \
  --normalize 1 \
  --query-limit 100 \
  --k 10 \
  --ef 100 \
  --threads 16 \
  --batch-size 100000 \
  --precisions fp32,fp16_true,fp16_fpma,fp8_e4m3,int16,int8 \
  --output recall.csv
```

Evaluation streams all indexed base rows and computes exact FP32 L2 top-k
ground truth. For each precision mode it reports:

- `recall_at_k`: overlap with exact ground truth;
- `recall_loss_vs_fp32`: FP32 recall minus the mode's recall;
- `overlap_vs_fp32`: overlap with the approximate FP32 HNSW result;
- query time and QPS.

FP32 is always evaluated first, even if omitted from `--precisions`.

## Reproduce paper Table 5

The standalone Table 5 workflow evaluates the four precision/ET variants and
derives the ET, risky-update, and boundary-decision-flip ratios from the
prepared seven-dataset cache:

```bash
./run_table5.sh
```

It uses the paper settings of 1,000 queries, `k=10`, and `efSearch=500`.
Queries and ground truth are selected with `TABLE5_CACHE_ROOT`, while indexes
come from `TABLE5_INDEX_ROOT` (default: `cpu_index`). All path overrides must
be relative to the repository root. Build, output, dataset-process, and
query-thread settings can also be overridden through environment variables. See
[`table5_reproduction/README.md`](table5_reproduction/README.md) for the cache
layout and complete instructions.

Generated tables, detailed counters, and logs are written below
`artifacts/table5_reproduction` and are not committed.

## Source provenance and license

The bundled fork came from
`MFANNS/recall_analysis/hnswlib`. Exact source hashes and the small standalone
repairs are documented in
[`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md). The current source manifest is
[`SOURCE_MANIFEST.sha256`](SOURCE_MANIFEST.sha256).

The hnswlib-derived files are distributed under Apache License 2.0; see
[`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
