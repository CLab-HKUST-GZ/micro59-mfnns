# BANG Figure 14 rerun

## Pinned source and CUDA environment

The recorded source is BANG commit
`c0ac04c4596a4a189c96173b881596ad00193cbe` from
`https://github.com/karthik86248/BANG-Billion-Scale-ANN`. BANG also requires a
PipeANN-compatible `build_disk_index` executable using the positional API
shown below. The local historical builder had SHA-256
`c59ae37e4dcafe7093dd3199f7855c829c17f82c9097f217f18eb9dc809ee4ea`;
the index build manifest records the actual builder path and SHA-256 used by
the reproducer.

BANG search is pinned to CUDA 12.8. The environment helper and contract
builder both reject any other `nvcc` version:

```bash
source GPU_Baseline/BANG/cuda12.8.env.sh
```

Override the installation path only when needed:

```bash
FIGURE14_BANG_CUDA_ROOT=/opt/cuda-12.8 \
  source GPU_Baseline/BANG/cuda12.8.env.sh
```

## Prepare data

```bash
RAW=/path/to/VectorDB
WORK=/local-ssd/$USER/figure14_gpu
CAGRA_PY=/path/to/python-with-numpy-cupy-cuvs

"$CAGRA_PY" GPU_Baseline/prepare_data.py \
  --profile bang --dataset all --raw-root "$RAW" \
  --output-root "$WORK/data" --gt-backend cupy --gpu 0 \
  --checksum
```

This produces the exact historical split-base queries for Deep, Wiki, and
Text2Img; the 100+900 Pubmed query mix; normalized bases; ID ground truth; and
BANG's IDs-plus-placeholder-distances GT container.

## Build the four search contracts

`MAX_R` and `BF_ENTRIES` are compile-time BANG constants. One mutable build
cannot represent all Figure 14 rows, so the builder copies the source into an
isolated temporary tree and publishes four independent binary/library pairs:

| Contract | `MAX_R` | `BF_ENTRIES` | CUDA |
|---|---:|---:|---:|
| `R16_BF399887` | 16 | 399887 | 12.8 |
| `R32_BF399887` | 32 | 399887 | 12.8 |
| `R32_BF99991` | 32 | 99991 | 12.8 |
| `R64_BF399887` | 64 | 399887 | 12.8 |

```bash
GPU_Baseline/BANG/build_contracts.sh \
  --source /path/to/BANG-Billion-Scale-ANN/BANG_Base \
  --output "$WORK/bin/bang" \
  --cuda-root /usr/local/cuda-12.8
```

Each contract directory contains `bang_search`, its matching `libbang.so`,
the rewritten `bang_search.cu`, `build.json`, and `SHA256SUMS`. The input
source tree is never modified.

## Build indexes and rewrap PQ pivots

The 21 selected rows require 9 unique graph/PQ indexes. Build them with:

```bash
python3 GPU_Baseline/BANG/build_indexes.py \
  --data-root "$WORK/data/bang" \
  --index-root "$WORK/index/bang" \
  --builder /path/to/PipeANN/tests/build_disk_index \
  --python python3 --threads 64 --build-memory-gb 64
```

The invoked `BANG/index_build.sh` performs all three required stages:

1. PipeANN graph/PQ build using
   `build_disk_index float BASE PREFIX R L PQ MEMORY_GB THREADS l2 pq`.
2. Conversion of `_disk.index` into BANG `_disk.bin` and
   `_disk_metadata.bin`.
3. PQ-pivot header rewrap into BANG's expected layout, followed by creation
   of the `_bang_*` prefix links used by `bang_search`.

Every stage is size/header validated and logged. Publication is atomic;
partial output is rejected, and complete output plus its build manifest is a
cache hit. Put `--index-root` on fast local storage. Use
`--only text2img1M_r10` or `--dry-run` for a focused check.

## Run the 21 frozen points

```bash
python3 GPU_Baseline/BANG/run_experiments.py \
  --data-root "$WORK/data/bang" \
  --index-root "$WORK/index/bang" \
  --binary-root "$WORK/bin/bang" \
  --output-root "$WORK/results/bang" \
  --cuda-root /usr/local/cuda-12.8 \
  --gpu 0 --omp-threads 64
```

The runner chooses the matching compile contract and index for each manifest
row, feeds the recorded search L to the interactive BANG driver, validates
query/GT/index containers, and records all measured rows plus the warm-run
median QPS. Append `--only text2img1M_r10` for a one-row smoke test.

## Regression

```bash
bash script/test_bang_index_build.sh
```

The regression uses a deterministic mock PipeANN builder to exercise index
construction, disk preprocessing, PQ rewrap, metadata, atomic publication,
and cache validation without requiring a multi-gigabyte real build.
