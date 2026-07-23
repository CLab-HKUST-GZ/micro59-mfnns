# 001 T2I-1M validation record

## Scope

Validate that the standalone package can:

1. compile without external hnswlib/Faiss/Python dependencies;
2. build and reload a normalized T2I HNSW index;
3. compute exact FP32 L2 ground truth;
4. quantify recall loss for the bundled precision modes;
5. reproduce a serial build byte for byte;
6. load the existing MFANNS T2I-1M index;
7. avoid the concurrency defects discovered during packaging.

## Environment

- Date: 2026-07-23
- Slurm partition: `debug`
- Final validation job: `10048003`, host `cpu1-75`
- Resources: 16 CPU cores, 16 GiB requested
- Compiler: GCC 9.4.0
- Additional compile check: Clang 10.0.0
- ThreadSanitizer final job: `10048001`

Inputs:

```text
base:  /hpc2hdd/home/rmeng603/vectordb/t2i/1M/base.1M.fbin
query: /hpc2hdd/home/rmeng603/vectordb/t2i/query/query.public.100K.fbin
full index:
  /hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_t2i_mantissa/
  hnsw_index_text2img1M_norm_M32_ef100.bin
```

Both base and query vectors were L2-normalized by the tool. Exact ground truth
was recomputed from the corresponding base rows; no cached labels were used.

## Commands

The reproducible command sequence is in `run_validation.sh`. It builds outside
the repository at `/tmp/mfnns_hnswlib_build`.

The final code was also compiled with:

```bash
clang++ -O2 -DNDEBUG -std=c++17 -fopenmp -march=native \
  -Wall -Wextra -Wpedantic -isystem src \
  src/mfnns_hnsw_tool.cpp \
  -o /tmp/mfnns_hnswlib_build/mfnns_hnsw_tool_clang -pthread
```

## Final results

### Fresh 20K smoke index

- Shape: 20,000 x 200
- Normalized: yes
- M: 16
- `ef_construction`: 100
- preprocessing threads: 16
- insertion threads: 1
- build time: 8.117177 s
- index size: 18,971,896 bytes
- FP32 recall@10 over 20 queries: 0.985

| Precision | Recall@10 | Absolute loss vs FP32 | Overlap vs FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 0.985 | 0.000 | 1.000 |
| FP16_TRUE | 0.985 | 0.000 | 1.000 |
| FP16_FPMA | 0.970 | 0.015 | 0.985 |
| FP8_E4M3 | 0.945 | 0.040 | 0.960 |
| INT16 | 0.985 | 0.000 | 1.000 |
| INT8 | 0.950 | 0.035 | 0.965 |

Full machine-readable output: `smoke_recall.csv`.

### Existing complete T2I-1M index

Header validation:

- elements: 1,000,000
- dimension: 200
- M: 32
- `ef_construction`: 100
- serialized size: 1,076,324,680 bytes

Evaluation used the first 10 public queries, recall@10, and `ef_search=100`.
This is deliberately a simple validation, not a statistically complete recall
study.

| Precision | Recall@10 | Absolute loss vs FP32 | Overlap vs FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 0.820 | 0.000 | 1.000 |
| FP16_TRUE | 0.820 | 0.000 | 0.990 |
| FP16_FPMA | 0.800 | 0.020 | 0.950 |
| FP8_E4M3 | 0.770 | 0.050 | 0.840 |
| INT16 | 0.820 | 0.000 | 1.000 |
| INT8 | 0.770 | 0.050 | 0.860 |

Full machine-readable output: `full_t2i1m_recall.csv`.

### Determinism

Two independent normalized 5K builds using M=16, `ef_construction=100`,
seed=100, and one insertion thread produced the same SHA-256:

```text
bef0760a22e18a8096d18b56d280e4ad57f01b46e3a8f3db4cde63e356bd44e9
```

The `cmp` check succeeded. See `single_thread_determinism.sha256`.

### ThreadSanitizer

The final 1K serial-insertion build completed with no reported data race.
Deadlock detection was disabled for that final run because the inherited code
has a static lock-order cycle even within one serial thread; public build code
never calls `addPoint` concurrently. See `run_tsan.sh`, `tsan.stdout.log`, and
`tsan.stderr.log`.

## Conclusion

All release paths passed. The package builds and reloads compatible indexes,
computes exact ground truth, reports precision recall loss, reproduces serial
indexes, and loads the existing full T2I-1M index.

The 10-query full-index result is only a smoke validation. A publication-grade
claim should rerun hundreds or thousands of queries and report confidence
intervals.
