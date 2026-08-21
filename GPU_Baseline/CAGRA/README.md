# CAGRA Figure 14 rerun

## Environment

Use an NVIDIA GPU and Python with NumPy, CuPy, and cuVS. The Nice-server smoke
test used Python 3.10, NumPy 1.26.4, CuPy 13.6.0, and cuVS 25.12.0. Configure a
machine-local interpreter symlink if desired:

```bash
GPU_Baseline/configure.sh --cagra-python /path/to/cuvs-env/bin/python
GPU_Baseline/CAGRA/python -c 'import numpy, cupy, cuvs'
```

The commands below use an explicit interpreter so they remain unambiguous.

## Prepare data

```bash
RAW=/path/to/VectorDB
WORK=/local-ssd/$USER/figure14_gpu
CAGRA_PY=/path/to/cuvs-env/bin/python

"$CAGRA_PY" GPU_Baseline/prepare_data.py \
  --profile cagra --dataset all --raw-root "$RAW" \
  --output-root "$WORK/data" --gt-backend cupy --gpu 0 \
  --pubmed-cagra-base /path/to/1m/pubmed_d2v/doc_vectors_norm.bin \
  --checksum
```

See the parent README for the public 500k Pubmed limitation. The expected
data root for the following steps is `$WORK/data/cagra`.

## Build indexes

`build_indexes.py` deduplicates the 21 search rows into 16 graph builds and
calls the atomic builder exposed as `CAGRA/index_build.sh`:

```bash
"$CAGRA_PY" GPU_Baseline/CAGRA/build_indexes.py \
  --data-root "$WORK/data/cagra" \
  --index-root "$WORK/index/cagra" \
  --python "$CAGRA_PY" --gpu 0
```

The builder validates FBIN size and finite values, refuses a busy GPU by
default, saves the dataset inside the serialized CAGRA index, reloads it for
validation, publishes atomically, and writes adjacent JSON metadata. An empty
`graph_build_algo` in the manifest means the cuVS default exactly; the key is
omitted instead of guessing an algorithm name.

Use `--only text2img_r10` to build or validate one selected graph. Existing
complete indexes are loaded and validated as cache hits; `--force` rebuilds.

## Run the 21 frozen points

```bash
"$CAGRA_PY" GPU_Baseline/CAGRA/run_experiments.py \
  --data-root "$WORK/data/cagra" \
  --index-root "$WORK/index/cagra" \
  --output-root "$WORK/results/cagra" \
  --python "$CAGRA_PY" --gpu 0
```

For a smoke test, append `--only text2img_r10`. Each row uses its recorded
graph degree, intermediate degree, metric/build algorithm, search width,
`itopk_size`, warm-ups, timed runs, repeated-query policy, and mmap policy.
The runner writes the exact command, stdout log, result JSON, and a JSONL
status ledger.
