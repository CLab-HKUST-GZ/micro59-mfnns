# Figure 14 GPU baselines

This directory supports a real rerun of the CAGRA and BANG points used in
Figure 14. It does not merely re-plot a shipped CSV. The 42 selected search
points are frozen in:

- `params/cagra.csv`: 7 datasets x Recall@{5,10,100}, including graph-build,
  search, warm-up, repeat, mmap, and historical result fields.
- `params/bang.csv`: 7 datasets x Recall@{5,10,100}, including R, build L, PQ,
  Bloom-filter contract, search L, and historical result fields.
- `params/bang_contracts.csv`: the four BANG compile-time contracts.

QPS depends on GPU clocks, host load, storage, CUDA/cuVS versions, and the
number of CPU threads. Recall and the selected parameters are the primary
reproduction checks; QPS is expected to vary across machines.

## 1. Prepare base, query, and ground truth

`prepare_data.py` reads the original FBIN/FVECS files, applies the historical
query-selection and normalization policy, and writes normalized FBIN base and
query files plus exact top-100 ground truth. For example on the Nice servers:

```bash
RAW=/home/CONNECT/rmeng603/data/VectorDB
WORK=/local-ssd/$USER/figure14_gpu
CAGRA_PY=/path/to/python-with-numpy-cupy-cuvs

"$CAGRA_PY" GPU_Baseline/prepare_data.py \
  --profile cagra --dataset text2img1M \
  --raw-root "$RAW" --output-root "$WORK/data" \
  --gt-backend cupy --gpu 0 --checksum

"$CAGRA_PY" GPU_Baseline/prepare_data.py \
  --profile bang --dataset all \
  --raw-root "$RAW" --output-root "$WORK/data" \
  --gt-backend cupy --gpu 0 --checksum
```

Use `--dataset all` for the full profile. CuPy reproduces the historical GPU
brute-force GT path, including its float32 tie behavior; NumPy is a portable
fallback. Outputs are cached and `prepare.json` records source paths, shapes,
policy, checksums, and whether the result is historically compatible.

The two profiles intentionally differ:

- CAGRA uses 100 seeded external queries for Deep/Wiki/Text2Img and repeats
  them to a batch of 1000 at search time. The other datasets use the first
  1000 external queries, except Pubmed, which has 100.
- BANG creates 1000 seeded queries from the base for Deep/Wiki/Text2Img and
  removes those vectors from the indexed base. Pubmed combines its 100 raw
  queries with 900 seeded base rows and removes the latter. The remaining
  datasets use the first 1000 external queries.
- The BANG GT distance section contains rank placeholders, matching the
  historical BANG input exactly; BANG recall evaluation consumes the IDs.

### Pubmed CAGRA limitation

The Figure 14 CAGRA point used the historical `pubmed_d2v` base with
1,000,000 x 768 rows. The public `VectorDB/pubmed/doc_vectors_norm.bin` in the
Nice data root is a different 500,000-row corpus, so the exact CAGRA Pubmed
input cannot be derived from that file. For an exact rerun, provide the
historical base explicitly:

```bash
"$CAGRA_PY" GPU_Baseline/prepare_data.py \
  --profile cagra --dataset all --raw-root "$RAW" \
  --output-root "$WORK/data" --gt-backend cupy --gpu 0 \
  --pubmed-cagra-base /path/to/1m/pubmed_d2v/doc_vectors_norm.bin
```

If only the public 500k corpus is available, add `--allow-pubmed-500k`. This
produces a runnable, explicitly marked variant, but the frozen Pubmed recall
and QPS values are not expected to match. This limitation does not apply to
the BANG Pubmed profile, which used the public 500k corpus.

## 2. Build and run

See `CAGRA/README.md` and `BANG/README.md` for the complete commands. Large
indexes should be placed on node-local SSD rather than in the Git checkout.
Both runners accept `--only ID[,ID...]` for a smoke test and refuse to start
on a GPU that already has a compute process unless `--allow-busy-gpu` is
given.

## 3. Validation status and Pubmed qualification

The workflow was validated on an RTX 6000 Ada server. For Text2Img, the
regenerated CAGRA and BANG base/query/GT files were byte-identical to the
historical prepared files. Representative live Recall@10 reruns matched the
recorded recall exactly:

```text
CAGRA Recall@10: 0.9090
BANG  Recall@10: 0.9386
```

All four BANG compile-time contracts were built with CUDA 12.8 and checked with
their adjacent source, library, binary, and checksum records. QPS is not
expected to be byte-for-byte reproducible across different GPU clocks, host
load, storage, CUDA/cuVS versions, and CPU thread counts; recall and parameter
identity are the primary portability checks.

With the historical 1M `pubmed_d2v` base supplied, all 42 GPU points retain
their historical dataset contract. With only the public 500k Pubmed corpus, 39
of 42 points do; the three CAGRA Pubmed runs are explicitly marked variants.
BANG Pubmed is unaffected because its historical input used the 500k corpus.

## 4. Fast regression tests

Run from the repository root:

```bash
bash script/test_gpu_baseline_repro.sh
bash script/test_bang_index_build.sh
```

The first test validates 21 CAGRA rows, 21 BANG rows, all four compile
contracts, and both conversion profiles on synthetic FVECS data. The second
test exercises BANG index construction, disk preprocessing, PQ-pivot rewrap,
atomic publication, and cache validation with a mock builder. Neither test
requires a real GPU or a multi-gigabyte index.

After supplying the two machine-local dependencies, validate the live
environment separately:

```bash
GPU_Baseline/configure.sh \
  --bang-builder /path/to/PipeANN/tests/build_disk_index \
  --cagra-python /path/to/cuvs-env/bin/python
GPU_Baseline/configure.sh --check
```
