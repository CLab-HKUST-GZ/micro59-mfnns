# Figure 14 simulator test

## Environment

- Python 3 with `simulator/run_case/figure14_recall_gt0895/requirements.txt`.
- A built `simulator/build/ramulator2` is required only to execute a case.
- Run from the repository root.

## Prepare normalized index, query, and GT

After the raw datasets have been prepared under `CPU_DATA_ROOT`, run:

```bash
script/cpu_index_build.sh \
  deep10m t2i1m wiki1m w2v1m glove2m sift1m pubmed
```

For each dataset, this one command:

- L2-normalizes every base vector before HNSW insertion;
- samples and persists 1,000 L2-normalized queries with seed 42; and
- computes exact normalized-L2 top-5, top-10, and top-100 ground truth.

These are the exact paths referenced by all 126 final YAMLs. The builder
rejects source vectors that cannot be normalized and refuses to reuse an
index without matching `normalization=l2` provenance.

## Validate and run one case

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
simulator/run_case/figure14_recall_gt0895/scripts/run_final_case.sh \
  simulator/run_case/figure14_recall_gt0895/configs/final/k100/glove2m/ndp_base.yaml
```

## Expected output

The pass criterion is
`PASS: 126 parsed, normalized, repository-relative Figure 14 YAMLs ...`.
The executed example case writes its configured stats file below
`simulator/run_case/figure14_recall_gt0895/results/k100/glove2m/`.

Exact GT generation is CPU-intensive, especially for billion-scale targets.
The command streams base batches rather than loading the full dataset, but
reviewers should use an appropriately sized CPU allocation.
