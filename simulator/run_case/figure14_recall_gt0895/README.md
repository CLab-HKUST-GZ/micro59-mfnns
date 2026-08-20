# Figure 14 simulator rerun

This directory reruns the 126 simulator-backed Figure 14 points. It is separate
from `ae/figure14/reproduce_figure14.sh`, which only validates and replots the
versioned 168-row CSV. BANG and CAGRA account for the remaining 42 plot rows and
are outside this simulator workflow.

Run every command below from the repository root.

## 1. Requirements

- Python 3 and `simulator/run_case/figure14_recall_gt0895/requirements.txt`;
- CMake 3.14+ and a C++20 compiler for `ramulator2`;
- the seven Figure 14 raw datasets;
- enough CPU time and storage for seven HNSW indexes and exact GT generation.

Install the Python dependency:

```bash
python3 -m pip install -r \
  simulator/run_case/figure14_recall_gt0895/requirements.txt
```

The historical seven indexes occupy about 18.2 GB in total. Exact GT is an
exhaustive normalized-L2 scan and is CPU-intensive, especially for Deep10M.
Use a compute allocation rather than a shared login node for the build and the
full 126-case run.

## 2. Prepare raw datasets

Choose a data root that is relative to the repository root. This example uses a
sibling directory:

```bash
script/dataset_prepare.sh --root ../figure14-data figure14
```

`dataset_prepare.sh` downloads and validates raw base/query files only. It does
not construct HNSW, select the final 1,000 queries, or compute normalized GT.

To inspect downloads without writing files:

```bash
script/dataset_prepare.sh --root ../figure14-data --dry-run figure14
```

## 3. Build all 35 simulator inputs

Use the same relative data root in `CPU_DATA_ROOT`:

```bash
CPU_DATA_ROOT=../figure14-data \
CPU_INDEX_THREADS=32 \
  script/cpu_index_build.sh \
    deep10m t2i1m wiki1m w2v1m glove2m sift1m pubmed
```

For each dataset, this single command:

- L2-normalizes every base vector before HNSW insertion;
- builds `hnsw_index_M32_ef100.bin`;
- selects and persists 1,000 normalized queries with seed 42; and
- computes exact normalized-L2 top-5, top-10, and top-100 GT.

The resulting flat layout is:

```text
mfnns_hnswlib/cpu_index/<dataset>/hnsw_index_M32_ef100.bin
mfnns_hnswlib/cpu_index/<dataset>/query_vectors_n1000_seed42.bin
mfnns_hnswlib/cpu_index/<dataset>/gt_labels_topk5_n1000_seed42.bin
mfnns_hnswlib/cpu_index/<dataset>/gt_labels_topk10_n1000_seed42.bin
mfnns_hnswlib/cpu_index/<dataset>/gt_labels_topk100_n1000_seed42.bin
```

PubMed has only 100 source queries. The builder normalizes those unique rows,
computes exact GT once, and repeats both deterministically to 1,000 rows. All
seven datasets use the same normalized-L2 contract; official GT from a
different metric is not reused.

Inspect the commands without datasets or a compute allocation:

```bash
script/cpu_index_build.sh --dry-run \
  deep10m t2i1m wiki1m w2v1m glove2m sift1m pubmed
```

## 4. Build the simulator

```bash
script/simulator_build.sh
```

This produces `simulator/build/ramulator2`. See `simulator/BUILD.md` for manual
and cluster-specific build commands.

## 5. Validate configs and generated inputs

The first check needs no datasets. The second reads the seven HNSW headers and
validates every query/GT header, payload size, query norm, GT label range, and
top-5/top-10 inclusion in top-100:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
python3 simulator/run_case/figure14_recall_gt0895/tools/check_final_inputs.py
```

Expected success messages are:

```text
PASS: 126 parsed, normalized, repository-relative Figure 14 YAMLs ...
INPUTS_OK datasets=7 indexes=7 queries=7 gt=21
```

## 6. Run one case or all 126

Run one case:

```bash
simulator/run_case/figure14_recall_gt0895/scripts/run_final_case.sh \
  simulator/run_case/figure14_recall_gt0895/configs/final/k100/glove2m/ndp_base.yaml
```

Audit the full batch command without inputs or `ramulator2`:

```bash
simulator/run_case/figure14_recall_gt0895/scripts/run_final_cases.sh --dry-run
```

Run all 126 cases synchronously in manifest order:

```bash
simulator/run_case/figure14_recall_gt0895/scripts/run_final_cases.sh
```

Resume without overwriting nonempty stats files:

```bash
simulator/run_case/figure14_recall_gt0895/scripts/run_final_cases.sh \
  --skip-existing
```

The batch runner also accepts one or more repository-relative final YAML paths.
It is synchronous; on a cluster, submit individual paths through the scheduler
to run cases concurrently. Each YAML writes its own stats file below
`simulator/run_case/figure14_recall_gt0895/results/<k>/<dataset>/`.

New stats do not mutate `ae/figure14/data/figure14_results.csv`. This separation
keeps the archived-paper plot reproducible while allowing an independent rerun
of the simulator-backed points.
