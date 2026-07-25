# Figure 18: billion-scale recall–QPS curves

This bundle reproduces the paper's DP1B/T2I1B Recall@10 and Recall@100
comparison among CPU, BANG, ANSMET, and MFNNS.  It also collects one runnable
simulator YAML for every plotted ANSMET/MFNNS point, the CPU benchmark source
and build/run workflow, and the synchronized BANG producers from the remote
AE branch.

## Reproduce the figure

From the repository root:

```bash
bash ae/figure18/reproduce_figure18.sh
```

Read-only validation:

```bash
bash ae/figure18/validate_figure18.sh
```

The default is `--recall-digits 2`.  Recall is rounded with decimal
`ROUND_HALF_UP` immediately before validation/plotting; the unrounded frozen
value remains in `data/figure18_recall_qps.csv` as `recall_raw`.

Outputs:

```text
output/figure18.pdf
output/figure18.png
output/figure18_summary.tsv
```

## CPU benchmark source and scripts

The source used by the 2026-06-15 CPU run is under `cpu/`:

```text
cpu/src/hnswlib_1b_qps.cpp
cpu/scripts/build_cpu_benchmark.sh
cpu/scripts/run_figure18_cpu_curves.sbatch
cpu/third_party/hnswlib/
```

Build from the bundled source and exact header snapshot:

```bash
bash ae/figure18/cpu/scripts/build_cpu_benchmark.sh
```

The CPU curve used 100 queries, 500 repeats, three trials per `ef`, 96
threads, a 10,000-search warmup, static OpenMP scheduling, and NUMA
interleaving. Every frozen CPU plot row is linked to an exact raw trial in:

```text
cpu/data/figure18_cpu_selected_trials.tsv
```

Validate all 31 mappings against the archived 96 raw trials:

```bash
python3 ae/figure18/cpu/scripts/validate_cpu_provenance.py --check-only
```

The paper curve froze individual trial values, not the later three-trial
median summary. Both the raw table and the median summary are included so this
historical choice remains visible.

## BANG scripts synchronized from the remote

The latest remote BANG files are preserved in this combined bundle:

```text
scripts/run_deep1b_bang_curve.sh
scripts/run_t2i1b_mips_l2_curve.sh
data/expected_deep1b_bang_curve.tsv
provenance/REMOTE_BANG_README.md
```

`REMOTE_BANG_README.md` is the byte-for-byte BANG-only README fetched from
the remote before this merge; this top-level README is the combined
CPU/BANG/ANSMET/MFNNS guide.

The Deep1B producer runs the official 10K queries and ground truth against an
existing BANG R64/Lbuild100/QD32 index. It checks `MAX_R=64`, index parts,
headers, and a GT sample before running the Recall@10/Recall@100 sweeps.
`expected_deep1b_bang_curve.tsv` is the synchronized 2026-06-15
`median_last3_qps` reference. It follows the same recall grid (within
`0.0001`) but is not a point-for-point QPS copy of the older frozen paper
curve: only three QPS rows are exactly equal. It must therefore remain
separate validation evidence and must not silently replace
`figure18_recall_qps.csv`.

```bash
export BANG_REPO=/path/to/BANG-Billion-Scale-ANN
export BASE=/path/to/base.1B.fbin
export QUERY=/path/to/query.public.10K.fbin
export INDEX_PREFIX=/path/to/diskann_deep1b_R64_L100_QD32
bash ae/figure18/scripts/run_deep1b_bang_curve.sh
```

The T2I1B script requires a BANG-compatible 201D MIPS-to-L2 transformed
index. It rejects a historical raw-200D L2 index because that index is not
metric-compatible with the official MIPS labels. A new T2I1B BANG run must
not be presented as the historical paper curve until its transformed-index
provenance and frontier are independently validated.

## Simulator configurations

The bundle contains 108 YAMLs:

| Method | Plotted points | YAMLs |
|---|---:|---:|
| ANSMET | 50 | 50 |
| MFNNS | 58 | 58 |

Their provenance status is:

| Status | Count | Meaning |
|---|---:|---|
| `verified_original_yaml_stats` | 88 | Original YAML and stats match dataset, method, k, ef, LBQueue, and plotted cycle. |
| `original_yaml_cycle_mismatch` | 1 | An original same-parameter YAML exists, but its recorded cycle differs from the frozen plot. |
| `reconstructed_from_same_panel_template` | 19 | No original same-parameter YAML was found; this is an explicitly marked rerun recipe. |

All 50 ANSMET configurations and 38 of 58 MFNNS configurations are in the
first category.  Reconstructed configurations are never treated as evidence
for the frozen Figure 18 values.  Exact paths, hashes, stats values, deltas,
and notes are in `data/simulator_provenance.tsv`.

The portable YAMLs preserve the historical input paths and all simulator
fields except `stat_path`, which is changed to a runtime-relative path.  For
the 19 reconstructed files, only `ef_search`, LBQueue size, and LBQueue warmup
are changed from a same-dataset/same-metric/same-method template.

## Select or submit simulator cases

Printing commands is the safe default:

```bash
python3 ae/figure18/run_simulator_configs.py \
  --dataset deep1b --recall-tag r10 --method mfnns
```

Only provenance-verified YAMLs are selected unless
`--include-unverified` is given.  A complete rerun is optional because DP1B
and T2I1B require the prepared billion-scale inputs and a big-memory node.

Submitting requires an explicit new numbered memory directory:

```bash
python3 ae/figure18/run_simulator_configs.py \
  --dataset deep1b --recall-tag r10 --method mfnns \
  --submit --result-root memory/YYYYMMDD/NNN_figure18_deep1b_r10_mfnns
```

After jobs finish:

```bash
python3 ae/figure18/summarize_simulator_results.py \
  memory/YYYYMMDD/NNN_figure18_deep1b_r10_mfnns
```

The runner reuses `simulator/build/ramulator2` and
`simulator/memory/run_yaml_case.py`; it does not create a new build directory.
The simulator source is already versioned under `simulator/`, and its
repository-relative compiler entry point is:

```bash
bash script/simulator_build.sh --cluster-modules --dry-run
```

Remove `--dry-run` only when the existing `simulator/build` needs rebuilding.
Outside the documented cluster environment, supply equivalent CMake/compiler
paths instead of `--cluster-modules`.

## Frozen-data boundary

The figure can be reproduced without the historical author workspace. CPU
and BANG do not use simulator YAMLs: CPU now has raw-trial, source, compile,
and execution provenance under `cpu/`, while BANG has a validated Deep1B
producer and a metric-correct T2I1B future-run producer. ANSMET/MFNNS QPS is
validated as:

```text
QPS = 100 queries × 2.4 GHz / simulated cycles
```

The author-only collector under `scripts/collect_author_sources.py` documents
how the frozen plot table and historical memory tree were consolidated.  It is
not needed to reproduce or validate the archived figure.
