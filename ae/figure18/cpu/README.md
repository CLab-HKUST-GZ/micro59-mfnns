# Figure 18 CPU producer and provenance

This directory packages the CPU hnswlib benchmark that produced the DP1B and
normalized-L2 T2I1B curves used by Figure 18.

## Build

The portable build uses the bundled benchmark source and the exact hnswlib
header snapshot that was present for the 2026-06-15 run:

```bash
bash ae/figure18/cpu/scripts/build_cpu_benchmark.sh
```

To validate compilation without leaving a build product in the repository:

```bash
tmp_bin="$(mktemp /tmp/figure18-cpu.XXXXXX)"
bash ae/figure18/cpu/scripts/build_cpu_benchmark.sh --output "${tmp_bin}"
file "${tmp_bin}"
```

The benchmark deliberately retains the historical `-march=native`,
OpenMP, fast-math, and unrolling flags because CPU QPS is sensitive to these
choices.

## Run

Export the seven prepared billion-scale input paths described by the script,
then submit:

```bash
sbatch ae/figure18/cpu/scripts/run_figure18_cpu_curves.sbatch
```

The defaults reproduce the main job's four curves:

```text
100 queries; 500 repeats; 3 trials per ef
96 threads; 10,000 warm-up searches
OpenMP static schedule, chunk 64
NUMA interleave
```

The historical scripts are preserved verbatim under `historical/`. They
contain author-workspace paths and are provenance records, not portable entry
points. The supplemental Deep1B Recall@100 `ef=1000` run is also archived,
although that point was not included in the frozen Figure 18 curve.

## Frozen curve mapping

`data/fresh_raw_all_trials.tsv` contains 96 raw rows: 32 search points with
three trials each. `data/figure18_cpu_selected_trials.tsv` proves that every
one of the 31 CPU rows in the Figure 18 plot table matches exactly one raw
trial.

The frozen paper curve selected individual trials rather than the later
three-trial median summary. This bundle preserves that historical choice
instead of silently replacing the plotted QPS values.

Validate the mapping:

```bash
python3 ae/figure18/cpu/scripts/validate_cpu_provenance.py --check-only
```

## Source dependency

The benchmark was compiled against hnswlib commit
`d9b3608c83d83b46c96e25088cb1d729b29dcfe9` plus a local profiling patch
already present before the Figure 18 run. The complete header snapshot,
including `build_profiler.h`, is bundled under `third_party/hnswlib/`;
licensing and hashes are recorded in `THIRD_PARTY_NOTICES.md` and
`data/SHA256SUMS`.
