# Figure 14 simulator results

Portable formal YAMLs write generated stats below this directory using:

```text
results/k<k>/<dataset>/<design>_stats.yml
```

Generated stats are ignored by Git. The historical result evidence remains
under `provenance/` and is not overwritten by this portable run interface.

## Paper energy trace summary

Figures 16 and 17 default to the tracked, fixed Recall@10 trace summary:

```text
results/paper_energy/k10/execution_traces.csv
```

This table contains the counters and DRAM-energy sums extracted from the
selected executions, together with repository-relative source references and
SHA-256 digests. The figure builders validate each row against the matching
final YAML under `../configs/final/k10/`. Generated `*_stats.yml` files remain
ignored and are not required to reproduce the archived paper figures.
