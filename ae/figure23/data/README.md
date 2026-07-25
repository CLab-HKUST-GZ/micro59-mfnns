# Figure 23 data and provenance

## `figure23_row_miss_ratio.csv`

This is the byte-identical 21-row author plotting table. For each dataset and
method it stores the historical source/run/Slurm paths, 32-memory counter
sums, and derived values.

The portable plotter independently validates:

```text
occurrences == 32
service_cycles_per_read == sum_service_cycles_0 / sum_num_read_reqs_0
row_miss_per_read == sum_row_misses_0 / sum_num_read_reqs_0
```

The absolute paths in this frozen table are historical provenance strings;
they are not dereferenced during AE reproduction.

## `config_provenance.tsv`

This 21-row manifest maps every plot row to:

- the archived YAML and its SHA-256;
- repository-relative historical YAML, stats, and Slurm references;
- stats and Slurm SHA-256 values from the author workspace;
- row policy, `ef_search`, LBQueue size, recall, query count, and `gt_k`.

The archived YAMLs are verified locally during every `--check-only` or plot
run. Historical stats and Slurm files are not duplicated in the compact
remote bundle; the plotting table freezes their raw 32-memory sums.
