# Figure 21: LBQueue-size sensitivity

## Reproduce and validate

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure21/validate_figure21.py
bash ae/figure21/reproduce_figure21.sh
```

Validation parses and verifies 243 MFNNS YAMLs (queue sizes 20--100 at
`ef_search` 20, 30, and 40), three ANSMET YAMLs, the frozen sweep, and the
included query/ground-truth digests. It prints
`CHECK_OK sweep=243 ansmet=3 configs=246 ...`. Reproduction writes:

```text
ae/figure21/output/figure21.pdf
ae/figure21/output/figure21.png
ae/figure21/output/figure21_plot_data.tsv
```

## Optional simulator rerun

The safe default only selects cases:

```bash
python3 ae/figure21/run_figure21_sweep.py \
  --method mfnns --ef 30 --queue 60
```

A real run needs the existing simulator build and the HNSW index at
`mfnns_hnswlib/cpu_index/t2i1m/hnsw_index_M32_ef100.bin`. Queries and ground
truth are included under `ae/figure21/inputs/`. Build the missing index with
`script/cpu_index_build.sh t2i1m`, then submit to an explicit numbered result
directory:

```bash
python3 ae/figure21/run_figure21_sweep.py \
  --method mfnns --ef 30 --queue 60 \
  --submit --result-root memory/YYYYMMDD/NNN_figure21_ef30_q60
```

The runner generates task-specific runtime YAMLs, reuses
`simulator/build/ramulator2`, and refuses an implicit full 246-case sweep.
Use `ae/figure21/summarize_rerun.py RESULT_ROOT` after jobs finish.
