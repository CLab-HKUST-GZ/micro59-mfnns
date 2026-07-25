# Figure 18 simulator YAMLs

Layout:

```text
configs/<dataset>/<r10|r100>/<ansmet|mfnns>/*.yaml
```

There are 108 configurations, one per plotted ANSMET/MFNNS point.  Every file
starts with a `provenance_status` comment and its source-YAML reference.

`verified_original_yaml_stats` files preserve the original configuration
semantics; only `stat_path` is made runtime-relative.  Files marked
`original_yaml_cycle_mismatch` or
`reconstructed_from_same_panel_template` are rerun recipes and must not be
presented as historical evidence until a new stats file is produced.

The index, query, and ground-truth paths point to the prepared author-workspace
billion-scale data.  Large inputs are intentionally not duplicated in `ae/`.
