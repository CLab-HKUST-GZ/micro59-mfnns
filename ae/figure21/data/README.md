# Figure 21 data

- `figure21_sweep_results.tsv`: the 243 completed MFNNS sweep rows consumed by
  the plotter.
- `simulator_provenance.tsv`: per-case YAML/stats/job provenance and hashes for
  243 MFNNS cases plus three ANSMET references.
- `ansmet_stats/`: the three complete raw ANSMET stats files used for reference
  recalls.
- `input_manifest.tsv`: exact input sizes and hashes, including the large index
  that is intentionally not duplicated in Git.
- `SHA256SUMS`: deterministic data, inputs, and exported-TSV inventory. Rendered
  PDF/PNG files are excluded because font and Matplotlib versions can change
  their bytes without changing the plotted values.

The plotter computes normalized throughput as:

```text
global maximum s_mem_cycle / case s_mem_cycle
```

For this sweep, the global maximum is `618147`.
