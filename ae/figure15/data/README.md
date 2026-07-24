# Figure 15 data

`area_specs.csv` records the six plotted methods, the current Figure 14 design
whose QPS is used, the synthesized DPE area, and whether the row is measured
or derived.

`figure15_area_efficiency.csv` is generated from:

```text
../../figure14/data/figure14_results.csv
area_specs.csv
```

It contains 42 rows: one for each combination of seven datasets and six
methods at Recall@10. Every row preserves the selected Figure 14 QPS, recall,
measurement status, source label, stats/config reference, and source note.

The derived metrics are:

```text
area_efficiency_kqps_per_mm2 = (QPS / 1000) / area_mm2
normalized_area_efficiency_vs_ansmet =
    area_efficiency(method) / area_efficiency(ANSMET)
```

The current Figure 14 source contains five direct Recall@10 NMP-FPMA results
and two explicit NMP-Base fallbacks for Wiki1M and PubMed. The data builder
validates those counts and requires all referenced k=10 source rows to have
recall above 0.895.

`SHA256SUMS` covers the area specification and generated CSV. Plot-file hashes
are not pinned because PDF metadata and font rendering can vary with
Matplotlib and system fonts; the plotting script validates all numeric values
before rendering.
