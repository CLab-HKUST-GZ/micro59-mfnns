# Figure 14 data

## Layout

```text
data/
  k5_k10_latest_metrics.csv
  figure14_results.csv
  test_results/
    k100/glove2m/ndp_base/
      config.yaml
      stats.yml
```

`figure14_results.csv` is the canonical 168-row plotting input. It contains one
row for every combination of:

- top-k: k5, k10, k100;
- dataset: Deep10M, GloVe2M, SIFT1M, T2I1M, W2V1M, Wiki1M, PubMed;
- design: CPU, CAGRA, BANG, ANSMET, NMP-Base, NMP-FPMA/FPSA,
  NMP-FPSA-ET, MFNNS.

The CSV contains QPS, the common Figure 14 CPU QPS for the same `(k, dataset)`,
normalized QPS, recall, measurement status, source label, portable source
references, and notes. Its status counts are:

```text
measured_completion                 34
measured_reused                     47
derived_copy_from_ndp_base           3
external_measurement                42
measured                            33
derived_copy_no_direct_test          7
manual_selected_measurement          2
```

`k5_k10_latest_metrics.csv` is the compact 84-row source inventory for the
six simulator designs. It records selected ef/queue values, query count,
cycles, QPS, recall, final YAML, source stats reference, and stats SHA-256.
All 84 plotted rows have recall above 0.895. Of the 14 k=5/10 NDP-FPMA rows,
11 are direct measurements and three are explicitly copied from the latest
NDP-Base result because their direct jobs were cancelled.

## Newly available raw result

`test_results/k100/glove2m/ndp_base/stats.yml` is the exact late result used to
replace the interpolation. Its SHA-256 is:

```text
f3548cf7f409598a25805bcf98a256b3e65784918ecd3d0cf975d03908eebd80
```

`config.yaml` is the portable final configuration with the same k, ef, queue,
parallel-query, and design parameters. Large indexes, queries, and ground truth
are not duplicated here; they follow the normalized layout documented by the
Figure 14 simulator case.

## Normalization

Figure 14 plots CPU-normalized QPS. Every row now uses the CPU QPS selected for
the same Figure 14 `(k, dataset)`. BANG's measured QPS is retained, but its
previous source-specific CPU denominator is not used in this combined figure.
Consequently, ratios between plotted bars are also direct ratios of their QPS.
