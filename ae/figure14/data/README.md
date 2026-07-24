# Figure 14 data

## Layout

```text
data/
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

The CSV contains QPS, the applicable CPU QPS, normalized QPS, recall,
measurement status, source label, portable source references, and notes. Its
status counts are:

```text
measured                         103
external_measurement              42
derived_copy_no_direct_test       21
manual_selected_measurement        2
```

The 21 derived rows are the seven NMP-FPMA copies at each of k5, k10, and k100.
They are retained to reproduce the paper's explicit NMP-FPMA plotting policy,
not presented as direct tests.

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

## Normalization caveat

Figure 14 plots CPU-normalized QPS. BANG data uses the CPU reference supplied
with the BANG measurements, which is why paper-level comparisons must use the
`qps_speedup_vs_cpu` column rather than recomputing every comparison from raw
QPS. The plotting script follows that definition.
