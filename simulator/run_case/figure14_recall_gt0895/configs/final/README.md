# Portable Figure 14 formal configurations

This directory contains one YAML for every point in the `k=5/10/100` Figure 14
simulator matrix:

```text
3 top-k values x 7 datasets x 6 designs = 126 YAML files
```

The hierarchy is:

```text
configs/final/k<k>/<dataset>/<design>.yaml
```

All filesystem fields are paths relative to the repository root. Run the
simulator from the repository root so these paths resolve consistently.
Prepared inputs use a normalized, variant-free layout:

```text
mfnns_hnswlib/cpu_index/<dataset>/
  hnsw_index_M32_ef100.bin
  query_vectors_n1000_seed42.bin
  gt_labels_topk5_n1000_seed42.bin
  gt_labels_topk10_n1000_seed42.bin
  gt_labels_topk100_n1000_seed42.bin
```

Simulator stats are written below:

```text
simulator/run_case/figure14_recall_gt0895/results/k<k>/<dataset>/
```

Validate the complete portable matrix from the repository root:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
```

The complete mapping is in
`simulator/run_case/figure14_recall_gt0895/manifests/final_cases.tsv`. The
portable configs were consolidated from the Figure 14 completion campaign
without retaining machine-specific paths. Three NDP-FPMA YAMLs have
`historical_result_status=not_run_cancelled`; the configuration files are
included to make the matrix complete, but no completed historical result is
claimed for them.

The k=100 configurations were matched against the legacy Figure speedup CSV
and the corresponding simulator-memory results. Their source map is
`simulator/run_case/figure14_recall_gt0895/manifests/k100_sources.tsv`. It
distinguishes 33 direct matches, two Figure manual overrides, and seven
NDP-FPMA rows copied from NDP-Base. The late GloVe2M k=100 NDP-Base result now
replaces the former Figure imputation.
