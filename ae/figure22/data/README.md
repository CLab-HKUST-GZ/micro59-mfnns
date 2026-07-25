# Figure 22 data

`figure22_latency_breakdown.csv` contains one row per final Recall@10
ANSMET/MFNNS operating point. The selected configurations come from
`ae/figure14/data/k5_k10_latest_metrics.csv`; the latency counters were
extracted from the matching formal Slurm outputs after requiring:

```text
Slurm Total Memory cycle == selected Figure 14 s_mem_cycle
```

All 14 rows passed that identity check. The source audit and old/new
comparison are recorded in
`memory/20260725/019_latency_breakdown_topk10_latest_data_audit/` in the
author workspace.
