# Figure 16 data

`figure16_energy_efficiency.csv` is generated from three repository inputs:

1. Recall@10 QPS/recall in `../../figure14/data/figure14_results.csv`;
2. the fixed simulator execution summary in
   `../../../simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10/execution_traces.csv`;
3. CAGRA/BANG measured power archived in `external_power.csv`.

For simulator methods, power is recomputed from the selected Figure 14 QPS
and energy per 1000-query batch. Consequently:

```text
power_W = total_energy_nJ * QPS / 1e12
QPS/W   = 1e12 / total_energy_nJ
```

CPU retains the documented historical `x30` DRAM-energy estimate. Non-CPU
methods use direct 1000-query DRAM energy.

The source strings in `external_power.csv` are repository-independent
provenance identifiers, not filesystem paths.
