# Figure 17: Recall@10 system-energy breakdown

This bundle reproduces the paper's Recall@10 system-energy comparison,
normalized independently to `NMP-Base = 1` for every dataset.

## Reproduce

From the repository root:

```bash
bash ae/figure17/reproduce_figure17.sh
```

Read-only validation:

```bash
python3 ae/figure17/build_figure17_data.py --check-only
python3 ae/figure17/plot_figure17.py --check-only
```

## Default trace and YAML inputs

Figure 17 uses the same fixed inputs as Figure 16:

```text
trace:
  simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10/execution_traces.csv

final YAMLs:
  simulator/run_case/figure14_recall_gt0895/configs/final/k10/<dataset>/<design>.yaml
```

The common `ae/energy_model.py` validates all 42 physical traces and final
YAMLs before deriving the seven plotted logical methods. In particular,
`NMP-Base-ET` reuses the `mfnns` trace; its memory energy, elapsed cycles,
phase counters, and QPS match MFNNS, while its compute-energy model remains
the Base-FP model.

## Outputs

```text
data/figure17_energy_breakdown.csv
output/figure17_summary.tsv
output/figure17.pdf
output/figure17.png
```

The summary reports both the historical summed-energy aggregation and the
geometric mean of per-dataset energy ratios. These are not interchangeable;
paper text must name the aggregation it uses.
