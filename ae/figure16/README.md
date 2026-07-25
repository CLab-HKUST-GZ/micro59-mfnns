# Figure 16: Recall@10 energy efficiency

This bundle reproduces the paper's system-level Recall@10 QPS/W comparison.
It follows the existing `ae/figureNN/` layout and contains no machine-specific
absolute paths.

## Reproduce

From the repository root:

```bash
bash ae/figure16/reproduce_figure16.sh
```

Read-only validation is available through:

```bash
python3 ae/figure16/build_figure16_data.py --check-only
python3 ae/figure16/plot_figure16.py --check-only
```

## Default trace and YAML inputs

The shared energy model reads the archived execution-trace summary from:

```text
simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10/execution_traces.csv
```

For every physical `(dataset, design)` trace, it then reads and validates:

```text
simulator/run_case/figure14_recall_gt0895/configs/final/k10/<dataset>/<design>.yaml
```

Each YAML must use 1000 queries, write stats below the portable
`results/k10/` directory, and enable DRAMPower. The trace table records the
source stats/energy references and their SHA-256 digests, simulator counters,
and the summed DRAM energy. Future results should replace this table only
after being extracted from the same fixed final-YAML matrix.

QPS and recall come from `ae/figure14/data/figure14_results.csv`. CAGRA and
BANG measured power values are archived in `data/external_power.csv`.

## Energy policy

- CPU retains the historical estimate: old nq1000 compute counters plus the
  nq32 DRAM-energy result scaled by `x30`.
- All non-CPU memory energy is taken directly from 1000-query, 32-bank traces.
- `NMP-Base-ET` and `NMP-FPSA-ET` share the same `ndp_et` trace, memory
  energy, elapsed cycles, and QPS. Only their compute-energy formulas differ.
- `NMP-FPMA` and `NMP-FPSA` share the selected `ndp_fpma` trace and differ
  only in modeled compute energy.

## Outputs

```text
data/figure16_energy_efficiency.csv
output/figure16_summary.tsv
output/figure16.pdf
output/figure16.png
```

The builder fails if the ET pair does not share the same memory trace or if
Base-ET does not have higher total energy and lower QPS/W than FPSA-ET.
