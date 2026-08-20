# Figure 14: throughput comparison

Figure 14 has two distinct reproduction claims. The default `ae/` command
validates and replots the shipped 168-row CSV. It does **not** execute the
simulator, rebuild index/query/GT inputs, or regenerate the CSV from new stats.

## Validate and replot the shipped data

Requirements are Python 3 and the packages in `ae/requirements.txt`. Run from
the repository root:

```bash
python3 ae/figure14/plot_figure14.py --check-only
bash ae/figure14/reproduce_figure14.sh
```

The check prints `DATA_OK rows=168 ...`. Reproduction writes:

```text
ae/figure14/output/figure14.pdf
ae/figure14/output/figure14.png
ae/figure14/output/figure14_summary.tsv
```

## Rerun the simulator experiments

The repository also contains 126 runnable simulator YAMLs:

```text
3 top-k values x 7 datasets x 6 simulator designs = 126 cases
```

They cover CPU, ANSMET, NDP-Base, NDP-FPMA, NDP-ET, and MFNNS. The other 42
rows in the 168-row plot are the external BANG and CAGRA baselines; those are
not simulator YAMLs and are not rerun by this workflow. Fresh simulator stats
are written below `simulator/run_case/figure14_recall_gt0895/results/`; they do
not automatically replace the shipped plotting CSV.

First validate the YAML matrix without building datasets or running a case:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
simulator/run_case/figure14_recall_gt0895/scripts/run_final_cases.sh --dry-run
```

Then follow
`simulator/run_case/figure14_recall_gt0895/README.md` to download raw datasets,
build all 35 normalized index/query/GT inputs, build `ramulator2`, preflight the
inputs, and run one or all 126 cases.
