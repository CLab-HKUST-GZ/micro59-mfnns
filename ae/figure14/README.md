# Figure 14: throughput comparison

Figure 14 has three distinct workflows: replot the shipped 168-row CSV, rerun
the 126 simulator-backed points, or rerun the 42 CAGRA/BANG points. The default
`ae/` command performs only the first workflow. It does **not** execute the
simulator or GPU baselines, rebuild their inputs, or regenerate the CSV from
new stats.

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
rows in the 168-row plot are the BANG and CAGRA baselines; those are not
simulator YAMLs and are handled by the separate GPU workflow below. Fresh
simulator stats are written below
`simulator/run_case/figure14_recall_gt0895/results/`; they do not automatically
replace the shipped plotting CSV.

First validate the YAML matrix without building datasets or running a case:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
simulator/run_case/figure14_recall_gt0895/scripts/run_final_cases.sh --dry-run
```

Then follow
`simulator/run_case/figure14_recall_gt0895/README.md` to download raw datasets,
build all 35 normalized index/query/GT inputs, build `ramulator2`, preflight the
inputs, and run one or all 126 cases.

## Rerun the CAGRA and BANG experiments

The remaining 42 points are frozen as 21 CAGRA and 21 BANG parameter rows under
`GPU_Baseline/params/`. Their source-data conversion, index construction,
compile-time contracts, smoke tests, and manifest-driven runners are documented
in:

```text
GPU_Baseline/README.md
GPU_Baseline/CAGRA/README.md
GPU_Baseline/BANG/README.md
```

Start with:

```bash
bash script/test_gpu_baseline_repro.sh
bash script/test_bang_index_build.sh
```

The exact 42-point rerun requires the historical 1,000,000-row `pubmed_d2v`
base for the three CAGRA Pubmed points. With only the public 500,000-row Pubmed
corpus, 39 points retain their historical dataset contract; the three CAGRA
Pubmed points can still run as an explicitly marked variant. See
`GPU_Baseline/README.md` for commands and the qualification.

GPU results are written to reviewer-selected output roots and do not overwrite
`ae/figure14/data/figure14_results.csv`.
