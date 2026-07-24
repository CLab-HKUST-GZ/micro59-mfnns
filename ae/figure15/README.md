# Figure 15: area-efficiency comparison

This bundle reproduces the paper's Recall@10 area-efficiency figure from the
canonical Figure 14 results already stored in the repository. It does not
duplicate simulator logs or configurations.

## Reproduce

From the repository root:

```bash
bash ae/figure15/reproduce_figure15.sh
```

Equivalent individual commands are:

```bash
python3 ae/figure15/build_figure15_data.py
python3 ae/figure15/plot_figure15.py
```

Use the read-only checks when generated files should not be changed:

```bash
python3 ae/figure15/build_figure15_data.py --check-only
python3 ae/figure15/plot_figure15.py --check-only
```

Requirements are Python 3, Matplotlib, and NumPy. The exact Python package
names are listed in `requirements.txt`.

## Inputs and outputs

The data builder reads:

```text
ae/figure14/data/figure14_results.csv
ae/figure15/data/area_specs.csv
```

It selects the Recall@10 rows for seven datasets and writes:

```text
ae/figure15/data/figure15_area_efficiency.csv
ae/figure15/output/figure15_summary.tsv
```

The plotting script validates the generated 7 datasets × 6 methods matrix and
writes:

```text
ae/figure15/output/figure15.pdf
ae/figure15/output/figure15.png
```

Figure 14 remains the source of QPS, recall, measurement status, source
references, and fallback notes. The corresponding final YAML files and test
evidence remain under `simulator/run_case/figure14_recall_gt0895/` and the
paths recorded by the Figure 14 CSV.

## Area and QPS mapping

| Figure 15 method | Figure 14 QPS row | Area (mm²) | Interpretation |
| --- | --- | ---: | --- |
| ANSMET | `ansmet` | 0.020913242 | measured QPS |
| NMP-Base | `ndp_base` | 0.020913242 | measured QPS |
| NMP-FPMA | `ndp_fpma` | 0.01208361 | direct QPS where available; explicit Figure 14 fallback otherwise |
| NMP-FPSA | `ndp_fpma` | 0.009678554 | derived QPS reuse; same DPE compute performance as NMP-FPMA |
| NMP-Base-ET | `mfnns` | 0.020913242 | derived current MFNNS QPS with NMP-Base area |
| MFNNS | `mfnns` | 0.009678554 | measured QPS |

The two Recall@10 NMP-FPMA fallbacks are Wiki1M and PubMed. Their direct jobs
were cancelled, so their selected Figure 14 rows explicitly copy NMP-Base.
The builder checks that these fallbacks remain explicit and exports their
provenance instead of treating them as independent measurements.

## Current result

MFNNS improves area efficiency over ANSMET by `3.184820410251080x` in
geometric mean (`3.18x` for paper display). The per-dataset plotted labels are:

```text
DP 2.3x, GV 4.2x, SF 2.3x, T2I 3.0x,
W2V 3.9x, WK 2.6x, PM 4.8x
```

The staged ratios are:

```text
NMP-FPMA / NMP-Base = 1.730711434745080x
NMP-FPSA / NMP-FPMA = 1.248493318320071x
MFNNS / NMP-FPSA = 2.064288747239976x
MFNNS / NMP-Base-ET = 2.160781662219377x
```

Internal paper-edit notes are intentionally not part of this remote bundle.
