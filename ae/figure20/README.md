# Figure 20: DPE area and power breakdown

This bundle reproduces Figure 20 in the AE manuscript:

> Area and power improvement of DPE in MFNNS.

It is the horizontal two-panel version: (a) area breakdown and (b) power
breakdown. All paths used by the reproduction command are repository-relative.

## Reproduce

From the repository root:

```bash
bash ae/figure20/reproduce_figure20.sh
```

Read-only data and derived-value validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure20/plot_figure20.py --check-only
```

Requirements are listed in `requirements.txt`. The script selects an
available Times-compatible serif font and therefore does not require a
machine-specific font path.

## Data and normalization

`data/figure20_area_power_breakdown.tsv` freezes the author plot's two raw
tables:

```text
2 metrics x 3 designs x 4 DPE components
```

For each panel, every component is normalized by the sum of ANSMET's four
components:

```text
normalized_component(metric, design, component)
  = raw_value(metric, design, component) / ANSMET_total(metric)
```

The stacked ANSMET bar is therefore 1.0. The plotter validates the complete
matrix and checks the paper annotations:

```text
Area:  NMP-FPMA / MFNNS = 1.2x; ANSMET / MFNNS = 2.2x
Power: NMP-FPMA / MFNNS = 1.2x; ANSMET / MFNNS = 2.7x
```

The raw table does not state physical units; Figure 20 reports only the
dimensionless normalized breakdown, so this bundle does not invent unit
labels.

## Outputs

```text
output/figure20.pdf
output/figure20.png
output/figure20_normalized.csv
output/figure20_summary.tsv
```

`figure20_normalized.csv` exposes all raw and derived component values.
`figure20_summary.tsv` records the total values and both annotated ratios for
each panel.

## Author-source provenance

The paper-matching source was identified in the author workspace as:

```text
MFANNS/figure/evaluation/A_P_Breakdown/data.txt
MFANNS/figure/evaluation/A_P_Breakdown/plot_ap_breakdown_singlecol_horizontal.py
MFANNS/figure/evaluation/A_P_Breakdown/ap_breakdown_singlecol_horizontal.{pdf,png}
```

The horizontal source is used because its layout and `1.2x/2.2x` and
`1.2x/2.7x` annotations match the compiled Figure 20. The same directory also
contains a vertical plotting variant, which is not the manuscript figure.
Source hashes and the portable data inventory are documented under `data/`.
