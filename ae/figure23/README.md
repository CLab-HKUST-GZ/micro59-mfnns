# Figure 23: DRAM row-miss ratio

This bundle reproduces the historical Recall@10 DRAM row-miss comparison
among ANSMET with open-row policy, NMP-FPSA-ET, and MFNNS. Per the artifact
freeze request, it preserves the original March/April test points and does
not replace them with later measurements.

## Reproduce

From the repository root:

```bash
bash ae/figure23/reproduce_figure23.sh
```

Read-only validation:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 ae/figure23/plot_figure23.py --check-only
```

Outputs:

```text
output/figure23.pdf
output/figure23.png
output/figure23_summary.tsv
```

## Frozen data

`data/figure23_row_miss_ratio.csv` is the byte-identical author plotting
table. It contains:

```text
7 datasets x {ANSMET-open, NMP-FPSA-ET, MFNNS}
top-k = 10
1000 queries
32 simulated memory instances
```

Although the historical author output stem contains `top5`, its default
`all-complete` mode included seven datasets:

```text
DP, GV, SF, T2I, W2V, WK, PM
```

The plotted quantity is recomputed and validated as:

```text
row_miss_per_read =
  sum(row_misses_0 over 32 memories) /
  sum(num_read_reqs_0 over 32 memories)
```

This is not the alternative DRAM-event fraction
`misses / (hits + misses + conflicts)`.

## Original simulator YAMLs

`configs/` contains the 21 original YAML files that produced the plotted
rows. They are copied byte-for-byte from the author experiment directories:

```text
ANSMET-open:
  simulator/memory/20260404/
  ansmet_rowpolicy_seven_dataset_k10_nq1000

NMP-FPSA-ET and MFNNS:
  simulator/memory/20260331/
  eight_config_k10_nq1000_noncpu_ndpfpma_nfmac16_emb017_stdfp16_finalize40_12
```

The YAMLs intentionally retain historical absolute input and `stat_path`
values so their digests remain identical to the original test artifacts.
They are provenance records, not portable one-command rerun recipes.

`data/config_provenance.tsv` maps every plotting row to its archived YAML,
original author-workspace YAML/stats/Slurm reference, configuration, recall,
row policy, and SHA-256 digests. The portable plotter validates all 21 YAML
digests before drawing.

## Historical producer

The original author entry point is preserved byte-for-byte at:

```text
scripts/historical/plot_row_miss_ratio_with_ansmet_open_top5.py
```

It depends on other author-workspace plotting modules and reads raw Slurm
logs. `plot_figure23.py` is the self-contained AE equivalent: it reads the
frozen CSV, recomputes every derived ratio, checks the YAML provenance
manifest, and reproduces the same figure without the large historical Slurm
tree.

## Historical-data boundary

Several frozen points predate the later final operating-point selection and
do not satisfy the later `Recall@10 > 0.895` policy. This bundle exposes those
recalls in `config_provenance.tsv` but deliberately does not modify or filter
them. It is a reproduction of the original Figure 23 evidence, not a
latest-data refresh.

Validate deterministic artifacts from `ae/figure23/data/`:

```bash
cd ae/figure23/data
sha256sum -c SHA256SUMS
```

The PDF is excluded from deterministic validation because Matplotlib embeds
its creation timestamp.
