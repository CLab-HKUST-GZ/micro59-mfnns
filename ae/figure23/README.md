# Figure 23: historical DRAM row-miss ratio

## Reproduce and validate

Run from the repository root:

```bash
python3 ae/figure23/plot_figure23.py --check-only
bash ae/figure23/reproduce_figure23.sh
```

The check parses and verifies all 21 archived YAML digests and prints
`CHECK_OK rows=21 yamls=21 ...`. Reproduction writes:

```text
ae/figure23/output/figure23.pdf
ae/figure23/output/figure23.png
ae/figure23/output/figure23_summary.tsv
```

## Historical-YAML boundary

The figure freezes seven datasets times ANSMET-open, NMP-FPSA-ET, and MFNNS.
`ae/figure23/configs/` contains the original byte-identical March/April YAMLs,
and `data/config_provenance.tsv` maps every plotted row to its configuration,
stats, Slurm source, and digests.

Those YAMLs intentionally retain author-machine absolute paths so that their
historical hashes remain intact. They are provenance records, not portable
one-command rerun recipes. The self-contained plotter recomputes row misses
per read from the frozen counters without accessing those historical paths.
Several points also predate the later `Recall@10 > 0.895` policy; Figure 23
reproduces the original evidence rather than silently replacing it.
