# Figure 14: throughput comparison

This bundle reproduces Figure 14 from a portable CSV. It is intentionally
limited to Figure 14 and does not modify the area, energy, or billion-scale
figures.

## Reproduce

From the repository root:

```bash
python3 ae/figure14/plot_figure14.py
```

Requirements are Python 3, Matplotlib, and NumPy. The command:

1. validates the exact 3 top-k × 7 dataset × 8 design matrix;
2. verifies that all data references are repository/workspace relative;
3. verifies all 84 k=5/10 simulator rows satisfy strict `recall > 0.895`;
4. verifies 11 direct NDP-FPMA measurements and exactly three explicit
   NDP-Base fallbacks;
5. verifies every method is normalized with the Figure 14 CPU for the same
   `(k, dataset)`;
6. verifies the measured GloVe2M k=100 NMP-Base point;
7. writes `output/figure14.pdf`, `output/figure14.png`, and
   `output/figure14_summary.tsv`.

Use `--check-only` for a read-only data and metric check:

```bash
python3 ae/figure14/plot_figure14.py --check-only
```

## k=5/10 selected-result update

The k=5/10 simulator slice now uses the selected July formal results:

```text
47 reused formal measurements
34 completion-batch measurements
 3 explicit NDP-FPMA fallbacks
84 plotted simulator rows
```

All 81 completed measurements satisfy strict `recall > 0.895`. Eleven
NDP-FPMA points use direct measurements. The three cancelled points—k=5
PubMed, k=10 Wiki1M, and k=10 PubMed—copy the latest selected NDP-Base metrics
for the same `(k, dataset)` and are marked `derived_copy_from_ndp_base`.

CAGRA, BANG, and k=100 retain their measured QPS. Normalized QPS is recomputed
against one Figure 14 CPU QPS per `(k, dataset)`, including BANG; this removes
the previous mixed-CPU-normalizer ambiguity. The previously added measured
GloVe2M k=100 NMP-Base result remains in use.

The canonical data and source inventory are documented in
[`data/README.md`](data/README.md). Internal paper-edit notes are intentionally
not included in the remote package.

## Relation to the author workspace

The author-workspace source table, exported plotting CSV, PDF, and PNG were
updated in:

```text
MFANNS/figure/evaluation/speedup/
```

The author-workspace generator consumes
`figure14_k5_k10_latest_metrics.csv` for the six simulator designs and retains
its historical source only for k=5/10 CAGRA QPS. The portable `ae/` plot
consumes the frozen 168-row CSV and therefore does not require the historical
memory tree.
