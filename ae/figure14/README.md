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
3. verifies the newly measured GloVe2M k=100 NMP-Base point;
4. keeps the NMP-FPMA point marked as a derived copy with no direct test;
5. writes `output/figure14.pdf`, `output/figure14.png`, and
   `output/figure14_summary.tsv`.

Use `--check-only` for a read-only data and metric check:

```bash
python3 ae/figure14/plot_figure14.py --check-only
```

## Measured-data update

The previous plot imputed the GloVe2M k=100 NMP-Base throughput from its k=10
speedup. A result completed later:

```text
s_mem_cycle = 701975198
s_num_query = 1000
QPS         = 1000 × 2.4e9 / 701975198 = 3418.924211
recall@100  = 0.90294
```

The normalized QPS changes from `12.307665×` to `12.133526×`. The existing
Figure 14 policy gives NMP-FPMA the same end-to-end QPS as NMP-Base because
their modeled compute throughput is the same. Its plotted value therefore
changes too, but it is still classified as `derived_copy_no_direct_test`; it
must not be described as a direct NMP-FPMA measurement.

The canonical data and raw stats are documented in
[`data/README.md`](data/README.md). Internal paper-edit notes are intentionally
not included in this repository package.

## Relation to the author workspace

The author-workspace source table, exported plotting CSV, PDF, and PNG were
updated in:

```text
MFANNS/figure/evaluation/speedup/
```

The author-workspace generator still assembles k=5/k=10 inputs from its
historical sources. This `ae/` bundle instead consumes the frozen 168-row CSV,
so it does not depend on absolute cluster paths or the historical memory tree.
