# Figure 22: Recall@10 latency breakdown

This bundle reproduces the paper's normalized ANSMET/MFNNS latency-breakdown
figure from the final Recall@10 operating points. It contains no
machine-specific absolute paths.

## Reproduce

From the repository root:

```bash
bash ae/figure22/reproduce_figure22.sh
```

Read-only validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure22/plot_figure22.py --check-only
```

## Data and decomposition

`data/figure22_latency_breakdown.csv` freezes the 14 selected rows:

```text
7 datasets x {ANSMET, MFNNS}
top-k = 10
recall > 0.895
1000 queries
```

Each row records the Figure 14 operating point, stats reference and digest,
formal Slurm reference, and the five raw latency-breakdown counters. The
plotter recomputes:

```text
distance = total - index - result_collection
hidden_ratio = hidden_FMAC_cycles / raw_FMAC_cycles
data_movement = distance * hidden_ratio
distance_computation = distance * (1 - hidden_ratio)
normalized_component = component / ANSMET_total_for_the_dataset
```

The `stats_ref` and `source_slurm_ref` fields are repository-relative
provenance identifiers from the full author workspace. The remote AE bundle
does not duplicate the large raw experiment directories; the frozen counters,
strict data checks, and SHA-256 inventory are sufficient for plot
reproduction.

## Headline values

- Arithmetic-mean total-latency reduction: `30.5144%`.
- Geometric-mean speedup: `1.4916x`.
- Arithmetic-mean visible distance-computation reduction: `55.3810%`.
- Arithmetic-mean data-movement reduction: `18.3429%`.
- Result collection and index traversal change by only `2.4353%` and
  `4.4768%` on average.

These replace the stale March-data values (`37.49%`, `1.63x`, and roughly
`67%`) previously associated with Figure 22.

## Outputs

```text
output/figure22.pdf
output/figure22.png
output/figure22_summary.tsv
```

Validate deterministic artifacts from `ae/figure22/data/`:

```bash
sha256sum -c SHA256SUMS
```

The PDF is excluded from the deterministic inventory because Matplotlib
embeds its creation timestamp. The PNG, numeric summary, and source CSV are
checked.
