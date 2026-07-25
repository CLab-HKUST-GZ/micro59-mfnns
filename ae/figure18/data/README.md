# Figure 18 data

## `figure18_recall_qps.csv`

This is the portable 177-row plotting table:

```text
T2I1B Recall@10    49
DP1B  Recall@10    48
T2I1B Recall@100   41
DP1B  Recall@100   39
```

`recall_raw` preserves the frozen author-workspace value.  `recall_2dp` is the
decimal `ROUND_HALF_UP` value used by the default Figure 18 plot.
ANSMET/MFNNS rows link to a portable YAML and one of the provenance statuses
below. CPU/BANG retain the historical `external_frozen_plot_input` label in
the canonical CSV for byte stability, but their newly recovered producers and
evidence are now packaged separately:

```text
CPU:  ../cpu/data/figure18_cpu_selected_trials.tsv
BANG: expected_deep1b_bang_curve.tsv
      ../scripts/run_deep1b_bang_curve.sh
      ../scripts/run_t2i1b_mips_l2_curve.sh
```

## `simulator_provenance.tsv`

The 108-row manifest maps every ANSMET/MFNNS plotting row to:

- method, dataset, recall metric, ef, and LBQueue;
- frozen recall, QPS, and cycle;
- portable configuration path and SHA-256;
- original YAML/stats workspace references and SHA-256;
- source stats recall/cycle/QPS;
- recall and cycle differences;
- provenance status and an explanatory note.

Status counts:

```text
verified_original_yaml_stats              88
original_yaml_cycle_mismatch               1
reconstructed_from_same_panel_template    19
```

The frozen plot recall is not silently replaced by stats recall.  Among the 88
cycle-verified rows, the maximum raw recall difference is 0.0084; 80 rows
remain equal after independent two-decimal rounding.  The manifest exposes
these differences for review.

## BANG reference

`expected_deep1b_bang_curve.tsv` is the 18-point, 2026-06-15 Deep1B
steady-state reference synchronized from the remote AE branch. It is frozen
comparison data, not output generated during figure rendering and not an
exact replacement for the paper's older BANG QPS rows. Its recall grid agrees
within `0.0001`; three QPS anchors are exact. The T2I1B BANG points in the
plot remain historical inputs; see the top-level README for the
metric-compatible 201D rerun requirement.

`SHA256SUMS` covers the three canonical Figure 18 data files. Each collected
YAML has its own SHA-256 in `simulator_provenance.tsv`; CPU source/evidence
has a separate inventory under `../cpu/data/SHA256SUMS`.
