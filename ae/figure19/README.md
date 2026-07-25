# Figure 19: MFNNS versus JUNO++ recall-QPS frontiers

This bundle reproduces the paper's four-panel Figure 19 and records the
point-level provenance of every plotted MFNNS frontier input.

## Result

The artifact also includes the complete 277-row JUNO++ Fig. 8 vector
extraction in `data/juno_fig8_designs.tsv`. The Figure 19 plot reads 353
selected rows:

- 164 JUNO++/HNSW vector-extracted rows;
- 189 MFNNS simulator frontier rows.

The plot validator reconstructs the 164 JUNO++/HNSW inputs from the 277-row
raw table using `plot_default=1` and requires exact equality with the combined
plot TSV. The remaining 113 PQ/raw-design points are retained for provenance
and alternative plotting.

All 189 MFNNS rows map one-to-one to a unique completed memory case, case
manifest row, YAML, and stats file. The YAML and stats audit verifies:

- dataset and recall tag;
- `ef_search`, lower-bound queue size, and warmup size;
- `gt_k=100`;
- `k_neighbors=1` for Recall@1 and `k_neighbors=100` for Recall@100;
- `nQueryLimit=nParallelQuery=1000`;
- `mfnnsEnable=true`, `earlyExitEnable=false`, and
  `dualQueueLowerBoundETEnable=true`;
- YAML `stat_path`;
- stats recall, memory cycles, and query count;
- `QPS = 1000 * 2.4 GHz / s_mem_cycle`;
- equality with both the merged outer frontier and the final plot TSV.

The 189 selected points come from four completed memory experiments:

| memory experiment | selected points | YAML-generation script |
| --- | ---: | --- |
| `20260612/002_sift_t2i_ansmet_mfnns_r1_r100_frontier` | 32 | `scripts/generate_submit_frontier.py` |
| `20260612/003_sift_mfnns_lowef_subef_pareto` | 44 | `scripts/generate_submit_sift_mfnns_refine.py` |
| `20260613/004_sift_t2i_mfnns_lowef_subef_frontier` | 21 | `scripts/generate_submit_t2i_mfnns_refine.py` |
| `20260613/006_mfnns_ansmet_lbq_juno_focused` | 92 | `scripts/generate_submit_lbq_tune.py` |

The final 189-point merge/frontier selection is performed by
`20260613/006.../scripts/summarize_lbq_tune.py`. RTC normalization and plot
data export are performed by `figure/evaluation/RTC/extract_rtc_data.py`.

## Portable reproduction

From the repository root:

```bash
bash ae/figure19/reproduce_figure19.sh
```

This uses only the frozen portable data in `data/` and writes:

```text
output/figure19.pdf
output/figure19.png
```

Read-only portable validation:

```bash
python3 ae/figure19/plot_figure19.py --check-only
```

## Author-memory provenance audit

The point-level audit requires the original author workspace. By default the
builder expects it as the sibling directory `../MFANNS`:

```bash
python3 ae/figure19/build_figure19_data.py
python3 ae/figure19/build_figure19_data.py --check-only
```

Use `--source-root` if the author workspace is elsewhere:

```bash
python3 ae/figure19/build_figure19_data.py \
  --source-root /path/to/MFANNS
```

The builder deliberately does not copy 189 full YAMLs or stats files into the
artifact. Instead, `data/figure19_mfnns_provenance.csv` stores their portable
author-workspace-relative references, SHA-256 digests, and all fields used by
the figure audit.

The JUNO++ paper PDF is not redistributed. The raw table records PDF page,
drawing index, point index, vector coordinates, recall, `log10(QPS)`, QPS and
selection status. Its source PDF SHA-256 and extraction boundary are documented
in `data/README.md`.

## Display-range detail

All 189 MFNNS points are plot inputs. Figure 19 displays recall from `0.4` to
`1.01`; 160 point coordinates lie inside that range and 29 lie below it.
Below-range points are retained because Matplotlib may use a neighboring point
when clipping a line segment at the left boundary. `point_within_xlim` records
this distinction without incorrectly labelling the 29 rows as unused.

## Interpretation boundary

JUNO++ QPS is extracted from Fig. 8 vector graphics in the JUNO++ paper.
MFNNS QPS is derived from simulator cycles at 2.4 GHz. These values are not
hardware-normalized. Figure 19 supports frontier-shape and raw-QPS comparison,
not a same-platform measured speedup claim.
