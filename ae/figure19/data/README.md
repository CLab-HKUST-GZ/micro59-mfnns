# Figure 19 data

## Files

`juno_fig8_designs.tsv`

: Complete 277-row numeric extraction of JUNO++ Fig. 8 vector paths, including
  PQ designs not selected for the default Figure 19 plot. Each row records PDF
  page, drawing index, point index, PDF coordinates, recall, `log10(QPS)`, QPS
  and `plot_default`. Exactly 164 rows have `plot_default=1` and are reproduced
  byte-for-byte as the JUNO++/HNSW portion of `figure19_plot_data.tsv`.

`figure19_plot_data.tsv`

: Frozen 353-row plotting input. It is byte-identical to the author workspace
  `figure/evaluation/RTC/data/rtc_recall_qps_plot_data.tsv`.

`figure19_mfnns_provenance.csv`

: One row for each of the 189 MFNNS plot inputs. It records the panel, measured
  recall/QPS/cycles, source memory experiment, TEST_RECORD, YAML generator,
  result summarizer, final selection script, case manifest, YAML and stats
  references and hashes, and the audited YAML/stats fields.

`figure19_source_experiments.csv`

: Four-row experiment-level summary with point contributions, recorded
  completion counts, script/manifest references, and SHA-256 digests.

`figure19_frontier_better_ranges.tsv`

: Common recall domains and intervals in which the interpolated MFNNS frontier
  is above the JUNO++ frontier.

`figure19_frontier_speedup_samples.tsv`

: Log-linearly interpolated QPS and raw MFNNS/JUNO++ ratios at recall 0.90 and
  0.95.

`SHA256SUMS`

: Digests for canonical data, the provenance summary, and generated PDF/PNG.
  Paths are relative to this `data/` directory.

## Source mapping

References beginning with `simulator/` or `figure/` are relative to the
original MFANNS author workspace. They intentionally contain no machine-
specific absolute prefix. The per-file hashes make the frozen mapping
auditable even when the large author memory tree is not shipped with the
portable artifact.

The source contribution matrix is:

| source memory | SIFT R@1 | T2I R@1 | SIFT R@100 | T2I R@100 | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260612/002 | 5 | 15 | 6 | 6 | 32 |
| 20260612/003 | 15 | 0 | 29 | 0 | 44 |
| 20260613/004 | 0 | 11 | 0 | 10 | 21 |
| 20260613/006 | 17 | 11 | 36 | 28 | 92 |
| **total** | **37** | **37** | **71** | **44** | **189** |

All four TEST_RECORD files state final completion with zero failed cases.

## JUNO++ source boundary

The raw table was produced by
`figure/evaluation/RTC/extract_rtc_data.py` from Fig. 8 on page 17
(zero-based PDF page 16) of `[TACO 25] JUNO++.pdf`, using PyMuPDF vector
drawing extraction and fixed panel/tick calibration.

Source PDF SHA-256:

```text
87ea7850149d7a4a99bb482eadd9323b768d03045e489efc1bf81eba11f8b869
```

The PDF itself is not redistributed. The numeric raw table is sufficient for
the plotting code and preserves the page/drawing/point provenance needed to
audit or replace the extraction.
