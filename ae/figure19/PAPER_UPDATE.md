# Figure 19 paper-facing summary

## Figure identity

Figure 19 is the 2x2 recall-QPS comparison between the MFNNS simulator
frontier and JUNO++ Fig. 8 vector data:

- SIFT1M Recall@1 (`R1@100` in the JUNO++ figure);
- T2I1M Recall@1 (`R1@100`);
- SIFT1M Recall@100 (`R100@1000`);
- T2I1M Recall@100 (`R100@1000`).

Use `output/figure19.pdf` as the paper graphic.

## MFNNS measurement provenance

The plot contains 189 distinct completed MFNNS simulator points:

```text
32  broad SIFT/T2I points from memory 20260612/002
44  SIFT low-ef/sub-ef refinement points from memory 20260612/003
21  T2I low-ef/sub-ef refinement points from memory 20260613/004
92  JUNO-focused LBQ-fill points from memory 20260613/006
---
189 MFNNS plot inputs
```

Each point has a unique YAML and stats file. All YAML, manifest, stats,
frontier, and plot fields pass the checks documented in `README.md`.

## Recomputed comparison values

MFNNS is above the log-linearly interpolated JUNO++ frontier over:

| panel | common recall domain | MFNNS-better interval |
| --- | --- | --- |
| SIFT1M Recall@1 | 0.440003–0.992000 | 0.819–0.992 |
| T2I1M Recall@1 | 0.470002–0.959997 | 0.512–0.960 |
| SIFT1M Recall@100 | 0.380003–0.989997 | 0.857–0.990 |
| T2I1M Recall@100 | 0.520002–0.959997 | 0.682–0.960 |

At recall 0.90, the four raw-QPS ratios range from `1.357598x` to
`4.336218x`, with geometric mean `2.197012x`.

At recall 0.95, the four ratios range from `1.497516x` to `4.333194x`, with
geometric mean `2.387686x` (`2.39x` for paper display).

## Required wording boundary

Do not call these same-hardware measured speedups. JUNO++ QPS is read from
paper vector graphics, whereas MFNNS QPS is computed from simulator batch
cycles using:

```text
QPS = 1000 queries * 2.4 GHz / s_mem_cycle
```

Appropriate wording is “raw-QPS frontier comparison” or “MFNNS/JUNO++
frontier ratio under the respective reported models.”
