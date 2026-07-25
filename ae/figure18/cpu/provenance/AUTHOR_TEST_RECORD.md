# Fresh HNSWLib CPU QPS-Recall Rerun

## Goal

Rerun CPU hnswlib QPS-recall curves for Deep1B and T2I1B without reusing previous result tables.

Metrics:

- Deep1B `R@10`
- Deep1B `R@100`
- T2I1B normalized-L2 `R@10`
- T2I1B normalized-L2 `R@100`

Target recall coverage: about `0.50` to `0.96`.

## Unified Configuration

- `nq=100` for all four curves.
- `query_repeats=500`.
- `trials_per_ef=3`.
- `threads=96`.
- `warmup_queries=10000` per trial.
- `omp_schedule=static`.
- `omp_chunk=64`.
- `numactl --interleave=all`.
- One Slurm bigmem node, `1600G` requested memory.

## Inputs

- Deep1B index: `/hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/hnsw_index_Deep1B_norm_M16_ef500_dynprec.bin`
- Deep1B queries: `/hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/query_vectors_fp32_norm_n100_seed42.bin`
- Deep1B R@10 GT: `/hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_Deep/gt_labels_1B_topk32_n100_seed42.bin`
- Deep1B R@100 GT: `/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator/memory/20260610/001_recall100_deep1b_t2i1b_scaled_curves_nq100/data/gt_labels_deep1b_topk100_n100_seed42_from_official.bin`
- T2I1B index: `/hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/cache_t2i/hnsw_index_text2img1B_norm_M16_ef500_dynprec.bin`
- T2I1B normalized queries: `memory/20260609/002_t2i1b_normalized_l2_gt_qps/results/query_vectors_norm_n100_first.bin`
- T2I1B normalized-L2 GT: `memory/20260609/002_t2i1b_normalized_l2_gt_qps/results/t2i1b_norm_l2_gt_top100_n100_first.bin`

## EF Selection

The EF lists were chosen by referring to previous memory records only to place points across the requested recall range. No previous QPS-recall rows are included in this rerun's output summaries.

- Deep1B `R@10`: `10,12,20,40,100,200,300,500`
- Deep1B `R@100`: `30,40,60,100,150,300,700`
- T2I1B normalized-L2 `R@10`: `30,40,60,100,150,300,1000,3000`
- T2I1B normalized-L2 `R@100`: `60,100,150,300,500,1500,3000,5000`

## Code Notes

- `src/hnswlib_1b_qps.cpp` now supports `--trials-per-ef`.
- Output TSV includes `trial` and `avg_results_per_query`.
- The previous `ef >= k` hard check has already been removed, and the benchmark keeps separate search/result queues for `ef < k`.

## Smoke Test

Deep10M smoke with `--trials-per-ef 2`, `ef=1,10`, `k=10` completed successfully.

Output:

- `results/smoke_deep10m_trials.tsv`

## Submit Command

```bash
sbatch memory/20260615/004_hnswlib_cpu_fresh_qps_recall/scripts/run_fresh_qps_recall_bigmem.sbatch
```

Submitted at 2026-06-15:

- JobID: `9842266`

Supplemental point submitted after the first job because Deep1B `R@100`
`ef=700` reached `0.9574`, slightly below the strict `0.96` threshold:

- JobID: `9842977`
- Metric: Deep1B `R@100`
- EF: `1000`
- Trials: `3`
- Output: `results/deep1b_r100_fresh_n100_extra_ef1000.tsv`

## Planned Raw Outputs

- `results/deep1b_r10_fresh_n100.tsv`
- `results/deep1b_r100_fresh_n100.tsv`
- `results/t2i1b_norml2_r10_fresh_n100.tsv`
- `results/t2i1b_norml2_r100_fresh_n100.tsv`

## Final Status

- Main JobID `9842266`: `COMPLETED`, exit `0:0`, elapsed `01:38:40`.
- Supplemental JobID `9842977`: `COMPLETED`, exit `0:0`, elapsed `00:20:43`.
- Final raw trial rows: `96`.
- Final EF summary rows: `32`.
- All EF points have `3` trials.
- SVG plots validated with Python XML parsing.

## Final Coverage

- Deep1B `R@10`: 8 points, EF `10..500`, recall `0.502..0.971`.
- Deep1B `R@100`: 8 points, EF `30..1000`, recall `0.499..0.9701`.
- T2I1B normalized-L2 `R@10`: 8 points, EF `30..3000`, recall `0.476..0.961`.
- T2I1B normalized-L2 `R@100`: 8 points, EF `60..5000`, recall `0.5016..0.9697`.

## Key Threshold Rows

Median QPS is used across the 3 trials for each EF.

- Deep1B `R@10` at recall >= `0.90`: EF `200`, recall `0.913`, QPS `71394.151`.
- Deep1B `R@10` at recall >= `0.96`: EF `500`, recall `0.971`, QPS `29346.683`.
- Deep1B `R@100` at recall >= `0.90`: EF `700`, recall `0.9574`, QPS `21079.183`.
- Deep1B `R@100` at recall >= `0.96`: EF `1000`, recall `0.9701`, QPS `14723.655`.
- T2I1B normalized-L2 `R@10` at recall >= `0.90`: EF `1000`, recall `0.912`, QPS `9458.636`.
- T2I1B normalized-L2 `R@10` at recall >= `0.96`: EF `3000`, recall `0.961`, QPS `3242.085`.
- T2I1B normalized-L2 `R@100` at recall >= `0.90`: EF `1500`, recall `0.9245`, QPS `6288.928`.
- T2I1B normalized-L2 `R@100` at recall >= `0.96`: EF `5000`, recall `0.9697`, QPS `1993.028`.

## Final Outputs

- `results/fresh_raw_all_trials.tsv`
- `results/fresh_summary_by_ef.tsv`
- `results/fresh_best_qps_by_recall_threshold_050_096.tsv`
- `plots/fresh_cpu_r10.svg`
- `plots/fresh_cpu_r100.svg`
