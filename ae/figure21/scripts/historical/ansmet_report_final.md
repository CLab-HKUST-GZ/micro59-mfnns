# ANSMET recall@10 ∈ [0.9, 0.91) ef_search Parameter Sweep

## Experiment Configuration

- **k_neighbors**: 10
- **gt_k**: 32 (ground truth top-32, recall computed as recall@min(k,gt_k)=recall@10)
- **Target recall band**: `[0.9, 0.91)`
- **Objective**: find ef_search that achieves target recall, then minimize `s_mem_cycle`
- **nFMAC**: 16
- **nQueryLimit**: 100 (nParallelQuery=100)
- **Binary**: `build_compute_sysgcc/ramulator2` with unclamped ef_search + recall@k denominator fix
- **Base YAMLs**: `20260326/multidataset_nine_config_fp16_hotrep_cfs/yamls/*_ansmet.yaml`
- **Original sweep cases**: 71 (54 stage 1 + 17 stage 2 refinement)
- **Manual extension**: +1 (`t2i1m_normalized_k10_ef40`, run on 2026-03-30, Slurm job `9616832`)

## Best Candidates

| Dataset | ef_search | recall@10 | s_mem_cycle | Status |
| --- | ---: | ---: | ---: | --- |
| **sift1m** | 10 | 0.924 | 224,847 | ❌ unreachable (ef=10 is minimum, 0.924) |
| **deep10m** | 12 | 0.900 | 226,272 | ✅ IN BAND (exactly 0.900) |
| **w2v1m** | 26 | 0.904 | 1,007,397 | ✅ IN BAND |
| **wiki1m** | 17 | 0.899 | 1,575,072 | ❌ unreachable (ef=17→0.899, ef=18→0.923) |
| **t2i1m** | 25 | 0.909 | 700,872 | ✅ IN BAND |
| **gist1m** | 41 | 0.903 | 5,590,647 | ✅ IN BAND |
| **pubmed** | 220 | 0.903 | 25,801,422 | ✅ IN BAND |
| **glove2m** | 500 | 0.904 | 27,166,797 | ✅ IN BAND |

**6/8 datasets achieved in-band recall.** 2 datasets (sift1m, wiki1m) have recall jumps that skip the band.

## Dataset Details

### sift1m / normalized

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 10 | **0.924** | **224,847** | no (closest, can't go below k=10) |
| 12 | 0.935 | 245,622 | no |
| 14 | 0.954 | 266,997 | no |
| 16 | 0.963 | 287,697 | no |
| 18 | 0.971 | 307,347 | no |
| 20 | 0.972 | 330,222 | no |
| 25 | 0.984 | 387,222 | no |

### deep10m / normalized

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 10 | 0.875 | 198,372 | no |
| 11 | 0.887 | 207,372 | no |
| 12 | **0.900** | **226,272** | ✅ yes |
| 14 | 0.920 | 254,322 | no |
| 16 | 0.932 | 277,722 | no |
| 18 | 0.938 | 306,597 | no |
| 20 | 0.943 | 327,672 | no |
| 25 | 0.956 | 387,522 | no |

### w2v1m / normalized

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 10 | 0.820 | 581,472 | no |
| 14 | 0.859 | 671,247 | no |
| 18 | 0.877 | 783,297 | no |
| 20 | 0.880 | 844,947 | no |
| 22 | 0.895 | 880,797 | no |
| 25 | 0.899 | 996,597 | no |
| 26 | **0.904** | **1,007,397** | ✅ yes |
| 27 | 0.912 | 1,045,197 | no |
| 28 | 0.910 | 1,071,972 | no |

### wiki1m / normalized

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 10 | 0.824 | 1,234,947 | no |
| 15 | 0.883 | 1,474,947 | no |
| 16 | 0.897 | 1,549,047 | no |
| 17 | **0.899** | **1,575,072** | no (closest below) |
| 18 | 0.923 | 1,638,072 | no |
| 19 | 0.917 | 1,685,322 | no |
| 20 | 0.932 | 1,723,872 | no |
| 22 | 0.920 | 1,834,722 | no |
| 25 | 0.916 | 1,985,547 | no |
| 28 | 0.938 | 2,152,647 | no |
| 32 | 0.930 | 2,413,422 | no |

### t2i1m / normalized

Manual follow-up on 2026-03-30 adds `ef=40`; it remains outside the target recall band and does not change the best in-band choice (`ef=25`).

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 10 | 0.796 | 407,472 | no |
| 15 | 0.850 | 504,297 | no |
| 20 | 0.893 | 600,297 | no |
| 25 | **0.909** | **700,872** | ✅ yes |
| 28 | 0.920 | 762,522 | no |
| 30 | 0.924 | 808,422 | no |
| 35 | 0.925 | 919,947 | no |
| 40 | 0.929 | 1,031,366 | no |

### gist1m / normalized

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 15 | 0.760 | 2,608,047 | no |
| 20 | 0.820 | 3,177,072 | no |
| 25 | 0.843 | 3,704,547 | no |
| 30 | 0.865 | 4,226,622 | no |
| 35 | 0.881 | 4,838,322 | no |
| 40 | 0.898 | 5,412,822 | no |
| 41 | **0.903** | **5,590,647** | ✅ yes |
| 42 | 0.899 | 5,687,022 | no |
| 43 | 0.896 | 5,815,572 | no |
| 44 | 0.897 | 5,945,547 | no |
| 45 | 0.907 | 6,103,047 | ✅ yes |

### pubmed / raw

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 50 | 0.672 | 6,407,022 | no |
| 80 | 0.757 | 9,931,572 | no |
| 100 | 0.788 | 12,259,122 | no |
| 150 | 0.859 | 17,977,722 | no |
| 200 | 0.895 | 23,616,897 | no |
| 210 | 0.896 | 24,704,022 | no |
| 220 | **0.903** | **25,801,422** | ✅ yes |
| 230 | 0.910 | 26,856,972 | no |
| 240 | 0.910 | 27,854,097 | no |
| 250 | 0.915 | 28,972,272 | no |

### glove2m / normalized

| ef | recall@10 | s_mem_cycle | in_band |
| ---: | ---: | ---: | --- |
| 100 | 0.801 | 4,564,422 | no |
| 200 | 0.847 | 10,021,422 | no |
| 300 | 0.879 | 15,850,797 | no |
| 400 | 0.890 | 21,600,072 | no |
| 450 | 0.888 | 24,429,072 | no |
| 480 | 0.899 | 26,108,922 | no |
| 500 | **0.904** | **27,166,797** | ✅ yes |
| 600 | 0.908 | 32,454,447 | ✅ yes |
