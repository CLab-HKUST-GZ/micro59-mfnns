# Figure 18: BANG Deep1B recall--QPS curve

This bundle contains the runnable BANG producer for the **DP1B** panels of
Figure 18. It uses the official Deep public 10K queries and ground truth, the
R64/Lbuild100/QD32 BANG index, and reports Recall@10 and Recall@100 for the
plotted search-budget points.

The T2I1B panels intentionally have no BANG producer here. The available
uploaded T2I1B index is raw 200-dimensional data while the official labels are
MIPS labels. The historical BANG invocation used its L2 path, so it is not a
valid Figure 18 T2I1B reproduction. A compatible 201-dimensional
MIPS-to-L2 index is required first.

## Prerequisites

The script must run on a host holding the full Deep1B files. It neither copies
nor builds the 1B index. Set the following paths before running:

~~~bash
export BANG_REPO=/path/to/BANG-Billion-Scale-ANN
export BASE=/mnt/smartssd0/CONNECT\rmeng603/bang_deep1b_20260606/data/base.1B.fbin
export QUERY=/mnt/smartssd0/CONNECT\rmeng603/bang_deep1b_20260606/data/query.public.10K.fbin
export INDEX_PREFIX=/mnt/smartssd1/CONNECT\rmeng603/deep1b_uploaded_index_pq_20260612_qd32/diskann_deep1b_R64_L100_QD32
export CUDA_VISIBLE_DEVICES=0
~~~

The BANG search binary must already be built for MAX_R=64. The graph/PQ prefix
must have _disk.bin, _disk_metadata.bin, _pq_compressed.bin, and
_pq_pivots.bin.

## Run

From the repository root:

~~~bash
bash ae/figure18/scripts/run_deep1b_bang_curve.sh
~~~

The default work directory is /local-ssd/<user>/figure18_bang_deep1b; override
it with WORK_DIR if needed. The script downloads the official 10K GT only when
missing, converts it to BANG's ids-plus-distances truthset, validates headers
plus a GT sample, then runs five repeats per point.

The default sweep is the union of the historical main and refinement sweeps:

~~~text
Recall@10:  L = 10 15 20 30 40 60 80 160 320 512
Recall@100: L = 100 130 160 200 240 280 320 512
~~~

results/summary_qps_recall_curve_q10000.csv records all raw BANG timing rows
and derives median_last3_qps, the historical steady-state metric.
data/expected_deep1b_bang_curve.tsv is the frozen 2026-06-15 reference curve,
not generated output.

## Historical agreement

The reference DP1B points agree with the purple BANG curve in Figure 18. For
Recall@10, (0.6331, 188679.25), (0.8706, 77519.38), (0.9327, 43668.12), and
(0.9795, 14577.26) are representative anchors. The paper's DP1B table also
uses 10,000 queries.

Timing is hardware- and load-sensitive. Compare recall--QPS frontiers under
an equivalent exclusive or documented shared load; do not expect bit-identical
individual timing rows.

## T2I1B script

scripts/run_t2i1b_mips_l2_curve.sh is the metric-compatible companion for the
T2I1B panels. It requires a **BANG-compatible transformed 201D index**:
base vectors must be transformed from x to [x, sqrt(R^2 - ||x||^2)], and the
script appends a zero coordinate to each official 200D query. This makes L2
ranking equivalent to the official maximum-inner-product labels.

Set BANG_REPO, INDEX_PREFIX, QUERY_RAW, and GT_RAW, then run:

~~~bash
export BANG_REPO=/path/to/BANG-Billion-Scale-ANN
export INDEX_PREFIX=/path/to/t2i1b_mips_l2_R64_QD32
export QUERY_RAW=/path/to/query.public.100K.fbin
export GT_RAW=/path/to/groundtruth.public.100K.ibin
bash ae/figure18/scripts/run_t2i1b_mips_l2_curve.sh
~~~

The script rejects 200D index metadata. Its output is a new benchmark run and
must not be represented as the historical Figure 18 curve until the transformed
index provenance and resulting frontier are independently validated.
