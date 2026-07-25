# Error Log

- 2026-03-29: 初始化实验目录，计划基于 `20260329/mfnns_etopt_hotrep_k10_ef_sweep` 的 t2i 配置，执行 `ef_search={20,30,40}` 与 `dualQueueLowerBoundQueueSize=20..100` 全扫。
- 2026-03-29: 首版 `commands.sh` 使用 `squeue -j <comma-separated ids>` 轮询时，未能稳定捕获本批 243 个作业，导致脚本过早进入汇总；现已改为 `squeue -u $USER` 后按 `job_id` 过滤，重新等待所有作业结束并重跑汇总。
- 2026-03-29: 除上述等待逻辑外，243/243 个作业均 `COMPLETED`，`summary_latest.tsv` 中全部为 `PASS`，未发现新的实验级错误。
