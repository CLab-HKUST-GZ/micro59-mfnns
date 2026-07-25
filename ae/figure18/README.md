# Figure 18: billion-scale recall--QPS curves

## Environment

- Linux, Bash, Python 3, GNU `sha256sum`, and a C++17 compiler with OpenMP.
- Install `ae/figure18/requirements.txt` (or the shared `ae/requirements.txt`).
- Run from the repository root.

## Reproduce the archived figure (no GPU)

```bash
bash ae/figure18/validate_figure18.sh
bash ae/figure18/reproduce_figure18.sh
```

Validation parses and checks 108 ANSMET/MFNNS YAMLs, maps 31 CPU plot rows to
96 raw trials, validates the frozen Deep1B BANG reference, syntax-checks the
CPU/BANG producer scripts, and dry-runs the CPU compiler command. It does not
execute a BANG search or request a GPU.

The validation ends with
`Figure 18 validation passed: plot, 108 YAMLs, CPU provenance/source, and BANG scripts.`
Reproduction writes:

```text
ae/figure18/output/figure18.pdf
ae/figure18/output/figure18.png
ae/figure18/output/figure18_summary.tsv
```

## Simulator YAMLs and optional CPU rerun

The bundle contains 50 ANSMET and 58 MFNNS YAMLs. Of these, 88 are linked to
matching original YAML/stats, one has a documented historical cycle mismatch,
and 19 are explicitly reconstructed rerun recipes. Exact status, hashes, and
deltas are in `data/simulator_provenance.tsv`.

The versioned YAMLs preserve historical absolute billion-scale input paths for
provenance. They are not submitted directly. The safe selector is read-only:

```bash
python3 ae/figure18/run_simulator_configs.py \
  --dataset deep1b --recall-tag r10 --method mfnns
```

To submit verified cases, provide the prepared inputs and a new numbered
result directory. The runner validates those inputs and writes task-specific
runtime YAMLs before invoking the existing simulator runner:

```bash
python3 ae/figure18/run_simulator_configs.py \
  --dataset deep1b --recall-tag r10 --method mfnns \
  --submit \
  --model-path /data/deep1b/index.bin \
  --query-path /data/deep1b/query_n100.bin \
  --gt-path /data/deep1b/gt_topk32_n100.bin \
  --result-root memory/YYYYMMDD/NNN_figure18_deep1b_r10_mfnns
```

DP1B/T2I1B simulations require prepared billion-scale inputs, the existing
`simulator/build/ramulator2`, and a big-memory CPU node. Reconstructed or
cycle-mismatch recipes are excluded unless `--include-unverified` is passed.
After completion:

```bash
python3 ae/figure18/summarize_simulator_results.py \
  memory/YYYYMMDD/NNN_figure18_deep1b_r10_mfnns
```

## Optional CPU and GPU producers

The standalone CPU source, exact hnswlib headers, compile command, and Slurm
producer are documented in `ae/figure18/cpu/README.md`.

The BANG scripts under `ae/figure18/scripts/` are optional GPU experiment
producers and are deliberately outside the default reproduction path.
Deep1B needs the official 10K inputs plus an existing R64/L100/QD32 BANG
index. T2I1B requires a metric-compatible transformed 201D MIPS-to-L2 index;
the script rejects the incompatible historical raw-200D L2 index. See
`ae/figure18/provenance/REMOTE_BANG_README.md` for the exact GPU workflow and
interpretation boundary.
