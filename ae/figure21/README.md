# Figure 21: LBQueue-size sensitivity

This bundle reproduces the T2I1M Recall@10 LBQueue-size sensitivity figure and
archives the simulator configurations and scripts used to generate its data.

## Reproduce the figure

From the repository root:

```bash
bash ae/figure21/reproduce_figure21.sh
```

Read-only validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 ae/figure21/validate_figure21.py
```

The generated files are:

```text
ae/figure21/output/figure21.pdf
ae/figure21/output/figure21.png
ae/figure21/output/figure21_plot_data.tsv
```

## Archived experiment

The main sweep contains 243 completed MFNNS cases:

```text
dataset                    T2I1M, L2-normalized
top-k                      10
queries / parallel queries 100 / 100
ef_search                  20, 30, 40
LBQueue size               every integer from 20 through 100
LBQueue warmup             queue size - 1
```

`configs/mfnns/` contains all 243 YAMLs. `configs/ansmet/` contains the three
ANSMET configurations whose recalls (`0.893`, `0.924`, and `0.929`) are used
as the per-ef upper bounds. `data/simulator_provenance.tsv` maps every portable
configuration to its historical YAML, stats file, Slurm job, source hashes,
recall, and memory cycles.

The portable YAMLs preserve all simulation parameters. Only `model_path`,
`query_path`, `gt_path`, and `stat_path` are replaced. Their historical source
hashes and portable hashes are both recorded.

Exact snapshots of the historical case generators, submission commands,
summarizers, manifests, reports, error logs, and author plotting script are
under `scripts/historical/`. They retain author-machine paths for provenance;
use the portable entry points in this directory for reruns.

## Rerun simulator cases

The safe default only selects and prints configurations:

```bash
python3 ae/figure21/run_figure21_sweep.py \
  --method mfnns --ef 30 --queue 60
```

Submission requires an explicit numbered memory directory:

```bash
python3 ae/figure21/run_figure21_sweep.py \
  --method mfnns --ef 30 --queue 60 \
  --submit \
  --result-root memory/YYYYMMDD/NNN_figure21_ef30_q60
```

The runner resolves all input paths in task-specific runtime YAMLs and reuses:

```text
simulator/build/ramulator2
simulator/memory/run_yaml_case.py
simulator/run-template/run.sh
```

Build `simulator/build/ramulator2` once by following `simulator/BUILD.md`
before submitting. The Figure 21 runner passes `--skip-build` and does not
create another build directory. After the Slurm jobs finish:

```bash
python3 ae/figure21/summarize_rerun.py \
  memory/YYYYMMDD/NNN_figure21_ef30_q60
```

To submit all 246 configurations, pass `--all`; the runner refuses an implicit
full sweep.

## Inputs and reproducibility boundary

The exact query and top-32 ground-truth files are included in `inputs/`.
The 1,076,324,284-byte HNSW index is not duplicated in Git. Its exact size,
SHA-256, and author source are recorded in `data/input_manifest.tsv`.

Place or build the index at:

```text
mfnns_hnswlib/cpu_index/t2i1m/hnsw_index_M32_ef100.bin
```

The raw T2I1M dataset can be downloaded with `script/dataset_prepare.sh`, and
the normalized M=32, efConstruction=100 index can be built with:

```bash
script/cpu_index_build.sh t2i1m
```

Simulator build instructions and the source-snapshot boundary are documented
in `simulator/BUILD.md` and `simulator/SOURCE_PROVENANCE.md`.

The historical run record did not store the March 2026 executable digest or a
clean source commit. Therefore the archived YAMLs, raw statistics, and plotted
data are exact, while a newly built simulator is a semantic rerun rather than
a claim of bit-identical reconstruction of the old binary.
