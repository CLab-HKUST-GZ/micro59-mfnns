# micro59-mfnns

## Index builders

The reproducible index builders are under `script/`:

- `cagra_index_build.sh` builds GPU CAGRA indexes.
- `bang_index_build.sh` builds a CPU DiskANN/PipeANN PQ graph, converts it to
  the BANG disk layout, and publishes the PQ files and BANG prefix symlinks.

Run `script/bang_index_build.sh --help` for the complete option list.

## BANG builder prerequisites

`bang_index_build.sh` requires:

- a little-endian float32 FBIN base file (`uint32 n`, `uint32 dim`, then
  exactly `n * dim` float32 values);
- Python 3 with NumPy;
- `flock` and `/usr/bin/time`;
- a compatible `build_disk_index` executable.

Two builder command-line APIs are supported:

- `pipeann` is the default and preserves the existing seven-dataset workflow;
- `diskann` uses the named options from the successful Deep1B QD64 build.

The graph build is a CPU/RAM/storage job. BANG GPU search is a separate step.

## Deep1B QD64: recorded one-billion-point configuration

The shortest safe invocation is the recorded preset:

```bash
script/bang_index_build.sh \
  --preset deep1b-qd64 \
  --base /path/to/deep/1B/base.1B.fbin \
  --dataset-name deep1b \
  --output-dir /large-local-storage/bang/deep1b \
  --build-disk-index /path/to/DiskANN/apps/build_disk_index
```

`deep1b-qd64` expands to:

| Parameter | Value |
| --- | ---: |
| builder API | `diskann` |
| points required | `1,000,000,000` |
| distance | float32 L2 |
| graph degree `R` | 64 |
| build list `L` | 100 |
| search DRAM budget `B` | 4 GiB |
| indexing RAM budget `M` | 32 GiB |
| builder threads `T` | 24 |
| build PQ bytes | 64 |
| quantized dimension `QD` | 64 |
| disk PQ bytes | 0 |
| normalization check | disabled |
| preserve failed stage | enabled |

These values correspond to the successful QD64 command:

```text
build_disk_index \
  --data_type float --dist_fn l2 \
  --data_path BASE --index_path_prefix PREFIX \
  -R 64 -L 100 -B 4 -M 32 -T 24 \
  --PQ_disk_bytes 0 --build_PQ_bytes 64 --QD 64
```

The recorded Deep1B build produced approximately:

- `disk.index`: 682.7 GB;
- BANG `disk.bin`: 644.0 GB;
- `pq_compressed.bin`: 64.0 GB.

Those are final files only. Reserve substantial extra space for builder
intermediates; 2.5 TB free space is a practical starting point, not a
guarantee. The recorded run used a 64-core/500-GB node and completed the
DiskANN build plus BANG conversion within a 48-hour allocation. Actual
requirements depend on the builder version, filesystem, and parameters.

Always validate the command before reserving a large node:

```bash
script/bang_index_build.sh \
  --preset deep1b-qd64 \
  --base /path/to/deep/1B/base.1B.fbin \
  --dataset-name deep1b \
  --output-dir /large-local-storage/bang/deep1b \
  --build-disk-index /path/to/build_disk_index \
  --dry-run
```

The dry run validates the complete FBIN byte length, samples finite values,
checks the expected point count, prints the exact builder command, and reports
the expected BANG and compressed-PQ sizes. It does not create index files.

## Custom billion-scale parameters

Every QD64 preset value can be overridden, regardless of whether the override
appears before or after `--preset`:

```bash
script/bang_index_build.sh \
  --builder-api diskann \
  --base /data/base.1B.fbin \
  --dataset-name custom1b \
  --output-dir /large-local-storage/bang/custom1b \
  --build-disk-index /opt/DiskANN/apps/build_disk_index \
  --graph-degree 64 \
  --build-l 100 \
  --search-dram-budget-gb 4 \
  --indexing-ram-budget-gb 32 \
  --threads 24 \
  --build-pq-bytes 64 \
  --quantized-dim 64 \
  --pq-disk-bytes 0 \
  --expect-points 1000000000 \
  --allow-non-normalized \
  --keep-staging-on-failure
```

Important controls:

- `--expect-points 0` disables an exact point-count requirement.
- `--require-normalized` checks sampled norms are in `[0.999, 1.001]`;
  `--allow-non-normalized` only checks finite values.
- `--build-pq-bytes` and `--quantized-dim` default to `--pq-chunks` when not
  specified.
- `--bf-entries` records the required BANG search compile value; it does not
  alter index bytes.
- `--force` backs up an existing published build before replacing it.

The DiskANN output prefix is:

```text
OUTPUT_DIR/NAME_R<R>_L<L>_QD<QD>
```

If build-PQ bytes and QD differ, `_BPQ<bytes>` is included. The runnable BANG
prefix has the same name plus `_bang`.

Successful publication includes:

- `<prefix>_disk.index`, `<prefix>_disk.bin`, and
  `<prefix>_disk_metadata.bin`;
- `<prefix>_pq_compressed.bin` and `<prefix>_pq_pivots.bin`;
- `<prefix>_bang_pq_pivots.bin` plus BANG-prefix symlinks for graph,
  metadata, and compressed PQ;
- `<prefix>.build.json` and per-stage logs.

## Failure recovery and staging

The build is assembled in a marked staging directory on the output
filesystem, then published with same-filesystem renames. For long builds,
preserve the stage on failure:

```bash
script/bang_index_build.sh ... --keep-staging-on-failure
```

The error output reports `STAGING_PRESERVED` and a `RESUME_WITH` argument.
Rerun the identical command with that directory:

```bash
script/bang_index_build.sh ... \
  --staging-dir /large-local-storage/bang/deep1b/.deep1b_R64_L100_QD64.tmp.ABCDEF
```

The script validates the stage identity and requires staging and output to be
on the same filesystem. If the builder outputs are already complete, it skips
the graph build and resumes conversion/validation. Do not point
`--staging-dir` at a general-purpose directory.

## Existing PipeANN workflow

The previous normalized-dataset interface remains the default:

```bash
BANG_BUILD_DISK_INDEX=/path/to/PipeANN/build/tests/build_disk_index \
script/bang_index_build.sh \
  --base /path/to/wiki_base.normalized.fbin \
  --dataset-name wiki1M \
  --output-dir /local-ssd/$USER/bang/index \
  --graph-degree 32 \
  --build-l 64 \
  --pq-chunks 128 \
  --build-memory-gb 64 \
  --threads 64 \
  --bf-entries 99991
```

For this API the builder command is:

```text
build_disk_index float BASE PREFIX R L PQ MEMORY_GB THREADS l2 pq
```

After either workflow completes, compile `bang_search` with:

```text
MAX_R=<graph-degree>
BF_ENTRIES=<bf-entries>
```

Run the fast interface/conversion regression before publishing script changes:

```bash
script/test_bang_index_build.sh
```
