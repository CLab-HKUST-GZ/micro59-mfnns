# GPU baseline index workspace

This directory is the repository-local entry point for building and storing
the seven GPU-baseline indexes. Index bytes are deliberately excluded from
Git.

## Layout

```text
GPU_Baseline/
├── BANG/
│   ├── index_build.sh -> ../../script/bang_index_build.sh
│   └── README.md
├── CAGRA/
│   ├── index_build.sh -> ../../script/cagra_index_build.sh
│   └── README.md
├── index/
│   ├── deep10M/
│   ├── glove2_2m/
│   ├── pubmed/
│   ├── sift1M/
│   ├── text2img1M/
│   ├── wiki1M/
│   └── word2vec/
├── build_index.sh
└── configure.sh
```

At runtime, `build_index.sh` writes each method below its dataset:

```text
GPU_Baseline/index/<dataset>/BANG/
GPU_Baseline/index/<dataset>/CAGRA/
```

Every dataset directory contains an allow-list `.gitignore`; generated
indexes, logs, manifests, staging directories, and runtime method
subdirectories therefore stay untracked.

## Configure dependencies

The tracked `BANG/index_build.sh` and `CAGRA/index_build.sh` links always
resolve to the canonical scripts under `script/`. Machine-specific
dependencies are configured once:

```bash
GPU_Baseline/configure.sh \
  --bang-builder /path/to/PipeANN/build/tests/build_disk_index \
  --cagra-python /path/to/cuvs-env/bin/python

GPU_Baseline/configure.sh --check
```

This creates ignored local links:

```text
GPU_Baseline/BANG/build_disk_index
GPU_Baseline/CAGRA/python
```

Optional source links can also be recorded without committing absolute paths:

```bash
GPU_Baseline/configure.sh \
  --bang-source /path/to/PipeANN \
  --cagra-source /path/to/cuvs-or-cagra-workspace
```

No BANG conversion source needs to be copied here: the canonical BANG builder
already embeds the recorded disk-index conversion and PQ-pivots rewrapping
code. CAGRA construction uses the installed `cuvs.neighbors.cagra` package.

## Build

The front-end selects the recorded unified/k=100 graph configuration and the
correct dataset index directory, then invokes the canonical build script:

```bash
# Inspect the exact BANG command without writing an index.
GPU_Baseline/build_index.sh bang wiki1M \
  --base /path/to/wiki1M_base.bin \
  --dry-run

# Build BANG.
GPU_Baseline/build_index.sh bang wiki1M \
  --base /path/to/wiki1M_base.bin

# Build CAGRA on physical GPU 1.
GPU_Baseline/build_index.sh cagra wiki1M \
  --base /path/to/wiki1M_base.bin \
  --gpu 1
```

List the frozen graph configurations:

```bash
GPU_Baseline/build_index.sh --list
```

Additional arguments are forwarded after the recorded defaults, so an
explicit later option can override a default when a controlled experiment
requires it.

## Storage placement

Repository-local output is the default. For large/high-I/O builds on the CLab
servers, point the same dataset layout at node-local SSD:

```bash
export GPU_BASELINE_INDEX_ROOT="/local-ssd/$(whoami)/GPU_Baseline/index"
GPU_Baseline/build_index.sh bang wiki1M --base /path/to/wiki1M_base.bin
```

Quote this variable because the domain-qualified username contains a
backslash. Local SSD is machine-local; copy only durable results elsewhere
before cleanup.
