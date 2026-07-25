# micro59-mfnns

This repository contains the software, simulator, hardware, and artifact-evaluation material for MFNNS.

## Directory overview

- `ae/`: plotting data, validation scripts, and generated outputs for Figures 14–23.
- `GPU_Baseline/`: BANG and CAGRA index-building entry points and per-dataset index workspace.
- `Hardware/`: the SpinalHDL MFNNS hardware design and its functional tests.
- `mfnns_hnswlib/`: the standalone HNSW index, recall-analysis, and Table 5 tools.
- `script/`: dataset preparation and CPU, GPU, and simulator build scripts.
- `simulator/`: the Ramulator2-based simulator source, portable configurations, runners, and result locations.

## Reproduce the paper figures

The CPU-only artifact path reproduces Figures 14--23 from the archived data
and validates all included simulator YAMLs. It does not launch the optional
GPU/BANG producers:

```bash
python3 -m pip install -r ae/requirements.txt
bash ae/reproduce_all_figures.sh
```

See [`ae/README.md`](ae/README.md) for per-figure commands, expected outputs,
YAML coverage, and the boundary between plot reproduction and source
experiment reruns.
