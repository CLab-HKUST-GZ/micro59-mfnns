# micro59-mfnns

This repository contains the software, simulator, hardware, and artifact-evaluation material for MFNNS.

## Directory overview

- `ae/`: plotting data, validation scripts, and generated outputs for Figures 14–23.
- `GPU_Baseline/`: BANG and CAGRA index-building entry points and per-dataset index workspace.
- `Hardware/`: the SpinalHDL MFNNS hardware design and its functional tests.
- `mfnns_hnswlib/`: the standalone HNSW index, recall-analysis, and Table 5 tools.
- `script/`: dataset preparation and CPU, GPU, and simulator build scripts.
- `simulator/`: the Ramulator2-based simulator source, portable configurations, runners, and result locations.
