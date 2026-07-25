# Artifact-evaluation material

This directory contains portable, paper-facing artifact material. Each figure
subdirectory owns its canonical plotting data, plotting code, generated output,
and provenance notes.

Current coverage:

- `figure14/`: throughput comparison for k=5, k=10, and k=100.
- `figure15/`: Recall@10 area-efficiency comparison derived from the canonical
  Figure 14 QPS table and synthesized DPE areas.
- `figure16/`: Recall@10 system energy-efficiency (QPS/W) comparison.
- `figure17/`: Recall@10 normalized system-energy breakdown.

Simulator configurations and their execution evidence remain under
`simulator/run_case/`; the `ae/` copy is the compact plotting interface.
Figures 16 and 17 share `ae/energy_model.py` and default to the fixed trace
summary under
`simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10/`.
