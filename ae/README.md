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
- `figure19/`: MFNNS versus JUNO++ recall-QPS frontiers, including the complete
  277-row JUNO++ Fig. 8 vector extraction and point-level MFNNS provenance.
- `figure21/`: T2I1M Recall@10 sensitivity to LBQueue size, with the complete
  243-case MFNNS sweep and three ANSMET upper-bound configurations.
- `figure22/`: final Recall@10 ANSMET/MFNNS normalized latency breakdown.

Simulator configurations and their execution evidence remain under
`simulator/run_case/`; the `ae/` copy is the compact plotting interface.
Figures 16 and 17 share `ae/energy_model.py` and default to the fixed trace
summary under
`simulator/run_case/figure14_recall_gt0895/results/paper_energy/k10/`.

Validate both energy figures without rewriting their data or plots:

```bash
bash ae/validate_energy_figures.sh
```
