# Artifact-evaluation figure reproduction

This directory is the reviewer-facing path for Figures 14--23. The default
workflow validates archived inputs and regenerates every PDF, PNG, and numeric
summary. It requires no paper-author workspace and launches no GPU job.

## Quick start (CPU only)

Run from the repository root:

```bash
python3 -m pip install -r ae/requirements.txt
bash ae/reproduce_all_figures.sh
```

The command exits nonzero on a missing/mismatched input, malformed YAML, or
plot validation failure. Successful outputs are under
`ae/figureNN/output/`. The tested plotting environment is Python 3.8,
Matplotlib 3.7.5, NumPy 1.24.4, and PyYAML 5.3.1; compatible versions are
declared in `ae/requirements.txt`.

## What the command covers

| Figure | Reproduction input | YAML/configuration coverage | Default command uses GPU? |
| --- | --- | --- | --- |
| 14 | frozen 168-row QPS table | validates 126 simulator YAMLs; 42 GPU rerun manifests are separate | no |
| 15 | Figure 14 Recall@10 QPS plus area specifications | inherits Figure 14 YAML provenance | no |
| 16--17 | Figure 14 QPS plus the fixed 42-trace energy table | validates the 42 final Recall@10 simulator YAMLs | no |
| 18 | frozen CPU/BANG/ANSMET/MFNNS recall--QPS curves | parses and validates 108 archived ANSMET/MFNNS YAMLs and CPU/BANG provenance | no |
| 19 | frozen JUNO++ extraction and MFNNS frontier data | point-level author-workspace YAML/stats hashes are recorded, not duplicated | no |
| 20 | frozen DPE area/power table | no simulator YAML is needed | no |
| 21 | frozen LBQueue sweep | parses and checksums all 246 portable simulator YAMLs | no |
| 22 | frozen final-point latency counters | source YAML/stats references and hashes are recorded in the CSV | no |
| 23 | frozen DRAM row-miss counters | parses and verifies 21 byte-identical historical YAMLs | no |

The Figure 18 BANG producers under `ae/figure18/scripts/` are optional GPU
source-experiment workflows. `reproduce_all_figures.sh` only syntax-checks
those scripts through the Figure 18 validator; it never executes them.

## Plot reproduction versus experiment rerun

The quick start reproduces the paper figures from the versioned data. Rerunning
the source experiments is a separate, resource-intensive task:

- Figure 14 uses repository-relative, directly runnable YAMLs.
  `script/cpu_index_build.sh` creates their indexes from normalized base
  vectors and simultaneously writes normalized queries and exact matching
  ground truth. The separate 21-point CAGRA and 21-point BANG reruns are under
  `GPU_Baseline/`; an exact CAGRA Pubmed rerun requires its documented
  historical 1M-row base.
- Figure 18 needs DP1B/T2I1B inputs and a big-memory node. Its archived YAMLs
  intentionally preserve historical input paths; the runner creates portable
  runtime copies from reviewer-supplied paths.
- Figure 21 includes queries and ground truth but not its 1.08 GB HNSW index.
  Its runner also creates task-specific runtime YAMLs.
- Figure 23 YAMLs are immutable historical provenance and are not advertised
  as portable rerun recipes.

Each figure README gives the exact read-only check, reproduction command,
expected output, and any optional rerun command.
