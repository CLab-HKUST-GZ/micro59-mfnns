# Summary

- Date: 2026-07-26
- Scope: reviewer-facing reproduction of remote AE Figures 14--23.
- Remote baseline: `micro59/main` at
  `1d3621486f84b6251125cfaceae8c25ebfb4a905`.
- Audit branch: `codex/ae-figure-reproduction-audit`.
- GPU policy: no GPU command, allocation, build, or experiment was executed.

## Outcome

The repository now has one CPU-only reviewer entry point:

```bash
python3 -m pip install -r ae/requirements.txt
bash ae/reproduce_all_figures.sh
```

It validates the archived inputs and regenerates Figures 14--23. Each figure
also has a documented `reproduce_figureNN.sh` entry point, expected outputs,
YAML/configuration coverage, provenance boundary, and optional source-rerun
requirements.

The final audit passed for all ten figures, all 501 bundled YAMLs, all output
formats, and repeated-output determinism. Within the requested scope--plot
reproduction from the versioned AE bundle and reviewer instructions--there
are no known unresolved defects.

## Material fixes

1. Added the root and `ae/README.md` quick starts, shared requirements, a
   unified CPU-only reproducer, and the missing Figure 14 wrapper.
2. Restored provenance and rerun boundaries removed by the shortened remote
   READMEs. Historical absolute-path YAMLs are no longer described as directly
   portable.
3. Corrected the Figure 14 `wiki1m/k5/mfnns` manifest parameters from
   `ef_search=18, queue_size=50` to the archived YAML/result values
   `ef_search=17, queue_size=30`.
4. Made the Figure 14, 18, 21, and 23 validators parse their YAMLs with
   `yaml.safe_load`; the validated counts are 126, 108, 246, and 21.
5. Changed the Figure 18 runner to require reviewer-supplied model, query, and
   ground-truth paths and to write runtime YAML copies. Archived provenance
   YAMLs remain unchanged.
6. Made runtime-YAML path replacement safe for spaces and YAML-special
   characters in both the Figure 18 and Figure 21 runners.
7. Restored the normalized, variant-free CPU index layout expected by all
   Figure 14 and Figure 21 YAMLs:
   `cpu_index/<dataset>/hnsw_index_...`.
8. Removed volatile PDF creation/modification timestamps from six plotters and
   regenerated those PDFs, making repeated reproduction byte deterministic.

## Explicit boundaries

- Figure reproduction uses frozen, versioned evidence. It does not claim to
  rerun every source experiment.
- Figure 18 optional BANG workflows require GPUs and were intentionally
  limited to static/syntax validation.
- Billion-scale Figure 18 CPU reruns require prepared external inputs and a
  big-memory node; no such rerun was needed for figure reproduction.
- Figure 19 and Figure 22 retain hash-level source provenance without
  duplicating the large author experiment trees.
- Figure 23 YAMLs are immutable historical provenance and intentionally keep
  their original absolute paths.
