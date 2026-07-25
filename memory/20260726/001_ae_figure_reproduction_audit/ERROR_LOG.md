# Error Log

## E01: Figure 14 manifest/YAML mismatch

Initial command:

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
```

Initial failure:

```text
RuntimeError: (5, 'wiki1m', 'mfnns'): ef_search mismatch
```

Cause: the final manifest said `ef_search=18, queue_size=50`, while both the
archived runnable YAML and the reused result used `17, 30`.

Fix: corrected the manifest to `17, 30` and `reused_result`.

Final result: all 126 YAMLs parse and validate.

## E02: non-deterministic PDF bytes

Initial observation: reproducing Figures 14, 15, 20, 21, 22, and 23 changed
their PDF hashes although the figures were visually identical.

Cause: Matplotlib embedded the current `CreationDate`; byte comparison showed
the timestamp as the only changing content.

Fix: passed deterministic PDF metadata with both date fields disabled and
regenerated the six PDFs.

Final result: all 20 PDF/PNG hashes are unchanged on a repeated full run.

## E03: Figure 18 archived paths were presented as runnable

Initial observation: the 108 versioned YAMLs parse but retain author-machine
absolute model/query/ground-truth paths. The runner previously submitted those
files directly.

Risk: a reviewer could submit invalid paths or accidentally mix one input set
across multiple dataset/recall panels.

Fix: the runner is read-only by default; submission requires one dataset and
recall panel plus three existing reviewer input files. It writes runtime YAML
copies under the numbered result root and leaves archived provenance intact.

Final result: selection, missing-input rejection, special-character paths, and
source-hash preservation all passed.

## E04: reviewer documentation was incomplete

Initial observation: there was no all-figure entry point, Figure 14 lacked a
wrapper, and shortened READMEs omitted YAML coverage, input boundaries, and
GPU/source-rerun distinctions.

Fix: added a root quick start, shared AE guide/requirements, unified
reproducer, Figure 14 wrapper, and per-figure provenance/rerun instructions.

Final result: every Figure 14--23 README names an existing one-figure
reproduction command; the unified command passes.

## E05: validators did not parse every YAML

Initial observation: some validators checked names, text patterns, or hashes
without proving that every YAML was syntactically valid.

Fix: added `yaml.safe_load` checks to Figures 14, 18, 21, and 23.

Final result: 501/501 relevant YAMLs parse as mappings.

## E06: CPU index builder disagreed with AE YAML paths

Initial observation: remote `script/cpu_index_build.sh` wrote
`cpu_index/<dataset>/<raw-or-normalized>/...`, while 126 Figure 14 and 246
Figure 21 YAMLs/runners reference `cpu_index/<dataset>/...`.

Fix: restored the single normalized, variant-free output policy and matching
tracked directory layout. PubMed remains unmodified because its source is
already normalized.

Final result: `--dry-run t2i1m` emits the exact Figure 21 path. No index was
built.

## E07: global Python environment warning

Command:

```bash
python3 -m pip check
```

Output:

```text
launchpadlib 1.10.13 requires testresources, which is not installed.
```

Assessment: unrelated system-package metadata; AE imports and the shared
requirements check pass. No global package was modified.

## E08: old pip lacks install dry-run support

Attempting `python3 -m pip install --dry-run ...` reported that `--dry-run` is
not an available option in this pip version.

Mitigation: used `pkg_resources.require` to validate the installed versions
against `ae/requirements.txt`, then executed every AE command. No package
installation was needed.

## Final unresolved errors

None within the requested non-GPU figure-reproduction scope. GPU source
experiments and external billion-scale source reruns remain intentionally
unexecuted and are documented as such.
