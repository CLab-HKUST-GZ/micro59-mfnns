# Test Record

## Environment

- Host repository worktree:
  `/tmp/micro27-ae-audit.pw3D1M`
- Python: 3.8.10
- Matplotlib: 3.7.5
- NumPy: 1.24.4
- PyYAML: 5.3.1
- Remote baseline:
  `micro59/main@1d3621486f84b6251125cfaceae8c25ebfb4a905`

The audit used a clean temporary worktree because the original workspace had
pre-existing user changes. Those changes were neither reset nor included.

## 1. Shared dependency check

```bash
python3 - <<'PY'
import pkg_resources
pkg_resources.require(open("ae/requirements.txt").read().splitlines())
print("REQUIREMENTS_OK")
PY
```

Result: `REQUIREMENTS_OK`.

## 2. Unified reviewer reproduction

```bash
bash ae/reproduce_all_figures.sh
```

Final result:

```text
AE_OK figures=14-23 gpu_jobs=0 outputs=ae/figureNN/output/
```

Per-figure results are recorded in `FIGURE_AUDIT.tsv`. The command regenerated
ten PDFs, ten PNGs, and the associated numeric summaries without submitting
any job.

## 3. YAML parsing and semantic validation

All YAMLs were loaded with `yaml.safe_load` and required to produce mappings:

```text
YAML_PARSE_OK counts=[126, 108, 246, 21] total=501
```

The groups are Figure 14 final configs, Figure 18 archived simulator configs,
Figure 21 sweep configs, and Figure 23 historical configs, respectively.
Figure-specific validators also checked manifests, parameter grids, hashes,
and plotted-data mappings.

## 4. Repeated-output determinism

```bash
find ae -path '*/output/*' -type f \
  \( -name '*.pdf' -o -name '*.png' \) -print0 |
  sort -z | xargs -0 sha256sum > /tmp/ae_outputs.before.sha256
bash ae/reproduce_all_figures.sh
find ae -path '*/output/*' -type f \
  \( -name '*.pdf' -o -name '*.png' \) -print0 |
  sort -z | xargs -0 sha256sum > /tmp/ae_outputs.after.sha256
cmp /tmp/ae_outputs.before.sha256 /tmp/ae_outputs.after.sha256
```

Result:

```text
DETERMINISM_OK all_versioned_outputs_unchanged
```

Final output digests are in `OUTPUT_SHA256SUMS`.

## 5. Output format and visual inspection

- `pdfinfo` reported one page for each of the ten PDFs.
- No PDF contains a `CreationDate` or `ModDate` field.
- `file` recognized all ten PNGs; dimensions ranged from 914x272 to 2565x631.
- A contact sheet containing Figures 14--23 was inspected visually. All panels,
  legends, labels, and axes rendered; no clipping or corrupted output was
  observed.

## 6. README entry-point check

A file/reference check required every `ae/figure14` through `ae/figure23`
directory to contain `reproduce_figureNN.sh` and its README to name that
script.

Result:

```text
README_ENTRYPOINT_OK figures=14-23
```

The commands documented by the READMEs are included in the unified reproducer
and passed in the final run.

## 7. Index-layout dry run

No index was built. The builder was inspected and invoked only in dry-run mode:

```bash
script/cpu_index_build.sh --list
script/cpu_index_build.sh --dry-run t2i1m
```

The generated command uses:

```text
mfnns_hnswlib/cpu_index/t2i1m/hnsw_index_M32_ef100.bin
```

This is the exact flat normalized path used by the Figure 21 runner and the
portable Figure 14/21 YAMLs. Result:

```text
INDEX_LAYOUT_OK figure14_figure21_flat_normalized_path
```

## 8. Runner safety tests

- Figure 18 read-only selection listed 13 verified configs and submitted no
  job.
- Figure 18 `--submit` without `--model-path`, `--query-path`, and `--gt-path`
  failed before creating its result root.
- Temporary Figure 18 and Figure 21 runtime YAMLs were generated with input
  paths containing spaces and `#`. `yaml.safe_load` recovered the exact paths.
- SHA-256 hashes of the archived source YAMLs were unchanged.

Result:

```text
RUNTIME_YAML_SPECIAL_PATH_OK figure18=2 figure21=1
```

## 9. Static checks

```bash
python3 -m py_compile <all modified Python files>
bash -n <all AE shell/Slurm scripts and script/cpu_index_build.sh>
git diff --check
```

Results:

```text
SYNTAX_OK python_and_shell
DIFF_CHECK_OK
```

## 10. GPU exclusion

No GPU binary or producer was executed. Figure 18 BANG files were checked only
by the repository validator for syntax, checksums, reference compatibility,
and expected anchors. The unified script reports `gpu_jobs=0`.
