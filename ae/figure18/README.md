# Figure 18 test

## Environment

- Linux, Bash, Python 3, GNU `sha256sum`, and a C++17 compiler with OpenMP.
- Install `ae/figure18/requirements.txt`.
- Run from the repository root.

## Test

```bash
bash ae/figure18/validate_figure18.sh
bash ae/figure18/reproduce_figure18.sh
```

## Expected output

Validation ends with `Figure 18 validation passed: plot, 108 YAMLs, CPU provenance/source, and BANG scripts.` The reproduction writes:

```text
ae/figure18/output/figure18.pdf
ae/figure18/output/figure18.png
ae/figure18/output/figure18_summary.tsv
```
