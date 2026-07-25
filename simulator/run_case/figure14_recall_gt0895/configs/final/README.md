# Figure 14 configuration test

## Environment

- Python 3.
- Run from the repository root.

## Test

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
```

## Expected output

The pass criterion is:

```text
PASS: 126 parsed, normalized, repository-relative Figure 14 YAMLs; k100 status counts 33 direct + 2 manual + 7 copied
```

Executed cases write stats below `simulator/run_case/figure14_recall_gt0895/results/k<k>/<dataset>/`.
