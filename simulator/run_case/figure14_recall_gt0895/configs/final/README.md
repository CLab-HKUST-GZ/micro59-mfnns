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
PASS: 126 normalized, repository-relative Figure 14 YAMLs; k100 status counts 33 direct + 2 manual + 7 copied
```

In the synchronized baseline, the command currently reports `(5, 'wiki1m', 'mfnns'): ef_search mismatch` because the manifest records `ef=18, queue=50`, while the YAML records `ef=17, queue=30`.

Executed cases write stats below `simulator/run_case/figure14_recall_gt0895/results/k<k>/<dataset>/`.
