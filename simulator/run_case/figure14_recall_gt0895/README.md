# Figure 14 simulator test

## Environment

- Python 3 with `simulator/run_case/figure14_recall_gt0895/requirements.txt`.
- A built `simulator/build/ramulator2` is required only to execute a case.
- The selected YAML's HNSW index, query, and ground-truth files must exist.
- Run from the repository root.

## Test

```bash
python3 simulator/run_case/figure14_recall_gt0895/tools/validate_final_configs.py
simulator/run_case/figure14_recall_gt0895/scripts/run_final_case.sh \
  simulator/run_case/figure14_recall_gt0895/configs/final/k100/glove2m/ndp_base.yaml
```

## Expected output

The pass criterion is `PASS: 126 normalized, repository-relative Figure 14 YAMLs ...`. In the synchronized baseline, validation currently stops at `(5, 'wiki1m', 'mfnns'): ef_search mismatch`: the manifest records `ef=18, queue=50`, while that YAML records `ef=17, queue=30`. The executed example case writes its configured stats file below `simulator/run_case/figure14_recall_gt0895/results/k100/glove2m/`.
