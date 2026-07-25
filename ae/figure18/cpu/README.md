# Figure 18 CPU test

## Environment

- Linux, Bash, Python 3, and a C++17 compiler with OpenMP.
- Run from the repository root.

## Test

```bash
python3 ae/figure18/cpu/scripts/validate_cpu_provenance.py --check-only
bash ae/figure18/cpu/scripts/build_cpu_benchmark.sh --dry-run
```

To compile the benchmark, omit `--dry-run`. Running the full curve also requires the prepared DP1B/T2I1B indexes, queries, and ground truth.

## Expected output

The provenance check prints `Validated Figure 18 CPU provenance: 31 plotted rows -> 96 raw trials ...`. The dry run prints the compiler command. A real build writes `ae/figure18/cpu/bin/hnswlib_1b_qps`; the Slurm producer writes its logs and TSV results to the paths configured in `ae/figure18/cpu/scripts/run_figure18_cpu_curves.sbatch`.
