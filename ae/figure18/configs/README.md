# Figure 18 configuration test

## Environment

- Linux, Bash, Python 3, and the packages in `ae/figure18/requirements.txt`.
- Run from the repository root.

## Test

```bash
bash ae/figure18/validate_figure18.sh
python3 ae/figure18/run_simulator_configs.py \
  --dataset deep1b --recall-tag r10 --method mfnns
```

## Expected output

Validation reports 108 valid simulator configurations and exits with status 0. The selector prints runnable commands without submitting jobs. With `--submit --result-root memory/YYYYMMDD/NNN_name`, run records and statistics are written below that explicit result directory.
