# Figure 18 data test

## Environment

- Linux, Bash, Python 3, GNU `sha256sum`, and the packages in `ae/figure18/requirements.txt`.
- Run from the repository root.

## Test

```bash
bash ae/figure18/validate_figure18.sh
```

## Expected output

The command validates the 177 plot rows, 108 simulator rows, CPU mapping, BANG reference, and both checksum inventories. Every checksum reports `OK`, and the command ends with the Figure 18 validation-passed message. Generated plots and the summary are under `ae/figure18/output/`.
