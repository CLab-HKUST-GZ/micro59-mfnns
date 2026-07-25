# Artifact-evaluation tests

## Environment

- Linux, Bash, and Python 3.
- Install the packages from the target figure's `requirements.txt`; Figure 14 requires NumPy and Matplotlib.
- Run all commands from the repository root.

## Test

```bash
python3 ae/figure14/plot_figure14.py --check-only
bash ae/figure15/reproduce_figure15.sh
bash ae/validate_energy_figures.sh
bash ae/figure18/validate_figure18.sh
bash ae/figure19/reproduce_figure19.sh
bash ae/figure20/reproduce_figure20.sh
bash ae/figure21/reproduce_figure21.sh
bash ae/figure22/reproduce_figure22.sh
bash ae/figure23/reproduce_figure23.sh
```

## Expected output

Every command exits with status 0. Generated plots and summaries are under `ae/figureNN/output/`; read-only checks print `DATA_OK`, `CHECK_OK`, or a figure-specific validation-passed message.
