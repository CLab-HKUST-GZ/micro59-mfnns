# Figure 21 simulator configurations

- `mfnns/`: 243 original-parameter sweep configurations.
- `ansmet/`: three ANSMET reference configurations.
- `SHA256SUMS`: digests for all 246 portable YAMLs.

Each YAML contains its historical source reference and source SHA-256 in the
header. The simulator parameters are unchanged; only the four filesystem paths
are portable. Use `../run_figure21_sweep.py` to resolve those paths and submit
cases.
