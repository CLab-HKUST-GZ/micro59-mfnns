# Figure 18 simulator configurations

## Validate

Run from the repository root:

```bash
bash ae/figure18/validate_figure18.sh
```

The command parses all 108 YAMLs, verifies every point-level digest and
parameter mapping, and verifies the split of 88 verified plus 20 rerun-only
cases.

## Select or submit

Selection is read-only and does not require the large inputs:

```bash
python3 ae/figure18/run_simulator_configs.py \
  --dataset deep1b --recall-tag r10 --method mfnns
```

The YAMLs intentionally retain historical absolute model/query/ground-truth
paths. A real submission must pass `--model-path`, `--query-path`, and
`--gt-path`; the runner creates portable runtime copies below the required
`--result-root memory/YYYYMMDD/NNN_name`. It never rewrites these archived
provenance files. See the parent README for the full command.
