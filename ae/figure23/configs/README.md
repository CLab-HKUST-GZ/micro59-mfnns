# Figure 23 original simulator configurations

This directory contains 21 byte-identical historical YAMLs:

```text
ansmet_open/   7
ndp_et/        7
mfnns/         7
```

The exact source path and SHA-256 for every file are recorded in:

```text
../data/config_provenance.tsv
```

The ANSMET-open YAMLs were generated from the corresponding March ANSMET
closed-row YAML by changing only:

```text
Frontend.stat_path
MemorySystem.Controller.RowPolicy.impl:
  ClosedRowPolicy -> OpenRowPolicy
```

All historical absolute paths are intentionally preserved. Rewriting
`stat_path` or input paths would make the files more portable but would no
longer archive the original YAML bytes requested for Figure 23.
