# CAGRA index builder

`index_build.sh` is a tracked relative link to
`../../script/cagra_index_build.sh`, the canonical cuVS CAGRA builder.

Configure a Python executable containing NumPy, CuPy, and cuVS:

```bash
GPU_Baseline/configure.sh \
  --cagra-python /path/to/cuvs-env/bin/python
```

Run builds through `../build_index.sh` so the unified k=100 graph parameters
and `index/<dataset>/CAGRA` cache directory are selected automatically.
CAGRA construction must run on a supported NVIDIA GPU.
