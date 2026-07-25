# CAGRA index-builder test

## Environment

- Linux with a supported NVIDIA GPU and `nvidia-smi`.
- Python 3 with NumPy, CuPy, and cuVS.
- Run from the repository root.

## Test

```bash
GPU_Baseline/configure.sh \
  --cagra-python /path/to/cuvs-env/bin/python
GPU_Baseline/CAGRA/python -c 'import numpy, cupy, cuvs'
GPU_Baseline/build_index.sh cagra wiki1M \
  --base /path/to/wiki1M_base.bin --gpu 0
```

## Expected output

The import check exits with status 0. A successful build prints JSON containing `"status": "built"` or `"status": "cache_hit"` and writes the serialized CAGRA index plus its JSON metadata under `GPU_Baseline/index/wiki1M/CAGRA/`.
