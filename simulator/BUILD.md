# Simulator build and run

ANSMET and MFNNS are configurations of the same `ramulator2` executable, so
the source only needs to be built once. The repository vendors its C++
dependencies under `simulator/ext/`.

## Requirements

- CMake 3.14 or newer
- A C++20 compiler (GCC 13.1 is used for the AE runs)
- GNU Make or another build tool supported by CMake

First verify the source snapshot from the repository root:

```bash
sha256sum -c simulator/SOURCE_MANIFEST.sha256
```

## Generic Linux build

```bash
cmake -S simulator -B simulator/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v g++)"
cmake --build simulator/build --target ramulator-exe --parallel 4
```

The executable is `simulator/build/ramulator2`. Both `simulator/build/` and
the generated `simulator/libramulator.so` are intentionally ignored by Git;
each machine should build them locally.

## Current cluster environment

On the cluster used for the AE runs, load the matching CMake and GCC modules
before configuring, building, or running:

```bash
source /etc/profile.d/modules.sh
module load cmake/3.27.0 compilers/gcc-13.1.0

cmake -S simulator -B simulator/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v g++)"
cmake --build simulator/build --target ramulator-exe -j 4
```

## Run a configuration

Run the binary directly when a machine does not use Slurm:

```bash
simulator/build/ramulator2 -f /path/to/case.yaml
```

The YAML's `model_path`, `query_path`, `gt_path`, and `stat_path` must point
to valid locations on the target machine. Dataset files are not part of this
source snapshot.

On a Slurm cluster, submit a case through the included runner instead of
executing a long simulation on a login node:

```bash
python3 simulator/memory/run_yaml_case.py \
  --skip-build \
  --partition YOUR_PARTITION \
  /path/to/case.yaml
```

The runner loads `compilers/gcc-13.1.0` by default. Use
`--gcc-module MODULE_NAME` for another module name, or `--no-module-load`
when the compiler runtime is already available. It records the Slurm job ID,
logs, command, and expected statistics path under
`simulator/memory/run_yaml_case_results/` unless `--result-root` is given.

To inspect the generated command and record without submitting a job, add:

```bash
--dry-run
```
