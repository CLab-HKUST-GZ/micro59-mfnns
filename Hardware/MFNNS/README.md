# MFNNS hardware functional test

## Environment

- Recommended container: `victorchan433/mfnns-dev`.
- Outside the container: Java, Coursier/SBT, and Icarus Verilog; dependencies are defined by `build.sbt`.
- Run the Docker command from the repository root; without Docker, run from `Hardware/MFNNS/`.

## Test

```bash
docker run --rm -it \
  -v "$PWD":/workspace/micro59-mfnns \
  -w /workspace/micro59-mfnns/Hardware/MFNNS \
  victorchan433/mfnns-dev \
  cs launch sbt -- "runMain MFNNS.Testing.OverallFunctionalTest"
```

Without Docker, run the same `cs launch sbt -- "runMain ..."` command in `Hardware/MFNNS/`.

## Expected output

The console ends with:

```text
[PASS] Functional test stages completed     : 3 / 3
[PASS] Overall bit-exact comparisons passed : 184,320 / 184,320
[PASS] OVERALL: ALL MFNNS FP16 FUNCTIONAL VERIFICATION TESTS PASSED
```

Simulator workspace and waveform files are generated under `Hardware/MFNNS/simWorkspace/`.
