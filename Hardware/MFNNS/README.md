# MFNNS: Hardware Design & Functional Verification


## Project Overview
This document outlines the hardware design and functional verification process for the MFNNS project. 
The core is implemented using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/master/index.html), 
a modern, high-level hardware description language that facilitates efficient and flexible hardware design.


## Directory Structure
Hardware files are organized as follows:

* **SpinalHDL Source Code:** `hw/spinal/MFNNS`
* **Verification & Testbenches:** `hw/spinal/MFNNS/Testing`
* **Generated Verilog RTL:** `hw/gen/MFNNS/` (This directory is created after running the generation)


## Environment Setup
### Recommended: Pre-configured Environment (only for the Hardware part)

We have prepared a ready-to-use development environment for you, which can be accessed via docker pull.

```bash
docker pull victorchan433/mfnns-dev

docker run --rm -it victorchan433/mfnns-dev /bin/bash

# After entered
cd /workspace
git clone https://github.com/CLab-HKUST-GZ/micro59-mfnns.git

# Navigate to the project's root directory
cd micro59-mfnns/Hardware/MFNNS/
```

## Generating Verilog RTL

Follow these steps to generate the Verilog RTL from the SpinalHDL source code.

The process uses SBT (Simple Build Tool) to compile the Scala-based SpinalHDL code and execute the generator.

All SpinalHDL components can generate Verilog RTL.
We provide some examples below:

```bash
# Launch the SBT (Simple Build Tool) interactive shell
cs launch sbt

# Within the sbt shell, compile the project's source code
compile

# Generate Verilog files for Conventional FPMA
runMain MFNNS.ConventionalFPMA.FPMA_FP16_Gen

# Generate Verilog files for Square FPMA
runMain MFNNS.SquareFPMA.FPMA_Square_FP16_Gen

# Wait for the "[success]" message, which indicates completion
```
The generated Verilog files can be found in the following output directory:
`hw/gen/MFNNS/`.


## Functional Verification with SpinalSim and Verilator

Use the SBT environment to launch the SpinalSim testbenches. Verilator is used as the RTL simulation backend.

The functional verification suite contains three complementary checks:

1. Conventional `FPMA_FP16` RTL against its bit-exact Golden Model.
2. Specialized `FPMA_Square_FP16` RTL against its bit-exact Golden Model.
3. Direct RTL-to-RTL cross-validation between `FPMA_FP16(x, x)` and `FPMA_Square_FP16(x)`.

Each test exhaustively checks all 61,440 normal FP16 input encodings. To keep the terminal output readable, only 64 seeded-random representative comparisons are displayed by each test. All non-displayed inputs are still checked, and every mismatch is counted.

### Running the Complete Test Suite
This command executes all three verification tests sequentially using a fail-fast policy. A successful run completes 122,880 Golden-Model checks and 61,440 direct RTL cross-checks, for a total of 184,320 bit-exact comparisons.

```bash
# Make sure you're still in the sbt shell

# (Optional) If you've modified the source code, re-compile the project first
compile

# Run the complete functional test suite
runMain MFNNS.Testing.OverallFunctionalTest
```

A successful complete run ends with:

```text
[PASS] Functional test stages completed     : 3 / 3
[PASS] Overall bit-exact comparisons passed : 184,320 / 184,320
[PASS] OVERALL: ALL MFNNS FP16 FUNCTIONAL VERIFICATION TESTS PASSED
```


### Running Individual Tests
The `OverallFunctionalTest` suite is composed of the three independent test modules below.

You can also execute these tests individually.

Make sure you are inside the SBT shell before running these commands.

```bash
# Make sure you're still in the sbt shell

# 1. Conventional FPMA_FP16 vs Golden Model
# Exhaustively checks FPMA_FP16(A=x, W=x) for all 61,440 normal FP16 encodings.
# Every DUT result is compared bit-for-bit with FPMA_FP16_Golden.
runMain MFNNS.Testing.TestCases.Test_FPMA_FP16

# 2. Specialized FPMA_Square_FP16 vs Golden Model
# Exhaustively checks FPMA_Square_FP16(x) for all 61,440 normal FP16 encodings.
# Every DUT result is compared bit-for-bit with FPMA_Square_FP16_Golden.
runMain MFNNS.Testing.TestCases.Test_FPMA_Square_FP16

# 3. Direct RTL-to-RTL Cross-Validation
# Compares FPMA_Square_FP16(x) directly against FPMA_FP16(A=x, W=x).
# Executes 61,440 paired comparisons and 122,880 total DUT evaluations.
# No Golden Model is used in this direct equivalence check.
runMain MFNNS.Testing.TestCases.Test_FPMA_Square_FP16_CrossValidation

# Tests are finished. To exit the sbt shell, type
exit
```
