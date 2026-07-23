set -euo pipefail
BUILD_DIR="${BUILD_DIR:-../build}"
RAMULATOR_BIN="${RAMULATOR_BIN:-./ramulator2}"

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "[INFO] SKIP_BUILD=1, skip make -C ${BUILD_DIR}"
else
  make -j -C "${BUILD_DIR}"
fi
echo "------------------------------" | tee stdout
time "${RAMULATOR_BIN}" -f "$1" 2>&1 | tee -a stdout
