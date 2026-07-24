#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 REPOSITORY_RELATIVE_CONFIG" >&2
  exit 2
fi

CONFIG_PATH="$1"
SIMULATOR_BIN="${FIGURE14_SIMULATOR_BIN:-simulator/build/ramulator2}"

case "${CONFIG_PATH}" in
  /*|"")
    echo "config path must be relative to the repository root: ${CONFIG_PATH}" >&2
    exit 2
    ;;
esac
case "${SIMULATOR_BIN}" in
  /*|"")
    echo "FIGURE14_SIMULATOR_BIN must be relative to the repository root: ${SIMULATOR_BIN}" >&2
    exit 2
    ;;
esac

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPOSITORY_ROOT}"

[[ -f "${CONFIG_PATH}" ]] || {
  echo "missing config: ${CONFIG_PATH}" >&2
  exit 2
}
[[ -x "${SIMULATOR_BIN}" ]] || {
  echo "missing simulator executable: ${SIMULATOR_BIN}" >&2
  exit 2
}

STAT_PATH="$(awk '$1 == "stat_path:" {print $2; exit}' "${CONFIG_PATH}")"
case "${STAT_PATH}" in
  /*|"")
    echo "YAML stat_path must be repository-relative: ${STAT_PATH}" >&2
    exit 2
    ;;
esac
mkdir -p "$(dirname "${STAT_PATH}")"

"${SIMULATOR_BIN}" -f "${CONFIG_PATH}"
