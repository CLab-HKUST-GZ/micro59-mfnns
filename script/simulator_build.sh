#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/simulator"
BUILD_DIR="${SOURCE_DIR}/build"
BUILD_TYPE="Release"
JOBS="${MFNNS_BUILD_JOBS:-4}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CXX_BIN="${CXX:-}"
CLEAN=0
DRY_RUN=0
VERIFY_SOURCE=1
SMOKE_TEST=1
LOAD_CLUSTER_MODULES=0
CMAKE_MODULE="${MFNNS_CMAKE_MODULE:-cmake/3.27.0}"
GCC_MODULE="${MFNNS_GCC_MODULE:-compilers/gcc-13.1.0}"

usage() {
    cat <<'EOF'
Usage: simulator_build.sh [options]

Options:
  --build-dir DIR          Build directory (default: simulator/build).
  --build-type TYPE        CMake build type (default: Release).
  --jobs N                 Parallel build jobs (default: 4).
  --compiler PATH          C++ compiler (default: CXX, g++, or c++).
  --cmake PATH             CMake executable (default: CMAKE_BIN or cmake).
  --clean                  Clean the selected build tree before compiling.
  --cluster-modules        Load the documented CMake 3.27 and GCC 13.1 modules.
  --skip-source-verify     Do not check SOURCE_MANIFEST.sha256.
  --skip-smoke-test        Do not run ramulator2 --help after building.
  --dry-run                Validate inputs and print build commands only.
  -h, --help               Show this help.

Environment overrides:
  MFNNS_BUILD_JOBS, MFNNS_CMAKE_MODULE, MFNNS_GCC_MODULE, CMAKE_BIN, CXX

The script resolves the repository relative to its own location, so it may be
invoked from any working directory.
EOF
}

log() {
    printf '[simulator-build] %s\n' "$*"
}

die() {
    printf '[simulator-build] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

print_command() {
    printf '[simulator-build] DRY-RUN'
    printf ' %q' "$@"
    printf '\n'
}

while (($#)); do
    case "$1" in
        --build-dir)
            (($# >= 2)) || die "--build-dir requires a directory"
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            (($# >= 2)) || die "--build-type requires a value"
            BUILD_TYPE="$2"
            shift 2
            ;;
        --jobs)
            (($# >= 2)) || die "--jobs requires a positive integer"
            JOBS="$2"
            shift 2
            ;;
        --compiler)
            (($# >= 2)) || die "--compiler requires a path or command"
            CXX_BIN="$2"
            shift 2
            ;;
        --cmake)
            (($# >= 2)) || die "--cmake requires a path or command"
            CMAKE_BIN="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --cluster-modules)
            LOAD_CLUSTER_MODULES=1
            shift
            ;;
        --skip-source-verify)
            VERIFY_SOURCE=0
            shift
            ;;
        --skip-smoke-test)
            SMOKE_TEST=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
require_command readlink
BUILD_DIR="$(readlink -m -- "$BUILD_DIR")"
case "$BUILD_DIR" in
    /|"$REPO_ROOT"|"$SOURCE_DIR")
        die "Refusing unsafe build directory: ${BUILD_DIR}"
        ;;
    "$SOURCE_DIR"/*)
        case "$BUILD_DIR" in
            "$SOURCE_DIR"/build|"$SOURCE_DIR"/build-*|"$SOURCE_DIR"/build/*)
                ;;
            *)
                die "A build directory inside simulator/ must be named build or build-*"
                ;;
        esac
        ;;
esac
[[ -d "$SOURCE_DIR" ]] || die "Simulator source directory not found: ${SOURCE_DIR}"
[[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] ||
    die "Simulator CMakeLists.txt not found: ${SOURCE_DIR}/CMakeLists.txt"

if ((LOAD_CLUSTER_MODULES)); then
    [[ -r /etc/profile.d/modules.sh ]] ||
        die "Environment Modules initialization not found"
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
    module load "$CMAKE_MODULE" "$GCC_MODULE"
    log "Loaded modules: ${CMAKE_MODULE} ${GCC_MODULE}"
fi

require_command "$CMAKE_BIN"
if [[ -z "$CXX_BIN" ]]; then
    if command -v g++ >/dev/null 2>&1; then
        CXX_BIN="$(command -v g++)"
    elif command -v c++ >/dev/null 2>&1; then
        CXX_BIN="$(command -v c++)"
    else
        die "No C++ compiler found; use --compiler or set CXX"
    fi
elif command -v "$CXX_BIN" >/dev/null 2>&1; then
    CXX_BIN="$(command -v "$CXX_BIN")"
elif [[ ! -x "$CXX_BIN" ]]; then
    die "C++ compiler is not executable: ${CXX_BIN}"
fi

log "Repository: ${REPO_ROOT}"
log "CMake: $("$CMAKE_BIN" --version | sed -n '1p')"
log "Compiler: $("$CXX_BIN" --version | sed -n '1p')"
log "Build directory: ${BUILD_DIR}"

if ((VERIFY_SOURCE)); then
    require_command sha256sum
    [[ -f "${SOURCE_DIR}/SOURCE_MANIFEST.sha256" ]] ||
        die "Source manifest not found: ${SOURCE_DIR}/SOURCE_MANIFEST.sha256"
    log "Verifying simulator source manifest"
    (
        cd "$REPO_ROOT"
        sha256sum --check --quiet simulator/SOURCE_MANIFEST.sha256
    )
    log "Source manifest verified"
fi

configure=(
    "$CMAKE_BIN"
    -S "$SOURCE_DIR"
    -B "$BUILD_DIR"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_CXX_COMPILER=${CXX_BIN}"
)
build=(
    "$CMAKE_BIN"
    --build "$BUILD_DIR"
    --target ramulator-exe
    --parallel "$JOBS"
)

if ((DRY_RUN)); then
    print_command "${configure[@]}"
    if ((CLEAN)); then
        print_command "$CMAKE_BIN" --build "$BUILD_DIR" --target clean
    fi
    print_command "${build[@]}"
    if ((SMOKE_TEST)); then
        print_command "${BUILD_DIR}/ramulator2" --help
    fi
    exit 0
fi

"${configure[@]}"
if ((CLEAN)); then
    "$CMAKE_BIN" --build "$BUILD_DIR" --target clean
fi
"${build[@]}"

executable="${BUILD_DIR}/ramulator2"
[[ -x "$executable" ]] || die "Expected executable was not produced: ${executable}"
if ((SMOKE_TEST)); then
    "$executable" --help >/dev/null
    log "Smoke test passed: ramulator2 --help"
fi

log "Build completed: ${executable}"
