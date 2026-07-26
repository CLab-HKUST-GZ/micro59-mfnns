#!/usr/bin/env bash
set -euo pipefail

ROOT=""
DRY_RUN=0
INCLUDE_BILLION=0
VERIFY=1
VERIFY_ONLY=0
LIST_ONLY=0
declare -a REQUESTED=()
declare -a SELECTED=()
declare -A SEEN=()

DEEP_BASE_10M_URL="${DEEP_BASE_10M_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin}"
DEEP_BASE_1B_URL="${DEEP_BASE_1B_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin}"
DEEP_QUERY_URL="${DEEP_QUERY_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin}"
DEEP_GT_URL="${DEEP_GT_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin}"

T2I_BASE_1M_URL="${T2I_BASE_1M_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin}"
T2I_BASE_1B_URL="${T2I_BASE_1B_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin}"
T2I_QUERY_URL="${T2I_QUERY_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin}"
T2I_GT_URL="${T2I_GT_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin}"

WIKI_ARCHIVE_URL="${WIKI_ARCHIVE_URL:-https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar}"
W2V_ARCHIVE_URL="${W2V_ARCHIVE_URL:-https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/word2vec.tar.gz}"
GLOVE_ARCHIVE_URL="${GLOVE_ARCHIVE_URL:-https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/glove2.2m.tar.gz}"
SIFT_ARCHIVE_URL="${SIFT_ARCHIVE_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz}"

PUBMED_BASE_URL="${PUBMED_BASE_URL:-https://drive.usercontent.google.com/download?id=1o8fTNw3mEtxBsbfwlnKFbArGkDei-U-x&export=download&confirm=t}"
PUBMED_QUERY_URL="${PUBMED_QUERY_URL:-https://drive.usercontent.google.com/download?id=1hqgwMBNVHn3LSeNHo1QNNuBp1H2bYLY_&export=download&confirm=t}"

usage() {
    cat <<'EOF'
Usage:
  dataset_prepare.sh --root DIR [options] DATASET...
  dataset_prepare.sh --root DIR [options] --datasets NAME[,NAME...]
  dataset_prepare.sh --list

Datasets:
  figure14  deep10m  t2i1m  wiki1m  w2v1m  glove2m  sift1m  pubmed
  deep1b    t2i1b    all

Options:
  --root DIR          Destination data root.
  --datasets LIST     Comma-separated dataset names.
  --include-billion   Permit Deep1B or T2I1B downloads.
  --dry-run           Print the plan without writing files.
  --verify-only       Verify existing files without downloading.
  --no-verify         Skip post-download binary layout verification.
  --list              List supported datasets and destination layouts.
  -h, --help          Show this help.

The "figure14" group contains the seven non-billion Figure 14 datasets.
The "all" group adds Deep1B and T2I1B. Billion-scale downloads require
--include-billion and more than 1.18 TB of space in total.

This script prepares and validates raw vector files. It does not normalize
vectors or build derived artifacts. Run script/cpu_index_build.sh afterward;
that command builds the index from normalized base vectors, persists
normalized queries, and recomputes exact ground truth in the same space.

Every URL can be overridden with its corresponding *_URL environment variable.
EOF
}

list_datasets() {
    cat <<'EOF'
NAME       BASE DATA                            METRIC          DESTINATION
deep10m    10,000,000 x 96 float32             Euclidean       deep/1M/base.10M.fbin
t2i1m      1,000,000 x 200 float32             Inner product   t2i/1M/base.1M.fbin
wiki1m     1,000,000 x 768 float32             Inner product   wiki/wiki1m/
w2v1m      1,000,000 x 300 float32             Angular         w2v/word2vec/
glove2m    2,196,017 x 300 float32             Angular         glove/glove2.2m/
sift1m     1,000,000 x 128 float32             Euclidean       sift/1M/sift/
pubmed     500,000 x 768 float32                Euclidean       pubmed/
deep1b     1,000,000,000 x 96 float32 (~384GB) Euclidean       deep/1B/base.1B.fbin
t2i1b      1,000,000,000 x 200 float32 (~800GB) Inner product  t2i/1B/base.1B.fbin
EOF
}

log() {
    printf '[dataset-prepare] %s\n' "$*"
}

die() {
    printf '[dataset-prepare] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

file_size() {
    stat -Lc '%s' "$1"
}

check_existing_file() {
    local path="$1"
    local expected_size="$2"
    local label="$3"

    if [[ ! -e "$path" ]]; then
        return 1
    fi
    [[ -f "$path" ]] || die "${label} exists but is not a regular file: ${path}"

    local actual_size
    actual_size="$(file_size "$path")"
    [[ "$actual_size" == "$expected_size" ]] ||
        die "${label} has size ${actual_size}, expected ${expected_size}: ${path}"
    return 0
}

ensure_disk_space() {
    local destination="$1"
    local required_bytes="$2"
    local directory available_kib available_bytes margin
    directory="$(dirname "$destination")"
    mkdir -p "$directory"
    available_kib="$(df -Pk "$directory" | awk 'NR == 2 {print $4}')"
    [[ "$available_kib" =~ ^[0-9]+$ ]] ||
        die "Could not determine free space for ${directory}"
    available_bytes=$((available_kib * 1024))
    margin=$((1024 * 1024 * 1024))
    if ((available_bytes < required_bytes + margin)); then
        die "Insufficient free space for ${destination}: need ${required_bytes} bytes plus a 1 GiB margin, have ${available_bytes} bytes"
    fi
}

download_file() {
    local url="$1"
    local destination="$2"
    local expected_size="$3"
    local label="$4"

    if check_existing_file "$destination" "$expected_size" "$label"; then
        log "Verified existing ${label}: ${destination}"
        return
    fi
    if ((DRY_RUN)); then
        log "DRY-RUN ${label}: ${url} -> ${destination} (${expected_size} bytes)"
        return
    fi

    local part="${destination}.part"
    local part_size=0
    mkdir -p "$(dirname "$destination")"
    if [[ -e "$part" ]]; then
        [[ -f "$part" ]] || die "Partial path is not a regular file: ${part}"
        part_size="$(file_size "$part")"
        ((part_size <= expected_size)) ||
            die "Partial file is larger than expected; move it aside and retry: ${part}"
        if ((part_size == expected_size)); then
            mv "$part" "$destination"
            log "Recovered complete partial file: ${destination}"
            return
        fi
    fi

    ensure_disk_space "$destination" $((expected_size - part_size))
    log "Downloading ${label}: ${url}"
    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 8 \
        --retry-delay 5 \
        --connect-timeout 30 \
        --output "$part" \
        "$url"

    local actual_size
    actual_size="$(file_size "$part")"
    [[ "$actual_size" == "$expected_size" ]] ||
        die "${label} download has size ${actual_size}, expected ${expected_size}: ${part}"
    mv "$part" "$destination"
    log "Completed ${label}: ${destination}"
}

check_archive() {
    local archive="$1"
    local archive_type="$2"
    local member
    local -a components

    case "$archive_type" in
        tar)
            tar -tf "$archive" >/dev/null
            ;;
        tgz)
            tar -tzf "$archive" >/dev/null
            ;;
        *)
            die "Unsupported archive type: ${archive_type}"
            ;;
    esac

    while IFS= read -r member; do
        [[ "$member" != /* ]] ||
            die "Archive contains an absolute path: ${member}"
        IFS='/' read -r -a components <<<"$member"
        local component
        for component in "${components[@]}"; do
            [[ "$component" != ".." ]] ||
                die "Archive contains a parent-directory component: ${member}"
        done
    done < <(
        if [[ "$archive_type" == "tar" ]]; then
            tar -tf "$archive"
        else
            tar -tzf "$archive"
        fi
    )
}

extract_archive_dataset() {
    local dataset="$1"
    local url="$2"
    local archive_name="$3"
    local archive_size="$4"
    local archive_type="$5"
    local extracted_size="$6"
    shift 6
    local -a specs=("$@")

    local spec source_name relative_path expected_size destination
    local missing=0
    for spec in "${specs[@]}"; do
        IFS='|' read -r source_name relative_path expected_size <<<"$spec"
        destination="${ROOT}/${relative_path}"
        if ! check_existing_file "$destination" "$expected_size" "${dataset}:${source_name}"; then
            missing=1
        fi
    done
    if ((missing == 0)); then
        log "All ${dataset} files are already valid"
        return
    fi

    local archive="${ROOT}/.downloads/${archive_name}"
    if ((DRY_RUN)); then
        log "DRY-RUN ${dataset} archive: ${url} -> ${archive} (${archive_size} bytes)"
        for spec in "${specs[@]}"; do
            IFS='|' read -r source_name relative_path expected_size <<<"$spec"
            destination="${ROOT}/${relative_path}"
            if [[ ! -e "$destination" ]]; then
                log "DRY-RUN extract ${source_name} -> ${destination} (${expected_size} bytes)"
            fi
        done
        return
    fi

    download_file "$url" "$archive" "$archive_size" "${dataset} archive"
    check_archive "$archive" "$archive_type"
    ensure_disk_space "${ROOT}/.extract/${dataset}" "$extracted_size"

    local extract_dir
    extract_dir="$(mktemp -d "${ROOT}/.extract/${dataset}.XXXXXX")"
    log "Extracting ${dataset} archive"
    if [[ "$archive_type" == "tar" ]]; then
        tar --no-same-owner --no-same-permissions -xf "$archive" -C "$extract_dir"
    else
        tar --no-same-owner --no-same-permissions -xzf "$archive" -C "$extract_dir"
    fi

    for spec in "${specs[@]}"; do
        IFS='|' read -r source_name relative_path expected_size <<<"$spec"
        destination="${ROOT}/${relative_path}"
        if check_existing_file "$destination" "$expected_size" "${dataset}:${source_name}"; then
            continue
        fi

        local -a matches=()
        mapfile -t matches < <(find "$extract_dir" -type f -name "$source_name" -print)
        ((${#matches[@]} == 1)) ||
            die "Expected exactly one ${source_name} in ${archive}, found ${#matches[@]}; extraction left at ${extract_dir}"
        check_existing_file "${matches[0]}" "$expected_size" "${dataset}:${source_name}"
        mkdir -p "$(dirname "$destination")"
        mv "${matches[0]}" "$destination"
        log "Installed ${dataset}:${source_name} -> ${destination}"
    done

    [[ "$extract_dir" == "${ROOT}/.extract/${dataset}."* ]] ||
        die "Refusing to remove unexpected temporary path: ${extract_dir}"
    rm -rf -- "$extract_dir"
}

fetch_deep_common() {
    download_file "$DEEP_QUERY_URL" "${ROOT}/deep/query/query.public.10K.fbin" 3840008 "Deep public queries"
    download_file "$DEEP_GT_URL" "${ROOT}/deep/gt/groundtruth.public.10K.ibin" 4000008 "Deep public ground truth"
}

fetch_t2i_common() {
    download_file "$T2I_QUERY_URL" "${ROOT}/t2i/query/query.public.100K.fbin" 80000008 "T2I public queries"
    download_file "$T2I_GT_URL" "${ROOT}/t2i/gt/groundtruth.public.100K.ibin" 40000008 "T2I public ground truth"
}

fetch_dataset() {
    local dataset="$1"
    case "$dataset" in
        deep10m)
            download_file "$DEEP_BASE_10M_URL" "${ROOT}/deep/1M/base.10M.fbin" 3840000008 "Deep10M base"
            fetch_deep_common
            ;;
        deep1b)
            download_file "$DEEP_BASE_1B_URL" "${ROOT}/deep/1B/base.1B.fbin" 384000000008 "Deep1B base"
            fetch_deep_common
            ;;
        t2i1m)
            download_file "$T2I_BASE_1M_URL" "${ROOT}/t2i/1M/base.1M.fbin" 800000008 "T2I1M base"
            fetch_t2i_common
            ;;
        t2i1b)
            download_file "$T2I_BASE_1B_URL" "${ROOT}/t2i/1B/base.1B.fbin" 800000000008 "T2I1B base"
            fetch_t2i_common
            ;;
        wiki1m)
            extract_archive_dataset \
                "wiki1m" "$WIKI_ARCHIVE_URL" "wiki_all_1M.tar" 3110727680 "tar" 3110727680 \
                "base.1M.fbin|wiki/wiki1m/base.1M.fbin|3072000008" \
                "queries.fbin|wiki/wiki1m/queries.fbin|30720008" \
                "groundtruth.1M.neighbors.ibin|wiki/wiki1m/groundtruth.1M.neighbors.ibin|4000008"
            ;;
        w2v1m)
            extract_archive_dataset \
                "w2v1m" "$W2V_ARCHIVE_URL" "word2vec.tar.gz" 566189722 "tgz" 1205204000 \
                "word2vec_base.fvecs|w2v/word2vec/word2vec_base.fvecs|1204000000" \
                "word2vec_query.fvecs|w2v/word2vec/word2vec_query.fvecs|1204000"
            ;;
        glove2m)
            extract_archive_dataset \
                "glove2m" "$GLOVE_ARCHIVE_URL" "glove2.2m.tar.gz" 2422999860 "tgz" 2645208468 \
                "glove2.2m_base.fvecs|glove/glove2.2m/glove2.2m_base.fvecs|2644004468" \
                "glove2.2m_query.fvecs|glove/glove2.2m/glove2.2m_query.fvecs|1204000"
            ;;
        sift1m)
            extract_archive_dataset \
                "sift1m" "$SIFT_ARCHIVE_URL" "sift.tar.gz" 168280445 "tgz" 600000000 \
                "sift_base.fvecs|sift/1M/sift/sift_base.fvecs|516000000" \
                "sift_query.fvecs|sift/1M/sift/sift_query.fvecs|5160000" \
                "sift_groundtruth.ivecs|sift/1M/sift/sift_groundtruth.ivecs|4040000"
            ;;
        pubmed)
            download_file "$PUBMED_BASE_URL" "${ROOT}/pubmed/doc_vectors_norm.bin" 1536000008 "PubMed base"
            download_file "$PUBMED_QUERY_URL" "${ROOT}/pubmed/query_vectors_norm.bin" 307208 "PubMed queries"
            ;;
        *)
            die "Unsupported dataset: ${dataset}"
            ;;
    esac
}

add_selected() {
    local dataset="$1"
    if [[ -z "${SEEN[$dataset]:-}" ]]; then
        SEEN["$dataset"]=1
        SELECTED+=("$dataset")
    fi
}

expand_requested() {
    local item
    for item in "${REQUESTED[@]}"; do
        case "$item" in
            figure14)
                add_selected deep10m
                add_selected t2i1m
                add_selected wiki1m
                add_selected w2v1m
                add_selected glove2m
                add_selected sift1m
                add_selected pubmed
                ;;
            all)
                add_selected deep10m
                add_selected t2i1m
                add_selected wiki1m
                add_selected w2v1m
                add_selected glove2m
                add_selected sift1m
                add_selected pubmed
                add_selected deep1b
                add_selected t2i1b
                ;;
            deep10m|t2i1m|wiki1m|w2v1m|glove2m|sift1m|pubmed|deep1b|t2i1b)
                add_selected "$item"
                ;;
            *)
                die "Unknown dataset or group: ${item}"
                ;;
        esac
    done
}

verify_selected() {
    python3 - "$ROOT" "${SELECTED[@]}" <<'PY'
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
selected = sys.argv[2:]
specs = {
    "deep10m": (
        ("deep/1M/base.10M.fbin", "bin", 10_000_000, 96),
        ("deep/query/query.public.10K.fbin", "bin", 10_000, 96),
        ("deep/gt/groundtruth.public.10K.ibin", "bin", 10_000, 100),
    ),
    "deep1b": (
        ("deep/1B/base.1B.fbin", "bin", 1_000_000_000, 96),
        ("deep/query/query.public.10K.fbin", "bin", 10_000, 96),
        ("deep/gt/groundtruth.public.10K.ibin", "bin", 10_000, 100),
    ),
    "t2i1m": (
        ("t2i/1M/base.1M.fbin", "bin", 1_000_000, 200),
        ("t2i/query/query.public.100K.fbin", "bin", 100_000, 200),
        ("t2i/gt/groundtruth.public.100K.ibin", "bin", 100_000, 100),
    ),
    "t2i1b": (
        ("t2i/1B/base.1B.fbin", "bin", 1_000_000_000, 200),
        ("t2i/query/query.public.100K.fbin", "bin", 100_000, 200),
        ("t2i/gt/groundtruth.public.100K.ibin", "bin", 100_000, 100),
    ),
    "wiki1m": (
        ("wiki/wiki1m/base.1M.fbin", "bin", 1_000_000, 768),
        ("wiki/wiki1m/queries.fbin", "bin", 10_000, 768),
        ("wiki/wiki1m/groundtruth.1M.neighbors.ibin", "bin", 10_000, 100),
    ),
    "w2v1m": (
        ("w2v/word2vec/word2vec_base.fvecs", "vecs", 1_000_000, 300),
        ("w2v/word2vec/word2vec_query.fvecs", "vecs", 1_000, 300),
    ),
    "glove2m": (
        ("glove/glove2.2m/glove2.2m_base.fvecs", "vecs", 2_196_017, 300),
        ("glove/glove2.2m/glove2.2m_query.fvecs", "vecs", 1_000, 300),
    ),
    "sift1m": (
        ("sift/1M/sift/sift_base.fvecs", "vecs", 1_000_000, 128),
        ("sift/1M/sift/sift_query.fvecs", "vecs", 10_000, 128),
        ("sift/1M/sift/sift_groundtruth.ivecs", "vecs", 10_000, 100),
    ),
    "pubmed": (
        ("pubmed/doc_vectors_norm.bin", "bin", 500_000, 768),
        ("pubmed/query_vectors_norm.bin", "bin", 100, 768),
    ),
}

unique = {}
for dataset in selected:
    for spec in specs[dataset]:
        unique[spec[0]] = spec

errors = []
for relative, kind, rows, dimensions in unique.values():
    path = root / relative
    try:
        if not path.is_file():
            raise ValueError(f"missing file: {path}")
        expected_size = (
            8 + rows * dimensions * 4
            if kind == "bin"
            else rows * (4 + dimensions * 4)
        )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {path}: got {actual_size}, expected {expected_size}"
            )
        with path.open("rb") as stream:
            if kind == "bin":
                header = stream.read(8)
                actual_rows, actual_dimensions = struct.unpack("<II", header)
                if (actual_rows, actual_dimensions) != (rows, dimensions):
                    raise ValueError(
                        f"header mismatch for {path}: got "
                        f"{actual_rows}x{actual_dimensions}, expected {rows}x{dimensions}"
                    )
            else:
                first_dimensions = struct.unpack("<i", stream.read(4))[0]
                stream.seek((rows - 1) * (4 + dimensions * 4))
                last_dimensions = struct.unpack("<i", stream.read(4))[0]
                if first_dimensions != dimensions or last_dimensions != dimensions:
                    raise ValueError(
                        f"row header mismatch for {path}: first={first_dimensions}, "
                        f"last={last_dimensions}, expected={dimensions}"
                    )
    except (OSError, ValueError, struct.error) as error:
        errors.append(str(error))

if errors:
    for error in errors:
        print(f"RAW_DATASETS_ERROR {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"RAW_DATASETS_OK datasets={len(selected)} files={len(unique)} root={root}")
PY
}

while (($#)); do
    case "$1" in
        --root)
            (($# >= 2)) || die "--root requires a directory"
            ROOT="$2"
            shift 2
            ;;
        --datasets)
            (($# >= 2)) || die "--datasets requires a comma-separated list"
            IFS=',' read -r -a parsed <<<"$2"
            REQUESTED+=("${parsed[@]}")
            shift 2
            ;;
        --include-billion)
            INCLUDE_BILLION=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        --no-verify)
            VERIFY=0
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            REQUESTED+=("$1")
            shift
            ;;
    esac
done

if ((LIST_ONLY)); then
    list_datasets
    exit 0
fi

[[ -n "$ROOT" ]] || die "--root is required"
require_command readlink
ROOT="$(readlink -m -- "$ROOT")"
[[ "$ROOT" != "/" ]] || die "Refusing to use the filesystem root as --root"
((${#REQUESTED[@]} > 0)) || die "At least one dataset or group is required"
((DRY_RUN == 0 || VERIFY_ONLY == 0)) ||
    die "--dry-run and --verify-only cannot be used together"
((VERIFY == 1 || VERIFY_ONLY == 0)) ||
    die "--verify-only and --no-verify cannot be used together"
expand_requested

if ((VERIFY_ONLY == 0)); then
    for dataset in "${SELECTED[@]}"; do
        if [[ "$dataset" == "deep1b" || "$dataset" == "t2i1b" ]]; then
            ((INCLUDE_BILLION)) ||
                die "${dataset} requires --include-billion"
        fi
    done
fi

require_command stat
if ((DRY_RUN == 0)); then
    require_command python3
fi
if ((DRY_RUN == 0 && VERIFY_ONLY == 0)); then
    require_command awk
    require_command curl
    require_command df
    require_command find
    require_command mktemp
    require_command tar
    mkdir -p "${ROOT}/.downloads" "${ROOT}/.extract"
fi

if ((VERIFY_ONLY == 0)); then
    for dataset in "${SELECTED[@]}"; do
        fetch_dataset "$dataset"
    done
fi

if ((DRY_RUN)); then
    log "Dry run completed; no files were written"
elif ((VERIFY)); then
    verify_selected
else
    log "Post-download verification was disabled"
fi

if ((DRY_RUN == 0)); then
    log "Raw dataset preparation completed"
    log "Next: run script/cpu_index_build.sh for normalized index/query/GT artifacts"
fi
