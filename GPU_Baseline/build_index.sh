#!/usr/bin/env bash
# Select a recorded graph configuration and invoke the canonical index builder.
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task_usage() {
    cat <<'EOF'
Usage:
  GPU_Baseline/build_index.sh --list
  GPU_Baseline/build_index.sh bang DATASET [BUILD_OPTIONS...]
  GPU_Baseline/build_index.sh cagra DATASET [BUILD_OPTIONS...]

Datasets:
  deep10M glove2_2m pubmed sift1M text2img1M wiki1M word2vec

The recorded unified/k=100 graph parameters and output directory are supplied
automatically. BUILD_OPTIONS must include --base FILE. Use --print-command to
show the resolved invocation without executing it.

Output root:
  GPU_BASELINE_INDEX_ROOT
      default: GPU_Baseline/index

Examples:
  GPU_Baseline/build_index.sh bang wiki1M \
    --base /path/to/wiki1M_base.bin --dry-run

  GPU_Baseline/build_index.sh cagra wiki1M \
    --base /path/to/wiki1M_base.bin --gpu 1
EOF
}

task_fail() {
    echo "ERROR: $*" >&2
    exit 2
}

task_list() {
    cat <<'EOF'
dataset       BANG R/L/PQ/BF             CAGRA gd/igd
deep10M       32/64/96/399887            28/56
glove2_2m     64/256/128/399887          12/60
pubmed        32/64/128/399887           64/128
sift1M        32/64/128/399887           64/128
text2img1M    16/64/128/399887           16/32
wiki1M        32/64/128/99991            6/12
word2vec      32/64/128/399887           64/128
EOF
}

task_dataset_config() {
    local task_name="$1"
    case "$task_name" in
        deep10M)
            task_bang_r=32
            task_bang_l=64
            task_bang_pq=96
            task_bang_bf=399887
            task_cagra_gd=28
            task_cagra_igd=56
            ;;
        glove2_2m)
            task_bang_r=64
            task_bang_l=256
            task_bang_pq=128
            task_bang_bf=399887
            task_cagra_gd=12
            task_cagra_igd=60
            ;;
        pubmed)
            task_bang_r=32
            task_bang_l=64
            task_bang_pq=128
            task_bang_bf=399887
            task_cagra_gd=64
            task_cagra_igd=128
            ;;
        sift1M)
            task_bang_r=32
            task_bang_l=64
            task_bang_pq=128
            task_bang_bf=399887
            task_cagra_gd=64
            task_cagra_igd=128
            ;;
        text2img1M)
            task_bang_r=16
            task_bang_l=64
            task_bang_pq=128
            task_bang_bf=399887
            task_cagra_gd=16
            task_cagra_igd=32
            ;;
        wiki1M)
            task_bang_r=32
            task_bang_l=64
            task_bang_pq=128
            task_bang_bf=99991
            task_cagra_gd=6
            task_cagra_igd=12
            ;;
        word2vec)
            task_bang_r=32
            task_bang_l=64
            task_bang_pq=128
            task_bang_bf=399887
            task_cagra_gd=64
            task_cagra_igd=128
            ;;
        *)
            task_fail "unsupported dataset: $task_name"
            ;;
    esac
}

task_has_option() {
    local task_wanted="$1"
    shift
    local task_arg
    for task_arg in "$@"; do
        if [[ "$task_arg" == "$task_wanted" ||
              "$task_arg" == "$task_wanted="* ]]; then
            return 0
        fi
    done
    return 1
}

task_print_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

if (($# == 0)); then
    task_usage
    exit 2
fi
case "$1" in
    --list)
        (($# == 1)) || task_fail "--list does not accept other arguments"
        task_list
        exit 0
        ;;
    -h|--help)
        task_usage
        exit 0
        ;;
esac

(($# >= 2)) || task_fail "METHOD and DATASET are required"
task_method="${1,,}"
task_dataset="$2"
shift 2
task_extra=("$@")
task_dataset_config "$task_dataset"

task_print_only=0
task_forward=()
for task_arg in "${task_extra[@]}"; do
    if [[ "$task_arg" == "--print-command" ]]; then
        task_print_only=1
    else
        task_forward+=("$task_arg")
    fi
done

task_index_root="${GPU_BASELINE_INDEX_ROOT:-${task_root}/index}"
if [[ "$task_index_root" != /* ]]; then
    task_index_root="$(pwd)/${task_index_root}"
fi

case "$task_method" in
    bang)
        task_builder="${task_root}/BANG/index_build.sh"
        [[ -x "$task_builder" ]] ||
            task_fail "BANG builder link is missing or not executable: $task_builder"
        task_output="${task_index_root}/${task_dataset}/BANG"
        task_command=(
            "$task_builder"
            --dataset-name "$task_dataset"
            --output-dir "$task_output"
            --graph-degree "$task_bang_r"
            --build-l "$task_bang_l"
            --pq-chunks "$task_bang_pq"
            --bf-entries "$task_bang_bf"
        )
        if ! task_has_option --build-disk-index "${task_forward[@]}" &&
           [[ -x "${task_root}/BANG/build_disk_index" ]]; then
            task_command+=(
                --build-disk-index "${task_root}/BANG/build_disk_index"
            )
        fi
        ;;
    cagra)
        task_builder="${task_root}/CAGRA/index_build.sh"
        [[ -x "$task_builder" ]] ||
            task_fail "CAGRA builder link is missing or not executable: $task_builder"
        task_output="${task_index_root}/${task_dataset}/CAGRA"
        task_command=(
            "$task_builder"
            --dataset-name "$task_dataset"
            --cache-dir "$task_output"
            --graph-degree "$task_cagra_gd"
            --intermediate-graph-degree "$task_cagra_igd"
            --metric sqeuclidean
            --build-algo nn_descent
        )
        if ! task_has_option --python "${task_forward[@]}" &&
           [[ -x "${task_root}/CAGRA/python" ]]; then
            task_command+=(--python "${task_root}/CAGRA/python")
        fi
        ;;
    *)
        task_fail "METHOD must be bang or cagra: $task_method"
        ;;
esac

task_command+=("${task_forward[@]}")
if ((task_print_only == 1)); then
    task_print_command "${task_command[@]}"
    exit 0
fi

mkdir -p "$task_output"
exec "${task_command[@]}"
