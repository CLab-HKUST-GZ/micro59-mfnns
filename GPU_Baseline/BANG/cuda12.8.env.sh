#!/usr/bin/env bash
# Source this file before manual Figure 14 BANG builds or runs.

task_cuda_root="${FIGURE14_BANG_CUDA_ROOT:-/usr/local/cuda-12.8}"
if [[ ! -x "$task_cuda_root/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.8 nvcc is missing: $task_cuda_root/bin/nvcc" >&2
    return 2 2>/dev/null || exit 2
fi
task_cuda_version="$($task_cuda_root/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -1)"
if [[ "$task_cuda_version" != "12.8" ]]; then
    echo "ERROR: Figure 14 BANG requires CUDA 12.8, got $task_cuda_version" >&2
    return 2 2>/dev/null || exit 2
fi

export CUDA_HOME="$task_cuda_root"
export CUDACXX="$task_cuda_root/bin/nvcc"
export PATH="$task_cuda_root/bin:$PATH"
export LD_LIBRARY_PATH="$task_cuda_root/lib64:${LD_LIBRARY_PATH:-}"
printf 'FIGURE14_BANG_CUDA=%s CUDA_HOME=%s\n' "$task_cuda_version" "$CUDA_HOME"
