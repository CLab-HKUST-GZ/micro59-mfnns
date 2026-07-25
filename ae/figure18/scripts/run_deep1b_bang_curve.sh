#!/usr/bin/env bash
# Reproduce the BANG DP1B Recall@10/Recall@100 frontier used by Figure 18.
set -euo pipefail

: "${BANG_REPO:?Set BANG_REPO to the BANG-Billion-Scale-ANN checkout.}"
: "${BASE:?Set BASE to base.1B.fbin.}"
: "${QUERY:?Set QUERY to query.public.10K.fbin.}"
: "${INDEX_PREFIX:?Set INDEX_PREFIX to the complete R64/Lbuild100/QD32 prefix.}"

BANG_SEARCH="${BANG_SEARCH:-${BANG_REPO}/BANG_Base/build/bang_search}"
WORK_DIR="${WORK_DIR:-/local-ssd/$(id -un)/figure18_bang_deep1b}"
DATA_DIR="${WORK_DIR}/data"
LOG_DIR="${WORK_DIR}/logs"
RESULT_DIR="${WORK_DIR}/results"
mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${RESULT_DIR}"

OFFICIAL_GT_URL="${OFFICIAL_GT_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin}"
GT_IBIN="${GT_IBIN:-${DATA_DIR}/groundtruth.public.10K.ibin}"
GT_BANG="${GT_BANG:-${DATA_DIR}/groundtruth.public.10K_top100_bang.bin}"
NUM_QUERIES="${NUM_QUERIES:-10000}"
K10_L_LIST="${K10_L_LIST:-10 15 20 30 40 60 80 160 320 512}"
K100_L_LIST="${K100_L_LIST:-100 130 160 200 240 280 320 512}"

export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export LD_LIBRARY_PATH="${BANG_REPO}/BANG_Base/build:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[ "${NUM_QUERIES}" = "10000" ] || fail "Figure 18 uses exactly NUM_QUERIES=10000"
[ -x "${BANG_SEARCH}" ] || fail "missing executable: ${BANG_SEARCH}"
[ -r "${BANG_REPO}/BANG_Base/bang_search.cu" ] || fail "missing BANG source configuration"
grep -Eq '^#define[[:space:]]+MAX_R[[:space:]]+64([[:space:]]|$)' \
  "${BANG_REPO}/BANG_Base/bang_search.cu" || fail "BANG source is not configured with MAX_R=64"

for suffix in _disk.bin _disk_metadata.bin _pq_compressed.bin _pq_pivots.bin; do
  [ -s "${INDEX_PREFIX}${suffix}" ] || fail "missing index artifact: ${INDEX_PREFIX}${suffix}"
done
[ -s "${BASE}" ] || fail "missing base file: ${BASE}"
[ -s "${QUERY}" ] || fail "missing query file: ${QUERY}"
for l in ${K10_L_LIST} ${K100_L_LIST}; do
  [ "${l}" -le 512 ] || fail "L=${l} exceeds the safe current BANG limit of 512"
done

if [ ! -s "${GT_IBIN}" ]; then
  curl -fL --retry 3 --retry-delay 5 "${OFFICIAL_GT_URL}" -o "${GT_IBIN}.tmp"
  mv "${GT_IBIN}.tmp" "${GT_IBIN}"
fi

GT_IBIN="${GT_IBIN}" GT_BANG="${GT_BANG}" python3 - <<'PY'
from array import array
import os
from pathlib import Path
import struct

src = Path(os.environ["GT_IBIN"])
dst = Path(os.environ["GT_BANG"])
with src.open("rb") as f:
    n, k = struct.unpack("<II", f.read(8))
    ids = f.read()
if (n, k) != (10000, 100):
    raise RuntimeError(f"unexpected official Deep GT header: {(n, k)}")
if len(ids) != n * k * 4:
    raise RuntimeError("unexpected official Deep GT payload length")
dists = array("f", [float(i) for _ in range(n) for i in range(k)])
with dst.open("wb") as f:
    f.write(struct.pack("<II", n, k))
    f.write(ids)
    f.write(dists.tobytes())
print(f"wrote {dst}: {dst.stat().st_size} bytes")
PY

BASE="${BASE}" QUERY="${QUERY}" GT_IBIN="${GT_IBIN}" \
  python3 - <<'PY' | tee "${RESULT_DIR}/official10k_gt_validation.txt"
import os
from pathlib import Path
import random
import struct

base, query, gt = (Path(os.environ[name]) for name in ("BASE", "QUERY", "GT_IBIN"))

def read_header(path):
    with path.open("rb") as f:
        return struct.unpack("<II", f.read(8))

def read_vector(handle, index, dim):
    handle.seek(8 + index * dim * 4)
    return struct.unpack("<" + "f" * dim, handle.read(dim * 4))

def l2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

base_n, base_d = read_header(base)
query_n, query_d = read_header(query)
gt_n, gt_k = read_header(gt)
if (base_n, base_d) != (1_000_000_000, 96):
    raise RuntimeError(f"unexpected base header: {(base_n, base_d)}")
if query_n < 10000 or query_d != 96:
    raise RuntimeError(f"unexpected query header: {(query_n, query_d)}")
if (gt_n, gt_k) != (10000, 100):
    raise RuntimeError(f"unexpected GT header: {(gt_n, gt_k)}")
with gt.open("rb") as f:
    f.seek(8)
    ids = struct.unpack("<" + "I" * (gt_n * gt_k), f.read(gt_n * gt_k * 4))

rng = random.Random(20260615)
random_ids = [rng.randrange(base_n) for _ in range(200)]
with base.open("rb") as bf:
    random_vectors = [read_vector(bf, index, base_d) for index in random_ids]

l2_failures = ip_failures = 0
with base.open("rb") as bf, query.open("rb") as qf:
    for q in range(1000):
        qv = read_vector(qf, q, query_d)
        gv = read_vector(bf, ids[q * gt_k], base_d)
        l2_failures += l2(qv, gv) > min(l2(qv, rv) for rv in random_vectors)
        ip_failures += dot(qv, gv) < max(dot(qv, rv) for rv in random_vectors)

print(f"base_header={base_n}x{base_d}")
print(f"query_header={query_n}x{query_d}")
print(f"gt_header={gt_n}x{gt_k}")
print("queries_checked=1000 random_candidates=200")
print(f"gt_top1_beaten_by_random_l2={l2_failures}")
print(f"gt_top1_beaten_by_random_ip={ip_failures}")
if l2_failures > 10 or ip_failures > 10:
    raise SystemExit("ground-truth sanity check failed")
PY

run_bang() {
  local k="$1"
  local l_list="$2"
  local tag
  tag="$(echo "${l_list}" | tr ' ' '_')"
  local log="${LOG_DIR}/deep1b_qd32_k${k}_q${NUM_QUERIES}_L${tag}.log"
  local raw_csv="${RESULT_DIR}/raw_k${k}_q${NUM_QUERIES}.csv"

  {
    echo "BEGIN $(date --iso-8601=seconds) host=$(hostname -f)"
    echo "BANG_SEARCH=${BANG_SEARCH}"
    echo "INDEX_PREFIX=${INDEX_PREFIX}"
    echo "QUERY=${QUERY}"
    echo "GT_BANG=${GT_BANG}"
    echo "NUM_QUERIES=${NUM_QUERIES} K=${k} L_LIST=${l_list}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    stat -c '%n %s' "${INDEX_PREFIX}"_disk.bin "${INDEX_PREFIX}"_disk_metadata.bin \
      "${INDEX_PREFIX}"_pq_compressed.bin "${INDEX_PREFIX}"_pq_pivots.bin "${QUERY}" "${GT_BANG}"
    read -r -a l_array <<< "${l_list}"
    {
      for index in "${!l_array[@]}"; do
        printf '%s\n' "${l_array[$index]}"
        [ "${index}" -lt "$(( ${#l_array[@]} - 1 ))" ] && printf 'y\n' || printf 'n\n'
      done
    } | /usr/bin/time -v "${BANG_SEARCH}" "${INDEX_PREFIX}" "${QUERY}" "${GT_BANG}" "${NUM_QUERIES}" "${k}" float l2
    echo "END $(date --iso-8601=seconds)"
  } 2>&1 | tee "${log}"

  awk -v k="${k}" '
    /^L[[:space:]]+Time/ { in_table=1; next }
    /^--/ { next }
    in_table && $1 ~ /^[0-9]+$/ && NF >= 4 { print k "," $1 "," $2 "," $3 "," $4 }
  ' "${log}" > "${raw_csv}.tmp"
  { echo "k,L,time_ms,qps,recall"; cat "${raw_csv}.tmp"; } > "${raw_csv}"
  rm -f "${raw_csv}.tmp"
}

run_bang 10 "${K10_L_LIST}"
run_bang 100 "${K100_L_LIST}"

RESULT_DIR="${RESULT_DIR}" NUM_QUERIES="${NUM_QUERIES}" python3 - <<'PY'
import csv
import os
from collections import defaultdict
from pathlib import Path
import statistics

result_dir = Path(os.environ["RESULT_DIR"])
num_queries = os.environ["NUM_QUERIES"]
rows = defaultdict(list)
for path in sorted(result_dir.glob(f"raw_k*_q{num_queries}.csv")):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(int(row["k"]), int(row["L"]))].append(row)

out = result_dir / f"summary_qps_recall_curve_q{num_queries}.csv"
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "L", "recall_percent", "runs", "qps_all", "median_warm_qps", "median_last3_qps"])
    for key, point_rows in sorted(rows.items()):
        qps = [float(row["qps"]) for row in point_rows]
        recalls = [float(row["recall"]) for row in point_rows]
        warm = qps[1:] if len(qps) > 1 else qps
        last3 = qps[-3:] if len(qps) >= 3 else qps
        writer.writerow([key[0], key[1], f"{statistics.median(recalls):.4f}", len(qps),
                         " ".join(f"{value:.2f}" for value in qps),
                         f"{statistics.median(warm):.2f}", f"{statistics.median(last3):.2f}"])
print(out)
PY

echo "Completed. Results: ${RESULT_DIR}"
