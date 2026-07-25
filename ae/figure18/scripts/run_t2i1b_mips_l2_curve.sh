#!/usr/bin/env bash
# Run a metric-compatible T2I1B BANG curve against official MIPS labels.
# The index must have been built on the 201D MIPS-to-L2 transformation.
set -euo pipefail

: "${BANG_REPO:?Set BANG_REPO to the BANG-Billion-Scale-ANN checkout.}"
: "${INDEX_PREFIX:?Set INDEX_PREFIX to a BANG-compatible transformed 201D prefix.}"
: "${QUERY_RAW:?Set QUERY_RAW to query.public.100K.fbin (raw 200D).}"
: "${GT_RAW:?Set GT_RAW to groundtruth.public.100K.ibin (official MIPS GT).}"

BANG_SEARCH="${BANG_SEARCH:-${BANG_REPO}/BANG_Base/build/bang_search}"
WORK_DIR="${WORK_DIR:-/local-ssd/$(id -un)/figure18_bang_t2i1b}"
DATA_DIR="${WORK_DIR}/data"
LOG_DIR="${WORK_DIR}/logs"
RESULT_DIR="${WORK_DIR}/results"
mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${RESULT_DIR}"

NUM_QUERIES="${NUM_QUERIES:-10000}"
GT_TOPK="${GT_TOPK:-100}"
K10_L_LIST="${K10_L_LIST:-10 20 40 80 160 320 512}"
K100_L_LIST="${K100_L_LIST:-100 160 240 320 512}"
QUERY_L2="${QUERY_L2:-${DATA_DIR}/query.public.first${NUM_QUERIES}.mips_l2_201d.fbin}"
GT_BANG="${GT_BANG:-${DATA_DIR}/groundtruth.public.first${NUM_QUERIES}.top${GT_TOPK}_bang.bin}"

export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export LD_LIBRARY_PATH="${BANG_REPO}/BANG_Base/build:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "${NUM_QUERIES}" = "10000" ] || fail "Figure 18 uses exactly NUM_QUERIES=10000"
[ -x "${BANG_SEARCH}" ] || fail "missing executable: ${BANG_SEARCH}"
grep -Eq '^#define[[:space:]]+MAX_R[[:space:]]+64([[:space:]]|$)' \
  "${BANG_REPO}/BANG_Base/bang_search.cu" || fail "BANG source is not configured with MAX_R=64"
for suffix in _disk.bin _disk_metadata.bin _pq_compressed.bin _pq_pivots.bin; do
  [ -s "${INDEX_PREFIX}${suffix}" ] || fail "missing index artifact: ${INDEX_PREFIX}${suffix}"
done
[ -s "${QUERY_RAW}" ] || fail "missing raw query file"
[ -s "${GT_RAW}" ] || fail "missing official GT file"
for l in ${K10_L_LIST} ${K100_L_LIST}; do
  [ "${l}" -le 512 ] || fail "L=${l} exceeds the safe current BANG limit of 512"
done

QUERY_RAW="${QUERY_RAW}" GT_RAW="${GT_RAW}" QUERY_L2="${QUERY_L2}" GT_BANG="${GT_BANG}" \
INDEX_META="${INDEX_PREFIX}_disk_metadata.bin" PQ_PIVOTS="${INDEX_PREFIX}_pq_pivots.bin" \
NUM_QUERIES="${NUM_QUERIES}" GT_TOPK="${GT_TOPK}" python3 - <<'PY' | tee "${RESULT_DIR}/input_validation.txt"
from array import array
import os
from pathlib import Path
import struct

query_raw = Path(os.environ["QUERY_RAW"])
gt_raw = Path(os.environ["GT_RAW"])
query_l2 = Path(os.environ["QUERY_L2"])
gt_bang = Path(os.environ["GT_BANG"])
meta = Path(os.environ["INDEX_META"])
pivots = Path(os.environ["PQ_PIVOTS"])
nq = int(os.environ["NUM_QUERIES"])
topk = int(os.environ["GT_TOPK"])

with meta.open("rb") as f:
    medoid, entry_len, dtype, dim, degree, n = struct.unpack("<QQIIII", f.read(32))
if (dtype, dim, degree, n) != (2, 201, 64, 1_000_000_000):
    raise RuntimeError(f"need transformed float 1B R64 metadata with D=201, got {(dtype, dim, degree, n)}")
if Path(str(meta).replace("_disk_metadata.bin", "_disk.bin")).stat().st_size != n * entry_len:
    raise RuntimeError("disk.bin size does not match metadata")
with pivots.open("rb") as f:
    noff, ncol = struct.unpack("<II", f.read(8))
    offsets = struct.unpack("<" + "Q" * noff, f.read(8 * noff))
    f.seek(offsets[0])
    pivot_rows, pivot_dim = struct.unpack("<II", f.read(8))
if (noff, ncol, pivot_rows, pivot_dim) != (4, 1, 256, 201):
    raise RuntimeError(f"PQ pivots are not transformed 201D: {(noff, ncol, pivot_rows, pivot_dim)}")

with query_raw.open("rb") as f:
    total, raw_dim = struct.unpack("<II", f.read(8))
    if total < nq or raw_dim != 200:
        raise RuntimeError(f"expected at least {nq} raw 200D queries, got {(total, raw_dim)}")
    raw = f.read(nq * raw_dim * 4)
with gt_raw.open("rb") as f:
    gt_total, gt_width = struct.unpack("<II", f.read(8))
    if gt_total < nq or gt_width < topk:
        raise RuntimeError(f"unexpected GT shape {(gt_total, gt_width)}")
    gt_all = f.read(nq * gt_width * 4)

with query_l2.open("wb") as f:
    f.write(struct.pack("<II", nq, 201))
    for row in struct.iter_unpack("<" + "f" * raw_dim, raw):
        f.write(struct.pack("<" + "f" * 201, *row, 0.0))
with gt_bang.open("wb") as f:
    f.write(struct.pack("<II", nq, topk))
    for q in range(nq):
        offset = q * gt_width * 4
        f.write(gt_all[offset:offset + topk * 4])
    array("f", [float(i) for _ in range(nq) for i in range(topk)]).tofile(f)

print(f"index_metadata=N={n} D={dim} R={degree} entry_len={entry_len}")
print(f"query_raw={total}x{raw_dim}; query_l2={nq}x201")
print(f"gt_raw={gt_total}x{gt_width}; bang_gt={nq}x{topk}")
PY

run_bang() {
  local k="$1" l_list="$2" tag log raw
  tag="$(echo "${l_list}" | tr ' ' '_')"
  log="${LOG_DIR}/t2i1b_mips_l2_k${k}_q${NUM_QUERIES}_L${tag}.log"
  raw="${RESULT_DIR}/raw_k${k}_q${NUM_QUERIES}.csv"
  {
    echo "BEGIN $(date --iso-8601=seconds) host=$(hostname -f)"
    echo "INDEX_PREFIX=${INDEX_PREFIX} QUERY=${QUERY_L2} GT=${GT_BANG} K=${k} L_LIST=${l_list}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    read -r -a l_array <<< "${l_list}"
    {
      for index in "${!l_array[@]}"; do
        printf '%s\n' "${l_array[$index]}"
        [ "${index}" -lt "$(( ${#l_array[@]} - 1 ))" ] && printf 'y\n' || printf 'n\n'
      done
    } | /usr/bin/time -v "${BANG_SEARCH}" "${INDEX_PREFIX}" "${QUERY_L2}" "${GT_BANG}" "${NUM_QUERIES}" "${k}" float l2
    echo "END $(date --iso-8601=seconds)"
  } 2>&1 | tee "${log}"
  awk -v k="${k}" '/^L[[:space:]]+Time/ { p=1; next } /^--/ { next } p && $1 ~ /^[0-9]+$/ && NF >= 4 { print k "," $1 "," $2 "," $3 "," $4 }' "${log}" > "${raw}.tmp"
  { echo "k,L,time_ms,qps,recall_percent"; cat "${raw}.tmp"; } > "${raw}"
  rm -f "${raw}.tmp"
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
rows = defaultdict(list)
for path in sorted(result_dir.glob(f"raw_k*_q{os.environ['NUM_QUERIES']}.csv")):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(int(row["k"]), int(row["L"]))].append(row)
out = result_dir / f"summary_qps_recall_curve_q{os.environ['NUM_QUERIES']}.csv"
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "L", "recall_percent", "runs", "qps_all", "median_warm_qps", "median_last3_qps"])
    for key, points in sorted(rows.items()):
        qps = [float(point["qps"]) for point in points]
        recall = statistics.median(float(point["recall_percent"]) for point in points)
        writer.writerow([key[0], key[1], f"{recall:.4f}", len(points),
                         " ".join(f"{value:.2f}" for value in qps),
                         f"{statistics.median((qps[1:] or qps)):.2f}",
                         f"{statistics.median(qps[-3:]):.2f}"])
print(out)
PY

echo "Completed. Results: ${RESULT_DIR}"
