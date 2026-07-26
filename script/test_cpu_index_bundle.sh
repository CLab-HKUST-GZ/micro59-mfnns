#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

fixture_dir="$(mktemp -d /tmp/mfnns-normalized-bundle-test.XXXXXX)"
cleanup() {
  case "${fixture_dir}" in
    /tmp/mfnns-normalized-bundle-test.*)
      rm -rf -- "${fixture_dir}"
      ;;
    *)
      echo "Refusing to remove unexpected fixture path: ${fixture_dir}" >&2
      exit 1
      ;;
  esac
}
trap cleanup EXIT

mkdir -p \
  "${fixture_dir}/data/t2i/1M" \
  "${fixture_dir}/data/t2i/query" \
  "${fixture_dir}/data/w2v/word2vec" \
  "${fixture_dir}/output"

python3 - "${fixture_dir}" <<'PY'
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = [
    (2.0, 0.0, 0.0),
    (0.0, 3.0, 0.0),
    (0.0, 0.0, 4.0),
    (1.0, 1.0, 0.0),
    (-2.0, 0.0, 0.0),
    (0.0, -5.0, 0.0),
]
queries = [
    (7.0, 1.0, 0.0),
    (0.0, 2.0, 2.0),
    (-3.0, 0.0, 0.0),
    (1.0, -4.0, 0.0),
]
for path, rows in (
    (root / "data/t2i/1M/base.1M.fbin", base),
    (root / "data/t2i/query/query.public.100K.fbin", queries),
):
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", len(rows), len(rows[0])))
        for row in rows:
            stream.write(struct.pack("<3f", *row))
for path, rows in (
    (root / "data/w2v/word2vec/word2vec_base.fvecs", base),
    (root / "data/w2v/word2vec/word2vec_query.fvecs", queries),
):
    with path.open("wb") as stream:
        for row in rows:
            stream.write(struct.pack("<i3f", len(row), *row))
PY

data_root="$(realpath --relative-to="${repo_root}" "${fixture_dir}/data")"
index_root="$(realpath --relative-to="${repo_root}" "${fixture_dir}/output")"
common_environment=(
  CPU_DATA_ROOT="${data_root}"
  CPU_INDEX_ROOT="${index_root}"
  CPU_QUERY_COUNT=6
  CPU_GT_TOPKS=1,2
  CPU_INDEX_THREADS=2
  CPU_INDEX_BATCH_SIZE=2
)

env "${common_environment[@]}" \
  script/cpu_index_build.sh t2i1m >"${fixture_dir}/first.log"
env "${common_environment[@]}" \
  script/cpu_index_build.sh t2i1m >"${fixture_dir}/cached.log"
env "${common_environment[@]}" \
  script/cpu_index_build.sh w2v1m >"${fixture_dir}/fvecs.log"
grep -Fq "status=generated" "${fixture_dir}/first.log"
grep -Fq "status=cached" "${fixture_dir}/cached.log"
grep -Fq "status=generated" "${fixture_dir}/fvecs.log"

python3 - "${fixture_dir}" <<'PY'
import math
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = root / "output/t2i1m"


def read_float_matrix(path):
    with path.open("rb") as stream:
        rows, cols = struct.unpack("<ii", stream.read(8))
        values = [
            struct.unpack(f"<{cols}f", stream.read(cols * 4))
            for _ in range(rows)
        ]
        assert stream.read() == b""
    return values


def read_uint_matrix(path):
    with path.open("rb") as stream:
        rows, cols = struct.unpack("<ii", stream.read(8))
        values = [
            struct.unpack(f"<{cols}I", stream.read(cols * 4))
            for _ in range(rows)
        ]
        assert stream.read() == b""
    return values


def normalize(row):
    norm = math.sqrt(sum(value * value for value in row))
    return tuple(value / norm for value in row)


base = [
    normalize(row)
    for row in read_float_matrix(root / "data/t2i/1M/base.1M.fbin")
]
source_queries = read_float_matrix(
    root / "data/t2i/query/query.public.100K.fbin"
)
queries = read_float_matrix(output / "query_vectors_n6_seed42.bin")
with (output / "query_indices_n6_seed42.bin").open("rb") as stream:
    count = struct.unpack("<i", stream.read(4))[0]
    indices = list(struct.unpack(f"<{count}i", stream.read(count * 4)))
assert indices == [0, 1, 2, 3, 0, 1], indices

expected_queries = [normalize(source_queries[index]) for index in indices]
for actual, expected in zip(queries, expected_queries):
    assert abs(math.sqrt(sum(value * value for value in actual)) - 1.0) <= 2e-7
    assert max(abs(left - right) for left, right in zip(actual, expected)) <= 1e-7

expected_gt = []
for query in queries:
    ranked = sorted(
        (
            sum((left - right) ** 2 for left, right in zip(query, point)),
            label,
        )
        for label, point in enumerate(base)
    )
    expected_gt.append(tuple(label for _, label in ranked[:2]))
gt1 = read_uint_matrix(output / "gt_labels_topk1_n6_seed42.bin")
gt2 = read_uint_matrix(output / "gt_labels_topk2_n6_seed42.bin")
assert gt2 == expected_gt, (gt2, expected_gt)
assert gt1 == [row[:1] for row in expected_gt]

index_path = output / "hnsw_index_M32_ef100.bin"
with index_path.open("rb") as stream:
    (
        _,
        _,
        current_elements,
        bytes_per_element,
        label_offset,
        data_offset,
    ) = struct.unpack("<6Q", stream.read(48))
    stream.read(4 + 4 + 8 + 8 + 8 + 8 + 8)
    header_bytes = stream.tell()
    dimension = (label_offset - data_offset) // 4
    norms = []
    for row in range(current_elements):
        stream.seek(header_bytes + row * bytes_per_element + data_offset)
        vector = struct.unpack(f"<{dimension}f", stream.read(dimension * 4))
        norms.append(math.sqrt(sum(value * value for value in vector)))
assert current_elements == 6
assert all(abs(norm - 1.0) <= 2e-7 for norm in norms), norms

index_metadata = (
    output / "hnsw_index_M32_ef100.bin.metadata.tsv"
).read_text()
bundle_metadata = (output / "bundle_metadata_n6_seed42.tsv").read_text()
assert "normalization\tl2\n" in index_metadata
assert "normalization\tl2\n" in bundle_metadata
assert "unique_query_count\t4\n" in bundle_metadata
index_fingerprint = next(
    line.split("\t", 1)[1]
    for line in index_metadata.splitlines()
    if line.startswith("base_fingerprint_fnv1a64\t")
)
bundle_fingerprint = next(
    line.split("\t", 1)[1]
    for line in bundle_metadata.splitlines()
    if line.startswith("base_fingerprint_fnv1a64\t")
)
assert index_fingerprint == bundle_fingerprint

w2v_output = root / "output/w2v1m"
assert (
    w2v_output / "query_vectors_n6_seed42.bin"
).read_bytes() == (
    output / "query_vectors_n6_seed42.bin"
).read_bytes()
for k in (1, 2):
    assert (
        w2v_output / f"gt_labels_topk{k}_n6_seed42.bin"
    ).read_bytes() == (
        output / f"gt_labels_topk{k}_n6_seed42.bin"
    ).read_bytes()
assert "base_format\tfvecs\n" in (
    w2v_output / "hnsw_index_M32_ef100.bin.metadata.tsv"
).read_text()
PY

metadata="${fixture_dir}/output/t2i1m/hnsw_index_M32_ef100.bin.metadata.tsv"
mv "${metadata}" "${metadata}.saved"
if env "${common_environment[@]}" \
  script/cpu_index_build.sh t2i1m >"${fixture_dir}/unverified.log" 2>&1; then
  echo "Expected an index without normalization metadata to be rejected" >&2
  exit 1
fi
grep -Fq "partial or unverified index cache" "${fixture_dir}/unverified.log"
mv "${metadata}.saved" "${metadata}"

python3 - "${fixture_dir}/output/t2i1m/query_vectors_n6_seed42.bin" <<'PY'
import struct
import sys

with open(sys.argv[1], "r+b") as stream:
    stream.seek(8)
    stream.write(struct.pack("<3f", 0.0, 0.0, 0.0))
PY
if env "${common_environment[@]}" \
  script/cpu_index_build.sh t2i1m >"${fixture_dir}/corrupt-query.log" 2>&1; then
  echo "Expected a non-normalized cached query to be rejected" >&2
  exit 1
fi
grep -Fq "Cached query row 0 is not unit-normalized" \
  "${fixture_dir}/corrupt-query.log"

python3 - "${fixture_dir}/zero.fbin" <<'PY'
import struct
import sys

with open(sys.argv[1], "wb") as stream:
    stream.write(struct.pack("<ii3f", 1, 3, 0.0, 0.0, 0.0))
PY
if mfnns_hnswlib/build/mfnns_hnsw_tool build \
  --base "${fixture_dir}/zero.fbin" \
  --index "${fixture_dir}/zero.index" \
  --normalize 1 --m 2 --ef-construction 2 \
  --threads 1 --insertion-threads 1 --batch-size 1 \
  >"${fixture_dir}/zero.log" 2>&1; then
  echo "Expected a zero-norm base vector to be rejected" >&2
  exit 1
fi
grep -Fq "Cannot L2-normalize zero-norm or non-finite vector" \
  "${fixture_dir}/zero.log"

echo "CPU_INDEX_BUNDLE_TEST_OK formats=fbin,fvecs normalized_index=6 normalized_queries=6 exact_gt=top1,top2"
