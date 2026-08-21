#!/usr/bin/env python3
"""Prepare the exact normalized Figure 14 CAGRA and BANG data contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np


DATASETS = {
    "deep10M": {
        "base": ["Deep/10M/base.10M.fbin", "deep/10M/base.10M.fbin", "deep/1M/base.10M.fbin"],
        "query": ["Deep/query/query.public.10K.fbin", "deep/query/query.public.10K.fbin"],
        "cagra_name": "deep",
        "cagra_query": ("cpp_seeded_query", 100),
        "cagra_normalization": "copy_unit",
        "bang_query": ("split_base", 1000),
    },
    "glove2_2m": {
        "base": ["glove2.2m/glove2.2m_base.fvecs", "glove/glove2.2m/glove2.2m_base.fvecs"],
        "query": ["glove2.2m/glove2.2m_query.fvecs", "glove/glove2.2m/glove2.2m_query.fvecs"],
        "cagra_name": "glove2.2m_norm",
        "cagra_query": ("first_query", 1000),
        "cagra_normalization": "cxx_float",
        "bang_query": ("first_query", 1000),
    },
    "pubmed": {
        "base": ["pubmed/doc_vectors_norm.bin"],
        "query": ["pubmed/query_vectors_norm.bin"],
        "cagra_name": "pubmed_d2v",
        "cagra_query": ("first_query", 100),
        "cagra_normalization": "copy_unit",
        "bang_query": ("mixed_query_base", 1000),
    },
    "sift1M": {
        "base": ["sift/1M/sift/sift_base.fvecs"],
        "query": ["sift/1M/sift/sift_query.fvecs"],
        "cagra_name": "sift1m",
        "cagra_query": ("first_query", 1000),
        "cagra_normalization": "numpy",
        "bang_query": ("first_query", 1000),
    },
    "text2img1M": {
        "base": ["text2img/1M/base.1M.fbin", "t2i/1M/base.1M.fbin"],
        "query": ["text2img/query/query.public.100K.fbin", "t2i/query/query.public.100K.fbin"],
        "cagra_name": "text2img",
        "cagra_query": ("cpp_seeded_query", 100),
        "cagra_normalization": "cxx_float",
        "bang_query": ("split_base", 1000),
    },
    "wiki1M": {
        "base": ["wiki/base.1M.fbin", "wiki/wiki1m/base.1M.fbin"],
        "query": ["wiki/queries.fbin", "wiki/wiki1m/queries.fbin"],
        "cagra_name": "wiki1m",
        "cagra_query": ("cpp_seeded_query", 100),
        "cagra_normalization": "cxx_float",
        "bang_query": ("split_base", 1000),
    },
    "word2vec": {
        "base": ["word2vec/word2vec_base.fvecs", "w2v/word2vec/word2vec_base.fvecs"],
        "query": ["word2vec/word2vec_query.fvecs", "w2v/word2vec/word2vec_query.fvecs"],
        "cagra_name": "word2vec",
        "cagra_query": ("first_query", 1000),
        "cagra_normalization": "cxx_float",
        "bang_query": ("first_query", 1000),
    },
}


class VectorFile:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        suffix = self.path.suffix.lower()
        self.format = "fvecs" if suffix == ".fvecs" else "fbin"
        if self.format == "fbin":
            with self.path.open("rb") as handle:
                header = handle.read(8)
            if len(header) != 8:
                raise ValueError(f"short FBIN header: {self.path}")
            self.rows, self.dim = struct.unpack("<II", header)
            expected = 8 + self.rows * self.dim * 4
            if self.path.stat().st_size != expected:
                raise ValueError(
                    f"FBIN size mismatch for {self.path}: expected {expected}, "
                    f"got {self.path.stat().st_size}"
                )
            self._vectors = np.memmap(
                self.path,
                dtype="<f4",
                mode="r",
                offset=8,
                shape=(self.rows, self.dim),
            )
        else:
            first = np.fromfile(self.path, dtype="<i4", count=1)
            if first.size != 1 or int(first[0]) <= 0:
                raise ValueError(f"invalid FVECS dimension: {self.path}")
            self.dim = int(first[0])
            row_bytes = (self.dim + 1) * 4
            if self.path.stat().st_size % row_bytes:
                raise ValueError(f"truncated FVECS payload: {self.path}")
            self.rows = self.path.stat().st_size // row_bytes
            words = np.memmap(
                self.path,
                dtype="<f4",
                mode="r",
                shape=(self.rows, self.dim + 1),
            )
            dimensions = words[:, 0].view("<i4")
            if not np.all(dimensions[: min(self.rows, 10000)] == self.dim):
                raise ValueError(f"inconsistent FVECS row dimensions: {self.path}")
            self._vectors = words[:, 1:]

    def rows_at(self, indices) -> np.ndarray:
        return np.asarray(self._vectors[indices], dtype=np.float32)

    def slice(self, start: int, end: int) -> np.ndarray:
        return np.asarray(self._vectors[start:end], dtype=np.float32)


def normalize(vectors: np.ndarray, mode: str) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if mode == "copy_unit":
        output = np.ascontiguousarray(vectors)
    elif mode == "numpy":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        output = np.ascontiguousarray(
            vectors / np.maximum(norms, np.float32(1e-12))
        )
    elif mode == "cxx_float":
        # Match the historical C++ loop: float accumulation in dimension order.
        squared = np.zeros(vectors.shape[0], dtype=np.float32)
        for column in range(vectors.shape[1]):
            squared += vectors[:, column] * vectors[:, column]
        norms = np.sqrt(squared).reshape(-1, 1)
        output = np.ascontiguousarray(
            vectors / np.maximum(norms, np.float32(1e-12))
        )
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    sample_norms = np.linalg.norm(output[: min(len(output), 10000)], axis=1)
    if not (float(sample_norms.min()) >= 0.999 and float(sample_norms.max()) <= 1.001):
        raise ValueError(
            f"{mode} output is not unit-normalized: "
            f"range={sample_norms.min()}..{sample_norms.max()}"
        )
    return output


def cpp_mt19937_shuffle_indices(size: int, count: int, seed: int) -> np.ndarray:
    """Match libstdc++ std::shuffle(std::mt19937(seed)) used by CAGRA."""
    generator = np.random.RandomState(seed)

    def raw() -> int:
        return int(generator.randint(0, 2**32, dtype=np.uint32))

    def uniform(exclusive: int) -> int:
        if exclusive <= 0 or exclusive > 2**32:
            raise ValueError(f"unsupported uniform range: {exclusive}")
        product = raw() * exclusive
        low = product & 0xFFFFFFFF
        if low < exclusive:
            threshold = ((-exclusive) & 0xFFFFFFFF) % exclusive
            while low < threshold:
                product = raw() * exclusive
                low = product & 0xFFFFFFFF
        return product >> 32

    values = np.arange(size, dtype=np.int64)
    if 0xFFFFFFFF // size >= size:
        position = 1
        if size % 2 == 0:
            other = uniform(2)
            values[position], values[other] = values[other], values[position]
            position += 1
        while position != size:
            first_range = position + 1
            combined = uniform(first_range * (first_range + 1))
            first = combined // (first_range + 1)
            second = combined % (first_range + 1)
            values[position], values[first] = values[first], values[position]
            position += 1
            values[position], values[second] = values[second], values[position]
            position += 1
    else:
        for position in range(1, size):
            other = uniform(position + 1)
            values[position], values[other] = values[other], values[position]
    return np.sort(values[:count])


def resolve_candidate(root: Path, candidates: list[str], kind: str) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path.resolve()
    formatted = "\n  ".join(str(root / item) for item in candidates)
    raise FileNotFoundError(f"could not resolve {kind}; tried:\n  {formatted}")


def select_policy(
    profile: str,
    config: dict,
    base: VectorFile,
    query: VectorFile,
    seed: int,
    normalization_mode: str,
):
    mode, count = config[f"{profile}_query"]
    rng = np.random.default_rng(seed)
    excluded = np.zeros(base.rows, dtype=np.bool_)
    selected_base = np.empty(0, dtype=np.int64)

    if mode == "cpp_seeded_query":
        if query.rows < count:
            raise ValueError(f"query source has {query.rows} rows; need {count}")
        query_indices = cpp_mt19937_shuffle_indices(query.rows, count, seed)
        query_vectors = normalize(query.rows_at(query_indices), normalization_mode)
    elif mode == "first_query":
        if query.rows < count:
            raise ValueError(f"query source has {query.rows} rows; need {count}")
        query_indices = np.arange(count, dtype=np.int64)
        query_vectors = normalize(query.rows_at(query_indices), normalization_mode)
    elif mode == "split_base":
        if base.rows <= count:
            raise ValueError(f"base source has {base.rows} rows; need more than {count}")
        selected_base = np.sort(rng.choice(base.rows, size=count, replace=False))
        excluded[selected_base] = True
        query_indices = selected_base.copy()
        query_vectors = normalize(base.rows_at(selected_base), normalization_mode)
    elif mode == "mixed_query_base":
        if query.rows >= count:
            raise ValueError(
                "mixed_query_base is only valid when the query file needs base rows"
            )
        fill = count - query.rows
        selected_base = np.sort(rng.choice(base.rows, size=fill, replace=False))
        excluded[selected_base] = True
        query_indices = np.concatenate(
            [
                np.arange(query.rows, dtype=np.int64),
                selected_base,
            ]
        )
        query_vectors = np.concatenate(
            [
                normalize(query.slice(0, query.rows), normalization_mode),
                normalize(base.rows_at(selected_base), normalization_mode),
            ]
        )
    else:
        raise AssertionError(mode)
    return mode, query_vectors, query_indices, selected_base, excluded


def atomic_fbin(path: Path, vectors: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        handle.write(struct.pack("<II", vectors.shape[0], vectors.shape[1]))
        handle.write(np.asarray(vectors, dtype="<f4", order="C").tobytes(order="C"))
    os.replace(temp, path)


def write_normalized_base(
    source: VectorFile,
    output: Path,
    excluded: np.ndarray,
    chunk_rows: int,
    normalization_mode: str,
) -> int:
    output_rows = source.rows - int(excluded.sum())
    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        handle.write(struct.pack("<II", output_rows, source.dim))
        written = 0
        for start in range(0, source.rows, chunk_rows):
            end = min(source.rows, start + chunk_rows)
            block = source.slice(start, end)
            if excluded.any():
                block = block[~excluded[start:end]]
            if block.size:
                normalized = normalize(block, normalization_mode)
                handle.write(normalized.astype("<f4", copy=False).tobytes(order="C"))
                written += normalized.shape[0]
            print(f"BASE_PROGRESS={end}/{source.rows}", flush=True)
    if written != output_rows:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"base row count mismatch: expected {output_rows}, wrote {written}")
    os.replace(temp, output)
    return output_rows


def read_fbin(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        rows, dim = struct.unpack("<II", handle.read(8))
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def merge_topk(
    old_scores: np.ndarray,
    old_ids: np.ndarray,
    new_scores: np.ndarray,
    new_ids: np.ndarray,
    k: int,
):
    scores = np.concatenate([old_scores, new_scores], axis=1)
    ids = np.concatenate([old_ids, new_ids], axis=1)
    chosen = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(scores, chosen, axis=1)
    ids = np.take_along_axis(ids, chosen, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return (
        np.take_along_axis(scores, order, axis=1),
        np.take_along_axis(ids, order, axis=1),
    )


def exact_gt(
    base_path: Path,
    queries: np.ndarray,
    k: int,
    backend: str,
    gpu: int,
    chunk_rows: int,
    query_batch_rows: int,
):
    base = read_fbin(base_path)
    if base.shape[0] < k:
        raise ValueError(f"base has {base.shape[0]} rows, fewer than GT k={k}")
    use_cupy = backend in {"auto", "cupy"}
    cp = None
    if use_cupy:
        try:
            import cupy as cp_module

            cp = cp_module
            cp.cuda.Device(gpu).use()
        except Exception:
            if backend == "cupy":
                raise
            cp = None
    actual_backend = "cupy" if cp is not None else "numpy"
    print(f"GT_BACKEND={actual_backend}", flush=True)

    nq = queries.shape[0]
    if chunk_rows <= 0:
        target_base_bytes = 512 * 1024 * 1024
        target_scores_bytes = 256 * 1024 * 1024
        by_base = max(1, target_base_bytes // (base.shape[1] * 4))
        by_scores = max(1, target_scores_bytes // (query_batch_rows * 4))
        chunk_rows = int(max(1024, min(by_base, by_scores)))

    # Keep the CuPy path byte-compatible with the historical Figure 14 GT
    # generator. In particular, the partition direction and all intermediate
    # merges stay on the GPU; changing either can alter IDs at float32 ties.
    if cp is not None:
        labels = np.empty((nq, k), dtype=np.uint32)
        all_scores = np.empty((nq, k), dtype=np.float32)
        for q_start in range(0, nq, query_batch_rows):
            q_end = min(nq, q_start + query_batch_rows)
            q_device = cp.asarray(queries[q_start:q_end])
            top_scores = cp.full((q_end - q_start, k), -cp.inf, dtype=cp.float32)
            top_ids = cp.full((q_end - q_start, k), -1, dtype=cp.int32)
            for start in range(0, base.shape[0], chunk_rows):
                end = min(base.shape[0], start + chunk_rows)
                block_device = cp.asarray(base[start:end])
                scores = q_device @ block_device.T
                width = min(k, end - start)
                if end - start > k:
                    local = cp.argpartition(scores, -k, axis=1)[:, -k:]
                    local_scores = cp.take_along_axis(scores, local, axis=1)
                else:
                    local = cp.arange(width, dtype=cp.int32)[cp.newaxis, :]
                    local = cp.broadcast_to(local, scores.shape)
                    local_scores = scores
                local = local.astype(cp.int32) + start
                merged_scores = cp.concatenate([top_scores, local_scores], axis=1)
                merged_ids = cp.concatenate([top_ids, local], axis=1)
                selected = cp.argpartition(merged_scores, -k, axis=1)[:, -k:]
                top_scores = cp.take_along_axis(merged_scores, selected, axis=1)
                top_ids = cp.take_along_axis(merged_ids, selected, axis=1)
            order = cp.argsort(top_scores, axis=1)[:, ::-1]
            labels[q_start:q_end] = cp.asnumpy(
                cp.take_along_axis(top_ids, order, axis=1)
            ).astype(np.uint32, copy=False)
            all_scores[q_start:q_end] = cp.asnumpy(
                cp.take_along_axis(top_scores, order, axis=1)
            )
            cp.cuda.Device().synchronize()
            print(f"GT_PROGRESS={q_end}/{nq}", flush=True)
        distances = np.maximum(0.0, 2.0 - 2.0 * all_scores).astype(np.float32)
        return labels, distances, actual_backend

    best_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    best_ids = np.full((nq, k), np.iinfo(np.int64).max, dtype=np.int64)
    for start in range(0, base.shape[0], chunk_rows):
        end = min(base.shape[0], start + chunk_rows)
        block = np.asarray(base[start:end], dtype=np.float32)
        for q_start in range(0, nq, query_batch_rows):
            q_end = min(nq, q_start + query_batch_rows)
            scores = queries[q_start:q_end] @ block.T
            width = min(k, block.shape[0])
            local = np.argpartition(-scores, kth=width - 1, axis=1)[:, :width]
            local_scores = np.take_along_axis(scores, local, axis=1)
            local_ids = local.astype(np.int64) + start
            merged = merge_topk(
                best_scores[q_start:q_end],
                best_ids[q_start:q_end],
                local_scores,
                local_ids,
                k,
            )
            best_scores[q_start:q_end], best_ids[q_start:q_end] = merged
        print(f"GT_PROGRESS={end}/{base.shape[0]}", flush=True)
    distances = np.maximum(0.0, 2.0 - 2.0 * best_scores).astype(np.float32)
    return best_ids.astype(np.uint32), distances, actual_backend


def atomic_ibin(path: Path, labels: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        handle.write(struct.pack("<II", labels.shape[0], labels.shape[1]))
        handle.write(labels.astype("<u4", copy=False).tobytes(order="C"))
    os.replace(temp, path)


def atomic_bang_gt(path: Path, labels: np.ndarray, distances: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        handle.write(struct.pack("<II", labels.shape[0], labels.shape[1]))
        handle.write(labels.astype("<u4", copy=False).tobytes(order="C"))
        handle.write(distances.astype("<f4", copy=False).tobytes(order="C"))
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_one(args, dataset: str) -> None:
    config = DATASETS[dataset]
    raw_root = args.raw_root.expanduser().resolve()
    base_path = resolve_candidate(raw_root, config["base"], f"{dataset} base")
    query_path = resolve_candidate(raw_root, config["query"], f"{dataset} query")
    historical_compatible = True
    if args.profile == "cagra" and dataset == "pubmed":
        if args.pubmed_cagra_base is not None:
            base_path = args.pubmed_cagra_base.expanduser().resolve()
        elif not args.allow_pubmed_500k:
            raise ValueError(
                "historical CAGRA Pubmed uses a 1,000,000-row base, while the "
                "public raw dataset has 500,000 rows; pass --pubmed-cagra-base "
                "or explicitly acknowledge the variant with --allow-pubmed-500k"
            )
        else:
            historical_compatible = False

    base = VectorFile(base_path)
    query = VectorFile(query_path)
    if base.dim != query.dim:
        raise ValueError(f"base/query dimension mismatch: {base.dim} vs {query.dim}")
    normalization_mode = (
        config["cagra_normalization"] if args.profile == "cagra" else "numpy"
    )
    mode, query_vectors, query_indices, selected_base, excluded = select_policy(
        args.profile, config, base, query, args.seed, normalization_mode
    )
    output_name = config["cagra_name"] if args.profile == "cagra" else dataset
    output = args.output_root.expanduser().resolve() / args.profile / output_name
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "base": output / "base.fbin",
        "query": output / "query.fbin",
        "gt100": output / "gt_top100.ibin",
        "metadata": output / "prepare.json",
        "indices": output / "query_indices.i64",
    }
    if args.profile == "cagra":
        paths["gt32"] = output / "gt_top32.ibin"
    else:
        paths["bang_gt"] = output / "gt_top100.bang.bin"

    expected = [paths["base"], paths["query"], paths["metadata"], paths["indices"]]
    if not args.skip_gt:
        expected += [paths["gt100"]]
        if args.profile == "cagra":
            expected += [paths["gt32"]]
        else:
            expected += [paths["bang_gt"]]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected) and not args.force:
        print(f"DATASET={dataset} STATUS=CACHE_HIT OUTPUT={output}")
        return
    if not args.force and any(path.exists() for path in expected):
        raise FileExistsError(f"incomplete output exists under {output}; inspect it or pass --force")

    started = time.time()
    output_rows = write_normalized_base(
        base, paths["base"], excluded, args.chunk_rows, normalization_mode
    )
    atomic_fbin(paths["query"], query_vectors)
    query_indices.astype("<i8", copy=False).tofile(paths["indices"])
    gt_backend = None
    if not args.skip_gt:
        labels, distances, gt_backend = exact_gt(
            paths["base"],
            query_vectors,
            args.gt_topk,
            args.gt_backend,
            args.gpu,
            args.gt_base_chunk_rows,
            args.gt_query_batch_rows,
        )
        atomic_ibin(paths["gt100"], labels)
        if args.profile == "cagra":
            atomic_ibin(paths["gt32"], labels[:, : min(32, labels.shape[1])])
        else:
            # BANG evaluates recall from IDs only. The historical converter
            # wrote rank placeholders in the distance section; preserve that
            # layout so regenerated input is byte-compatible.
            rank_placeholders = np.broadcast_to(
                np.arange(labels.shape[1], dtype=np.float32), labels.shape
            ).copy()
            atomic_bang_gt(paths["bang_gt"], labels, rank_placeholders)

    payload = {
        "format": "figure14-gpu-normalized-v1",
        "profile": args.profile,
        "dataset": dataset,
        "output_dataset": output_name,
        "historical_compatible": historical_compatible,
        "seed": args.seed,
        "query_policy": mode,
        "raw_base": str(base_path),
        "raw_base_format": base.format,
        "raw_base_rows": base.rows,
        "raw_query": str(query_path),
        "raw_query_format": query.format,
        "raw_query_rows": query.rows,
        "dimension": base.dim,
        "prepared_base_rows": output_rows,
        "prepared_query_rows": query_vectors.shape[0],
        "selected_base_query_rows": int(selected_base.size),
        "normalization": normalization_mode,
        "gt_metric": None if args.skip_gt else "normalized squared L2 (inner-product equivalent)",
        "gt_topk": None if args.skip_gt else args.gt_topk,
        "gt_backend": gt_backend,
        "gt_base_chunk_rows": None if args.skip_gt else args.gt_base_chunk_rows,
        "gt_query_batch_rows": None if args.skip_gt else args.gt_query_batch_rows,
        "base_sha256": sha256_file(paths["base"]) if args.checksum else None,
        "query_sha256": sha256_file(paths["query"]) if args.checksum else None,
        "query_indices_sha256": (
            sha256_file(paths["indices"]) if args.checksum else None
        ),
        "gt_top100_sha256": (
            sha256_file(paths["gt100"])
            if args.checksum and not args.skip_gt
            else None
        ),
        "gt_top32_sha256": (
            sha256_file(paths["gt32"])
            if args.checksum and not args.skip_gt and args.profile == "cagra"
            else None
        ),
        "bang_gt_sha256": (
            sha256_file(paths["bang_gt"])
            if args.checksum and not args.skip_gt and args.profile == "bang"
            else None
        ),
        "elapsed_seconds": time.time() - started,
    }
    temp_metadata = paths["metadata"].with_name(
        f".{paths['metadata'].name}.tmp-{os.getpid()}"
    )
    temp_metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_metadata, paths["metadata"])
    print(
        f"DATASET={dataset} STATUS=BUILT PROFILE={args.profile} "
        f"BASE_ROWS={output_rows} QUERY_ROWS={query_vectors.shape[0]} OUTPUT={output}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cagra", "bang"), required=True)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS.keys()),
        default="all",
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pubmed-cagra-base", type=Path)
    parser.add_argument("--allow-pubmed-500k", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=100000)
    parser.add_argument("--gt-topk", type=int, default=100)
    parser.add_argument("--gt-backend", choices=("auto", "cupy", "numpy"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--gt-base-chunk-rows",
        type=int,
        default=0,
        help="0 uses the historical 512 MiB base / 256 MiB score heuristic",
    )
    parser.add_argument("--gt-query-batch-rows", type=int, default=1000)
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in (
        "chunk_rows",
        "gt_topk",
        "gt_query_batch_rows",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gt_base_chunk_rows < 0:
        parser.error("--gt-base-chunk-rows must be non-negative")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepare_one(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
