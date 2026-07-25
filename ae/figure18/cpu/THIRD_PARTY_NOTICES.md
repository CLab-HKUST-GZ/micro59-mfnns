# Third-party notices

The files under `third_party/hnswlib/` are a source snapshot derived from:

```text
project: hnswlib
upstream: https://github.com/nmslib/hnswlib
base commit: d9b3608c83d83b46c96e25088cb1d729b29dcfe9
license: Apache License 2.0
```

The snapshot includes the local `hnswalg.h` profiling changes and
`build_profiler.h` that were present in the author workspace before the
2026-06-15 CPU benchmark binary was compiled. Those changes are retained
because they are part of the actual source dependency used by the measured
Figure 18 CPU run.

See `third_party/hnswlib/LICENSE` for the full license text. File-level hashes
are recorded in `data/SHA256SUMS`.
