# Source provenance

Snapshot date: 2026-07-23

The hnswlib-derived headers were copied from:

```text
/hpc2hdd/home/rmeng603/workspace/MFANNS/recall_analysis/hnswlib
```

The source repository did not contain a commit-tracked historical snapshot for
all of these modified headers. The following hashes identify the exact local
inputs used for this standalone package:

| File | Source SHA-256 |
| --- | --- |
| `hnswlib.h` | `54403db81f55fd28246114c5c9f683e663a2a55ad64903b56578c73e6db5ad2c` |
| `hnswalg.h` | `d6014df0339fb8063315a42a45e3501a9af440b84ee225ca683cdd12485c6094` |
| `visited_list_pool.h` | `8e5bab8d12b5ef26e3b603b656b8a1acba6738498d1d98f44c9fdb1cb0e25110` |
| `space_l2.h` | `c599d024657896412250b4fde1a95e39f8bc55cccd263f36c8948fa35779edf1` |
| `space_ip.h` | `6a89b5275527fe3592a9b02ea4fc50293fdc68cbd9f7c1ec13683581e1ebbaf7` |
| `stop_condition.h` | `930da73eead77dff0c8c9f3281c5d12250051c90ef98fcfbab8f5f307bdae8d9` |
| `bruteforce.h` | `da1f179e99d8e5a4c39b5db0c681bccf3c63e797aaa4982620519f8e001cb2c6` |
| `space_l2_dynamic_precision.h` | `30d81f64ff0d3a457c19009187a8c0cea91c957bb10966768620efb10763e8c5` |
| `space_l2_dynamic_precision_et.h` | `912df7347d8c626c109391e08e58eacb14e945edf7247af6f3529fee9d296751` |

## Standalone repairs

The package intentionally makes these narrow changes:

1. `early_termination_interface.h` breaks the original ET header cycle, so both
   GCC and Clang see a complete dynamic-cast target.
2. The HNSW level/update random generators are protected by mutexes.
3. `L2SpaceDynamicPrecision` initializes members in declaration order.
4. The public build tool serializes `addPoint`. ThreadSanitizer exposed a
   potential link-lock/global-lock inversion in concurrent insertion; keeping
   insertion serial removes that reachable risk and provides deterministic
   output. Normalization, exact ground truth, and query evaluation remain
   parallel.
5. `mfnns_hnsw_tool.cpp` adds strict FBIN validation, streaming construction,
   index-header validation, exact ground truth, and recall-loss reporting.
6. Trailing whitespace was removed mechanically from the copied headers; this
   does not change their semantics.

The package does not claim byte-for-byte identity with an upstream hnswlib
release. It preserves the MFANNS fork's native serialized-index layout.
