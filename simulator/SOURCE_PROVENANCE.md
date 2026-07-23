# Simulator source provenance

This directory is a source snapshot copied on 2026-07-13 from:

`/hpc2hdd/home/rmeng603/workspace/MFANNS/simulator`

The source repository reported Git commit:

`e81c563ba2b6f090210e97c93529645e89b80347`

The requested source tree was not clean. The copied working tree included
uncommitted changes in these simulator files:

- `src/base/anns.h`
- `src/frontend/impl/anns_ndp/emb_unit.cpp`
- `src/frontend/impl/anns_ndp/hnsw.h`
- `src/frontend/impl/anns_ndp/trav_unit.cpp`

Those changes were newer than the historical 2026-03-31 experiment and
amounted to 635 insertions and 86 deletions at copy time. The historical
record did not preserve its source diff or executable hash, so bit-for-bit
reconstruction of the old executable is not possible. This repository uses
the current working-tree source explicitly requested for the AE rerun; the
copied files, not the external repository, are the authoritative snapshot.

Third-party source revisions in the original nested repositories were:

- argparse: `997da9255618311d1fcb0135ce86022729d1f1cb`
- spdlog: `ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36`
- yaml-cpp: `0579ae3d976091d7d664aa9d2527e0d0cff25763`

Nested `.git` directories and all old build outputs were deliberately not
copied. Old CMake caches and runtime search paths refer to the external
MFANNS tree and are not relocatable. `SOURCE_MANIFEST.sha256` records every
file under `src/` and `ext/` in this snapshot.
