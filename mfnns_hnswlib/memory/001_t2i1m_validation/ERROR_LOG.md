# Error and repair log

## 1. ET incomplete-type compilation warnings

Symptom: the original fork performed `dynamic_cast` to
`L2SpaceDynamicPrecisionET` while that type was still incomplete because of a
header cycle. GCC accepted it with warnings; this was not portable.

Repair: introduced `L2EarlyTerminationInterface`, cast to that complete
interface in `hnswalg.h`, and made the ET space implement it. Both GCC 9 and
Clang 10 then compiled the tool.

Evidence: `validation_attempt1.stderr.log`.

## 2. Sanitizer flags replaced required compiler flags

Symptom: overriding `CXXFLAGS` for ThreadSanitizer removed `-std=c++17` and
`-fopenmp`, so `<filesystem>` failed to compile.

Repair: separated overridable `CXXFLAGS` from unconditional
`REQUIRED_CXXFLAGS` in the Makefile.

## 3. OpenMP/ThreadSanitizer synchronization false positive

Symptom: GCC ThreadSanitizer reported a stack race between two consecutive
OpenMP regions. The first region has an implicit barrier, but the installed
`libgomp` lacks the sanitizer synchronization annotations needed to model it.

Repair for diagnosis: disabled normalization in the isolated HNSW insertion
sanitizer run. The original report is preserved in
`tsan_openmp_false_positive.stderr.log`.

## 4. Concurrent `addPoint` lock-order inversion

Symptom: with four concurrent insertions, ThreadSanitizer found a potential
cycle between a per-node link lock and the HNSW global lock.

Risk: historical large MFANNS builds completed, but that does not prove the
cycle can never deadlock.

Repair: the public build path now serializes all HNSW insertion. OpenMP remains
enabled for vector normalization, exact ground truth, and queries. The report
is preserved in `tsan_parallel_add_lock_inversion.stderr.log`.

## 5. Static lock-order warning during serial insertion

Symptom: ThreadSanitizer's deadlock detector records inconsistent lock order
even when both acquisitions occur on the only insertion thread.

Resolution: final race testing uses `detect_deadlocks=0`; this does not hide a
reachable cross-thread insertion deadlock because public insertion is serial.
The intermediate report is preserved in
`tsan_serial_lock_order.stderr.log`.
