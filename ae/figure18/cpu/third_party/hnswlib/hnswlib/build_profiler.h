#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>

#include "hnswlib.h"

namespace hnswlib {
namespace build_profiler {

enum Stage {
    kOther = 0,
    kUpperGreedy = 1,
    kSearchBaseLayer = 2,
    kHeuristic = 3,
    kLinkMutation = 4,
    kStageCount = 5
};

enum Event {
    kSearchCalls = 0,
    kSearchLayer0Calls = 1,
    kSearchUpperLayerCalls = 2,
    kSearchNodeExpansions = 3,
    kSearchNeighborChecks = 4,
    kSearchVisitedSkips = 5,
    kHeuristicCalls = 6,
    kHeuristicEarlyReturns = 7,
    kHeuristicInputCandidates = 8,
    kHeuristicPairChecks = 9,
    kLinkConnectCalls = 10,
    kLinkSelectedNeighbors = 11,
    kLinkSaturatedRewrites = 12,
    kUpperNodeExpansions = 13,
    kUpperNeighborChecks = 14,
    kUpperHops = 15,
    kAddPointCalls = 16,
    kEventCount = 17
};

enum SearchBreakdown {
    kSbInitialDistance = 0,
    kSbCandidateTop = 1,
    kSbCandidatePop = 2,
    kSbAdjacencyLockRead = 3,
    kSbPrefetch = 4,
    kSbVisitedCheck = 5,
    kSbDataPointer = 6,
    kSbSimdDistance = 7,
    kSbCandidateFilter = 8,
    kSbCandidateQueuePush = 9,
    kSbResultQueueUpdate = 10,
    kSbReleaseVisited = 11,
    kSbCount = 12
};

inline const char *stageName(size_t idx) {
    switch (idx) {
        case kOther: return "other";
        case kUpperGreedy: return "upper_greedy";
        case kSearchBaseLayer: return "search_base_layer";
        case kHeuristic: return "heuristic";
        case kLinkMutation: return "link_mutation";
        default: return "unknown";
    }
}

inline const char *eventName(size_t idx) {
    switch (idx) {
        case kSearchCalls: return "search_calls";
        case kSearchLayer0Calls: return "search_layer0_calls";
        case kSearchUpperLayerCalls: return "search_upper_layer_calls";
        case kSearchNodeExpansions: return "search_node_expansions";
        case kSearchNeighborChecks: return "search_neighbor_checks";
        case kSearchVisitedSkips: return "search_visited_skips";
        case kHeuristicCalls: return "heuristic_calls";
        case kHeuristicEarlyReturns: return "heuristic_early_returns";
        case kHeuristicInputCandidates: return "heuristic_input_candidates";
        case kHeuristicPairChecks: return "heuristic_pair_checks";
        case kLinkConnectCalls: return "link_connect_calls";
        case kLinkSelectedNeighbors: return "link_selected_neighbors";
        case kLinkSaturatedRewrites: return "link_saturated_rewrites";
        case kUpperNodeExpansions: return "upper_node_expansions";
        case kUpperNeighborChecks: return "upper_neighbor_checks";
        case kUpperHops: return "upper_hops";
        case kAddPointCalls: return "add_point_calls";
        default: return "unknown";
    }
}

inline const char *searchBreakdownName(size_t idx) {
    switch (idx) {
        case kSbInitialDistance: return "initial_distance";
        case kSbCandidateTop: return "candidate_top_and_stop_check";
        case kSbCandidatePop: return "candidate_pop";
        case kSbAdjacencyLockRead: return "adjacency_lock_read";
        case kSbPrefetch: return "prefetch";
        case kSbVisitedCheck: return "visited_check";
        case kSbDataPointer: return "data_pointer";
        case kSbSimdDistance: return "simd_distance";
        case kSbCandidateFilter: return "candidate_filter";
        case kSbCandidateQueuePush: return "candidate_queue_push";
        case kSbResultQueueUpdate: return "result_queue_update";
        case kSbReleaseVisited: return "release_visited_list";
        default: return "unknown";
    }
}

#ifdef HNSWLIB_BUILD_PROFILE

inline uint64_t nowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Snapshot {
    uint64_t stage_ns[kStageCount];
    uint64_t distance_calls[kStageCount];
    uint64_t distance_ns[kStageCount];
    uint64_t search_breakdown_ns[kSbCount];
    uint64_t events[kEventCount];
    uint64_t add_point_ns;

    Snapshot() : add_point_ns(0) {
        std::memset(stage_ns, 0, sizeof(stage_ns));
        std::memset(distance_calls, 0, sizeof(distance_calls));
        std::memset(distance_ns, 0, sizeof(distance_ns));
        std::memset(search_breakdown_ns, 0, sizeof(search_breakdown_ns));
        std::memset(events, 0, sizeof(events));
    }
};

struct GlobalCounters {
    std::atomic<uint64_t> stage_ns[kStageCount];
    std::atomic<uint64_t> distance_calls[kStageCount];
    std::atomic<uint64_t> distance_ns[kStageCount];
    std::atomic<uint64_t> search_breakdown_ns[kSbCount];
    std::atomic<uint64_t> events[kEventCount];
    std::atomic<uint64_t> add_point_ns;

    GlobalCounters() : add_point_ns(0) {
        reset();
    }

    void reset() {
        for (size_t i = 0; i < kStageCount; i++) {
            stage_ns[i].store(0, std::memory_order_relaxed);
            distance_calls[i].store(0, std::memory_order_relaxed);
            distance_ns[i].store(0, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < kSbCount; i++) {
            search_breakdown_ns[i].store(0, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < kEventCount; i++) {
            events[i].store(0, std::memory_order_relaxed);
        }
        add_point_ns.store(0, std::memory_order_relaxed);
    }
};

struct ThreadCounters {
    uint64_t stage_ns[kStageCount];
    uint64_t distance_calls[kStageCount];
    uint64_t distance_ns[kStageCount];
    uint64_t search_breakdown_ns[kSbCount];
    uint64_t events[kEventCount];
    uint64_t add_point_ns;
    int current_stage;
    uint64_t current_stage_start_ns;

    ThreadCounters() { reset(); }

    void reset() {
        std::memset(stage_ns, 0, sizeof(stage_ns));
        std::memset(distance_calls, 0, sizeof(distance_calls));
        std::memset(distance_ns, 0, sizeof(distance_ns));
        std::memset(search_breakdown_ns, 0, sizeof(search_breakdown_ns));
        std::memset(events, 0, sizeof(events));
        add_point_ns = 0;
        current_stage = -1;
        current_stage_start_ns = 0;
    }
};

inline GlobalCounters &globals() {
    static GlobalCounters counters;
    return counters;
}

inline ThreadCounters &local() {
    static thread_local ThreadCounters counters;
    return counters;
}

inline void reset() {
    globals().reset();
    local().reset();
}

inline void flushThread() {
    ThreadCounters &t = local();
    GlobalCounters &g = globals();
    for (size_t i = 0; i < kStageCount; i++) {
        g.stage_ns[i].fetch_add(t.stage_ns[i], std::memory_order_relaxed);
        g.distance_calls[i].fetch_add(t.distance_calls[i], std::memory_order_relaxed);
        g.distance_ns[i].fetch_add(t.distance_ns[i], std::memory_order_relaxed);
    }
    for (size_t i = 0; i < kSbCount; i++) {
        g.search_breakdown_ns[i].fetch_add(t.search_breakdown_ns[i], std::memory_order_relaxed);
    }
    for (size_t i = 0; i < kEventCount; i++) {
        g.events[i].fetch_add(t.events[i], std::memory_order_relaxed);
    }
    g.add_point_ns.fetch_add(t.add_point_ns, std::memory_order_relaxed);
    t.reset();
}

inline Snapshot snapshot() {
    Snapshot s;
    GlobalCounters &g = globals();
    for (size_t i = 0; i < kStageCount; i++) {
        s.stage_ns[i] = g.stage_ns[i].load(std::memory_order_relaxed);
        s.distance_calls[i] = g.distance_calls[i].load(std::memory_order_relaxed);
        s.distance_ns[i] = g.distance_ns[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < kSbCount; i++) {
        s.search_breakdown_ns[i] = g.search_breakdown_ns[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < kEventCount; i++) {
        s.events[i] = g.events[i].load(std::memory_order_relaxed);
    }
    s.add_point_ns = g.add_point_ns.load(std::memory_order_relaxed);
    return s;
}

inline void record(Event event, uint64_t value = 1) {
    local().events[event] += value;
}

inline void recordSearchBreakdown(SearchBreakdown bucket, uint64_t value) {
    local().search_breakdown_ns[bucket] += value;
}

class ScopedStage {
 public:
    explicit ScopedStage(Stage stage) : previous_stage_(-1), active_(true) {
        ThreadCounters &t = local();
        const uint64_t now = nowNs();
        previous_stage_ = t.current_stage;
        previous_stage_start_ns_ = t.current_stage_start_ns;
        if (previous_stage_ >= 0) {
            t.stage_ns[previous_stage_] += now - previous_stage_start_ns_;
        }
        t.current_stage = stage;
        t.current_stage_start_ns = now;
    }

    ~ScopedStage() {
        if (!active_) return;
        ThreadCounters &t = local();
        const uint64_t now = nowNs();
        if (t.current_stage >= 0) {
            t.stage_ns[t.current_stage] += now - t.current_stage_start_ns;
        }
        t.current_stage = previous_stage_;
        t.current_stage_start_ns = now;
    }

    ScopedStage(const ScopedStage &) = delete;
    ScopedStage &operator=(const ScopedStage &) = delete;

 private:
    int previous_stage_;
    uint64_t previous_stage_start_ns_;
    bool active_;
};

class ScopedAddPoint {
 public:
    ScopedAddPoint() : start_ns_(nowNs()) {
        record(kAddPointCalls);
    }

    ~ScopedAddPoint() {
        local().add_point_ns += nowNs() - start_ns_;
    }

    ScopedAddPoint(const ScopedAddPoint &) = delete;
    ScopedAddPoint &operator=(const ScopedAddPoint &) = delete;

 private:
    uint64_t start_ns_;
};

template <typename dist_t>
inline dist_t profiledDistance(DISTFUNC<dist_t> func, const void *a, const void *b, void *param) {
    ThreadCounters &t = local();
    const int stage = t.current_stage >= 0 ? t.current_stage : kOther;
    t.distance_calls[stage]++;
#ifdef HNSWLIB_PROFILE_DISTANCE_TIME
    const uint64_t start = nowNs();
    dist_t result = func(a, b, param);
    t.distance_ns[stage] += nowNs() - start;
    return result;
#else
    return func(a, b, param);
#endif
}

#else

struct Snapshot {
    uint64_t stage_ns[kStageCount];
    uint64_t distance_calls[kStageCount];
    uint64_t distance_ns[kStageCount];
    uint64_t search_breakdown_ns[kSbCount];
    uint64_t events[kEventCount];
    uint64_t add_point_ns;

    Snapshot() : add_point_ns(0) {
        std::memset(stage_ns, 0, sizeof(stage_ns));
        std::memset(distance_calls, 0, sizeof(distance_calls));
        std::memset(distance_ns, 0, sizeof(distance_ns));
        std::memset(search_breakdown_ns, 0, sizeof(search_breakdown_ns));
        std::memset(events, 0, sizeof(events));
    }
};

inline void reset() {}
inline void flushThread() {}
inline Snapshot snapshot() { return Snapshot(); }
inline void record(Event, uint64_t = 1) {}
inline void recordSearchBreakdown(SearchBreakdown, uint64_t) {}

class ScopedStage {
 public:
    explicit ScopedStage(Stage) {}
};

class ScopedAddPoint {
 public:
    ScopedAddPoint() {}
};

template <typename dist_t>
inline dist_t profiledDistance(DISTFUNC<dist_t> func, const void *a, const void *b, void *param) {
    return func(a, b, param);
}

#endif

}  // namespace build_profiler
}  // namespace hnswlib
