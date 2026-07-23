#ifndef     HNSW_ACCEL_H
#define     HNSW_ACCEL_H

#include "frontend/frontend.h"
#include "memory_system/impl/multi_generic_DRAM_system.cpp"
#include "base/anns.h"
#include "cache/hardware_cache.h"
#include "hnswlib/fp16_fpma.h"
#include <list>
#include <deque>
#include <unordered_map>
#include <array>
#include <limits>

namespace Ramulator {

// Per-rank LRU cache for vector data (caches m vectors by candId)
class VectorLRUCache {
public:
    uint64_t s_cache_hit = 0;
    uint64_t s_cache_miss = 0;
    uint64_t s_cache_access = 0;

    VectorLRUCache() : m_capacity(0) {}

    void init(size_t capacity) {
        m_capacity = capacity;
    }

    bool contains(PointId candId) {
        if (m_capacity == 0) return false;  // cache disabled

        s_cache_access++;
        auto it = m_cache_map.find(candId);
        if (it != m_cache_map.end()) {
            // Cache hit: move to front (MRU)
            s_cache_hit++;
            m_lru_list.erase(it->second);
            m_lru_list.push_front(candId);
            m_cache_map[candId] = m_lru_list.begin();
            return true;
        }
        s_cache_miss++;
        return false;
    }

    void fill(PointId candId) {
        if (m_capacity == 0) return;
        auto it = m_cache_map.find(candId);
        if (it != m_cache_map.end()) {
            m_lru_list.erase(it->second);
            m_lru_list.push_front(candId);
            m_cache_map[candId] = m_lru_list.begin();
            return;
        }
        if (m_lru_list.size() >= m_capacity) {
            PointId evict_id = m_lru_list.back();
            m_lru_list.pop_back();
            m_cache_map.erase(evict_id);
        }
        m_lru_list.push_front(candId);
        m_cache_map[candId] = m_lru_list.begin();
    }

    // Backward-compatible access+fill-on-miss path.
    bool access(PointId candId) {
        if (contains(candId)) {
            return true;
        }
        fill(candId);
        return false;
    }

    size_t size() const { return m_lru_list.size(); }
    size_t capacity() const { return m_capacity; }

private:
    size_t m_capacity;
    std::list<PointId> m_lru_list;  // front = MRU, back = LRU
    std::unordered_map<PointId, std::list<PointId>::iterator> m_cache_map;
};

#define print(args...) \
{ \
    fprintf(stdout, args); \
    fprintf(stdout, "\n"); \
    fflush(stdout); \
}

struct MaxHeapByFirst {
    bool operator()(const PointDistId& a, const PointDistId& b) const {
        return a.first < b.first; // max-heap
    }
};

struct MinHeapByFirst {
    bool operator()(const PointDistId& a, const PointDistId& b) const {
        return a.first < b.first; // min-heap
    }
};

template <class T>
int read_vec(std::string name, std::vector<T> &data, uint32_t &rows, uint32_t &cols) {
    // Open the binary file for reading
    std::ifstream file(name, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << name << std::endl;
        return 1;
    }

    // Read header: nvecs (rows) and dim (cols)
    int32_t read_rows, read_cols;
    file.read(reinterpret_cast<char*>(&read_rows), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&read_cols), sizeof(int32_t));

    if (read_cols != cols) {
        print("Input cols %d and read_col %d not match!", cols, read_cols);
        std::cerr << "Error read in file: " << name << std::endl;
        fflush(stdout);
    }
    assert(read_cols == cols);

    rows = read_rows;

    // Read all data continuously
    data.resize(rows * cols);
    file.read(reinterpret_cast<char*>(data.data()), rows * cols * sizeof(T));

    if (!file) {
        std::cerr << "Failed to read vector data from file: " << name << std::endl;
        file.close();
        return 1;
    }

    // Close the file
    file.close();

    return 0;
}


using ANNSSearchQueue = std::priority_queue<PointDistId, std::vector<PointDistId>, MinHeapByFirst>;
// using ANNSResultQueue = std::priority_queue<PointDistId, std::vector<PointDistId>, MaxHeapByFirst>;
class ANNSResultQueue : public std::priority_queue<PointDistId, std::vector<PointDistId>, MaxHeapByFirst> {
public:
    std::vector<PointDistId> c;
};

struct Query {
    struct BufferedCandidateResult {
        PointId candId = 0;
        Type pDistance = 0;
        Type candidateLowerBound = 0;
        bool dualQueuePruned = false;
        uint64_t num_access = 0;
        uint64_t batchOrder = 0;
    };

    struct DuplicateAcceptDebugEvent {
        uint64_t acceptSeq = 0;
        PointId candId = 0;
        Type pDistance = 0;
        Type upperbound = 0;
        uint32_t level = 0;
        uint32_t resultSizeBefore = 0;
        uint32_t searchSizeBefore = 0;
        uint32_t inflightSizeBefore = 0;
        uint32_t acceptCountAfter = 0;
    };

    struct CandidateIssueDebugStat {
        uint32_t level = 0;
        PointId candId = 0;
        uint64_t seenCount = 0;
        uint64_t baseVisitSkipCount = 0;
        uint64_t baseVisitMarkCount = 0;
        uint64_t candidateIssueCount = 0;
        uint64_t disreqIssueCount = 0;
        PointId firstParentCandId = 0;
        PointId lastParentCandId = 0;
        bool hasParent = false;
    };

    struct CandidateIssueDebugEvent {
        uint64_t seq = 0;
        std::string kind;
        uint32_t level = 0;
        PointId parentCandId = 0;
        PointId candId = 0;
        bool baseVisitBefore = false;
        uint64_t seenCountAfter = 0;
        uint64_t candidateIssueCountAfter = 0;
        uint64_t disreqIssueCountAfter = 0;
        uint32_t embUnitId = 0;
        uint32_t vDimBase = 0;
        Type upperbound = 0;
    };

    uint64_t annsId;
    std::string type;
    uint32_t level;
    ANNSSearchQueue search;
    ANNSResultQueue result;
    ANNSResultQueue finalResult;
    Point query;
    Type querySquaredNorm = 0;
    std::unordered_set<PointId> inflightCands;
    std::unordered_map<PointId, std::unordered_set<uint64_t>> inflightDisReqs;
    std::unordered_map<PointId, Type> pDistance;
    ANNSResultQueue lowerBoundResult;
    std::unordered_map<PointId, Type> candidateLowerBounds;
    std::vector<BufferedCandidateResult> bufferedLevel0DualQueueResults;
    std::unordered_map<PointId, uint64_t> level0DualQueueBatchOrder;
    uint64_t nextLevel0DualQueueBatchOrder = 0;

    // for construction
    uint32_t consLevel;
    PointId vectorId;

    // latency stats
    uint64_t startCycle = 0;
    uint64_t indexCycles = 0;
    uint64_t disCompCycles = 0;
    uint64_t disCompAccCycles = 0;
    uint64_t disCompRejCycles = 0;
    uint64_t disOffloadCycles = 0;
    uint64_t disGatherCycles = 0;
    uint64_t perQueryTravNodeReq = 0;
    uint64_t perQueryTravEdgeReq = 0;
    uint64_t perQueryDisreqCompleted = 0;
    uint64_t perQueryLevel0BaseVisitMark = 0;
    uint64_t perQueryLevel0BaseVisitSkip = 0;
    uint64_t perQueryCandidateIssue = 0;
    uint64_t perQueryRecallHits = 0;
    uint64_t perQueryRecallResultCount = 0;
    uint64_t perQueryRecallCompareK = 0;
    uint64_t perQueryLatencyCycles = 0;
    bool sameRankRunValid = false;
    uint32_t sameRankLast = 0;
    uint32_t sameRankCurrentLength = 0;

    bool topChanged;
    bool* baseVisitArray;
    bool debugDuplicateAcceptActive = false;
    bool debugDuplicateAcceptDumped = false;
    uint64_t debugDuplicateAcceptSeq = 0;
    uint64_t debugDuplicateAcceptTotal = 0;
    uint64_t debugDuplicateAcceptDuplicateEvents = 0;
    std::unordered_map<uint64_t, uint32_t> debugAcceptedCountsByLevelKey;
    std::vector<DuplicateAcceptDebugEvent> debugDuplicateAcceptEvents;
    bool debugIssueTraceActive = false;
    bool debugIssueTraceDumped = false;
    uint64_t debugIssueTraceSeq = 0;
    uint64_t debugIssueTraceSeenTotal = 0;
    uint64_t debugIssueTraceBaseVisitSkipTotal = 0;
    uint64_t debugIssueTraceBaseVisitMarkTotal = 0;
    uint64_t debugIssueTraceCandidateIssueTotal = 0;
    uint64_t debugIssueTraceDisreqIssueTotal = 0;
    std::unordered_map<uint64_t, CandidateIssueDebugStat> debugIssueStatsByLevelKey;
    std::vector<CandidateIssueDebugEvent> debugIssueEvents;

    Query(uint64_t _annsId, std::string _type, Point& _query, PointId _epId, uint32_t _level):
        annsId(_annsId), type(_type), query(_query), level(_level) {
        assert(type == "search");
        for (const Type value : query) {
            querySquaredNorm += value * value;
        }
        search.push(std::make_pair(-1U, _epId));
        result.push(std::make_pair(-1U, _epId));
        baseVisitArray = nullptr;
    }

    Query(uint64_t _annsId, std::string _type, Point& _query):
        annsId(_annsId), type(_type), query(_query) {
        assert(type == "construct");
        for (const Type value : query) {
            querySquaredNorm += value * value;
        }
        baseVisitArray = nullptr;
    }

    ~Query() {
        if (baseVisitArray) delete[] baseVisitArray;
    }
};

class HNSWTraversalUnit;

class HNSWEmbUnit : public IFrontEnd, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, HNSWEmbUnit, "HNSWEmbUnit", "HNSWEmbUnit")
private:
    friend class HNSWTraversalUnit;
    uint32_t m_id;
    std::string name;
    Clk_t m_clk;
    Logger_t m_logger;
    std::function<void(Request&)> pdis_callback;
    HardwareCache g_cache;

    struct PendingPDisReqState {
        PDisReq req;
        uint32_t remainingCallbacks = 0;
        uint64_t sendCycle = 0;
        uint64_t latestReturnCycle = 0;
    };

    struct PredictedBGTgt {
        uint32_t bankgroup = 0;
        uint32_t bank = 0;
        uint64_t row = 0;
    };

public:
    struct ReplicaDispatchProbe {
        uint32_t inflightLoad = 0;
        uint32_t lineCount = 0;
        uint32_t sumBgOccupancy = 0;
        uint32_t sumBankOccupancy = 0;
        uint32_t rowHitPred = 0;
        uint32_t bgRowHitPred = 0;
    };

private:

    struct PendingPDisReqSelection {
        std::vector<uint64_t> uniqueRowKeys;
        uint32_t phase = 0;
        uint32_t lineCount = 0;
        uint32_t maxOccupancy = 0;
        uint32_t sumOccupancy = 0;
        uint32_t maxBankOccupancy = 0;
        uint32_t sumBankOccupancy = 0;
        uint32_t minCredit = 0;
        uint32_t minBankCredit = 0;
        uint32_t rowHitPred = 0;
        uint32_t rowBurstPred = 0;
        uint32_t congestedBGs = 0;
        uint32_t maxBankToken = 0;
        uint32_t maxBGToken = 0;
        uint32_t throttledBanks = 0;
        uint32_t throttledBGs = 0;
        int64_t thresholdGainScaled = 0;
        bool zeroMem = false;
        bool fineDeferredByCongestion = false;
        bool thresholdQueueFull = false;
        bool ageForced = false;
        uint64_t enqueueAge = 0;
    };

    // parameters
    // uint32_t nCompUnit;
    uint32_t nParallelDisReq = 0;
    uint32_t nParallelPDisReq = 0;
    uint32_t nDim = 0, nBit = 0;
    uint32_t dimSize = 0;
    uint32_t configuredVectorDataBitWidth = 0;
    uint32_t nativeDataBitWidth = 0;
    uint32_t modeledReadBitWidth = 0;
    uint32_t dimStep = 0, bitStep = 0;
    uint32_t dimStep1 = 0, bitStep1 = 0;
    uint32_t dimStep2 = 0, bitStep2 = 0;
    std::string dimbitMode;
    std::string disMethod;
    hnswlib::FP16L2SquareMethod fp16L2SquareMethod = hnswlib::FP16L2SquareMethod::Standard;
    bool earlyExitEnable = false;
    bool simulateFp16DataRead = false;
    bool ansmetRuntimeTrueFp16BitChopEnable = false;
    bool enableLogging = false;

    bool dualQueueBGAwareScheduleEnable = false;
    bool dualQueueFineBankRowAwareScheduleEnable = false;
    bool dualQueueCrossLevelNMPEnable = false;
    std::string dualQueueCoarseNMPLevel = "rank";
    std::string dualQueueFineNMPLevel = "rank";
    uint32_t dualQueueFineSubarrayWays = 1;
    uint32_t dualQueueFineSubarrayCount = 256;
    uint32_t dualQueueFineSubarrayInterleaveRows = 1;
    uint32_t dualQueueBGAwareCreditLimit = 32;
    uint32_t dualQueueBGAwareCongestionThreshold = 24;
    uint32_t dualQueueBGAwareBalanceWeight = 4;
    uint32_t dualQueueBGAwareLocalityWeight = 8;
    uint32_t dualQueueBGAwareFineDeferPenalty = 128;
    uint32_t dualQueueFineBankCreditLimit = 16;
    uint32_t dualQueueFineBankTokenLimit = 8;
    uint32_t dualQueueFineBGTokenLimit = 16;
    uint32_t dualQueueFineBalanceWeight = 4;
    uint32_t dualQueueFineLocalityWeight = 16;
    uint32_t dualQueueFineThresholdGainWeight = 16;
    uint32_t dualQueueFineAgeWeight = 256;
    uint32_t dualQueueFineAgeThreshold = 256;
    uint32_t dualQueueFineTokenPenalty = 64;
    uint32_t dualQueueFineRowBurstCap = 4;

    // Batch dispatch + row-sweep scheduling parameters
    uint32_t maxPDisReqDispatchPerCycle = 1;   // Max PDisReqs dispatched per tick (1 = legacy)
    bool rowSweepEnable = false;               // Enable row-sweep coalescing after best dispatch
    uint32_t maxRowSweepBatch = 4;             // Max additional same-row requests per sweep

    bool dualQueueStagedAdmissionEnable = false;
    uint32_t dualQueueCoarseAdmissionWindow = 0;
    uint32_t dualQueueFineAdmissionWindow = 0;
    static constexpr uint32_t kBGAwareNumBankGroups = 4;
    static constexpr uint32_t kBGAwareNumBanksPerGroup = 4;
    static constexpr uint32_t kBGAwareNumBanks = kBGAwareNumBankGroups * kBGAwareNumBanksPerGroup;
    std::array<uint32_t, kBGAwareNumBankGroups> m_bgOutstanding = {0, 0, 0, 0};
    std::array<uint32_t, kBGAwareNumBankGroups> m_bgCredits = {0, 0, 0, 0};
    std::array<int64_t, kBGAwareNumBankGroups> m_bgRecentRow = {-1, -1, -1, -1};
    std::array<int32_t, kBGAwareNumBankGroups> m_bgRecentBank = {-1, -1, -1, -1};
    std::array<uint32_t, kBGAwareNumBanks> m_bankOutstanding = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, kBGAwareNumBanks> m_bankCredits = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<int64_t, kBGAwareNumBanks> m_bankRecentRow = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    std::array<uint32_t, kBGAwareNumBanks> m_bankFineTokens = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<uint32_t, kBGAwareNumBankGroups> m_bgFineTokens = {0, 0, 0, 0};

    struct KernelTimingCfg {
        uint32_t lanes = 1;
        uint32_t issueInterval = 1;
        uint32_t pipelineLatency = 1;
        uint32_t setupCycles = 0;
        uint32_t finalizeCycles = 0;
        bool pipelined = true;
    };

    // Compute configuration
    uint32_t nFMAC;              // Legacy/default lane count for distance evaluation
    uint32_t cyclesPerFMAC;      // Legacy/default standard-FP issue interval in EmbUnit cycles
    uint32_t multiplierLatencyCycles;
    bool fmacPipelined;          // Legacy/default standard-FP pipelined flag
    double embComputeCycleRatio;            // EmbUnit cycle -> memory-domain cycle ratio
    uint32_t dualQueueCoarseOpsPerDim;      // Modeled coarse lower-bound ops per dimension
    std::string dualQueueCoarseComputeModel; // coarse_lb | std_fp
    uint32_t dualQueueCoarseStdFpOpsPerDim; // Modeled std-FP ops per dim when phase-1 uses std_fp
    uint32_t dualQueueFineOpsPerDim;        // Modeled dual fine-distance ops per dimension
    uint32_t dualQueueCoarseLanes;
    uint32_t dualQueueCoarseCyclesPerOp;
    uint32_t dualQueueCoarseLatencyCycles;
    uint32_t dualQueueCoarseSetupCycles;
    uint32_t dualQueueCoarseFinalizeCycles;
    bool dualQueueCoarsePipelined;
    uint32_t dualQueueFineLanes;
    uint32_t dualQueueFineCyclesPerOp;
    uint32_t dualQueueFineLatencyCycles;
    uint32_t dualQueueFineSetupCycles;
    uint32_t dualQueueFineFinalizeCycles;
    bool dualQueueFinePipelined;
    uint32_t stdFpLanes;
    uint32_t stdFpSetupCycles;
    uint32_t stdFpFinalizeCycles;
    bool zeroMemFinalizeSeparatePoolEnable;
    uint32_t zeroMemFinalizeWorkItemsPerTask;
    uint32_t zeroMemFinalizeModeledOpsPerTask;
    uint32_t zeroMemFinalizeLanes;
    uint32_t zeroMemFinalizeIssueInterval;
    uint32_t zeroMemFinalizeLatencyCycles;
    uint32_t zeroMemFinalizeSetupCycles;
    uint32_t zeroMemFinalizeFinalizeCycles;
    bool zeroMemFinalizePipelined;

    // Per-rank vector LRU cache
    VectorLRUCache m_vector_cache;
    uint32_t vectorCacheSize = 0;    // Number of vectors to cache per rank (0 = disabled)
    bool vectorCacheEnabled = false;
    std::string vectorCacheMode = "full_object";
    VectorLRUCache m_phase1_signexp_cache;
    uint32_t phase1SignExpCacheSize = 0;
    bool phase1SignExpCacheEnabled = false;

    // stat counters
    uint64_t s_num_disreq = 0;
    uint64_t s_num_pdisreq = 0;
    uint64_t s_num_early_exit = 0;
    uint64_t s_num_similarity_mul = 0;
    uint64_t s_num_redundant_mul = 0;
    uint64_t s_modeled_read_bits = 0;
    uint64_t s_dualq_phase1_pdisreq = 0;
    uint64_t s_dualq_phase2_pdisreq = 0;
    uint64_t s_dualq_phase1_modeled_read_bits = 0;
    uint64_t s_dualq_phase2_modeled_read_bits = 0;
    uint64_t s_pdis_mem_service_cycles = 0;
    uint64_t s_dualq_phase1_pdis_mem_service_cycles = 0;
    uint64_t s_dualq_phase2_pdis_mem_service_cycles = 0;
    uint64_t s_fmac_raw_cycles = 0;
    uint64_t s_fmac_hidden_cycles = 0;
    uint64_t s_fmac_queue_cycles = 0;
    uint64_t s_fmac_compute_cycles = 0;  // Critical-path compute delay after memory return (queue + exposed service)
    uint64_t s_dualq_phase1_exposed_compute_cycles = 0;
    uint64_t s_dualq_phase2_exposed_compute_cycles = 0;
    uint64_t s_dualq_phase2_early_exit = 0;
    uint64_t s_mfnns_linear_early_exit = 0;
    uint64_t s_fmac_ops = 0;             // Total FMAC operations (mul+add pairs)
    uint64_t s_dualq_phase1_fmac_ops = 0; // Phase-1 FMAC ops (MFNNS dual-queue coarse)
    uint64_t s_dualq_phase2_fmac_ops = 0; // Phase-2 FMAC ops (MFNNS dual-queue fine)
    uint64_t s_dualq_phase1_modeled_ops = 0; // Phase-1 modeled compute ops (coarse LB comparison)
    uint64_t s_dualq_phase2_modeled_ops = 0; // Phase-2 modeled compute ops (fine FP distance)
    uint64_t s_ansmet_et_fmac_ops = 0;    // FMAC ops for early-terminated candidates (ANSMET)
    uint64_t s_ansmet_total_fmac_ops = 0; // Total FMAC ops in ANSMET path
    uint64_t s_num_zero_mem_pdisreq = 0;
    uint64_t s_num_zero_mem_finalize_tasks = 0;
    uint64_t s_zero_mem_finalize_cycles = 0;
    uint64_t s_dualq_bgaware_reordered = 0;
    uint64_t s_dualq_bgaware_fine_deferred = 0;
    uint64_t s_dualq_bgaware_rowhit_selected = 0;
    uint64_t s_dualq_rowaware_reordered = 0;
    uint64_t s_dualq_rowaware_rowhit_selected = 0;
    uint64_t s_dualq_rowaware_rowburst_selected = 0;
    uint64_t s_dualq_rowaware_threshold_selected = 0;
    uint64_t s_dualq_rowaware_age_forced = 0;
    uint64_t s_dualq_rowaware_token_throttled = 0;
    uint64_t s_dualq_crosslevel_coarse_bg_tasks = 0;
    uint64_t s_dualq_crosslevel_fine_bank_tasks = 0;
    uint64_t s_dualq_crosslevel_fine_subarray_tasks = 0;
    uint64_t s_dualq_stage_admission_coarse_blocked = 0;
    uint64_t s_dualq_stage_admission_fine_blocked = 0;
    uint64_t s_dualq_stage_admission_fine_deferred = 0;
    uint64_t s_dualq_stage_admission_fine_promoted = 0;
    uint64_t s_dualq_stage_admission_fine_wait_cycles = 0;
    uint64_t s_batch_dispatch_total = 0;       // Total dispatches from batch mode
    uint64_t s_batch_dispatch_extra = 0;       // Extra dispatches beyond first-per-cycle
    uint64_t s_row_sweep_dispatched = 0;       // Dispatches from row-sweep coalescing
    uint64_t s_row_sweep_opportunities = 0;    // Times row-sweep found same-row candidates
    enum class ComputeTaskKind {
        None = 0,
        DualQueueCoarseLowerBound,
        DualQueueFineApproxDistance,
        StandardFpDistance,
        ZeroMemFinalize,
    };

    static constexpr size_t kNumComputePools = 4;

    struct ComputeTaskModel {
        ComputeTaskKind kind = ComputeTaskKind::None;
        uint32_t workItems = 0;
        uint32_t modeledOps = 0;
        uint32_t fmacOps = 0;
    };

    // runtime queues
    std::unordered_map<uint64_t, DisReq> inflightDisReqs;
    std::unordered_map<uint64_t, DisResp> finishDisReqs;
    // std::vector<HNSWCompUnit*> CompUnits;
    std::deque<PDisReq> pendPDisReqs;
    std::queue<PDisReq> pendCacheHitPDisReqs;  // Cache hit requests to process async
    std::unordered_map<uint64_t, PendingPDisReqState> inflightPDisReqCallbacks;
    uint64_t nextPDisReqToken = 1;
    std::deque<uint64_t> dualQueuePendingFineAdmissions;
    uint32_t dualQueueActiveCoarseDisReqs = 0;
    uint32_t dualQueueActiveFineDisReqs = 0;

    struct PendingComputeTask {
        uint64_t taskId = 0;
        PDisReq req;
        ComputeTaskKind kind = ComputeTaskKind::None;
        uint32_t workItems = 0;
        uint32_t modeledOps = 0;
        uint32_t fmacOps = 0;
        uint64_t readyCycle = 0;
        uint64_t startCycle = 0;
        uint64_t completionCycle = 0;
        uint64_t rawComputeCycles = 0;
        uint64_t hiddenComputeCycles = 0;
        uint64_t queueCycles = 0;
        uint64_t serviceExposedComputeCycles = 0;
        uint64_t exposedComputeCycles = 0;
        size_t poolIndex = 0;
        uint32_t resourceUnit = 0;
    };
    struct PendingComputeTaskCompletionCmp {
        bool operator()(const PendingComputeTask& lhs, const PendingComputeTask& rhs) const {
            if (lhs.completionCycle != rhs.completionCycle) {
                return lhs.completionCycle > rhs.completionCycle;
            }
            return lhs.taskId > rhs.taskId;
        }
    };
    std::array<std::queue<PendingComputeTask>, kNumComputePools> readyComputeTasks;
    uint64_t nextComputeTaskId = 1;
    std::array<bool, kNumComputePools> hasActiveComputeTask = {false, false, false, false};
    std::array<PendingComputeTask, kNumComputePools> activeComputeTask;
    std::array<std::vector<uint64_t>, kNumComputePools> computeResourceReadyCycles;
    std::priority_queue<PendingComputeTask, std::vector<PendingComputeTask>, PendingComputeTaskCompletionCmp> pendingComputeCompletions;

    HNSWTraversalUnit* travUnit = nullptr;

    // Helper functions to model EmbUnit compute latency
    KernelTimingCfg getKernelTimingCfg(ComputeTaskKind kind) const;
    uint64_t calculateComputeLatency(const ComputeTaskModel& task) const;
    ComputeTaskModel describeBlockCompute(DisReq& disReq, const PDisReq& pDisReq) const;
    size_t getComputePoolIndex(ComputeTaskKind kind) const;
    const char* getComputeTaskKindName(ComputeTaskKind kind) const;
    uint32_t getComputeResourceCount(size_t poolIndex) const;
    uint32_t predictSubarrayId(uint64_t row) const;
    void assignPredictedComputeTarget(PDisReq& pDisReq, const std::vector<Addr_t>& requestAddrs) const;
    uint32_t selectComputeResourceUnit(const DisReq& disReq, const PendingComputeTask& task) const;
    void schedulePDisReqCompute(PDisReq& pDisReq);
    void tryStartNextComputeTask(uint64_t curCycle, size_t poolIndex);
    void tryStartNextComputeTasks(uint64_t curCycle);
    bool hasPendingComputeWork() const;
    void handleCompletedPDisReq(const PendingComputeTask& task);
    int getOutlierColC0(uint32_t candId) const;

    void recordMultiplications(uint32_t nDim, DisReq* source);
    hnswlib::FP16L2SquareMethod getEffectiveL2SquareMethod(const std::string& spacetype, bool allowApprox) const;
    Type computeDistanceValue(const Type* x, const Type* y, uint32_t nDim, const std::string& spacetype, bool allowApprox) const;
    Type bitRound(Type dis, Type x, Type y, uint32_t correctBits);
    Type mergeDistance(DisReq& disReq, PDisReq& resp);
    Type mergeDistance1(DisReq& disReq, PDisReq& resp);

    int preprocess(PointId candId);

    void getNextDimBit(DisReq& disReq);
    bool isLastDimBit(DisReq& disReq);
    Addr_t mapVectorLineAddr(Addr_t logicalLineAddr, uint32_t phase) const;
    uint32_t getDualQueueModeledFetchBits(uint32_t phase) const;
    bool usesAnsmetFp16FullDimFetch(const DisReq& disReq) const;
    uint32_t getLinearStageBitStep(const DisReq& disReq, uint32_t curBit);
    uint32_t getLinearRequestDimStep(const DisReq& disReq, uint32_t curBit);
    std::vector<Addr_t> buildRequestAddrs(DisReq& disReq, uint32_t curDim, uint32_t curBit, bool commitChargedState);
    std::vector<Addr_t> getRequestAddrs(DisReq& disReq, uint32_t curDim, uint32_t curBit);
    bool isDualQueueScheduleEnabled() const;
    bool needsDualQueueMemoryTracking() const;
    uint32_t flattenBank(const PredictedBGTgt& tgt) const;
    PredictedBGTgt predictBGTgt(Addr_t addr) const;
    ReplicaDispatchProbe predictReplicaDispatchProbe(PointId candId,
                                                     uint32_t vDimBase,
                                                     uint32_t vDimEnd,
                                                     bool dualQueueTwoPhase);
    void updateBGOutstandingOnIssue(const std::vector<Addr_t>& addrs, uint32_t phase);
    void updateBGOutstandingOnReturn(Addr_t addr, uint32_t phase);
    PendingPDisReqSelection analyzePendingPDisReq(const PDisReq& pDisReq, uint64_t curCycle, bool anyCoarseReady);
    size_t selectNextPendingPDisReqIndex(uint64_t curCycle);
    void dispatchPendingPDisReqs(uint64_t curCycle);
    bool isDualQueueStagedAdmissionEnabled() const;
    uint32_t getDualQueueFineAdmissionOccupancy() const;
    bool canAdmitDualQueueCoarseDisReq() const;
    void markDualQueueCoarseAdmitted(DisReq& disReq);
    void releaseDualQueueCoarseAdmission(DisReq& disReq);
    bool tryActivateDualQueueFineAdmission(DisReq& disReq, uint64_t curCycle);
    void deferDualQueueFineAdmission(DisReq& disReq, uint64_t curCycle);
    void releaseDualQueueFineAdmission(DisReq& disReq);
    void startDualQueueFinePhase(DisReq& disReq);
    void tryPromoteDeferredDualQueueFineAdmissions(uint64_t curCycle);
    void handlePDisReqLineReturn(Request& req);
    bool sendPDisReq(PDisReq &pDisReq);
    void trySendPDisReq(uint64_t disreqId, uint32_t curDim, uint32_t curBit,int num_extra_sned);
    bool canLookupVectorCache(const DisReq& disReq) const;
    bool canFillVectorCache(const DisReq& disReq) const;
    Type computeDualQueueDistanceFromScore(Type scoreAccum, Type querySquaredNorm) const;
    Type computeDualQueueExactScoreContribution(Type cand, Type query, DisReq* source = nullptr);
    bool usesAnsmetRuntimeTrueFp16BitChop(bool dtype) const;

public:
    void printBinary(Type value,int isint_tmp);
    Type getDistance(const Type* x, const Type* y, uint32_t nDim, DisReq* source = nullptr);
    Type getDistance(const Type* x, const Type* y, uint32_t nDim, std::string spacetype, DisReq* source = nullptr);
    Type bitChop(Type x,Type q, uint32_t curBit, uint32_t bitEnd,int up);
    Type bitChop(Type x,Type q, uint32_t curBit, uint32_t bitEnd,int up,bool dtype,bool dataunsigned);
    Type adjustCandPartialData(Type* queryData, size_t dim, uint32_t curBit, Type* candData, uint32_t bitEnd,Type* candFormatData, uint32_t candId);
    Type adjustCandPartialData0(Type* queryData, size_t dim, uint32_t curBit, Type* candData, uint32_t bitEnd,bool dtype,std::string spacetype, bool dataunsigned);
    bool sendDisReq(DisReq& disReq);
    uint32_t sendResultProbe();
    uint64_t get_num_disreq() { return s_num_disreq; }
    uint64_t get_num_pdisreq() { return s_num_pdisreq; }
    uint64_t get_num_similarity_mul() { return s_num_similarity_mul; }
    uint64_t get_num_redundant_mul() { return s_num_redundant_mul; }
    uint32_t get_inflight_disreq() { return inflightDisReqs.size(); }
    uint32_t getFinishDisReqs() { return finishDisReqs.size(); }

    // Vector cache statistics
    uint64_t get_vector_cache_hit() { return m_vector_cache.s_cache_hit; }
    uint64_t get_vector_cache_miss() { return m_vector_cache.s_cache_miss; }
    uint64_t get_vector_cache_access() { return m_vector_cache.s_cache_access; }
    uint64_t get_phase1_signexp_cache_hit() { return m_phase1_signexp_cache.s_cache_hit; }
    uint64_t get_phase1_signexp_cache_miss() { return m_phase1_signexp_cache.s_cache_miss; }
    uint64_t get_phase1_signexp_cache_access() { return m_phase1_signexp_cache.s_cache_access; }

    void setup(uint32_t id, HNSWTraversalUnit* frontend, MultiGenericDRAMSystem* memory_system);
    void init() override;
    void tick() override;
    bool is_finished() override {
        return pendPDisReqs.empty() && pendCacheHitPDisReqs.empty() && inflightPDisReqCallbacks.empty() && inflightDisReqs.empty() && !hasPendingComputeWork();
    }
};

class HNSWTraversalUnit : public IFrontEnd, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, HNSWTraversalUnit, "HNSWTraversalUnit", "HNSWTraversalUnit")

public:
    struct GraphMemLoc {
        uint32_t rankId = 0;
        Addr_t addr = 0;
    };
    struct GraphTravSchedInfo {
        uint32_t rankId = 0;
        uint64_t pageId = 0;
        bool valid = false;
    };
    std::unordered_set<Addr_t> top_layer_addr;
    uint64_t s_top_layer_access = 0;
    uint64_t s_cache_hit = 0;
    uint64_t s_total_pdisreq = 0;
    uint64_t top_layer_size_limit = 131072; // 131072 * 64B = 8MB

    std::unordered_set<Addr_t> unlimited_top_layer_addr;
    uint64_t s_unlimited_top_layer_access = 0;
    std::vector<uint64_t> s_top_layer_access_per_embunit;
    std::vector<uint64_t> s_num_total_pdisreq;

private:
    int m_id;
    Clk_t m_clk;
    Logger_t m_logger;
    std::function<void(Request&)> travNodeCallback, travEdgeCallback;
    std::function<void(Request&)> pollCallback;

    // parameters
    uint32_t nData, nQuery;
    uint32_t nDim;
    uint32_t vDimSize;
    uint32_t k_neighbors;
    uint32_t ef_search;
    bool enableLogging;
    bool earlyExitEnable;
    bool topkThresholdETEnable;
    bool dualQueueLowerBoundETEnable;
    uint32_t nParallelQuery;
    uint32_t nEmbUnit = 8; // aka memory ranks
    uint32_t nParallelDisReq = 16;
    bool traversalMemAcc;
    bool debug_minDisReqMap;
    bool debug_epCopy;
    bool debugIssueTraceEnable = false;
    uint32_t debugIssueTraceQueryId = 0;
    std::string debugIssueTracePath;
    bool perQuerySummaryEnable = false;
    std::string perQuerySummaryPath;
    bool debugDuplicateAcceptEnable = false;
    uint32_t debugDuplicateAcceptQueryId = 0;
    std::string debugDuplicateAcceptPath;
    bool mfnnsEnable = false;
    std::unordered_map<PointId, uint64_t> disreqMap;
    uint32_t nPackingVec;
    std::unordered_map<PointId, uint32_t> vecPackingMap;
    std::string statFile;
    uint32_t resProbeEpoch;
    bool pollingSimulation;
    bool adaptiveResultProbe;
    std::string vectorLayoutMode;
    std::string vectorLogicalLayoutMode;
    bool layout2GlobalCoarseThenFineEnable = false;
    bool dualQueueAllowLinearFullReadLayout = false;
    std::string vectorPhysicalPlacementMode;
    std::string vectorRankPlacementMode;
    uint32_t vectorRankPlacementChunkSize = 128;
    uint32_t vectorRankPlacementSuperChunkFactor = 1;
    uint32_t vectorRankPlacementStripeGroupSize = 1;
    uint32_t vectorRankPlacementStripeBlockSize = 1;
    bool vectorRankPlacementLevelAwareStripeEnable = false;
    uint32_t vectorRankPlacementTopLevelCount = 0;
    uint32_t vectorRankPlacementTopStripeGroupSize = 1;
    uint32_t vectorRankPlacementTopStripeBlockSize = 1;
    std::string vectorAddressMappingMode;
    std::string vectorAddressMappingPhase1Mode;
    std::string vectorAddressMappingPhase2Mode;
    bool vectorPhaseBankPartitionEnable = false;
    uint32_t vectorPhase1DedicatedBGCount = 2;
    std::string vectorReplicaDispatchMode;
    uint32_t hotNodeReplicationTopLevelCount = 0;
    std::string graphStaticScheduleMode;
    std::string graphVertexReorderMode;
    bool graphTraversalBatchingEnable = false;
    uint32_t graphTraversalBatchingWindow = 32;
    uint64_t graphTraversalBatchingAgeThreshold = 256;
    uint32_t graphTraversalBatchingPageWeight = 8;
    uint32_t graphTraversalBatchingRankWeight = 2;
    uint32_t graphTraversalBatchingLastPageBonus = 4;
    uint32_t graphTraversalBatchingLastRankBonus = 1;
    std::string resolvedVectorLayoutMode;
    std::string resolvedVectorAddressMappingMode;
    std::string resolvedVectorAddressMappingPhase1Mode;
    std::string resolvedVectorAddressMappingPhase2Mode;
    std::string resolvedVectorReplicaDispatchMode;
    uint32_t hotNodeReplicationCount = 0;
    bool hotReplicaRowAwareEnable = false;
    uint32_t hotReplicaLoadWeight = 1;
    uint32_t hotReplicaBgPenaltyWeight = 4;
    uint32_t hotReplicaBankPenaltyWeight = 8;
    uint32_t hotReplicaRowHitBonus = 32;
    uint32_t hotReplicaBgRowHitBonus = 16;
    uint32_t graphPlacementPageLines = 64;
    std::unordered_set<PointId> hotReplicaNodes;
    std::vector<uint8_t> graphLevel0RankId;
    std::vector<uint32_t> graphLevel0NodeLineBase;
    std::vector<uint32_t> graphLevel0EdgeLineBase;
    std::vector<uint16_t> vectorRankLocalId;
    std::vector<uint32_t> vectorLocalSlot;
    std::vector<uint32_t> vectorLocalRankCount;
    std::vector<uint32_t> vectorReplicaSlot;
    uint32_t vectorReplicaCount = 0;
    bool graphTravLastIssueValid = false;
    uint32_t graphTravLastIssueRankId = 0;
    uint64_t graphTravLastIssuePageId = 0;

    // runtime info
    std::vector<HNSWEmbUnit*> embUnits;
    std::queue<Query*> pendQueries;
    std::unordered_map<uint32_t, Query*> queries; // inflight queries
    std::vector<std::queue<DisReq>> pendDisReqs; // pendDisReq of each embUnit
    std::unordered_map<PointId, uint64_t> prof_candAcc;
    std::string query_path;
    std::vector<uint32_t> gt;
    uint32_t gt_k;
    bool checkedRecall = false;
    uint64_t disreqId; // DisReq version id
    const Addr_t graphNodeAddrBase = 0x1000;
    Addr_t graphEdgeAddrBase;
    std::vector<Addr_t> graphEdgeAddrOffset;
    std::deque<TravReq> pendTravNodeReqs;
    std::deque<TravReq> pendTravEdgeReqs;
    std::queue<Request> pendPollReqs;

    uint32_t dimStep_trav, bitStep_trav;
    uint32_t dimStep_trav1, bitStep_trav1;
    uint32_t dimStep_trav2, bitStep_trav2;
    uint32_t preprocessVectorDataBitWidth = 0;
    float outlier_percentage;
    bool allowSample;
    bool allowoutlier;
    std::string dimbitMode;
    bool limit_bs_num;
    int bs_num;

    // stats counter
    uint64_t s_num_query = 0;
    uint64_t s_num_construct = 0;
    uint64_t s_num_travnode_req = 0;
    uint64_t s_num_travedge_req = 0;
    uint64_t s_total_latency = 0;
    std::vector<uint64_t> s_query_latency_bin;
    uint64_t s_bd_index_cycles = 0;
    uint64_t s_bd_disOffload_cycles = 0;
    uint64_t s_bd_disComp_cycles = 0;
    uint64_t s_bd_disCompAcc_cycles = 0;
    uint64_t s_bd_disCompRej_cycles = 0;
    uint64_t s_bd_disGather_cycles = 0;
    uint64_t s_bd_polling_cycles = 0;
    const uint64_t latBinSize = 1000, nLatBin = 10;
    double s_disreq_average_latency = 0;
    uint64_t s_trav_earlyexit = 0;
    std::vector<uint64_t> s_num_total_disreq;
    std::vector<uint64_t> s_num_packed_disreq;
    std::vector<uint64_t> s_num_packed_saving;
    uint64_t s_total_disreq = 0;
    uint64_t s_disreq_time = 0;
    uint64_t s_disreq_compute_time = 0;
    uint64_t s_disreq_req_idle_time = 0;
    uint64_t s_disreq_resp_idle_time = 0;
    uint64_t s_disreq_mem_service_cycles = 0;
    uint64_t s_disreq_raw_compute_cycles = 0;
    uint64_t s_disreq_hidden_compute_cycles = 0;
    uint64_t s_disreq_compute_queue_cycles = 0;
    uint64_t s_disreq_exposed_compute_cycles = 0;
    uint64_t s_num_result_probe = 0;
    uint64_t s_mem_cycle = 0;
    uint64_t s_mem_read_req = 0;
    uint64_t s_mem_write_req = 0;
    uint64_t s_num_polling = 0;
    uint64_t s_num_recall = 0;
    uint64_t s_num_recall_1 = 0;
    double s_avg_total_latency = 0.0;
    double s_recall_rate = 0.0;
    uint64_t s_dual_queue_lb_et = 0;
    uint64_t s_hot_replica_remap = 0;
    uint64_t s_hot_replica_rowaware_selected = 0;
    uint64_t s_hot_replica_rowhit_selected = 0;
    uint64_t s_hot_replica_bg_rowhit_selected = 0;
    uint64_t s_graph_level0_layout_total_lines = 0;
    uint64_t s_graph_level0_layout_padding_lines = 0;
    uint64_t s_graph_level0_degree_bfs_neighbor_visits = 0;
    uint64_t s_graph_trav_batch_reordered = 0;
    uint64_t s_graph_trav_batch_page_match_selected = 0;
    uint64_t s_graph_trav_batch_rank_match_selected = 0;
    uint64_t s_graph_trav_batch_age_forced = 0;
    std::vector<uint32_t> s_same_rank_run_lengths;
    double s_same_rank_run_length_mean = 0.0;
    uint64_t s_same_rank_run_length_p95 = 0;
    uint32_t dualQueueLowerBoundQueueSize = 0;
    uint32_t dualQueueLowerBoundWarmupSize = 0;

    void prepareHotReplicaNodes();
    void prepareGraphStaticSchedule();
    std::vector<PointId> buildGraphLevel0Order();
    std::vector<PointId> buildVectorPlacementOrder();
    void buildKMeansBalancedPartition(
        uint32_t n_v_emb_unit,
        std::vector<uint16_t>& outRankLocalId,
        std::vector<uint32_t>& outLocalSlot,
        std::vector<uint32_t>& outLocalRankCount);
    void buildKMeansClusterRoundRobinPartition(
        uint32_t n_v_emb_unit,
        std::vector<uint16_t>& outRankLocalId,
        std::vector<uint32_t>& outLocalSlot,
        std::vector<uint32_t>& outLocalRankCount);
    void buildMultiStartBalancedPartition(
        uint32_t n_v_emb_unit,
        std::vector<uint16_t>& outRankLocalId,
        std::vector<uint32_t>& outLocalSlot,
        std::vector<uint32_t>& outLocalRankCount);
    void buildModuloBfsSlotReorder(
        uint32_t n_v_emb_unit,
        std::vector<uint16_t>& outRankLocalId,
        std::vector<uint32_t>& outLocalSlot,
        std::vector<uint32_t>& outLocalRankCount);
    void prepareVectorRankPlacement();
    uint32_t getGraphLevel0Degree(PointId candId) const;
    GraphTravSchedInfo getGraphTravSchedInfo(const TravReq& travReq, bool isEdgeReq);
    size_t selectTravReqIndex(const std::deque<TravReq>& queue, bool isEdgeReq);
    bool trySendTravReq(std::deque<TravReq>& queue, bool isEdgeReq);
    uint32_t selectLeastLoadedEmbUnitInVerticalGroup(uint32_t verticalId);
    uint32_t selectReplicaEmbUnitInVerticalGroup(PointId candId,
                                                 uint32_t verticalId,
                                                 uint32_t vDimBase,
                                                 uint32_t vDimEnd,
                                                 bool dualQueueTwoPhase);
    uint32_t mapDisReq(PointId candId,
                       uint32_t verticalId,
                       uint32_t vDimBase,
                       uint32_t vDimEnd,
                       bool dualQueueTwoPhase);
    uint32_t getGraphRankId(Addr_t addr);
    GraphMemLoc getGraphNodeMemLoc(TravReq& travReq);
    GraphMemLoc getGraphEdgeMemLoc(TravReq& travReq);
    Addr_t getGraphNodeAddress(TravReq& travReq);
    Addr_t getGraphEdgeAddress(TravReq& travReq);
    uint32_t getTravNodeSegment(PointId candId, uint32_t queryLevel);
    Type computeDualQueueLowerBound(const Query* query, PointId candId);
    void updateFinalResultQueue(Query* query, PointDistId result);
    Type getDualQueueLowerBoundThreshold(const Query* query);
    Type getTopKThreshold(const Query* query);
    Type getExactDistanceThreshold(const Query* query);
    Type getEarlyTerminationThreshold(const Query* query);
    std::vector<std::pair<Type, hnswlib::tableint>> PrioQueueToVector(ANNSResultQueue &pq);
    uint32_t getVectorPackingGroup(PointId candId);
    bool shouldBufferLevel0DualQueueResults(const Query* query) const;
    void beginLevel0DualQueueBatch(Query* query);
    void recordLevel0DualQueueCandidateOrder(Query* query, PointId candId);
    bool shouldDebugIssueTraceForQuery(uint64_t annsId) const;
    bool shouldDebugDuplicateAcceptForQuery(uint64_t annsId) const;
    void recordIssueTraceSeen(Query* query,
                              uint32_t level,
                              PointId parentCandId,
                              PointId candId,
                              bool baseVisitBefore,
                              bool skippedByBaseVisit,
                              bool markedBaseVisit);
    void recordIssueTraceDisreqIssue(Query* query,
                                     uint32_t level,
                                     PointId parentCandId,
                                     PointId candId,
                                     uint32_t embUnitId,
                                     uint32_t vDimBase,
                                     Type upperbound);
    void dumpIssueTraceDebug(Query* query, const char* reason);
    void dumpPerQuerySummary(Query* query, const char* reason);
    void recordAcceptedCandidateDebugEvent(Query* query,
                                           PointId candId,
                                           Type pDistance,
                                           Type upperbound);
    void dumpAcceptedCandidateDebug(Query* query, const char* reason);
    void applyCompletedCandidateResult(Query* query,
                                       PointId candId,
                                       Type pDistance,
                                       Type candidateLowerBound,
                                       bool dualQueuePruned,
                                       uint64_t num_access);
    void flushBufferedLevel0DualQueueResults(Query* query);

    void handleTravNodeReq(TravReq& travReq);
    void handleTravEdgeReq(TravReq& travReq);
    bool sendTravNodeReq(TravReq& travReq);
    bool sendTravEdgeReq(TravReq& travReq);

    void handlePollReq(Request& req);
    void recordSameRankDispatch(Query* query, uint32_t rankId);
    void flushSameRankRun(Query* query);

    void finishQueryCheck(uint32_t annsId);
    void handleSearchLayer(TravReq& travReq);
    void searchLayer(uint32_t annsId);

    void finishConstructCheck(uint32_t annsId);
    void handleAddPoint(TravReq& travReq);
    void addPoint(uint32_t annsId);

    void preprocess_vecPacking();
    int preprocess();

    void checkRecall(Query* query);

public:
    bool outlier_burden;
    std::string datatype;
    bool dataunsigned;
    std::string spacetype;
    std::vector<int> bitStep_array;
    hnswlib::HierarchicalNSW<Type>* hnsw = nullptr;
    Type* getEmbData(PointId candId);
    bool mergeDisReq(Query* query, DisResp& resp);
    void handleDisReq(DisResp& resp);
    uint32_t getbitstep();
    uint32_t getdimstep();
    uint32_t getbitstep1();
    uint32_t getdimstep1();
    uint32_t getbitstep2();
    uint32_t getdimstep2();
    std::string getDimbitMode() {
        return dimbitMode;
    };
    const std::string& getVectorLayoutMode() const {
        return resolvedVectorLayoutMode;
    }
    const std::string& getVectorLogicalLayoutMode() const {
        return vectorLogicalLayoutMode;
    }
    uint32_t getEmbUnitCount() const {
        return nEmbUnit;
    }
    uint32_t getDataCount() const {
        return nData;
    }
    const std::string& getVectorPhysicalPlacementMode() const {
        return vectorPhysicalPlacementMode;
    }
    const std::string& getVectorRankPlacementMode() const {
        return vectorRankPlacementMode;
    }
    const std::string& getVectorAddressMappingMode() const {
        return resolvedVectorAddressMappingMode;
    }
    const std::string& getVectorAddressMappingMode(uint32_t phase) const {
        if (phase == 1) {
            return resolvedVectorAddressMappingPhase1Mode;
        }
        if (phase == 2) {
            return resolvedVectorAddressMappingPhase2Mode;
        }
        return resolvedVectorAddressMappingMode;
    }
    const std::string& getVectorReplicaDispatchMode() const {
        return resolvedVectorReplicaDispatchMode;
    }
    bool usesTopLayerReplicaPlacement() const {
        return vectorPhysicalPlacementMode == "toplayer_replicated";
    }
    uint32_t getHotNodeReplicationTopLevelCount() const {
        return hotNodeReplicationTopLevelCount;
    }
    bool usesCoarseFineSplitVectorLayout() const {
        return vectorLogicalLayoutMode == "coarse_fine_split" ||
               vectorLogicalLayoutMode == "row_aligned_coarse_fine_split" ||
               vectorLogicalLayoutMode == "global_coarse_fine_split";
    }
    bool usesRowAlignedCoarseFineVectorLayout() const {
        return vectorLogicalLayoutMode == "row_aligned_coarse_fine_split";
    }
    bool usesGlobalCoarseFineVectorLayout() const {
        return vectorLogicalLayoutMode == "global_coarse_fine_split";
    }
    bool usesGraphStaticPagePlacement() const {
        return graphStaticScheduleMode == "level0_page_blocked";
    }
    bool usesGraphClusteredVectorRankPlacement() const {
        return vectorRankPlacementMode == "graph_clustered_balanced";
    }
    bool usesKMeansBalancedPlacement() const {
        return vectorRankPlacementMode == "kmeans_balanced";
    }
    bool usesKMeansClusterRoundRobinPlacement() const {
        return vectorRankPlacementMode == "kmeans_cluster_rr";
    }
    bool usesGraphBalancedMultistartPlacement() const {
        return vectorRankPlacementMode == "graph_balanced_multistart";
    }
    bool usesModuloBfsSlotPlacement() const {
        return vectorRankPlacementMode == "modulo_bfs_slot";
    }
    bool usesGraphDegreeBfsReorder() const {
        return graphVertexReorderMode == "degree_bfs";
    }
    bool usesGraphTraversalBatching() const {
        return graphTraversalBatchingEnable;
    }
    bool usesStripedAddressMapping() const {
        return resolvedVectorAddressMappingMode == "striped";
    }
    bool usesStripedAddressMapping(uint32_t phase) const {
        return getVectorAddressMappingMode(phase) == "striped";
    }
    bool usesVectorPhaseBankPartitioning() const {
        return vectorPhaseBankPartitionEnable;
    }
    uint32_t getVectorPhase1DedicatedBGCount() const {
        return std::min<uint32_t>(3U, std::max<uint32_t>(1U, vectorPhase1DedicatedBGCount));
    }
    uint32_t getVectorPhasePartitionBGCount(uint32_t phase) const {
        const uint32_t phase1_bg = getVectorPhase1DedicatedBGCount();
        const uint32_t phase2_bg = 4U - phase1_bg;
        if (phase == 1) {
            return phase1_bg;
        }
        if (phase == 2) {
            return phase2_bg;
        }
        return 4U;
    }
    uint32_t getVectorPhasePartitionBGBase(uint32_t phase) const {
        if (phase == 2) {
            return getVectorPhase1DedicatedBGCount();
        }
        return 0U;
    }
    bool usesStripedVectorLayout() const {
        return usesStripedAddressMapping();
    }
    bool needsHotReplicaNodeSelection() const {
        const bool has_selection = hotNodeReplicationCount > 0 || hotNodeReplicationTopLevelCount > 0;
        return has_selection &&
               (resolvedVectorReplicaDispatchMode == "hot_node" || usesTopLayerReplicaPlacement());
    }
    bool usesHotNodeReplicaDispatch() const {
        return resolvedVectorReplicaDispatchMode == "hot_node" &&
               (hotNodeReplicationCount > 0 || hotNodeReplicationTopLevelCount > 0);
    }
    bool usesHotNodeReplication() const {
        return usesHotNodeReplicaDispatch() || usesTopLayerReplicaPlacement();
    }
    bool isHotReplicaRowAwareEnabled() const {
        return hotReplicaRowAwareEnable;
    }
    uint32_t getHotReplicaLoadWeight() const {
        return hotReplicaLoadWeight;
    }
    uint32_t getHotReplicaBgPenaltyWeight() const {
        return hotReplicaBgPenaltyWeight;
    }
    uint32_t getHotReplicaBankPenaltyWeight() const {
        return hotReplicaBankPenaltyWeight;
    }
    uint32_t getHotReplicaRowHitBonus() const {
        return hotReplicaRowHitBonus;
    }
    uint32_t getHotReplicaBgRowHitBonus() const {
        return hotReplicaBgRowHitBonus;
    }
    bool isHotReplicaNode(PointId candId) const {
        return hotReplicaNodes.count(candId) > 0;
    }
    bool hasVectorReplicaSlot(PointId candId) const {
        return candId < vectorReplicaSlot.size() &&
               vectorReplicaSlot[candId] != std::numeric_limits<uint32_t>::max();
    }
    uint32_t getVectorReplicaSlot(PointId candId) const {
        if (hasVectorReplicaSlot(candId)) {
            return vectorReplicaSlot[candId];
        }
        return 0;
    }
    uint32_t getVectorReplicaCount() const {
        return vectorReplicaCount;
    }
    bool isDualQueueLowerBoundETEnabled() const {
        return dualQueueLowerBoundETEnable;
    }
    bool isMFNNSEnabled() const {
        return mfnnsEnable;
    }
    uint32_t getVectorRankLocalEmbUnitCount() const {
        const uint32_t vertical_groups = std::max<uint32_t>(1U, nDim / vDimSize);
        return std::max<uint32_t>(1U, nEmbUnit / vertical_groups);
    }
    uint32_t getVectorRankLocalId(PointId candId) const {
        if (candId < vectorRankLocalId.size()) {
            return vectorRankLocalId[candId];
        }
        const uint32_t n_v_emb_unit = getVectorRankLocalEmbUnitCount();
        return static_cast<uint32_t>(candId) % n_v_emb_unit;
    }
    uint32_t getVectorLocalSlot(PointId candId) const {
        if (candId < vectorLocalSlot.size()) {
            return vectorLocalSlot[candId];
        }
        const uint32_t n_v_emb_unit = getVectorRankLocalEmbUnitCount();
        return static_cast<uint32_t>(candId) / std::max<uint32_t>(1U, n_v_emb_unit);
    }
    uint32_t getVectorLocalRankCount(uint32_t localRank) const {
        if (localRank < vectorLocalRankCount.size()) {
            return vectorLocalRankCount[localRank];
        }
        const uint32_t n_v_emb_unit = getVectorRankLocalEmbUnitCount();
        if (n_v_emb_unit == 0 || localRank >= n_v_emb_unit || nData == 0) {
            return 0;
        }
        return (nData > localRank)
            ? (static_cast<uint32_t>((static_cast<uint64_t>(nData - 1U - localRank) / n_v_emb_unit) + 1ULL))
            : 0U;
    }
    uint32_t getVectorRankPlacementBaseChunkSize() const {
        return std::max<uint32_t>(1U, vectorRankPlacementChunkSize);
    }
    uint32_t getVectorRankPlacementSuperChunkFactor() const {
        return std::max<uint32_t>(1U, vectorRankPlacementSuperChunkFactor);
    }
    uint32_t getVectorRankPlacementStripeGroupSize() const {
        return std::max<uint32_t>(1U, vectorRankPlacementStripeGroupSize);
    }
    uint32_t getVectorRankPlacementStripeBlockSize() const {
        return std::max<uint32_t>(1U, vectorRankPlacementStripeBlockSize);
    }
    bool usesVectorRankPlacementLevelAwareStripe() const {
        return vectorRankPlacementLevelAwareStripeEnable && vectorRankPlacementTopLevelCount > 0;
    }
    uint32_t getVectorRankPlacementTopLevelCount() const {
        return vectorRankPlacementTopLevelCount;
    }
    uint32_t getVectorRankPlacementTopStripeGroupSize() const {
        return std::max<uint32_t>(1U, vectorRankPlacementTopStripeGroupSize);
    }
    uint32_t getVectorRankPlacementTopStripeBlockSize() const {
        return std::max<uint32_t>(1U, vectorRankPlacementTopStripeBlockSize);
    }
    uint32_t getVectorRankPlacementEffectiveChunkSize() const {
        const uint64_t effective =
            static_cast<uint64_t>(getVectorRankPlacementBaseChunkSize()) *
            static_cast<uint64_t>(getVectorRankPlacementSuperChunkFactor());
        return static_cast<uint32_t>(std::min<uint64_t>(
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()), std::max<uint64_t>(1ULL, effective)));
    }
    Type getCurrentUpperbound(uint32_t annsId) const {
        auto it = queries.find(annsId);
        if (it == queries.end() || it->second == nullptr || it->second->result.empty()) {
            return std::numeric_limits<Type>::infinity();
        }
        const Query* query = it->second;
        if (query->type == "search" && query->level == 0) {
            if (query->result.size() < ef_search || query->finalResult.size() < k_neighbors) {
                return std::numeric_limits<Type>::infinity();
            }
            return std::max(query->result.top().first, query->finalResult.top().first);
        }
        return query->result.top().first;
    }
    uint32_t getResultQueueLimit(uint32_t annsId) const {
        auto it = queries.find(annsId);
        if (it == queries.end() || it->second == nullptr) {
            return 0;
        }
        Query* query = it->second;
        return (query->type == "search") ? ef_search : hnsw->ef_construction_;
    }
    bool isResultQueueFull(uint32_t annsId) const {
        auto it = queries.find(annsId);
        if (it == queries.end() || it->second == nullptr) {
            return false;
        }
        Query* query = it->second;
        if (query->type == "search" && query->level == 0) {
            return query->result.size() >= ef_search && query->finalResult.size() >= k_neighbors;
        }
        const uint32_t limit = (query->type == "search") ? ef_search : hnsw->ef_construction_;
        return query->result.size() >= limit;
    }
    void init() override;
    void updatebitSteparray();
    void tick() override;
    bool is_finished() override {
        bool finish = pendQueries.empty();
        finish &= queries.empty();
        for (uint32_t i = 0; i < nEmbUnit; i++) {
            finish &= pendDisReqs[i].empty();
            if (!finish) return false;
        }
        finish &= pendTravNodeReqs.empty();
        finish &= pendTravEdgeReqs.empty();
        return finish;
    };
    void connect_memory_system(IMemorySystem* memory_system) override {
        m_memory_system = memory_system;
        assert(embUnits.size() == nEmbUnit);
        MultiGenericDRAMSystem* mem_system = dynamic_cast<MultiGenericDRAMSystem*>(memory_system);
        assert(mem_system);
        for (uint32_t i = 0; i < embUnits.size(); i++) {
            embUnits[i]->setup(i, this, mem_system);
        }
    };
    void finalize();
};


} // namespace Ramulator

#endif   // HNSW_ACCEL_H
