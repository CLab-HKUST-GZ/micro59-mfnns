#ifndef     ANNS_H
#define     ANNS_H

#include "../hnswlib/hnswlib.h"

using Type = float;
using PointId = hnswlib::tableint;
using PointDistId = std::pair<Type, PointId>;
using Point = std::vector<Type>;

struct ANNSCpuRequest {
    // request
    uint32_t id;
    Point query;
    Type upperbound;
    PointId candId;
    uint32_t curDim;
    uint32_t curBit;
    // response
    bool positive;
    Type result;
};

struct ANNSRequest {
    // request
    uint32_t annsId;
    Point query;
    // response
    std::vector<PointId> cands;
};

struct DimBit{
    uint32_t dim;
    uint32_t bit;
    uint32_t ncurstep;
    uint32_t ncurstep_last;
};

struct TravReq {
    uint32_t annsId = 0;
    PointId candId = 0;
    uint32_t queryLevel = 0;
    uint32_t segmentId = 0;
    uint64_t reqSendCycle = 0;
    TravReq() = default;
    TravReq(uint32_t _annsId, PointId _candId, uint32_t _queryLevel, uint32_t _segmentId, uint64_t _reqSendCycle)
        : annsId(_annsId),
          candId(_candId),
          queryLevel(_queryLevel),
          segmentId(_segmentId),
          reqSendCycle(_reqSendCycle) {}
};

struct DisReq {
    uint64_t disreqId;
    uint32_t annsId;
    PointId candId;
    Point query;
    Type querySquaredNorm = 0;
    Type upperbound;
    Type dualQueuePhase1Upperbound = 0;
    // at runtime
    std::vector<Type> curDis;
    Type pDistanceAccum = 0;
    std::vector<DimBit> pends;
    DimBit last;
    uint32_t vDimBase, vDimEnd;
    uint32_t nDim;
    // stats
    uint64_t sendCycle = 0, recvCycle = 0;
    uint32_t layer;
    uint64_t num_access = 0;
    uint64_t mulCount = 0;
    uint64_t fmacOpsAccum = 0;  // Accumulator for total FMAC ops across all blocks
    uint64_t memServiceCyclesAccum = 0;
    uint64_t rawComputeCyclesAccum = 0;
    uint64_t hiddenComputeCyclesAccum = 0;
    uint64_t computeQueueCyclesAccum = 0;
    uint64_t exposedComputeCyclesAccum = 0;
    bool dualQueueTwoPhase = false;
    uint32_t dualQueuePhase = 0; // 0: disabled, 1: sign+exp, 2: mantissa
    Type dualQueueScoreAccum = 0;
    Type dualQueueExactDistanceAccum = 0;
    Type dualQueueCoarseScoreAccum = 0;
    int64_t dualQueuePhase1ChargedEndBit = -1;
    int64_t dualQueuePhase2ChargedEndBit = -1;
    int64_t ansmetLinearChargedEndBit = -1;
    Type dualQueueLowerBound = 0;
    uint32_t dualQueueAdmissionStage = 0; // 0: none, 1: coarse window, 2: fine window
    bool dualQueueWaitingFineAdmission = false;
    uint64_t dualQueueFineReadyCycle = 0;
    DisReq() {}
    DisReq(uint32_t _disreqId, uint32_t _annsId, PointId _candId, Point& _query, Type _upperbound, uint32_t _vDimBase, uint32_t _vDimEnd, uint32_t _nDim, uint64_t _sendCycle, uint32_t _layer):
        disreqId(_disreqId), annsId(_annsId), candId(_candId), query(_query), upperbound(_upperbound), vDimBase(_vDimBase), vDimEnd(_vDimEnd), nDim(_nDim), sendCycle(_sendCycle), layer(_layer) {
            dualQueuePhase1Upperbound = _upperbound;
            curDis = std::vector<Type>(nDim, 0);
            pDistanceAccum = 0;
            pends = std::vector<DimBit>();
            last = DimBit{vDimBase, 0};
            mulCount = 0;
            fmacOpsAccum = 0;
            dualQueueScoreAccum = 0;
            dualQueueExactDistanceAccum = 0;
            dualQueueCoarseScoreAccum = 0;
            for (const Type value : query) {
                querySquaredNorm += value * value;
            }
    }
};

struct DisResp {
    uint32_t annsId;
    PointId candId;
    bool positive;
    Type distance;
    uint64_t disreqId;
    // stats
    uint64_t reqSendCycle;
    uint64_t reqRecvCycle;
    uint64_t respSendCycle;
    uint64_t respRecvCycle;
    uint64_t num_access;
    uint64_t memServiceCycles = 0;
    uint64_t rawComputeCycles = 0;
    uint64_t hiddenComputeCycles = 0;
    uint64_t computeQueueCycles = 0;
    uint64_t exposedComputeCycles = 0;
    bool dualQueuePruned = false;
    bool dualQueueLowerBoundValid = false;
    Type dualQueueLowerBound = 0;
    DisResp() {}
    DisResp(uint32_t _annsId, PointId _candId, bool _positive, Type _distance, uint64_t _disreqId, uint64_t _respSendCycle):
        annsId(_annsId), candId(_candId), positive(_positive), distance(_distance), disreqId(_disreqId), respSendCycle(_respSendCycle) {}
};

struct PDisReq {
    // facilitates
    uint32_t embUnitId = 0;
    // request
    uint64_t disreqId = 0;
    uint32_t annsId = 0;
    PointId candId = 0;
    Point query;
    uint32_t curDim = 0;
    uint32_t curBit = 0;
    //uint32_t ncurStep;
    // response
    Type pDistance = 0;
    int num_extra_send = 0;
    // dim/bit step
    uint32_t dimStep = 0;
    uint32_t bitStep = 0;
    uint32_t phase = 0;
    uint64_t modeledReadBits = 0;
    uint64_t token = 0;
    uint64_t enqueueCycle = 0;
    uint64_t issueCycle = 0;
    uint64_t memServiceCycles = 0;
    uint32_t callbackLineCount = 0;
    bool computeTargetValid = false;
    uint32_t targetBankGroup = 0;
    uint32_t targetFlatBank = 0;
    uint64_t targetRow = 0;
    uint32_t targetSubarray = 0;
};

#endif   // ANNS_H
