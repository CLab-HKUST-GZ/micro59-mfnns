#include <unordered_set>
#include <type_traits>

#include "frontend/frontend.h"
#include "base/anns.h"

namespace Ramulator {

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
        printf("Input cols %d and read_col %d not match!", cols, read_cols);
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

class HNSWCpuFrontend : public IFrontEnd, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, HNSWCpuFrontend, "HNSWCpuFrontend", "HNSWCpuFrontend")

private:
    bool finish;
    int m_id;
    std::function<void(Request&)> m_callback;
    Logger_t m_logger;

    using ANNSSearchQueue = std::priority_queue<PointDistId, std::vector<PointDistId>, MinHeapByFirst>;
    using ANNSResultQueue = std::priority_queue<PointDistId, std::vector<PointDistId>, MaxHeapByFirst>;

    class Query {
    public:
        uint32_t level;
        ANNSSearchQueue search;
        ANNSResultQueue result;
        Point query;
        std::unordered_set<PointId> inflightCands;
        std::unordered_set<PointId> pendCands;

        bool topChanged;
        bool* baseVisitArray;

        Query(Point& _query, PointId _epId, uint32_t _level):
            query(_query), level(_level) {
            search.push(std::make_pair(-1U, _epId));
            result.push(std::make_pair(-1U, _epId));
            baseVisitArray = nullptr;
        }

        ~Query() {
            if (baseVisitArray) delete[] baseVisitArray;
        }
    };

    hnswlib::HierarchicalNSW<Type>* hnsw = nullptr;
    uint32_t nData, nQuery;
    uint32_t nDim, nBit;
    uint32_t k_neighbors;
    uint32_t dimSize;
    uint32_t dimStep, bitStep;
    uint32_t ef;
    std::unordered_map<uint32_t, Query*> queries;
    std::queue<ANNSCpuRequest> pendReqs;
    bool earlyExitEnable;
    bool enableLogging;
    uint32_t nParallelQuery;
    uint32_t nParallelCand;
    std::unordered_set<uint32_t> pendQueries;

    uint64_t s_total_query = 0;
    uint64_t s_total_mem_req = 0;
    uint64_t s_total_early_exit = 0;

    void load_hnsw_graph_from_file(std::string path, uint32_t dims) {
        hnswlib::L2Space space(nDim);
        hnsw = new hnswlib::HierarchicalNSW<Type>(&space, path);
        nData = hnsw->cur_element_count;
    }

    Addr_t getAddress(uint32_t annsId, PointId candId, uint32_t curDim, uint32_t curBit) {
        Addr_t addr = (candId * nDim + curDim) * dimSize;
        // printf("send addr %lx\n", addr);
        // fflush(stdout);
        // printf("[getAddress] annsId %d candId %x curDim %d curBit %d addr %lx\n", annsId, candId, curDim, curBit, addr);
        return addr;
    }

    bool sendRequest(uint32_t queryId, ANNSCpuRequest &info) {
        assert(queries.count(queryId));
        Type upperbound = queries[queryId]->result.top().first;
        Addr_t addr = getAddress(queryId, info.candId, info.curDim, info.curBit);
        bool ok = m_memory_system->send(Request(addr,
                                      Request::Type::Read,
                                      m_id,
                                      m_callback,
                                      info));
        if (ok) s_total_mem_req += 1;
        return ok;
    }

    void searchLayer(uint32_t queryId) {
        Query* query = queries[queryId];
        assert(query->search.size() > 0);
        assert(query->result.size() > 0);
        PointDistId curNode = query->search.top();
        query->search.pop();
        Type upperbound = query->result.top().first;
        if (query->level > 0) query->topChanged = false;
        if (query->level == 0) assert(query->baseVisitArray);
        // TODO: CSR graph needs memory access
        PointId curNodeId = curNode.second;
        unsigned int* data = (unsigned int *) hnsw->get_linklist_at_level(curNodeId, query->level);
        uint32_t size = hnsw->getListCount(data);
        PointId* datal = (PointId*) (data + 1);
        m_logger->info("[searchLayer] queryId {} curNode {} level {} neighbor num {}", queryId, curNodeId, query->level, size);
        if (query->level > 0 && size == 0) {
            // Fix cornercase of entrypoint no neighbors
            query->pendCands.insert(curNodeId);
            m_logger->info("[searchLayer] queryId {} curNode {} no neighbor send request to curNode", queryId, curNodeId);
        } else {
            for (uint32_t i = 0; i < size; i++) {
                PointId candId = datal[i];
                if (candId < 0 || candId > hnsw->max_elements_)
                    throw std::runtime_error("candId error");
                uint32_t curDim = 0, curBit = 0;
                if (query->level == 0) {
                    if (query->baseVisitArray[candId]) continue;
                    query->baseVisitArray[candId] = true;
                }
                query->pendCands.insert(candId);
                m_logger->info("[searchLayer] queryId {} send candId {} request upperbound {}", queryId, candId, upperbound);
            }
            m_logger->info("[searchLayer] queryId {} curNode {} pend total {} requests", queryId, curNodeId, query->pendCands.size());
            if (query->inflightCands.empty() && query->pendCands.empty()) {
                finishQueryCheck(queryId);
            }
        }
    }

    enum Result {
        Accept,
        Reject,
        Unidentified,
    };

    Type getDistance(Type* x, Type* y, uint32_t nDim) {
        Type res = 0;
        for (uint32_t i = 0; i < nDim; i++)
            res += (x[i]-y[i]) * (x[i]-y[i]);
        return res;
    }

    Type bitTransform(Type x, uint32_t bitEnd) {
        typedef union {
            float f;
            struct {
                unsigned int mantisa : 23;
                unsigned int exponent : 8;
                unsigned int sign : 1;
            } parts;
        } float32_cast;

        bool isFloat = std::is_same<Type, float>::value;
        assert(isFloat);
        if (bitEnd == 16) {
            float32_cast d1 = { .f = x };
            // printf("old f %f mantisa = %x\n", d1.f, d1.parts.mantisa);
            d1.parts.mantisa = (d1.parts.mantisa >> 16) << 16;
            // printf("new f %f new mantisa = %x\n", d1.f, d1.parts.mantisa);
            return d1.f;
        } else if (bitEnd == 32) {
            return x;
        }
        assert(false);
        return 0;
    }

    Result earlyExitCheck(Query* query, Request& resp) {
        // get partial result
        Type* candData = (Type*)hnsw->getDataByInternalId(resp.cpuanns.candId);
        uint32_t dimEnd = std::min(resp.cpuanns.curDim + dimStep, nDim);
        uint32_t bitEnd = std::min(resp.cpuanns.curBit + bitStep, nBit);
        Type* candPartialData = (Type*) malloc(sizeof(Type) * dimEnd);
        for (size_t dim = 0; dim < dimEnd; dim++) {
            *(candPartialData + dim) = bitTransform(*(candData + dim), bitEnd);
        }
        Type pResult = getDistance(resp.cpuanns.query.data(), candPartialData, dimEnd);
        Type fResult = getDistance(resp.cpuanns.query.data(), candData, nDim);
        free(candPartialData);
        resp.cpuanns.result = pResult;
        // dim-level
        // m_logger->info("[earlyExitCheck] queryId {} candId {} curDim {} pResult {} fullResult {} upperbound {}", resp.cpuanns.id, resp.cpuanns.candId, resp.cpuanns.curDim, resp.cpuanns.result, fResult, resp.cpuanns.upperbound);
        if (earlyExitEnable && resp.cpuanns.result >= resp.cpuanns.upperbound) {
            s_total_early_exit += 1;
            return Result::Reject;
        }
        // TODO: early accept
        uint32_t dim = resp.cpuanns.curDim, bit = resp.cpuanns.curBit;
        bool fullData = getNextDimBit(dim, bit);
        if (fullData && resp.cpuanns.result >= resp.cpuanns.upperbound) return Result::Reject;
        if (fullData && resp.cpuanns.result < resp.cpuanns.upperbound) return Result::Accept;
        return Result::Unidentified;
    }

    bool getNextDimBit(uint32_t &dim, uint32_t &bit) {
        bit += bitStep;
        if (bit >= nBit) {
            bit = 0;
            dim += dimStep;
        }
        return dim >= nDim;
    }

    void trySendRequest(uint32_t queryId, PointId candId, uint32_t dim, uint32_t bit) {
        assert(queries.count(queryId));
        Point& query = queries[queryId]->query;
        Type upperbound = queries[queryId]->result.top().first;
        ANNSCpuRequest info = {queryId,
                             query,
                             upperbound,
                             candId,
                             dim,
                             bit};
        if (!sendRequest(queryId, info)) {
            m_logger->info("Memory refuses requests query {} candId {}. Pending", queryId, candId);
            pendReqs.push(info);
        }
    }

    void finishQueryCheck(uint32_t queryId) {
        Query* query = queries[queryId];
        if (query->level > 0 && query->topChanged)
            searchLayer(queryId);
        else if (query->level == 0 && !query->search.empty())
            searchLayer(queryId);
        else if (query->level > 0 && !query->topChanged) {
            // end of this layer
            PointDistId nxEntryPoint = query->result.top();
            query->result.pop();
            query->search = ANNSSearchQueue();
            query->result = ANNSResultQueue();
            query->search.push(nxEntryPoint);
            query->result.push(nxEntryPoint);
            query->level -= 1;
            if (query->level == 0) {
                query->baseVisitArray = new bool[nData];
                std::fill(query->baseVisitArray, query->baseVisitArray + nData, false);
            }
            m_logger->info("[handleQuery] finish query {} level. Next level {} Next EP {}", queryId, query->level, nxEntryPoint.second);
            // printf("[handleQuery] finish query %d next level %d\n", queryId, query->level);
            // fflush(stdout);
            searchLayer(queryId);
        } else if (query->level == 0 && query->search.empty()) {
            // end of this query
            assert(queries.count(queryId));
            delete queries[queryId];
            queries.erase(queryId);
            assert(pendQueries.count(queryId));
            pendQueries.erase(queryId);
            s_total_query += 1;
            m_logger->info("[handleQuery] finish query {}", queryId);
            printf("[handleQuery] finish query %d\n", queryId);
            fflush(stdout);
            return;
        }
    }

    void handleQuery(Query* query, Request& resp) {
        PointId& candId = resp.cpuanns.candId;
        uint32_t& queryId = resp.cpuanns.id;
        assert(query->inflightCands.count(candId));
        query->inflightCands.erase(candId);
        assert(!query->pendCands.count(candId));
        if (resp.cpuanns.positive) {
            assert(query->result.size() > 0);
            Type upperbound = query->result.top().first;
            query->search.emplace(resp.cpuanns.result, candId);
            query->result.emplace(resp.cpuanns.result, candId);
            if (query->level > 0) {
                while (query->result.size() > 1) query->result.pop();
                while (query->search.size() > 1) query->search.pop();
            } else {
                uint32_t limit = std::min(ef, k_neighbors);
                if (query->result.size() > limit) query->result.pop();
            }
            query->topChanged = true;
            m_logger->info("[handleQuery] query {} candId {} result {} upperbound {} accept. inflightCands {} pendCands {}", queryId, candId, resp.cpuanns.result, upperbound, query->inflightCands.size(), query->pendCands.size());
        } else {
            Type upperbound = query->result.top().first;
            m_logger->info("[handleQuery] query {} candId {} result {} upperbound {} reject. inflightCands {} pendCands {}", queryId, candId, resp.cpuanns.result, upperbound, query->inflightCands.size(), query->pendCands.size());
        }
        if (query->inflightCands.empty() && query->pendCands.empty()) {
            finishQueryCheck(queryId);
        }
    }

    void handleRequest(Request& req) {
        assert(queries.count(req.cpuanns.id));
        Query* query = queries[req.cpuanns.id];
        Result result = earlyExitCheck(query, req);
        if (result == Result::Unidentified) {
            bool fullData = getNextDimBit(req.cpuanns.curDim, req.cpuanns.curBit);
            assert(!fullData);
            trySendRequest(req.cpuanns.id, req.cpuanns.candId, req.cpuanns.curDim, req.cpuanns.curBit);
        } else if (result == Result::Accept) {
            req.cpuanns.positive = true;
            handleQuery(query, req);
        } else if (result == Result::Reject) {
            req.cpuanns.positive = false;
            handleQuery(query, req);
        }
    };

public:
    void init() override {
        // initilize general config
        finish = false;
        m_id = 0;
        m_logger = Logging::create_logger("HNSWTraversalUnit");
        m_callback = [this](Request& req) { return this->handleRequest(req); };
        enableLogging = param<bool>("enableLogging").desc("enableLogging").default_val(false);
        if (!enableLogging) m_logger->set_level(spdlog::level::off);
        nParallelQuery = param<uint32_t>("nParallelQuery").desc("nParallelQuery").required();
        nParallelCand = param<uint32_t>("nParallelCand").desc("nParallelCand").default_val(1);

        // initilize anns config
        k_neighbors = 2;
        earlyExitEnable = param<bool>("earlyExitEnable").desc("earlyExitEnable").required();
        nDim = param<uint32_t>("nDim").desc("nDim").required();
        dimSize = sizeof(Type);
        nBit = sizeof(Type) * 8;
        if (earlyExitEnable) {
            dimStep = param<uint32_t>("dimStep").desc("dimStep").required();
            bitStep = param<uint32_t>("bitStep").desc("bitStep").required();
        } else {
            dimStep = 64 / dimSize;
            bitStep = 32;
        }
        assert(dimStep * bitStep == 64 * 8);
        printf("earlyExitEnable %x\n", earlyExitEnable);
        fflush(stdout);

        // load model and query
        std::string model_path = param<std::string>("model_path").desc("Path to the load HNSW graph.").required();
        load_hnsw_graph_from_file(model_path, nDim);
        printf("Load index from %s, with %d elements and %d dims\n", model_path.c_str(), nData, nDim);
        fflush(stdout);
        std::string query_path = param<std::string>("query_path").desc("Path to the load HNSW graph.").required();
        std::vector<Type> all_queries;
        read_vec<Type>(query_path, all_queries, nQuery, nDim);
        uint32_t nQueryLimit = param<uint32_t>("nQueryLimit").desc("nQueryLimit").default_val(-1U);
        nQuery = std::min(nQuery, nQueryLimit);
        printf("Load %d queries from %s\n", nQuery, query_path.c_str());
        fflush(stdout);
        assert(nDim % dimStep == 0);
        assert(nBit % bitStep == 0);
        for (uint32_t i = 0; i < nQuery; i++) {
            Point q;
            for (uint32_t d = 0; d < nDim; d++) {
                q.push_back(all_queries[i * nDim + d]);
            }
            queries[i] = new Query(q, hnsw->enterpoint_node_, hnsw->maxlevel_);
        }

        // initialize stats
        register_stat(s_total_early_exit).name("s_total_early_exit");
        register_stat(s_total_mem_req).name("s_total_mem_req");
        register_stat(s_total_query).name("s_total_query");
    };

    void tick() override {
        if (is_finished()) return;
        // search the next ready query
        if (pendQueries.size() < nParallelQuery) {
            for (auto& [queryId, query] : queries) {
                if (pendQueries.count(queryId)) continue;
                if (query->inflightCands.empty() && query->pendCands.empty()) {
                    m_logger->info("[tick] find ready query {} level {}", queryId, query->level);
                    searchLayer(queryId);
                    pendQueries.insert(queryId);
                    break;
                }
            }
        }
        // search pend cands in inflight queries
        for (auto& [queryId, query] : queries) {
            if (!pendQueries.count(queryId)) continue;
            if (!query->pendCands.empty()) {
                PointId cand = *query->pendCands.begin();
                // m_logger->info("[tick] find pend cand: queryId {} candId {} inflightSize {}", queryId, cand, query->inflightCands.size());
                if (query->inflightCands.size() < nParallelCand) {
                    query->pendCands.erase(cand);
                    query->inflightCands.insert(cand);
                    trySendRequest(queryId, cand, 0, 0);
                }
            }
        }
        // send pending reqs
        if (pendReqs.size() > 0) {
            ANNSCpuRequest& cur_req = pendReqs.front();
            pendReqs.pop();
            trySendRequest(cur_req.id, cur_req.candId, cur_req.curDim, cur_req.curBit);
        }
    };

    bool is_finished() override {
        return queries.size() == 0;
    };
};

} // namespace Ramulator