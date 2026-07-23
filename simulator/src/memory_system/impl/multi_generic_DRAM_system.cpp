#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "dram_controller/controller.h"
#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"
#include "dram_controller/impl/generic_dram_controller.cpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace Ramulator {

namespace {

template <typename T>
double compute_cv(const std::vector<T>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (mean == 0.0) {
    return 0.0;
  }
  double sq_sum = 0.0;
  for (const auto& value : values) {
    const double delta = static_cast<double>(value) - mean;
    sq_sum += delta * delta;
  }
  const double variance = sq_sum / static_cast<double>(values.size());
  return std::sqrt(variance) / mean;
}

template <typename T>
T compute_p95(std::vector<T> values) {
  if (values.empty()) {
    return T{};
  }
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(std::ceil(static_cast<double>(values.size()) * 0.95)) - 1U;
  return values[std::min(index, values.size() - 1U)];
}

template <typename T>
void emit_yaml_seq(YAML::Emitter& emitter, const std::vector<T>& values) {
  emitter << YAML::BeginSeq;
  for (const auto& value : values) {
    emitter << value;
  }
  emitter << YAML::EndSeq;
}

}  // namespace

class MultiGenericDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, MultiGenericDRAMSystem, "MultiGenericDRAM", "Multiple generic DRAM-based memory system.");

  public:
    struct RankDashboardSnapshot {
      std::vector<uint64_t> rank_total_reqs;
      std::vector<uint64_t> rank_service_cycles;
      std::vector<uint64_t> rank_busy_cycles;
      std::vector<double> rank_avg_queue_len;
      std::vector<uint64_t> rank_max_queue_len;
      std::vector<double> rank_weighted_row_hit_rate;
      std::vector<double> rank_weighted_row_conflict_rate;
      std::vector<uint64_t> act_count_per_rank;
      std::vector<uint32_t> active_ranks_per_epoch;
      double rank_cv_reqs = 0.0;
      double rank_cv_service_cycles = 0.0;
      double hottest_rank_share = 0.0;
      double avg_active_ranks_per_epoch = 0.0;
      uint32_t p95_active_ranks_per_epoch = 0;
    };

  protected:
    Clk_t m_clk = 0;
    std::vector<IDRAM*>  m_drams;
    // IDRAM* m_dram;
    IAddrMapper*  m_addr_mapper;
    uint32_t nMemory;
    std::string assignMethod;
    bool m_dram_access_trace_enable = false;
    std::string m_dram_access_trace_path;
    uint64_t m_dram_access_trace_max_records = 0;
    uint64_t m_dram_access_trace_line_bytes = 64;
    uint64_t s_dram_access_trace_records = 0;
    uint64_t s_dram_access_trace_dropped = 0;
    std::ofstream m_dram_access_trace_file;
    uint64_t m_rank_active_epoch_cycles = 5000;
    uint64_t m_rank_active_epoch_progress = 0;
    std::vector<uint64_t> s_rank_total_reqs;
    std::vector<uint64_t> s_rank_busy_cycles;
    std::vector<uint64_t> s_rank_max_queue_len;
    std::vector<uint8_t> m_rank_active_in_epoch;
    std::vector<uint32_t> s_active_ranks_per_epoch;

  public:
    std::vector<std::vector<IDRAMController*>> m_memctrls;
    int s_num_read_requests = 0;
    int s_num_write_requests = 0;
    int s_num_other_requests = 0;
    size_t hashAddress(Addr_t address) {
        std::hash<Addr_t> hasher;
        size_t hash = hasher(address / 64);
        // Mix up the bits with XOR
        size_t mixedHash = (hash >> 32) ^ (hash & 0xFFFFFFFF);
        return mixedHash;
    }

  public:
    void init() override {
      nMemory = param<uint>("nMemory").required();
      assert(nMemory > 0);
      assignMethod = param<std::string>("assignMethod").required();
      m_dram_access_trace_enable = param<bool>("dramAccessTraceEnable")
                                       .desc("Enable per-request DRAM access trace logging")
                                       .default_val(false);
      m_dram_access_trace_path = param<std::string>("dramAccessTracePath")
                                     .desc("CSV path for per-request DRAM access trace")
                                     .default_val("");
      m_dram_access_trace_max_records = param<uint64_t>("dramAccessTraceMaxRecords")
                                            .desc("Maximum number of DRAM access records to emit (0 = unlimited)")
                                            .default_val(0);
      m_dram_access_trace_line_bytes = param<uint64_t>("dramAccessTraceLineBytes")
                                           .desc("Modeled request payload size in bytes for each accepted DRAM request")
                                           .default_val(64);
      m_rank_active_epoch_cycles = std::max<uint64_t>(
          1ULL,
          param<uint64_t>("rankActiveEpochCycles")
              .desc("Sampling window in cycles for active-rank dashboard stats")
              .default_val(5000));

      for (int i = 0; i < nMemory; i++) {
        IDRAM* m_dram = create_child_ifce<IDRAM>();
        m_dram->m_impl->set_id(fmt::format("DRAM {}", i));
        m_drams.push_back(m_dram);
      }
      // m_dram = create_child_ifce<IDRAM>();
      m_addr_mapper = create_child_ifce<IAddrMapper>();

      int num_channels = m_drams[0]->get_level_size("channel");
      // int num_channels = m_dram->get_level_size("channel");

      // Create memory controllers
      m_memctrls = std::vector<std::vector<IDRAMController*>>(nMemory);
      for (int mem_id = 0; mem_id < nMemory; mem_id++) {
        m_memctrls[mem_id] = std::vector<IDRAMController*>(num_channels);
        for (int i = 0; i < num_channels; i++) {
          IDRAMController* controller = create_child_ifce<IDRAMController>();
          controller->m_impl->set_id(fmt::format("DRAM {}-Channel {}", mem_id, i));
          controller->m_channel_id = i;
          m_memctrls[mem_id][i] = controller;
        }
      }

      m_clock_ratio = param<uint>("clock_ratio").required();

      register_stat(m_clk).name("memory_system_cycles");
      register_stat(s_num_read_requests).name("total_num_read_requests");
      register_stat(s_num_write_requests).name("total_num_write_requests");
      register_stat(s_num_other_requests).name("total_num_other_requests");
      register_stat(s_dram_access_trace_records).name("dram_access_trace_records");
      register_stat(s_dram_access_trace_dropped).name("dram_access_trace_dropped");
      s_rank_total_reqs.assign(nMemory, 0);
      s_rank_busy_cycles.assign(nMemory, 0);
      s_rank_max_queue_len.assign(nMemory, 0);
      m_rank_active_in_epoch.assign(nMemory, 0);
      init_trace_file();
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    bool send(Request req) override {
    if(req.nocallback){
        m_clk++;
        bool is_success = true;
        if (is_success) {
        switch (req.type_id) {
          case Request::Type::Read: {
            s_num_read_requests++;
            break;
          }
          case Request::Type::Write: {
            s_num_write_requests++;
            break;
          }
          default: {
            s_num_other_requests++;
            break;
          }
        }
      }
      return is_success;
      }else{
      uint32_t memId = assignMethod == "source"? req.source_id:
                       assignMethod == "random"? hashAddress(req.addr) % nMemory:
                       assignMethod == "interleave"? (req.addr / 4096) % nMemory:
                       0;
      m_addr_mapper->apply(req);
      req.issue_time = m_clk;
      int channel_id = req.addr_vec[0];
      // printf("addr %lx memId %d channelId %d\n", req.addr, memId, channel_id);
      // fflush(stdout);
      bool is_success;
      if (req.is_anns_graph) {
        // printf("send priority request %lx\n", req.addr);
        // fflush(stdout);
        is_success = m_memctrls[memId][channel_id]->priority_send(req);
      }
      else
        is_success = m_memctrls[memId][channel_id]->send(req);

      if (is_success) {
        emit_trace_record(req, memId);
        if (memId < s_rank_total_reqs.size()) {
          s_rank_total_reqs[memId] += 1;
        }
        switch (req.type_id) {
          case Request::Type::Read: {
            s_num_read_requests++;
            break;
          }
          case Request::Type::Write: {
            s_num_write_requests++;
            break;
          }
          default: {
            s_num_other_requests++;
            break;
          }
        }
      }

      return is_success;
      }
    };

    void tick() override {
      m_clk++;
      for (auto dram: m_drams) {
        dram->tick();
      }
      // m_dram->tick();
      for (size_t mem_id = 0; mem_id < m_memctrls.size(); ++mem_id) {
        bool rank_busy = false;
        uint64_t rank_queue_len = 0;
        for (auto* controller : m_memctrls[mem_id]) {
          controller->tick();
          rank_busy = rank_busy || controller->was_busy_last_cycle();
          rank_queue_len += controller->get_current_queue_len();
        }
        if (rank_busy) {
          s_rank_busy_cycles[mem_id] += 1;
          m_rank_active_in_epoch[mem_id] = 1;
        }
        s_rank_max_queue_len[mem_id] = std::max<uint64_t>(s_rank_max_queue_len[mem_id], rank_queue_len);
      }
      m_rank_active_epoch_progress += 1;
      if (m_rank_active_epoch_progress >= m_rank_active_epoch_cycles) {
        flush_rank_active_epoch();
      }
    };

    float get_tCK() override {
      return m_drams[0]->m_timing_vals("tCK_ps") / 1000.0f;
      // return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    }

    Clk_t get_clk() { return m_clk; }
    int get_num_read() { return s_num_read_requests; }
    int get_num_write() { return s_num_write_requests; }
    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };

    uint32_t get_num_memory() {
      return nMemory;
    }

    AddrVec_t decode_address_vec(Addr_t addr) {
      Request req(addr, Request::Type::Read);
      m_addr_mapper->apply(req);
      return req.addr_vec;
    }

    void finalize() override {
      flush_rank_active_epoch();
      for (auto component : m_components) {
        component->finalize();
      }
      // uint32_t tot_read = 0, tot_write = 0;
      // size_t tot_rd = 0, tot_wr = 0, tot_act = 0, tot_pre = 0;
      // for (uint32_t i = 0; i < nMemory; i++) {
      //   printf("Memory %d read req: %d\twrite req: %d\n", i, m_memory[i]->get_num_read(), m_memory[i]->get_num_write());
      //   printf("Memory %d assigned req: %d\n", i, m_acc_cnt[i]);
      //   printf("Memory %d cmd ACT: %ld\tPRE: %ld\n", i, m_memory[i]->get_num_act(), m_memory[i]->get_num_pre());
      //   printf("Memory %d idle cycle:%ld\n", i, m_memory[i]->get_num_idle_cycle());
      //   tot_read += m_memory[i]->get_num_read();
      //   tot_write += m_memory[i]->get_num_write();
      //   tot_rd += m_memory[i]->get_num_rd();
      //   tot_wr += m_memory[i]->get_num_wr();
      //   tot_act += m_memory[i]->get_num_act();
      //   tot_pre += m_memory[i]->get_num_pre();
      // }
      for(int i = 0; i < nMemory; i++){
        //printf("mdram: %d\n", i);
        fflush(stdout);
      }

      printf("Total Memory cycle %lu\n", m_clk);
      printf("Total Memory read req %d\n", s_num_read_requests);
      printf("Total Memory write req %d\n", s_num_write_requests);
      // printf("Memory cmd ACT %ld\n", tot_act);
      // printf("Memory cmd PRE %ld\n", tot_pre);
      fflush(stdout);
      for(int i = 0; i<nMemory; i++){
        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        m_drams[i]->m_impl->print_stats(emitter);
        emitter << YAML::EndMap;
        std::cout << emitter.c_str() << std::endl;
      }
      for (auto memctrls : m_memctrls) {
        for (auto controller : memctrls) {
          YAML::Emitter emitter;
          emitter << YAML::BeginMap;
          if (auto* generic_controller = dynamic_cast<GenericDRAMController*>(controller)) {
            generic_controller->emit_bank_profile(emitter);
          } else {
            controller->m_impl->print_stats(emitter);
          }
          emitter << YAML::EndMap;
          std::cout << emitter.c_str() << std::endl;
        }
      }
      {
        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        emit_rank_dashboard(emitter);
        emitter << YAML::EndMap;
        std::cout << emitter.c_str() << std::endl;
      }
      if (m_dram_access_trace_file.is_open()) {
        m_dram_access_trace_file.flush();
        m_dram_access_trace_file.close();
      }
      printf("***\n");
      // YAML::Emitter emitter;
      // emitter << YAML::BeginMap;
      // m_impl->print_stats(emitter);
      // emitter << YAML::EndMap;
      // // std::cout << emitter.c_str() << std::endl;
      // std::ofstream file(statFile);
      // if (file.is_open()) {
      //     file << emitter.c_str();
      //     file.close();
      // } else {
      //     m_logger->warn("Failed to write stats to file {}", statFile.c_str());
      // }
    };

    RankDashboardSnapshot capture_rank_dashboard() const {
      RankDashboardSnapshot snapshot;
      snapshot.rank_total_reqs = s_rank_total_reqs;
      snapshot.rank_service_cycles.assign(nMemory, 0);
      snapshot.rank_busy_cycles = s_rank_busy_cycles;
      snapshot.rank_avg_queue_len.assign(nMemory, 0.0);
      snapshot.rank_max_queue_len = s_rank_max_queue_len;
      snapshot.rank_weighted_row_hit_rate.assign(nMemory, 0.0);
      snapshot.rank_weighted_row_conflict_rate.assign(nMemory, 0.0);
      snapshot.act_count_per_rank.assign(nMemory, 0);
      snapshot.active_ranks_per_epoch = s_active_ranks_per_epoch;

      std::vector<double> reqs_as_double(nMemory, 0.0);
      std::vector<double> service_as_double(nMemory, 0.0);
      uint64_t total_rank_reqs = 0;
      uint64_t max_rank_reqs = 0;

      for (size_t mem_id = 0; mem_id < m_memctrls.size(); ++mem_id) {
        uint64_t rank_queue_len_sum = 0;
        uint64_t row_hits = 0;
        uint64_t row_misses = 0;
        uint64_t row_conflicts = 0;
        for (auto* controller : m_memctrls[mem_id]) {
          snapshot.rank_service_cycles[mem_id] += controller->get_total_service_cycles();
          rank_queue_len_sum += controller->get_queue_len_sum();
          row_hits += controller->get_row_hits();
          row_misses += controller->get_row_misses();
          row_conflicts += controller->get_row_conflicts();
          snapshot.act_count_per_rank[mem_id] += controller->get_act_count();
        }
        snapshot.rank_avg_queue_len[mem_id] =
            m_clk == 0 ? 0.0 : static_cast<double>(rank_queue_len_sum) / static_cast<double>(m_clk);
        const uint64_t row_total = row_hits + row_misses + row_conflicts;
        snapshot.rank_weighted_row_hit_rate[mem_id] =
            row_total == 0 ? 0.0 : static_cast<double>(row_hits) / static_cast<double>(row_total);
        snapshot.rank_weighted_row_conflict_rate[mem_id] =
            row_total == 0 ? 0.0 : static_cast<double>(row_conflicts) / static_cast<double>(row_total);
        reqs_as_double[mem_id] = static_cast<double>(snapshot.rank_total_reqs[mem_id]);
        service_as_double[mem_id] = static_cast<double>(snapshot.rank_service_cycles[mem_id]);
        total_rank_reqs += snapshot.rank_total_reqs[mem_id];
        max_rank_reqs = std::max<uint64_t>(max_rank_reqs, snapshot.rank_total_reqs[mem_id]);
      }

      snapshot.rank_cv_reqs = compute_cv(reqs_as_double);
      snapshot.rank_cv_service_cycles = compute_cv(service_as_double);
      snapshot.hottest_rank_share =
          total_rank_reqs == 0 ? 0.0 : static_cast<double>(max_rank_reqs) / static_cast<double>(total_rank_reqs);
      if (!snapshot.active_ranks_per_epoch.empty()) {
        const double active_sum =
            std::accumulate(snapshot.active_ranks_per_epoch.begin(), snapshot.active_ranks_per_epoch.end(), 0.0);
        snapshot.avg_active_ranks_per_epoch =
            active_sum / static_cast<double>(snapshot.active_ranks_per_epoch.size());
        snapshot.p95_active_ranks_per_epoch = compute_p95(snapshot.active_ranks_per_epoch);
      }
      return snapshot;
    }

    void emit_rank_dashboard(YAML::Emitter& emitter) const {
      const RankDashboardSnapshot snapshot = capture_rank_dashboard();
      std::vector<double> act_count_as_double(snapshot.act_count_per_rank.size(), 0.0);
      uint64_t act_count_total = 0;
      uint32_t hottest_rank_id = 0;
      uint64_t hottest_rank_reqs = 0;
      for (size_t idx = 0; idx < snapshot.act_count_per_rank.size(); ++idx) {
        act_count_total += snapshot.act_count_per_rank[idx];
        act_count_as_double[idx] = static_cast<double>(snapshot.act_count_per_rank[idx]);
        if (idx < snapshot.rank_total_reqs.size() && snapshot.rank_total_reqs[idx] > hottest_rank_reqs) {
          hottest_rank_reqs = snapshot.rank_total_reqs[idx];
          hottest_rank_id = static_cast<uint32_t>(idx);
        }
      }
      emitter << YAML::Key << "rank_active_epoch_cycles" << YAML::Value << m_rank_active_epoch_cycles;
      emitter << YAML::Key << "rank_cv_reqs" << YAML::Value << snapshot.rank_cv_reqs;
      emitter << YAML::Key << "rank_cv_service_cycles" << YAML::Value << snapshot.rank_cv_service_cycles;
      emitter << YAML::Key << "hottest_rank_share" << YAML::Value << snapshot.hottest_rank_share;
      emitter << YAML::Key << "hottest_rank_id" << YAML::Value << hottest_rank_id;
      emitter << YAML::Key << "hottest_rank_reqs" << YAML::Value << hottest_rank_reqs;
      emitter << YAML::Key << "avg_active_ranks_per_epoch" << YAML::Value << snapshot.avg_active_ranks_per_epoch;
      emitter << YAML::Key << "p95_active_ranks_per_epoch" << YAML::Value << snapshot.p95_active_ranks_per_epoch;
      emitter << YAML::Key << "act_count_total" << YAML::Value << act_count_total;
      emitter << YAML::Key << "act_count_per_rank_cv" << YAML::Value << compute_cv(act_count_as_double);
    }

  private:
    void flush_rank_active_epoch() {
      if (m_rank_active_epoch_progress == 0) {
        return;
      }
      uint32_t active_count = 0;
      for (uint8_t active : m_rank_active_in_epoch) {
        active_count += active ? 1U : 0U;
      }
      s_active_ranks_per_epoch.push_back(active_count);
      std::fill(m_rank_active_in_epoch.begin(), m_rank_active_in_epoch.end(), static_cast<uint8_t>(0));
      m_rank_active_epoch_progress = 0;
    }

    void init_trace_file() {
      if (!m_dram_access_trace_enable) {
        return;
      }
      if (m_dram_access_trace_path.empty()) {
        m_dram_access_trace_path = "dram_access_trace.csv";
      }
      std::filesystem::path trace_path(m_dram_access_trace_path);
      if (trace_path.has_parent_path()) {
        std::filesystem::create_directories(trace_path.parent_path());
      }
      m_dram_access_trace_file.open(trace_path, std::ios::out | std::ios::trunc);
      if (!m_dram_access_trace_file.is_open()) {
        throw ConfigurationError("Failed to open dram access trace path {}", m_dram_access_trace_path);
      }
      m_dram_access_trace_file
          << "trace_idx,mem_clk,mem_id,addr_hex,size_bytes,req_type,source_id,is_graph,dualq_phase,disreq_id,anns_id,cand_id,cur_dim,cur_bit,"
          << "graph_level,graph_segment,channel,rank,bankgroup,bank,row,col\n";
    }

    int get_level_index(const std::string& level_name) const {
      try {
        return m_drams[0]->m_levels(level_name);
      } catch (const std::out_of_range&) {
        return -1;
      }
    }

    int get_level_value(const Request& req, int level_index) const {
      if (level_index < 0 || static_cast<size_t>(level_index) >= req.addr_vec.size()) {
        return -1;
      }
      return req.addr_vec[level_index];
    }

    const char* get_req_type_name(int type_id) const {
      switch (type_id) {
        case Request::Type::Read:
          return "read";
        case Request::Type::Write:
          return "write";
        default:
          return "other";
      }
    }

    void emit_trace_record(const Request& req, uint32_t mem_id) {
      if (!m_dram_access_trace_enable || req.nocallback) {
        return;
      }
      if (m_dram_access_trace_max_records != 0 && s_dram_access_trace_records >= m_dram_access_trace_max_records) {
        s_dram_access_trace_dropped += 1;
        return;
      }
      const int rank_idx = get_level_index("rank");
      const int bg_idx = get_level_index("bankgroup");
      const int bank_idx = get_level_index("bank");
      const int row_idx = get_level_index("row");
      const int col_idx = get_level_index("column");
      m_dram_access_trace_file
          << s_dram_access_trace_records << ','
          << m_clk << ','
          << mem_id << ','
          << fmt::format("0x{:x}", static_cast<uint64_t>(req.addr)) << ','
          << m_dram_access_trace_line_bytes << ','
          << get_req_type_name(req.type_id) << ','
          << req.source_id << ','
          << (req.is_anns_graph ? 1 : 0) << ','
          << req.anns.phase << ','
          << req.anns.disreqId << ','
          << req.anns.annsId << ','
          << req.anns.candId << ','
          << req.anns.curDim << ','
          << req.anns.curBit << ','
          << req.trav.queryLevel << ','
          << req.trav.segmentId << ','
          << get_level_value(req, 0) << ','
          << get_level_value(req, rank_idx) << ','
          << get_level_value(req, bg_idx) << ','
          << get_level_value(req, bank_idx) << ','
          << get_level_value(req, row_idx) << ','
          << get_level_value(req, col_idx)
          << '\n';
      s_dram_access_trace_records += 1;
    }

};

}   // namespace
