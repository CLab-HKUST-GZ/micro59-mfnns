#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "dram_controller/controller.h"
#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"
#include <filesystem>
#include <fstream>

namespace Ramulator {

class GenericDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, GenericDRAMSystem, "GenericDRAM", "A generic DRAM-based memory system.");

  protected:
    Clk_t m_clk = 0;
    IDRAM*  m_dram;
    IAddrMapper*  m_addr_mapper;
    std::vector<IDRAMController*> m_controllers;
    bool m_dram_access_trace_enable = false;
    std::string m_dram_access_trace_path;
    uint64_t m_dram_access_trace_max_records = 0;
    uint64_t m_dram_access_trace_line_bytes = 64;
    uint64_t s_dram_access_trace_records = 0;
    uint64_t s_dram_access_trace_dropped = 0;
    std::ofstream m_dram_access_trace_file;

  public:
    int s_num_read_requests = 0;
    int s_num_write_requests = 0;
    int s_num_other_requests = 0;


  public:
    void init() override { 
      // Create device (a top-level node wrapping all channel nodes)
      m_dram = create_child_ifce<IDRAM>();
      m_addr_mapper = create_child_ifce<IAddrMapper>();

      int num_channels = m_dram->get_level_size("channel");   

      // Create memory controllers
      for (int i = 0; i < num_channels; i++) {
        IDRAMController* controller = create_child_ifce<IDRAMController>();
        controller->m_impl->set_id(fmt::format("{}-Channel {}", m_id.c_str(), i));
        controller->m_channel_id = i;
        m_controllers.push_back(controller);
      }

      m_clock_ratio = param<uint>("clock_ratio").required();
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

      register_stat(m_clk).name("memory_system_cycles");
      register_stat(s_num_read_requests).name("total_num_read_requests");
      register_stat(s_num_write_requests).name("total_num_write_requests");
      register_stat(s_num_other_requests).name("total_num_other_requests");
      register_stat(s_dram_access_trace_records).name("dram_access_trace_records");
      register_stat(s_dram_access_trace_dropped).name("dram_access_trace_dropped");
      init_trace_file();
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    bool send(Request req) override {
      m_addr_mapper->apply(req);
      req.issue_time = m_clk;
      int channel_id = req.addr_vec[0];
      bool is_success = m_controllers[channel_id]->send(req);

      if (is_success) {
        emit_trace_record(req, /* mem_id */ 0);
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
        // printf("[%s] cycle %ld receive memory request addr %lx\n", get_id().c_str(), m_clk, req.addr);
        // fflush(stdout);
      }

      return is_success;
    };
    
    void tick() override {
      m_clk++;
      m_dram->tick();
      // printf("[%s] tick dram_system cycle %ld\n", get_id().c_str(), m_clk);
      for (auto controller : m_controllers) {
        controller->tick();
      }
    };

    float get_tCK() override {
      return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    }

    Clk_t get_clk() { return m_clk; }
    int get_num_read() { return s_num_read_requests; }
    int get_num_write() { return s_num_write_requests; }
    size_t get_num_act() {
      size_t num_act = 0;
      for (auto controller : m_controllers) {
        num_act += controller->s_num_act;
      }
      return num_act;
    }
    size_t get_num_pre() {
      size_t num_pre = 0;
      for (auto controller : m_controllers) {
        num_pre += controller->s_num_pre;
      }
      return num_pre;
    }
    size_t get_num_rd() {
      size_t num_rd = 0;
      for (auto controller : m_controllers) {
        num_rd += controller->s_num_rd;
      }
      return num_rd;
    }
    size_t get_num_wr() {
      size_t num_wr = 0;
      for (auto controller : m_controllers) {
        num_wr += controller->s_num_wr;
      }
      return num_wr;
    }
    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };

    void finalize() override {
      if (m_dram_access_trace_file.is_open()) {
        m_dram_access_trace_file.flush();
        m_dram_access_trace_file.close();
      }
    }

  private:
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
        return m_dram->m_levels(level_name);
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

