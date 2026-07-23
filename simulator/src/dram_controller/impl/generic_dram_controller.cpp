#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class GenericDRAMController final : public IDRAMController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, GenericDRAMController, "Generic", "A generic DRAM controller.");
  private:
    std::deque<Request> pending;          // A queue for read requests that are about to finish (callback after RL)

    ReqBuffer m_active_buffer;            // Buffer for requests being served. This has the highest priority 
    ReqBuffer m_priority_buffer;          // Buffer for high-priority requests (e.g., maintenance like refresh).
    ReqBuffer m_read_buffer;              // Read request buffer
    ReqBuffer m_write_buffer;             // Write request buffer

    int m_rank_addr_idx = -1;
    int m_bankgroup_addr_idx = -1;
    int m_bank_addr_idx = -1;

    size_t m_num_ranks = 1;
    size_t m_num_bankgroups = 1;
    size_t m_num_banks = 1;
    size_t m_num_bankgroup_slots = 1;
    size_t m_num_bank_slots = 1;

    float m_wr_low_watermark;
    float m_wr_high_watermark;
    bool  m_is_write_mode = false;

    size_t s_row_hits = 0;
    size_t s_row_misses = 0;
    size_t s_row_conflicts = 0;
    size_t s_read_row_hits = 0;
    size_t s_read_row_misses = 0;
    size_t s_read_row_conflicts = 0;
    size_t s_write_row_hits = 0;
    size_t s_write_row_misses = 0;
    size_t s_write_row_conflicts = 0;

    size_t m_num_cores = 0;
    std::vector<size_t> s_read_row_hits_per_core;
    std::vector<size_t> s_read_row_misses_per_core;
    std::vector<size_t> s_read_row_conflicts_per_core;

    size_t s_num_read_reqs = 0;
    size_t s_num_write_reqs = 0;
    size_t s_num_other_reqs = 0;
    uint64_t s_service_cycles = 0;
    uint64_t s_busy_cycles = 0;
    size_t s_queue_len = 0;
    size_t s_read_queue_len = 0;
    size_t s_write_queue_len = 0;
    size_t s_priority_queue_len = 0;
    size_t s_queue_len_max = 0;
    float s_queue_len_avg = 0;
    float s_read_queue_len_avg = 0;
    float s_write_queue_len_avg = 0;
    float s_priority_queue_len_avg = 0;

    std::vector<size_t> s_bank_req_distribution;
    std::vector<size_t> s_bank_row_hits;
    std::vector<size_t> s_bank_row_misses;
    std::vector<size_t> s_bank_row_conflicts;
    std::vector<size_t> s_bank_queue_len_sum;
    std::vector<float> s_bank_queue_len_avg;
    std::vector<float> s_bank_row_hit_rate;

    std::vector<size_t> s_bankgroup_req_distribution;
    std::vector<size_t> s_bankgroup_row_hits;
    std::vector<size_t> s_bankgroup_row_misses;
    std::vector<size_t> s_bankgroup_row_conflicts;
    std::vector<size_t> s_bankgroup_queue_len_sum;
    std::vector<float> s_bankgroup_queue_len_avg;
    std::vector<float> s_bankgroup_row_hit_rate;

    std::vector<size_t> s_dualq_phase1_bank_req_distribution;
    std::vector<size_t> s_dualq_phase1_bank_row_hits;
    std::vector<size_t> s_dualq_phase1_bank_row_misses;
    std::vector<size_t> s_dualq_phase1_bank_row_conflicts;
    std::vector<size_t> s_dualq_phase2_bank_req_distribution;
    std::vector<size_t> s_dualq_phase2_bank_row_hits;
    std::vector<size_t> s_dualq_phase2_bank_row_misses;
    std::vector<size_t> s_dualq_phase2_bank_row_conflicts;

    std::vector<size_t> s_dualq_phase1_bankgroup_req_distribution;
    std::vector<size_t> s_dualq_phase1_bankgroup_row_hits;
    std::vector<size_t> s_dualq_phase1_bankgroup_row_misses;
    std::vector<size_t> s_dualq_phase1_bankgroup_row_conflicts;
    std::vector<size_t> s_dualq_phase2_bankgroup_req_distribution;
    std::vector<size_t> s_dualq_phase2_bankgroup_row_hits;
    std::vector<size_t> s_dualq_phase2_bankgroup_row_misses;
    std::vector<size_t> s_dualq_phase2_bankgroup_row_conflicts;

    std::vector<size_t> m_bank_queue_occupancy;
    std::vector<size_t> m_bankgroup_queue_occupancy;

    size_t s_read_latency = 0;
    float s_avg_read_latency = 0;
    bool m_was_busy_last_cycle = false;


  public:

    void emit_bank_profile(YAML::Emitter& emitter) const {
      emitter << YAML::Key << "Controller";
      emitter << YAML::Value;
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "impl" << YAML::Value << get_name();
      if (get_id() != "_default_id") {
        emitter << YAML::Key << "id" << YAML::Value << get_id();
      }
      emitter << YAML::Key << fmt::format("num_read_reqs_{}", m_channel_id) << YAML::Value << s_num_read_reqs;
      emitter << YAML::Key << fmt::format("num_write_reqs_{}", m_channel_id) << YAML::Value << s_num_write_reqs;
      emitter << YAML::Key << fmt::format("num_other_reqs_{}", m_channel_id) << YAML::Value << s_num_other_reqs;
      emitter << YAML::Key << fmt::format("service_cycles_{}", m_channel_id) << YAML::Value << s_service_cycles;
      emitter << YAML::Key << fmt::format("busy_cycles_{}", m_channel_id) << YAML::Value << s_busy_cycles;
      emitter << YAML::Key << fmt::format("queue_len_avg_{}", m_channel_id) << YAML::Value << s_queue_len_avg;
      emitter << YAML::Key << fmt::format("queue_len_max_{}", m_channel_id) << YAML::Value << s_queue_len_max;
      emitter << YAML::Key << fmt::format("read_queue_len_avg_{}", m_channel_id) << YAML::Value << s_read_queue_len_avg;
      emitter << YAML::Key << fmt::format("write_queue_len_avg_{}", m_channel_id) << YAML::Value << s_write_queue_len_avg;
      emitter << YAML::Key << fmt::format("priority_queue_len_avg_{}", m_channel_id) << YAML::Value << s_priority_queue_len_avg;
      emitter << YAML::Key << fmt::format("row_hits_{}", m_channel_id) << YAML::Value << s_row_hits;
      emitter << YAML::Key << fmt::format("row_misses_{}", m_channel_id) << YAML::Value << s_row_misses;
      emitter << YAML::Key << fmt::format("row_conflicts_{}", m_channel_id) << YAML::Value << s_row_conflicts;
      for (size_t rank_id = 0; rank_id < m_num_ranks; rank_id++) {
        for (size_t bankgroup_id = 0; bankgroup_id < m_num_bankgroups; bankgroup_id++) {
          const size_t bankgroup_slot = rank_id * m_num_bankgroups + bankgroup_id;
          emitter << YAML::Key << fmt::format("bankgroup_req_dist_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_req_distribution[bankgroup_slot];
          emitter << YAML::Key << fmt::format("bankgroup_row_hits_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_row_hits[bankgroup_slot];
          emitter << YAML::Key << fmt::format("bankgroup_row_misses_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_row_misses[bankgroup_slot];
          emitter << YAML::Key << fmt::format("bankgroup_row_conflicts_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_row_conflicts[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase1_bankgroup_req_dist_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase1_bankgroup_req_distribution[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase1_bankgroup_row_hits_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase1_bankgroup_row_hits[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase1_bankgroup_row_misses_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase1_bankgroup_row_misses[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase1_bankgroup_row_conflicts_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase1_bankgroup_row_conflicts[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase2_bankgroup_req_dist_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase2_bankgroup_req_distribution[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase2_bankgroup_row_hits_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase2_bankgroup_row_hits[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase2_bankgroup_row_misses_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase2_bankgroup_row_misses[bankgroup_slot];
          emitter << YAML::Key << fmt::format("dualq_phase2_bankgroup_row_conflicts_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_dualq_phase2_bankgroup_row_conflicts[bankgroup_slot];
          emitter << YAML::Key << fmt::format("bankgroup_queue_len_sum_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_queue_len_sum[bankgroup_slot];
          emitter << YAML::Key << fmt::format("bankgroup_queue_len_avg_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_queue_len_avg[bankgroup_slot];
          emitter << YAML::Key << fmt::format("bankgroup_row_hit_rate_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id) << YAML::Value << s_bankgroup_row_hit_rate[bankgroup_slot];
          for (size_t bank_id = 0; bank_id < m_num_banks; bank_id++) {
            const size_t bank_slot = bankgroup_slot * m_num_banks + bank_id;
            emitter << YAML::Key << fmt::format("bank_req_dist_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_req_distribution[bank_slot];
            emitter << YAML::Key << fmt::format("bank_row_hits_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_row_hits[bank_slot];
            emitter << YAML::Key << fmt::format("bank_row_misses_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_row_misses[bank_slot];
            emitter << YAML::Key << fmt::format("bank_row_conflicts_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_row_conflicts[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase1_bank_req_dist_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase1_bank_req_distribution[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase1_bank_row_hits_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase1_bank_row_hits[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase1_bank_row_misses_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase1_bank_row_misses[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase1_bank_row_conflicts_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase1_bank_row_conflicts[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase2_bank_req_dist_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase2_bank_req_distribution[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase2_bank_row_hits_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase2_bank_row_hits[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase2_bank_row_misses_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase2_bank_row_misses[bank_slot];
            emitter << YAML::Key << fmt::format("dualq_phase2_bank_row_conflicts_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_dualq_phase2_bank_row_conflicts[bank_slot];
            emitter << YAML::Key << fmt::format("bank_queue_len_sum_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_queue_len_sum[bank_slot];
            emitter << YAML::Key << fmt::format("bank_queue_len_avg_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_queue_len_avg[bank_slot];
            emitter << YAML::Key << fmt::format("bank_row_hit_rate_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id) << YAML::Value << s_bank_row_hit_rate[bank_slot];
          }
        }
      }
      emitter << YAML::EndMap;
      emitter << YAML::Newline;
    }

    bool req_buffer_full(Request& req) {

      if (req.type_id == Request::Type::Read) {
        return m_read_buffer.full();
      }
      return m_write_buffer.full();
    }

    void init() override {
      m_wr_low_watermark =  param<float>("wr_low_watermark").desc("Threshold for switching back to read mode.").default_val(0.2f);
      m_wr_high_watermark = param<float>("wr_high_watermark").desc("Threshold for switching to write mode.").default_val(0.8f);

      m_scheduler = create_child_ifce<IScheduler>();
      m_refresh = create_child_ifce<IRefreshManager>();    
      m_rowpolicy = create_child_ifce<IRowPolicy>();    

      if (m_config["plugins"]) {
        YAML::Node plugin_configs = m_config["plugins"];
        for (YAML::iterator it = plugin_configs.begin(); it != plugin_configs.end(); ++it) {
          m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
        }
      }

      register_stat(s_num_act).name("s_num_act");
      register_stat(s_num_pre).name("s_num_pre");
      register_stat(s_num_rd).name("s_num_rd");
      register_stat(s_num_wr).name("s_num_wr");
      // register_stat(s_acc_lat_bin).name("s_acc_lat_bin");
      s_acc_lat_bin.resize(10);
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      //std::string desired_dram_id = "DRAM " + std::to_string(1);  // 根据通道 ID 构造 DRAM 标识符
      
      std::string id_str = get_id().c_str();
      std::string desired_dram_id;
      size_t pos = id_str.find("DRAM ");
      if (pos != std::string::npos) {
          size_t start = pos + 5; 
          size_t end = id_str.find_first_not_of("0123456789", start);
          std::string dram_number = id_str.substr(start, end - start);
          desired_dram_id = "DRAM " + dram_number;
          //std::cout << "Extracted DRAM ID: " << desired_dram_id << std::endl;
      } else {
          std::cerr << "Error: DRAM ID not found!" << std::endl;
      }

      //printf("memorysystem: %s\n", *memory_system->get_id().c_str());
      m_dram = memory_system->get_ifce<IDRAM>(desired_dram_id);  // 根据通道 ID 获取 DRAM 实例
      // m_dram = memory_system->get_ifce<IDRAM>("1");
      try {
        m_rank_addr_idx = m_dram->m_levels("rank");
      } catch (const std::out_of_range&) {
        m_rank_addr_idx = -1;
      }
      try {
        m_bankgroup_addr_idx = m_dram->m_levels("bankgroup");
      } catch (const std::out_of_range&) {
        m_bankgroup_addr_idx = -1;
      }
      try {
        m_bank_addr_idx = m_dram->m_levels("bank");
      } catch (const std::out_of_range&) {
        m_bank_addr_idx = -1;
      }
      m_num_ranks = m_dram->get_level_size("rank") > 0 ? static_cast<size_t>(m_dram->get_level_size("rank")) : 1;
      m_num_bankgroups = m_dram->get_level_size("bankgroup") > 0 ? static_cast<size_t>(m_dram->get_level_size("bankgroup")) : 1;
      m_num_banks = m_dram->get_level_size("bank") > 0 ? static_cast<size_t>(m_dram->get_level_size("bank")) : 1;
      m_num_bankgroup_slots = m_num_ranks * m_num_bankgroups;
      m_num_bank_slots = m_num_bankgroup_slots * m_num_banks;
      m_priority_buffer.max_size = 512*3 + 32;

      m_num_cores = frontend->get_num_cores();

      s_read_row_hits_per_core.resize(m_num_cores, 0);
      s_read_row_misses_per_core.resize(m_num_cores, 0);
      s_read_row_conflicts_per_core.resize(m_num_cores, 0);

      s_bank_req_distribution.resize(m_num_bank_slots, 0);
      s_bank_row_hits.resize(m_num_bank_slots, 0);
      s_bank_row_misses.resize(m_num_bank_slots, 0);
      s_bank_row_conflicts.resize(m_num_bank_slots, 0);
      s_bank_queue_len_sum.resize(m_num_bank_slots, 0);
      s_bank_queue_len_avg.resize(m_num_bank_slots, 0.0f);
      s_bank_row_hit_rate.resize(m_num_bank_slots, 0.0f);

      s_bankgroup_req_distribution.resize(m_num_bankgroup_slots, 0);
      s_bankgroup_row_hits.resize(m_num_bankgroup_slots, 0);
      s_bankgroup_row_misses.resize(m_num_bankgroup_slots, 0);
      s_bankgroup_row_conflicts.resize(m_num_bankgroup_slots, 0);
      s_bankgroup_queue_len_sum.resize(m_num_bankgroup_slots, 0);
      s_bankgroup_queue_len_avg.resize(m_num_bankgroup_slots, 0.0f);
      s_bankgroup_row_hit_rate.resize(m_num_bankgroup_slots, 0.0f);

      s_dualq_phase1_bank_req_distribution.resize(m_num_bank_slots, 0);
      s_dualq_phase1_bank_row_hits.resize(m_num_bank_slots, 0);
      s_dualq_phase1_bank_row_misses.resize(m_num_bank_slots, 0);
      s_dualq_phase1_bank_row_conflicts.resize(m_num_bank_slots, 0);
      s_dualq_phase2_bank_req_distribution.resize(m_num_bank_slots, 0);
      s_dualq_phase2_bank_row_hits.resize(m_num_bank_slots, 0);
      s_dualq_phase2_bank_row_misses.resize(m_num_bank_slots, 0);
      s_dualq_phase2_bank_row_conflicts.resize(m_num_bank_slots, 0);

      s_dualq_phase1_bankgroup_req_distribution.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase1_bankgroup_row_hits.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase1_bankgroup_row_misses.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase1_bankgroup_row_conflicts.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase2_bankgroup_req_distribution.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase2_bankgroup_row_hits.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase2_bankgroup_row_misses.resize(m_num_bankgroup_slots, 0);
      s_dualq_phase2_bankgroup_row_conflicts.resize(m_num_bankgroup_slots, 0);

      m_bank_queue_occupancy.resize(m_num_bank_slots, 0);
      m_bankgroup_queue_occupancy.resize(m_num_bankgroup_slots, 0);

      register_stat(s_row_hits).name("row_hits_{}", m_channel_id);
      register_stat(s_row_misses).name("row_misses_{}", m_channel_id);
      register_stat(s_row_conflicts).name("row_conflicts_{}", m_channel_id);
      register_stat(s_read_row_hits).name("read_row_hits_{}", m_channel_id);
      register_stat(s_read_row_misses).name("read_row_misses_{}", m_channel_id);
      register_stat(s_read_row_conflicts).name("read_row_conflicts_{}", m_channel_id);
      register_stat(s_write_row_hits).name("write_row_hits_{}", m_channel_id);
      register_stat(s_write_row_misses).name("write_row_misses_{}", m_channel_id);
      register_stat(s_write_row_conflicts).name("write_row_conflicts_{}", m_channel_id);

      for (size_t core_id = 0; core_id < m_num_cores; core_id++) {
        register_stat(s_read_row_hits_per_core[core_id]).name("read_row_hits_core_{}", core_id);
        register_stat(s_read_row_misses_per_core[core_id]).name("read_row_misses_core_{}", core_id);
        register_stat(s_read_row_conflicts_per_core[core_id]).name("read_row_conflicts_core_{}", core_id);
      }

      register_stat(s_num_read_reqs).name("num_read_reqs_{}", m_channel_id);
      register_stat(s_num_write_reqs).name("num_write_reqs_{}", m_channel_id);
      register_stat(s_num_other_reqs).name("num_other_reqs_{}", m_channel_id);
      register_stat(s_service_cycles).name("service_cycles_{}", m_channel_id);
      register_stat(s_busy_cycles).name("busy_cycles_{}", m_channel_id);
      register_stat(s_queue_len).name("queue_len_{}", m_channel_id);
      register_stat(s_queue_len_max).name("queue_len_max_{}", m_channel_id);
      register_stat(s_read_queue_len).name("read_queue_len_{}", m_channel_id);
      register_stat(s_write_queue_len).name("write_queue_len_{}", m_channel_id);
      register_stat(s_priority_queue_len).name("priority_queue_len_{}", m_channel_id);
      register_stat(s_queue_len_avg).name("queue_len_avg_{}", m_channel_id);
      register_stat(s_read_queue_len_avg).name("read_queue_len_avg_{}", m_channel_id);
      register_stat(s_write_queue_len_avg).name("write_queue_len_avg_{}", m_channel_id);
      register_stat(s_priority_queue_len_avg).name("priority_queue_len_avg_{}", m_channel_id);

      register_stat(s_read_latency).name("read_latency_{}", m_channel_id);
      register_stat(s_avg_read_latency).name("avg_read_latency_{}", m_channel_id);

      for (size_t rank_id = 0; rank_id < m_num_ranks; rank_id++) {
        for (size_t bankgroup_id = 0; bankgroup_id < m_num_bankgroups; bankgroup_id++) {
          size_t bankgroup_slot = rank_id * m_num_bankgroups + bankgroup_id;
          register_stat(s_bankgroup_req_distribution[bankgroup_slot]).name("bankgroup_req_dist_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_bankgroup_row_hits[bankgroup_slot]).name("bankgroup_row_hits_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_bankgroup_row_misses[bankgroup_slot]).name("bankgroup_row_misses_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_bankgroup_row_conflicts[bankgroup_slot]).name("bankgroup_row_conflicts_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase1_bankgroup_req_distribution[bankgroup_slot]).name("dualq_phase1_bankgroup_req_dist_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase1_bankgroup_row_hits[bankgroup_slot]).name("dualq_phase1_bankgroup_row_hits_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase1_bankgroup_row_misses[bankgroup_slot]).name("dualq_phase1_bankgroup_row_misses_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase1_bankgroup_row_conflicts[bankgroup_slot]).name("dualq_phase1_bankgroup_row_conflicts_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase2_bankgroup_req_distribution[bankgroup_slot]).name("dualq_phase2_bankgroup_req_dist_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase2_bankgroup_row_hits[bankgroup_slot]).name("dualq_phase2_bankgroup_row_hits_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase2_bankgroup_row_misses[bankgroup_slot]).name("dualq_phase2_bankgroup_row_misses_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_dualq_phase2_bankgroup_row_conflicts[bankgroup_slot]).name("dualq_phase2_bankgroup_row_conflicts_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_bankgroup_queue_len_sum[bankgroup_slot]).name("bankgroup_queue_len_sum_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_bankgroup_queue_len_avg[bankgroup_slot]).name("bankgroup_queue_len_avg_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          register_stat(s_bankgroup_row_hit_rate[bankgroup_slot]).name("bankgroup_row_hit_rate_ch{}_rk{}_bg{}", m_channel_id, rank_id, bankgroup_id);
          for (size_t bank_id = 0; bank_id < m_num_banks; bank_id++) {
            size_t bank_slot = bankgroup_slot * m_num_banks + bank_id;
            register_stat(s_bank_req_distribution[bank_slot]).name("bank_req_dist_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_bank_row_hits[bank_slot]).name("bank_row_hits_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_bank_row_misses[bank_slot]).name("bank_row_misses_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_bank_row_conflicts[bank_slot]).name("bank_row_conflicts_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase1_bank_req_distribution[bank_slot]).name("dualq_phase1_bank_req_dist_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase1_bank_row_hits[bank_slot]).name("dualq_phase1_bank_row_hits_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase1_bank_row_misses[bank_slot]).name("dualq_phase1_bank_row_misses_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase1_bank_row_conflicts[bank_slot]).name("dualq_phase1_bank_row_conflicts_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase2_bank_req_distribution[bank_slot]).name("dualq_phase2_bank_req_dist_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase2_bank_row_hits[bank_slot]).name("dualq_phase2_bank_row_hits_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase2_bank_row_misses[bank_slot]).name("dualq_phase2_bank_row_misses_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_dualq_phase2_bank_row_conflicts[bank_slot]).name("dualq_phase2_bank_row_conflicts_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_bank_queue_len_sum[bank_slot]).name("bank_queue_len_sum_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_bank_queue_len_avg[bank_slot]).name("bank_queue_len_avg_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
            register_stat(s_bank_row_hit_rate[bank_slot]).name("bank_row_hit_rate_ch{}_rk{}_bg{}_b{}", m_channel_id, rank_id, bankgroup_id, bank_id);
          }
        }
      }
    };

    bool send(Request& req) override {
      req.final_command = m_dram->m_request_translations(req.type_id);
      req.n_command = 0;
      req.command_history = "";

      switch (req.type_id) {
        case Request::Type::Read: {
          s_num_read_reqs++;
          break;
        }
        case Request::Type::Write: {
          s_num_write_reqs++;
          break;
        }
        default: {
          s_num_other_reqs++;
          break;
        }
      }

      // Forward existing write requests to incoming read requests
      if (req.type_id == Request::Type::Read) {
        auto compare_addr = [req](const Request& wreq) {
          return wreq.addr == req.addr;
        };
        if (std::find_if(m_write_buffer.begin(), m_write_buffer.end(), compare_addr) != m_write_buffer.end()) {
          // The request will depart at the next cycle
          req.depart = m_clk + 1;
          pending.push_back(req);
          return true;
        }
      }

      // Else, enqueue them to corresponding buffer based on request type id
      bool is_success = false;
      req.arrive = m_clk;
      if        (req.type_id == Request::Type::Read) {
        is_success = m_read_buffer.enqueue(req);
      } else if (req.type_id == Request::Type::Write) {
        is_success = m_write_buffer.enqueue(req);
      } else {
        throw std::runtime_error("Invalid request type!");
      }
      if (!is_success) {
        // We could not enqueue the request
        req.arrive = -1;
        return false;
      }

      enqueue_bank_queue_counters(req);
      return true;
    };

    bool priority_send(Request& req) override {
      if (req.final_command >= 0) {
      } else if (req.type_id >= 0 && req.type_id < static_cast<int>(m_dram->m_requests.size())) {
        req.final_command = m_dram->m_request_translations(req.type_id);
      } else {
        req.final_command = req.type_id;
      }
      req.n_command = 0;
      req.command_history = "";
      req.arrive = m_clk;

      bool is_success = false;
      is_success = m_priority_buffer.enqueue(req);
      if (is_success) {
        enqueue_bank_queue_counters(req);
      } else {
        req.arrive = -1;
      }
      return is_success;
    }

    void tick() override {
      m_clk++;
      // printf("[%s] tick dram_controller cycle %ld\n", get_id().c_str(), m_clk);

      // Update statistics
      const size_t sampled_queue_len = get_sampled_queue_len();
      s_queue_len += sampled_queue_len;
      s_read_queue_len += m_read_buffer.size() + pending.size();
      s_write_queue_len += m_write_buffer.size();
      s_priority_queue_len += m_priority_buffer.size();
      s_queue_len_max = std::max(s_queue_len_max, sampled_queue_len);
      m_was_busy_last_cycle = has_any_outstanding_work();
      if (m_was_busy_last_cycle) {
        s_busy_cycles += 1;
      }
      accumulate_bank_queue_stats();

      // 1. Serve completed reads
      serve_completed_reads();

      m_refresh->tick();

      // 2. Try to find a request to serve.
      ReqBuffer::iterator req_it;
      ReqBuffer* buffer = nullptr;
      bool request_found = schedule_request(req_it, buffer);

      // 2.1 Take row policy action
      m_rowpolicy->update(request_found, req_it);

      // 3. Update all plugins
      for (auto plugin : m_plugins) {
        plugin->update(request_found, req_it);
      }

      // 4. Finally, issue the commands to serve the request
      if (request_found) {
        // If we find a real request to serve
        auto command = m_dram->m_commands(req_it->command);
        //printf("[%s] cycle %ld issue command %s addr %lx\n", get_id().c_str(), m_clk, command.data(), req_it->addr);
        //printf("m_bank_addr_idx %d\n", m_bank_addr_idx);
        //fflush(stdout);
        if (command == "RD") s_num_rd++;
        else if (command == "WR") s_num_wr++;
        else if (command == "ACT") s_num_act++;
        else if (command == "PRE") s_num_pre++;//TODO:check
        if (req_it->is_stat_updated == false) {
          update_request_stats(req_it);
        }
        m_dram->issue_command(req_it->command, req_it->addr_vec);
        req_it->n_command += 1;
        req_it->command_history += std::string(command) + std::string(" ");

        // If we are issuing the last command, set depart clock cycle and move the request to the pending queue
        if (req_it->command == req_it->final_command) {
          if (req_it->type_id == Request::Type::Read) {
            req_it->depart = m_clk + m_dram->m_read_latency;
            pending.push_back(*req_it);
          } else {
            req_it->depart = m_clk;
            s_service_cycles += static_cast<uint64_t>(req_it->depart - req_it->arrive);
          }
          assert(req_it->depart >= req_it->arrive);
          Clk_t lat = req_it->depart - req_it->arrive;
          // printf("[%s] cycle %ld finish request addr %lx\n", get_id().c_str(), m_clk, req_it->addr);
          s_acc_lat_bin[std::min(lat / 10, 9L)]++;
          dequeue_bank_queue_counters(*req_it);
          buffer->remove(req_it);
        } else {
          if (m_dram->m_command_meta(req_it->command).is_opening) {
            m_active_buffer.enqueue(*req_it);
            buffer->remove(req_it);
          }
        }

      }

    };


  private:
    /**
     * @brief    Helper function to check if a request is hitting an open row
     * @details
     * 
     */
    bool is_row_hit(ReqBuffer::iterator& req)
    {
        return m_dram->check_rowbuffer_hit(req->final_command, req->addr_vec);
    }
    /**
     * @brief    Helper function to check if a request is opening a row
     * @details
     * 
    */
    bool is_row_open(ReqBuffer::iterator& req)
    {
        return m_dram->check_node_open(req->final_command, req->addr_vec);
    }

    bool has_valid_bank_target(const AddrVec_t& addr_vec) const {
      if (m_bank_addr_idx < 0 || addr_vec[m_bank_addr_idx] < 0) {
        return false;
      }
      if (m_bankgroup_addr_idx >= 0 && addr_vec[m_bankgroup_addr_idx] < 0) {
        return false;
      }
      if (m_rank_addr_idx >= 0 && addr_vec[m_rank_addr_idx] < 0) {
        return false;
      }
      return true;
    }

    size_t flatten_bankgroup_slot(const AddrVec_t& addr_vec) const {
      const size_t rank_id = m_rank_addr_idx >= 0 ? static_cast<size_t>(addr_vec[m_rank_addr_idx]) : 0;
      const size_t bankgroup_id = m_bankgroup_addr_idx >= 0 ? static_cast<size_t>(addr_vec[m_bankgroup_addr_idx]) : 0;
      return rank_id * m_num_bankgroups + bankgroup_id;
    }

    size_t flatten_bank_slot(const AddrVec_t& addr_vec) const {
      const size_t bankgroup_slot = flatten_bankgroup_slot(addr_vec);
      const size_t bank_id = static_cast<size_t>(addr_vec[m_bank_addr_idx]);
      return bankgroup_slot * m_num_banks + bank_id;
    }

    uint32_t get_tracked_dualq_phase(const Request& req) const {
      if (req.type_id != Request::Type::Read) {
        return 0;
      }
      return (req.anns.phase == 1 || req.anns.phase == 2) ? req.anns.phase : 0;
    }

    void enqueue_bank_queue_counters(const Request& req) {
      if (!has_valid_bank_target(req.addr_vec)) {
        return;
      }
      const size_t bankgroup_slot = flatten_bankgroup_slot(req.addr_vec);
      const size_t bank_slot = flatten_bank_slot(req.addr_vec);
      m_bankgroup_queue_occupancy[bankgroup_slot]++;
      m_bank_queue_occupancy[bank_slot]++;
      s_bankgroup_req_distribution[bankgroup_slot]++;
      s_bank_req_distribution[bank_slot]++;
      const uint32_t dualq_phase = get_tracked_dualq_phase(req);
      if (dualq_phase == 1) {
        s_dualq_phase1_bankgroup_req_distribution[bankgroup_slot]++;
        s_dualq_phase1_bank_req_distribution[bank_slot]++;
      } else if (dualq_phase == 2) {
        s_dualq_phase2_bankgroup_req_distribution[bankgroup_slot]++;
        s_dualq_phase2_bank_req_distribution[bank_slot]++;
      }
    }

    void dequeue_bank_queue_counters(const Request& req) {
      if (!has_valid_bank_target(req.addr_vec)) {
        return;
      }
      const size_t bankgroup_slot = flatten_bankgroup_slot(req.addr_vec);
      const size_t bank_slot = flatten_bank_slot(req.addr_vec);
      if (m_bankgroup_queue_occupancy[bankgroup_slot] > 0) {
        m_bankgroup_queue_occupancy[bankgroup_slot]--;
      }
      if (m_bank_queue_occupancy[bank_slot] > 0) {
        m_bank_queue_occupancy[bank_slot]--;
      }
    }

    void accumulate_bank_queue_stats() {
      for (size_t bankgroup_slot = 0; bankgroup_slot < m_num_bankgroup_slots; bankgroup_slot++) {
        s_bankgroup_queue_len_sum[bankgroup_slot] += m_bankgroup_queue_occupancy[bankgroup_slot];
      }
      for (size_t bank_slot = 0; bank_slot < m_num_bank_slots; bank_slot++) {
        s_bank_queue_len_sum[bank_slot] += m_bank_queue_occupancy[bank_slot];
      }
    }

    /**
     * @brief    
     * @details
     * 
     */
    void update_request_stats(ReqBuffer::iterator& req)
    {
      req->is_stat_updated = true;

      const bool bank_target_valid = has_valid_bank_target(req->addr_vec);
      const size_t bankgroup_slot = bank_target_valid ? flatten_bankgroup_slot(req->addr_vec) : 0;
      const size_t bank_slot = bank_target_valid ? flatten_bank_slot(req->addr_vec) : 0;
      const uint32_t dualq_phase = get_tracked_dualq_phase(*req);

      auto has_valid_core_source = [&]() {
        return req->source_id >= 0 &&
               static_cast<size_t>(req->source_id) < s_read_row_hits_per_core.size();
      };

      if (req->type_id == Request::Type::Read) 
      {
        if (is_row_hit(req)) {
          s_read_row_hits++;
          s_row_hits++;
          if (bank_target_valid) {
            s_bankgroup_row_hits[bankgroup_slot]++;
            s_bank_row_hits[bank_slot]++;
            if (dualq_phase == 1) {
              s_dualq_phase1_bankgroup_row_hits[bankgroup_slot]++;
              s_dualq_phase1_bank_row_hits[bank_slot]++;
            } else if (dualq_phase == 2) {
              s_dualq_phase2_bankgroup_row_hits[bankgroup_slot]++;
              s_dualq_phase2_bank_row_hits[bank_slot]++;
            }
          }
          if (has_valid_core_source())
            s_read_row_hits_per_core[req->source_id]++;
        } else if (is_row_open(req)) {
          s_read_row_conflicts++;
          s_row_conflicts++;
          if (bank_target_valid) {
            s_bankgroup_row_conflicts[bankgroup_slot]++;
            s_bank_row_conflicts[bank_slot]++;
            if (dualq_phase == 1) {
              s_dualq_phase1_bankgroup_row_conflicts[bankgroup_slot]++;
              s_dualq_phase1_bank_row_conflicts[bank_slot]++;
            } else if (dualq_phase == 2) {
              s_dualq_phase2_bankgroup_row_conflicts[bankgroup_slot]++;
              s_dualq_phase2_bank_row_conflicts[bank_slot]++;
            }
          }
          if (has_valid_core_source())
            s_read_row_conflicts_per_core[req->source_id]++;
        } else {
          s_read_row_misses++;
          s_row_misses++;
          if (bank_target_valid) {
            s_bankgroup_row_misses[bankgroup_slot]++;
            s_bank_row_misses[bank_slot]++;
            if (dualq_phase == 1) {
              s_dualq_phase1_bankgroup_row_misses[bankgroup_slot]++;
              s_dualq_phase1_bank_row_misses[bank_slot]++;
            } else if (dualq_phase == 2) {
              s_dualq_phase2_bankgroup_row_misses[bankgroup_slot]++;
              s_dualq_phase2_bank_row_misses[bank_slot]++;
            }
          }
          if (has_valid_core_source())
            s_read_row_misses_per_core[req->source_id]++;
        } 
      } 
      else if (req->type_id == Request::Type::Write) 
      {
        if (is_row_hit(req)) {
          s_write_row_hits++;
          s_row_hits++;
          if (bank_target_valid) {
            s_bankgroup_row_hits[bankgroup_slot]++;
            s_bank_row_hits[bank_slot]++;
          }
        } else if (is_row_open(req)) {
          s_write_row_conflicts++;
          s_row_conflicts++;
          if (bank_target_valid) {
            s_bankgroup_row_conflicts[bankgroup_slot]++;
            s_bank_row_conflicts[bank_slot]++;
          }
        } else {
          s_write_row_misses++;
          s_row_misses++;
          if (bank_target_valid) {
            s_bankgroup_row_misses[bankgroup_slot]++;
            s_bank_row_misses[bank_slot]++;
          }
        }
      }
    }

    /**
     * @brief    Helper function to serve the completed read requests
     * @details
     * This function is called at the beginning of the tick() function.
     * It checks the pending queue to see if the top request has received data from DRAM.
     * If so, it finishes this request by calling its callback and poping it from the pending queue.
     */
    void serve_completed_reads() {
      if (pending.size()) {
        // Check the first pending request
        auto& req = pending[0];
        if (req.depart <= m_clk) {
          // Request received data from dram
          if (req.depart - req.arrive > 1) {
            // Check if this requests accesses the DRAM or is being forwarded.
            // TODO add the stats back
            s_read_latency += req.depart - req.arrive;
          }

          // printf("[memctr] cycle %ld send finished read addr %lx last %ld", m_clk, req.addr, req.depart - req.arrive);
          if (req.callback && !req.nocallback) {
            // If the request comes from outside (e.g., processor), call its callback
            req.callback(req);
          }
          s_service_cycles += static_cast<uint64_t>(req.depart - req.arrive);
          // Finally, remove this request from the pending queue
          pending.pop_front();
        }
      };
    };

    size_t get_sampled_queue_len() const {
      return m_read_buffer.size() + m_write_buffer.size() + m_priority_buffer.size() + pending.size();
    }

    bool has_any_outstanding_work() const {
      return get_sampled_queue_len() + m_active_buffer.size() > 0;
    }

  public:
    uint64_t get_total_reqs() const override {
      return static_cast<uint64_t>(s_num_read_reqs + s_num_write_reqs + s_num_other_reqs);
    }

    uint64_t get_total_service_cycles() const override {
      return s_service_cycles;
    }

    uint64_t get_busy_cycles() const override {
      return s_busy_cycles;
    }

    uint64_t get_queue_len_sum() const override {
      return s_queue_len;
    }

    double get_queue_len_avg_live() const override {
      return m_clk == 0 ? 0.0 : static_cast<double>(s_queue_len) / static_cast<double>(m_clk);
    }

    uint64_t get_current_queue_len() const override {
      return static_cast<uint64_t>(get_sampled_queue_len());
    }

    uint64_t get_max_queue_len() const override {
      return s_queue_len_max;
    }

    bool was_busy_last_cycle() const override {
      return m_was_busy_last_cycle;
    }

    uint64_t get_row_hits() const override {
      return s_row_hits;
    }

    uint64_t get_row_misses() const override {
      return s_row_misses;
    }

    uint64_t get_row_conflicts() const override {
      return s_row_conflicts;
    }

    uint64_t get_act_count() const override {
      return s_num_act;
    }


    /**
     * @brief    Checks if we need to switch to write mode
     * 
     */
    void set_write_mode() {
      if (!m_is_write_mode) {
        if ((m_write_buffer.size() > m_wr_high_watermark * m_write_buffer.max_size) || m_read_buffer.size() == 0) {
          m_is_write_mode = true;
        }
      } else {
        if ((m_write_buffer.size() < m_wr_low_watermark * m_write_buffer.max_size) && m_read_buffer.size() != 0) {
          m_is_write_mode = false;
        }
      }
    };


    /**
     * @brief    Helper function to find a request to schedule from the buffers.
     * 
     */
    bool schedule_request(ReqBuffer::iterator& req_it, ReqBuffer*& req_buffer) {
      bool request_found = false;
      // 2.1    First, check the act buffer to serve requests that are already activating (avoid useless ACTs)
      if (req_it= m_scheduler->get_best_request(m_active_buffer); req_it != m_active_buffer.end()) {
        if (m_dram->check_ready(req_it->command, req_it->addr_vec)) {
          request_found = true;
          req_buffer = &m_active_buffer;
        }
      }

      // 2.2    If no requests can be scheduled from the act buffer, check the rest of the buffers
      if (!request_found) {
        // 2.2.1    We first check the priority buffer to prioritize e.g., maintenance requests
        if (m_priority_buffer.size() != 0) {
          req_buffer = &m_priority_buffer;
          req_it = m_priority_buffer.begin();
          req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
          
          request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
          if (!request_found & m_priority_buffer.size() != 0) {
            return false;
          }
        }

        // 2.2.1    If no request to be scheduled in the priority buffer, check the read and write buffers.
        if (!request_found) {
          // Query the write policy to decide which buffer to serve
          set_write_mode();
          auto& buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
          if (req_it = m_scheduler->get_best_request(buffer); req_it != buffer.end()) {
            request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
            req_buffer = &buffer;
          }
        }
      }

      // 2.3 If we find a request to schedule, we need to check if it will close an opened row in the active buffer.
      if (request_found) {
        if (m_dram->m_command_meta(req_it->command).is_closing) {
          auto& rowgroup = req_it->addr_vec;
          for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end(); _it++) {
            auto& _it_rowgroup = _it->addr_vec;
            bool is_matching = true;
            for (int i = 0; i < m_bank_addr_idx + 1 ; i++) {
              if (_it_rowgroup[i] != rowgroup[i] && _it_rowgroup[i] != -1 && rowgroup[i] != -1) {
                is_matching = false;
                break;
              }
            }
            if (is_matching) {
              request_found = false;
              break;
            }
          }
        }
      }

      return request_found;
    }

    void finalize() override {
      s_avg_read_latency = s_num_read_reqs == 0 ? 0.0f : (float) s_read_latency / (float) s_num_read_reqs;

      s_queue_len_avg = m_clk == 0 ? 0.0f : (float) s_queue_len / (float) m_clk;
      s_read_queue_len_avg = m_clk == 0 ? 0.0f : (float) s_read_queue_len / (float) m_clk;
      s_write_queue_len_avg = m_clk == 0 ? 0.0f : (float) s_write_queue_len / (float) m_clk;
      s_priority_queue_len_avg = m_clk == 0 ? 0.0f : (float) s_priority_queue_len / (float) m_clk;

      for (size_t bank_slot = 0; bank_slot < m_num_bank_slots; bank_slot++) {
        const size_t bank_total = s_bank_row_hits[bank_slot] + s_bank_row_misses[bank_slot] + s_bank_row_conflicts[bank_slot];
        s_bank_queue_len_avg[bank_slot] = m_clk == 0 ? 0.0f : (float) s_bank_queue_len_sum[bank_slot] / (float) m_clk;
        s_bank_row_hit_rate[bank_slot] = bank_total == 0 ? 0.0f : (float) s_bank_row_hits[bank_slot] / (float) bank_total;
      }
      for (size_t bankgroup_slot = 0; bankgroup_slot < m_num_bankgroup_slots; bankgroup_slot++) {
        const size_t bankgroup_total = s_bankgroup_row_hits[bankgroup_slot] + s_bankgroup_row_misses[bankgroup_slot] + s_bankgroup_row_conflicts[bankgroup_slot];
        s_bankgroup_queue_len_avg[bankgroup_slot] = m_clk == 0 ? 0.0f : (float) s_bankgroup_queue_len_sum[bankgroup_slot] / (float) m_clk;
        s_bankgroup_row_hit_rate[bankgroup_slot] = bankgroup_total == 0 ? 0.0f : (float) s_bankgroup_row_hits[bankgroup_slot] / (float) bankgroup_total;
      }

      return;
    }

};
  
}   // namespace Ramulator
