#ifndef RAMULATOR_CONTROLLER_CONTROLLER_H
#define RAMULATOR_CONTROLLER_CONTROLLER_H

#include <vector>
#include <deque>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "base/base.h"
#include "dram/dram.h"
#include "dram_controller/scheduler.h"
#include "dram_controller/plugin.h"
#include "dram_controller/refresh.h"
#include "dram_controller/rowpolicy.h"


namespace Ramulator {

class IDRAMController : public Clocked<IDRAMController> {
  RAMULATOR_REGISTER_INTERFACE(IDRAMController, "Controller", "Memory Controller Interface");

  public:
    IDRAM*  m_dram = nullptr;          
    IScheduler*   m_scheduler = nullptr;
    IRefreshManager*   m_refresh = nullptr;
    IRowPolicy*   m_rowpolicy = nullptr;
    std::vector<IControllerPlugin*> m_plugins;

    int m_channel_id = -1;

  // stats
  public:
    size_t s_num_row_hits = 0;
    size_t s_num_row_misses = 0;
    size_t s_num_row_conflicts = 0;
    size_t s_num_act = 0;
    size_t s_num_pre = 0;
    size_t s_num_rd = 0;
    size_t s_num_wr = 0;
    std::vector<size_t> s_acc_lat_bin;

  public:
    /**
     * @brief       Send a request to the memory controller.
     * 
     * @param    req        The request to be enqueued.
     * @return   true       Successful.
     * @return   false      Failed (e.g., buffer full).
     */
    virtual bool send(Request& req) = 0;

    /**
     * @brief       Send a high-priority request to the memory controller.
     * 
     */
    virtual bool priority_send(Request& req) = 0;

    /**
     * @brief       Ticks the memory controller.
     * 
     */
    virtual void tick() = 0;

    virtual uint64_t get_total_reqs() const { return 0; }
    virtual uint64_t get_total_service_cycles() const { return 0; }
    virtual uint64_t get_busy_cycles() const { return 0; }
    virtual uint64_t get_queue_len_sum() const { return 0; }
    virtual double get_queue_len_avg_live() const { return 0.0; }
    virtual uint64_t get_current_queue_len() const { return 0; }
    virtual uint64_t get_max_queue_len() const { return 0; }
    virtual bool was_busy_last_cycle() const { return false; }
    virtual uint64_t get_row_hits() const { return 0; }
    virtual uint64_t get_row_misses() const { return 0; }
    virtual uint64_t get_row_conflicts() const { return 0; }
    virtual uint64_t get_act_count() const { return 0; }
   
};

}       // namespace Ramulator

#endif  // RAMULATOR_CONTROLLER_CONTROLLER_H
