#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "memory_system/impl/generic_DRAM_system.cpp"

namespace Ramulator {

class MultiMemorySystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, MultiMemorySystem, "MultiMemory", "Multiple generic DRAM-based memory system.");

  protected:
    Clk_t m_clk = 0;
    std::vector<IMemorySystem*> m_memory;
    std::vector<uint32_t> m_acc_cnt;
    std::string assignMethod;
    uint32_t prior_assign_id;
    uint32_t nMemory;

  private:
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
      for (int i = 0; i < nMemory; i++) {
        IMemorySystem* memory = create_child_ifce<IMemorySystem>();
        GenericDRAMSystem* memory1 = dynamic_cast<GenericDRAMSystem*>(memory);
        memory1->set_id(fmt::format("Memory {}", i));
        m_memory.push_back(memory);
      }
      m_acc_cnt = std::vector<uint32_t>(nMemory, 0);
      prior_assign_id = 0;
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    bool send(Request req) override {
      assert(req.source_id < nMemory);
      uint32_t assignId = assignMethod == "source"? req.source_id:
                          assignMethod == "random"? hashAddress(req.addr) % nMemory:
                          assignMethod == "interleave"? (req.addr / 4096) % nMemory:
                          assignMethod == "rr"? (prior_assign_id+1) % nMemory:
                          0;
      bool success = m_memory[assignId]->send(req);
      // printf("[multiGeneric] cycle %ld addr %lx hash %lx assignId %d reqBufFull %d success %x\n", m_memory[0]->get_clk(), req.addr, hashAddress(req.addr), assignId, m_memory[assignId]->get_reqbuf_full(req), success);
      // fflush(stdout);
      if (success) {
        m_acc_cnt[assignId]++;
        prior_assign_id = assignId;
      }
      return success;
    };

    void tick() override {
      for (auto memory : m_memory) {
        memory->tick();
      }
    };

    float get_tCK() override {
      assert(!m_memory.empty());
      return m_memory[0]->get_tCK();
    }

    uint32_t get_num_memory() {
      return nMemory;
    }

    void finalize() override {
      uint32_t tot_read = 0, tot_write = 0;
      size_t tot_rd = 0, tot_wr = 0, tot_act = 0, tot_pre = 0;
      for (uint32_t i = 0; i < nMemory; i++) {
        printf("Memory %d read req: %d\twrite req: %d\n", i, m_memory[i]->get_num_read(), m_memory[i]->get_num_write());
        printf("Memory %d assigned req: %d\n", i, m_acc_cnt[i]);
        printf("Memory %d cmd ACT: %ld\tPRE: %ld\n", i, m_memory[i]->get_num_act(), m_memory[i]->get_num_pre());
        tot_read += m_memory[i]->get_num_read();
        tot_write += m_memory[i]->get_num_write();
        tot_rd += m_memory[i]->get_num_rd();
        tot_wr += m_memory[i]->get_num_wr();
        tot_act += m_memory[i]->get_num_act();
        tot_pre += m_memory[i]->get_num_pre();
      }
      for (uint32_t i = 0; i < nMemory; i++) {
      }
      printf("Memory cycle %lu\n", m_memory[0]->get_clk());
      printf("Memory read req %d\n", tot_read);
      printf("Memory write req %d\n", tot_write);
      printf("Memory cmd ACT %ld\n", tot_act);
      printf("Memory cmd PRE %ld\n", tot_pre);
      fflush(stdout);
    };


    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };
};

}   // namespace 

