#pragma once
#include <cstdint>
#include <vector>
#include <list>

class HardwareCache {
public:
    uint64_t s_num_cache_hit = 0;
    uint64_t s_num_cache_access = 0;

    HardwareCache() : sets(NumSets) { }

    // Return true if hit, false if miss
    bool access(uint64_t addr) {
        s_num_cache_access++;
        uint64_t block_addr = addr / BlockSize;
        size_t set_index = block_addr % NumSets;
        auto &cache_set = sets[set_index];
        // Search for the block tag in the set
        for(auto it = cache_set.begin(); it != cache_set.end(); ++it) {
            if(*it == block_addr) {
                s_num_cache_hit++;
                // Move to front (MRU)
                cache_set.erase(it);
                cache_set.push_front(block_addr);
                return true;
            }
        }
        // Miss: Insert block using LRU eviction if necessary
        if(cache_set.size() == Associativity) {
            cache_set.pop_back();
        }
        cache_set.push_front(block_addr);
        return false;
    }

private:
    static constexpr size_t CacheSize = 8 * 1024 * 1024; // 8MB
    static constexpr size_t BlockSize = 64;          // 64B
    static constexpr size_t NumSets = 16384;
    static constexpr size_t Associativity = 8;
    std::vector<std::list<uint64_t>> sets;
};
