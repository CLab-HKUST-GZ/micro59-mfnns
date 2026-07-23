#pragma once

namespace hnswlib {

class L2EarlyTerminationInterface {
 public:
    virtual ~L2EarlyTerminationInterface() = default;
    virtual bool get_et_enabled() const = 0;
    virtual float dist_func_et(const void* query, const void* point, float threshold) = 0;
    virtual float compute_full_l1_lower_bound(const void* query, const void* point) const = 0;
};

}  // namespace hnswlib
