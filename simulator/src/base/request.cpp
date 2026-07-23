#include "base/request.h"

namespace Ramulator {

Request::Request(Addr_t addr, int type): addr(addr), type_id(type) {};

Request::Request(AddrVec_t addr_vec, int type): addr_vec(addr_vec), type_id(type) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback):
addr(addr), type_id(type), source_id(source_id), callback(callback) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, PDisReq anns):
addr(addr), type_id(type), source_id(source_id), callback(callback), anns(anns) {};


Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, PDisReq anns,bool nocallback):
addr(addr), type_id(type), source_id(source_id), callback(callback), anns(anns), nocallback(nocallback) {};


Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, ANNSCpuRequest cpuanns):
addr(addr), type_id(type), source_id(source_id), callback(callback), cpuanns(cpuanns) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, TravReq trav) :
addr(addr), type_id(type), source_id(source_id), callback(callback), trav(trav) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, uint64_t id):
addr(addr), type_id(type), source_id(source_id), callback(callback), id(id) {};

}        // namespace Ramulator

