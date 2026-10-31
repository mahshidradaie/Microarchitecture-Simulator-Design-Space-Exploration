// core.hpp - parameterizable timing model: replays the committed trace through an
// in-order or out-of-order superscalar core (dispatch width, ROB, issue/commit width,
// memory ports, misprediction redirect penalty) using the real predictor and cache models.
#pragma once
#include <cstdint>
#include <vector>
#include "bpred.hpp"
#include "cache.hpp"
#include "isa.hpp"

namespace asim {

struct CoreCfg {
    bool     inorder = false;
    uint32_t width = 4;          // fetch = dispatch = issue = commit width
    uint32_t rob = 192;
    uint32_t mem_ports = 2;
    uint32_t penalty = 10;       // cycles from branch resolution until correct-path dispatch resumes
    uint32_t mul_lat = 3;
    std::string bp = "gshare";
    unsigned bp_bits = 12;
};

struct CoreStats {
    uint64_t insts = 0, cycles = 0, branches = 0, mispredicts = 0;
    uint64_t loads = 0, stores = 0;
    double ipc() const { return cycles ? double(insts) / double(cycles) : 0.0; }
};

inline CoreStats run_core(const std::vector<TraceInst> &trace, const CoreCfg &cc, MemSys &mem) {
    CoreStats st;
    auto bp = make_predictor(cc.bp, cc.bp_bits);
    const uint32_t W = cc.width;
    std::vector<uint64_t> reg_ready(32, 0), commit_ring(cc.rob, 0);
    std::vector<uint8_t> slots, mslots;
    auto at = [](std::vector<uint8_t> &v, uint64_t c) -> uint8_t & { if (c >= v.size()) v.resize(c * 2 + 64, 0); return v[c]; };

    uint64_t disp_cycle = 0, redirect = 0, last_issue = 0, last_commit = 0;
    uint32_t disp_cnt = 0, commit_cnt = 0;

    for (size_t i = 0; i < trace.size(); ++i) {
        const TraceInst &t = trace[i];
        // ---- front end: in-order dispatch, W per cycle, limited by ROB capacity and redirects
        if (disp_cnt >= W) { ++disp_cycle; disp_cnt = 0; }
        if (disp_cycle < redirect) { disp_cycle = redirect; disp_cnt = 0; }
        if (i >= cc.rob) { const uint64_t freed = commit_ring[i % cc.rob]; if (disp_cycle < freed) { disp_cycle = freed; disp_cnt = 0; } }
        const uint64_t D = disp_cycle; ++disp_cnt;

        // ---- issue: operands ready, free issue slot (and memory port), in-order if requested
        uint64_t R = D + 1;
        if (t.rs1 >= 0) R = std::max(R, reg_ready[t.rs1]);
        if (t.rs2 >= 0) R = std::max(R, reg_ready[t.rs2]);
        if (cc.inorder) R = std::max(R, last_issue);
        const bool is_mem = (t.kind == Kind::LOAD || t.kind == Kind::STORE);
        uint64_t I = R;
        for (;; ++I) {
            if (at(slots, I) >= W) continue;
            if (is_mem && at(mslots, I) >= cc.mem_ports) continue;
            break;
        }
        ++at(slots, I); if (is_mem) ++at(mslots, I);
        last_issue = I;

        // ---- execute
        uint64_t lat = 1;
        switch (t.kind) {
        case Kind::MUL: lat = cc.mul_lat; break;
        case Kind::LOAD: lat = mem.access(t.addr, t.pc, I, true); ++st.loads; break;
        case Kind::STORE: mem.access(t.addr, t.pc, I, false); ++st.stores; break;   // store buffer hides latency
        default: break;
        }
        const uint64_t C = I + lat;
        if (t.rd >= 0) reg_ready[t.rd] = C;

        if (t.kind == Kind::BRANCH) {
            ++st.branches;
            const bool pred = bp->predict(t.pc, t.addr);
            bp->update(t.pc, t.taken);
            if (pred != t.taken) { ++st.mispredicts; redirect = std::max<uint64_t>(redirect, C + cc.penalty); }
        }

        // ---- in-order commit, W per cycle
        uint64_t K = std::max(C + 1, last_commit);
        if (K == last_commit) { if (commit_cnt >= W) { ++K; commit_cnt = 0; } } else commit_cnt = 0;
        ++commit_cnt;
        last_commit = K;
        commit_ring[i % cc.rob] = K;
    }
    st.insts = trace.size();
    st.cycles = last_commit;
    return st;
}

}  // namespace asim
