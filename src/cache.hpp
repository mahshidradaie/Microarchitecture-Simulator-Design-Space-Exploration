// cache.hpp - set-associative cache (LRU / tree-PLRU / Random / SRRIP) and a two-level
// hierarchy with MSHR-limited misses and next-line / stride prefetchers.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace asim {

enum class Repl { LRU, PLRU, RANDOM, SRRIP, BRRIP };
inline Repl parse_repl(const std::string &s) {
    if (s == "lru") return Repl::LRU;
    if (s == "plru") return Repl::PLRU;
    if (s == "random") return Repl::RANDOM;
    if (s == "srrip") return Repl::SRRIP;
    if (s == "brrip") return Repl::BRRIP;   // bimodal RRIP: thrash-resistant insertion
    throw std::runtime_error("unknown replacement policy: " + s);
}

struct CacheCfg { uint32_t size = 32768, ways = 8, line = 64; Repl repl = Repl::LRU; };

class Cache {
public:
    struct Line { bool valid = false, pf = false; uint32_t tag = 0; uint64_t ready = 0, stamp = 0; uint8_t rrpv = 3; };

    explicit Cache(const CacheCfg &c) : cfg(c) {
        auto pow2 = [](uint32_t v) { return v && !(v & (v - 1)); };
        if (!pow2(c.size) || !pow2(c.ways) || !pow2(c.line) || c.size < c.ways * c.line)
            throw std::runtime_error("cache size/ways/line must be powers of two");
        sets_ = c.size / (c.ways * c.line);
        lines_.resize(static_cast<size_t>(sets_) * c.ways);
        plru_.assign(sets_, 0);
    }
    uint32_t set_of(uint32_t a) const { return (a / cfg.line) & (sets_ - 1); }
    uint32_t tag_of(uint32_t a) const { return (a / cfg.line) / sets_; }
    Line &line(uint32_t set, int way) { return lines_[static_cast<size_t>(set) * cfg.ways + way]; }

    int lookup(uint32_t a) {
        const uint32_t s = set_of(a), t = tag_of(a);
        for (uint32_t w = 0; w < cfg.ways; ++w) { Line &l = line(s, w); if (l.valid && l.tag == t) return static_cast<int>(w); }
        return -1;
    }
    void on_hit(uint32_t set, int way) {
        Line &l = line(set, way);
        switch (cfg.repl) {
        case Repl::LRU: l.stamp = ++clock_; break;
        case Repl::PLRU: plru_touch(set, way); break;
        case Repl::SRRIP: case Repl::BRRIP: l.rrpv = 0; break;
        case Repl::RANDOM: break;
        }
    }
    // Install a line; returns true if a valid line was evicted.
    bool install(uint32_t a, uint64_t ready, bool pf) {
        const uint32_t s = set_of(a);
        int w = -1;
        for (uint32_t i = 0; i < cfg.ways; ++i) if (!line(s, i).valid) { w = static_cast<int>(i); break; }
        const bool evict = (w < 0);
        if (evict) w = victim(s);
        Line &l = line(s, w);
        l.valid = true; l.tag = tag_of(a); l.ready = ready; l.pf = pf;
        switch (cfg.repl) {
        case Repl::LRU: l.stamp = ++clock_; break;
        case Repl::PLRU: plru_touch(s, w); break;
        case Repl::SRRIP: l.rrpv = 2; break;
        case Repl::BRRIP: l.rrpv = (next_rand() % 32 == 0) ? 2 : 3; break;   // insert "distant" 31/32 of the time
        case Repl::RANDOM: break;
        }
        return evict;
    }
    CacheCfg cfg;

private:
    int victim(uint32_t s) {
        switch (cfg.repl) {
        case Repl::LRU: {
            int v = 0; for (uint32_t w = 1; w < cfg.ways; ++w) if (line(s, w).stamp < line(s, v).stamp) v = static_cast<int>(w);
            return v; }
        case Repl::PLRU: {
            uint32_t node = 1;
            while (node < cfg.ways) node = node * 2 + ((plru_[s] >> node) & 1);
            return static_cast<int>(node - cfg.ways); }
        case Repl::RANDOM:
            return static_cast<int>(next_rand() % cfg.ways);
        case Repl::SRRIP: case Repl::BRRIP:
            for (;;) {
                for (uint32_t w = 0; w < cfg.ways; ++w) if (line(s, w).rrpv == 3) return static_cast<int>(w);
                for (uint32_t w = 0; w < cfg.ways; ++w) ++line(s, w).rrpv;
            }
        }
        return 0;
    }
    uint64_t next_rand() { rng_ ^= rng_ << 13; rng_ ^= rng_ >> 7; rng_ ^= rng_ << 17; return rng_; }
    void plru_touch(uint32_t s, int way) {
        uint32_t node = static_cast<uint32_t>(way) + cfg.ways;
        while (node > 1) {
            const uint32_t parent = node / 2;
            if (node & 1) plru_[s] &= ~(1ull << parent); else plru_[s] |= (1ull << parent);
            node = parent;
        }
    }
    uint32_t sets_;
    std::vector<Line> lines_;
    std::vector<uint64_t> plru_;
    uint64_t clock_ = 0, rng_ = 88172645463325252ull;
};

enum class Prefetch { NONE, NEXTLINE, STRIDE };
inline Prefetch parse_pf(const std::string &s) {
    if (s == "none") return Prefetch::NONE;
    if (s == "nextline") return Prefetch::NEXTLINE;
    if (s == "stride") return Prefetch::STRIDE;
    throw std::runtime_error("unknown prefetcher: " + s);
}

struct MemCfg {
    CacheCfg l1{32768, 8, 64, Repl::LRU};
    CacheCfg l2{524288, 8, 64, Repl::LRU};
    uint32_t l1_lat = 2, l2_lat = 12, mem_lat = 100, mshr = 8;
    Prefetch pf = Prefetch::NONE;
    uint32_t pf_degree = 4;
};

struct MemStats {
    uint64_t l1_acc = 0, l1_miss = 0, l2_acc = 0, l2_miss = 0;
    uint64_t pf_issued = 0, pf_useful = 0, pf_late = 0;
    uint64_t loads = 0, load_lat_sum = 0;
};

class MemSys {
public:
    explicit MemSys(const MemCfg &c) : cfg(c), l1(c.l1), l2(c.l2), stride_(256) {}

    // Returns the access latency in cycles for an access issued at cycle `now`.
    uint64_t access(uint32_t addr, uint32_t pc, uint64_t now, bool is_load) {
        ++st.l1_acc;
        uint64_t lat;
        const uint32_t s = l1.set_of(addr);
        const int w = l1.lookup(addr);
        bool miss = false;
        if (w >= 0) {
            Cache::Line &L = l1.line(s, w);
            l1.on_hit(s, w);
            lat = cfg.l1_lat;
            if (L.ready > now) { lat += L.ready - now; if (L.pf) ++st.pf_late; }   // fill still in flight
            if (L.pf) { ++st.pf_useful; L.pf = false; }
        } else {
            miss = true; ++st.l1_miss;
            while (!mshr_.empty() && mshr_.top() <= now) mshr_.pop();
            uint64_t start = now;
            if (mshr_.size() >= cfg.mshr) { start = mshr_.top(); mshr_.pop(); }   // wait for a free MSHR
            lat = (start - now) + cfg.l1_lat + l2_access(addr, start + cfg.l1_lat);
            l1.install(addr, now + lat, false);
            mshr_.push(now + lat);
        }
        if (is_load) { ++st.loads; st.load_lat_sum += lat; }
        train(addr, pc, now, miss);
        return lat;
    }
    MemCfg cfg;
    Cache l1, l2;
    MemStats st;

private:
    uint64_t l2_access(uint32_t addr, uint64_t t) {
        ++st.l2_acc;
        const uint32_t s = l2.set_of(addr);
        const int w = l2.lookup(addr);
        if (w >= 0) {
            l2.on_hit(s, w);
            uint64_t lat = cfg.l2_lat; const uint64_t r = l2.line(s, w).ready;
            if (r > t) lat += r - t;
            return lat;
        }
        ++st.l2_miss;
        const uint64_t lat = cfg.l2_lat + cfg.mem_lat;
        l2.install(addr, t + lat, false);
        return lat;
    }
    void issue_prefetch(uint32_t addr, uint64_t now) {
        if (addr >= (4u << 20)) return;
        if (l1.lookup(addr) >= 0) return;
        ++st.pf_issued;
        const uint64_t lat = l2_access_quiet(addr, now + cfg.l1_lat);
        l1.install(addr, now + cfg.l1_lat + lat, true);
    }
    uint64_t l2_access_quiet(uint32_t addr, uint64_t t) {     // prefetch fills do not count as demand L2 accesses
        const uint32_t s = l2.set_of(addr);
        const int w = l2.lookup(addr);
        if (w >= 0) { l2.on_hit(s, w); uint64_t lat = cfg.l2_lat; const uint64_t r = l2.line(s, w).ready; if (r > t) lat += r - t; return lat; }
        const uint64_t lat = cfg.l2_lat + cfg.mem_lat;
        l2.install(addr, t + lat, false);
        return lat;
    }
    void train(uint32_t addr, uint32_t pc, uint64_t now, bool miss) {
        const uint32_t line = cfg.l1.line;
        if (cfg.pf == Prefetch::NEXTLINE) {
            if (miss) for (uint32_t k = 1; k <= cfg.pf_degree; ++k) issue_prefetch((addr & ~(line - 1)) + k * line, now);
        } else if (cfg.pf == Prefetch::STRIDE) {
            StrideEntry &e = stride_[(pc >> 2) & 255];
            if (e.pc != pc) { e = {pc, addr, 0, 0}; return; }
            const int32_t d = static_cast<int32_t>(addr - e.last);
            if (d == e.stride && d != 0) { if (e.conf < 3) ++e.conf; } else { e.stride = d; e.conf = 0; }
            e.last = addr;
            if (e.conf >= 2) {
                int32_t step = e.stride;
                if (std::abs(step) < static_cast<int32_t>(line)) step = step > 0 ? static_cast<int32_t>(line) : -static_cast<int32_t>(line);
                for (uint32_t k = 1; k <= cfg.pf_degree; ++k) issue_prefetch(addr + static_cast<uint32_t>(step * static_cast<int32_t>(k)), now);
            }
        }
    }
    struct StrideEntry { uint32_t pc = 0, last = 0; int32_t stride = 0; uint8_t conf = 0; };
    std::vector<StrideEntry> stride_;
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> mshr_;
};

}  // namespace asim
