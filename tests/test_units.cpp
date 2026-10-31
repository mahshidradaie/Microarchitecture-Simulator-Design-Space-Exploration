// Unit tests for the cache, predictor and core timing models (synthetic, analytically checkable cases).
#include <cstdio>
#include <cstdlib>
#include "core.hpp"

using namespace asim;
static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); ++failures; } else std::printf("ok:   %s\n", msg); } while (0)

static bool touch(Cache &c, uint32_t a) {          // returns true on hit
    const int w = c.lookup(a);
    if (w >= 0) { c.on_hit(c.set_of(a), w); return true; }
    c.install(a, 0, false); return false;
}

static void test_caches() {
    { CacheCfg cfg{256, 2, 64, Repl::LRU};             // 2 sets x 2 ways; lines 0,2,4 map to set 0
      Cache c(cfg); const uint32_t A = 0, B = 128, C = 256;
      touch(c, A); touch(c, B); touch(c, A); touch(c, C);   // LRU victim must be B
      CHECK(touch(c, A) && !touch(c, B), "LRU evicts the least recently used line"); }
    { CacheCfg cfg{256, 4, 64, Repl::PLRU}; Cache c(cfg);   // 1 set x 4 ways
      for (uint32_t i = 0; i < 4; ++i) touch(c, i * 64);
      touch(c, 0); touch(c, 4 * 64);
      CHECK(c.lookup(0) >= 0, "tree-PLRU keeps the most recently touched line"); }
    { auto run = [](Repl r) {                          // hot set of 3 lines (reused twice per round) + streaming scan, 4-way single set
          CacheCfg cfg{256, 4, 64, r}; Cache c(cfg); int hits = 0;
          for (int rep = 0; rep < 50; ++rep) {
              for (int pass = 0; pass < 2; ++pass)
                  for (uint32_t h = 0; h < 3; ++h) hits += touch(c, h * 64) ? 1 : 0;
              for (uint32_t s = 0; s < 3; ++s) touch(c, (100 + rep * 3 + s) * 64);
          }
          return hits; };
      CHECK(run(Repl::SRRIP) > run(Repl::LRU) + 50, "SRRIP is scan-resistant (hot lines survive a streaming scan, LRU loses them)"); }
    { auto run = [](Repl r) {                          // cyclic scan over 6 lines in a 4-way set: LRU always misses
          CacheCfg cfg{256, 4, 64, r}; Cache c(cfg); int hits = 0;
          for (int rep = 0; rep < 200; ++rep) for (uint32_t l = 0; l < 6; ++l) hits += touch(c, l * 64) ? 1 : 0;
          return hits; };
      CHECK(run(Repl::LRU) == 0 && run(Repl::BRRIP) > 200, "BRRIP resists thrashing where LRU gets zero hits on a cyclic scan"); }
    { MemCfg m; MemSys ms(m);
      const uint64_t miss = ms.access(0x20000, 0, 0, true), hit = ms.access(0x20000, 0, 1000, true);
      CHECK(miss == 2 + 12 + 100 && hit == 2, "cold miss costs L1+L2+memory latency, then hits in L1"); }
    { MemCfg m; m.pf = Prefetch::STRIDE; m.pf_degree = 4; MemSys ms(m);
      for (uint32_t i = 0; i < 64; ++i) ms.access(0x30000 + i * 64, 0x400, i * 400, true);
      CHECK(ms.st.pf_useful > 40 && ms.st.l1_miss < 16, "stride prefetcher covers a unit-stride stream"); }
}

static uint32_t rnd_state = 1;
static bool rnd() { rnd_state = rnd_state * 1664525u + 1013904223u; return (rnd_state >> 16) & 1; }

static double acc(const std::string &name, const std::function<bool(int)> &outcome, int n, uint32_t pcA, uint32_t pcB, bool secondOnly) {
    auto bp = make_predictor(name, 12); int ok = 0, tot = 0;
    for (int i = 0; i < n; ++i) {
        const bool a = outcome(i);
        const bool pa = bp->predict(pcA, pcA - 8); bp->update(pcA, a);
        if (!secondOnly && i > n / 2) { ok += pa == a; ++tot; }
        if (pcB) { const bool pb = bp->predict(pcB, pcB - 8); bp->update(pcB, a);   // B is perfectly correlated with A
                   if (secondOnly && i > n / 2) { ok += pb == a; ++tot; } }
    }
    return double(ok) / tot;
}

static void test_predictors() {
    auto loop4 = [](int i) { return i % 4 != 3; };      // T T T N repeating
    CHECK(acc("gshare", loop4, 4000, 0x100, 0, false) > 0.98, "gshare learns a period-4 pattern");
    CHECK(acc("bimodal", loop4, 4000, 0x100, 0, false) < 0.80, "bimodal cannot (it saturates to the majority outcome)");
    rnd_state = 7; std::vector<bool> r(8000); for (auto &&b : r) b = rnd();
    auto rnd_out = [&](int i) { return static_cast<bool>(r[i]); };
    CHECK(acc("gshare", rnd_out, 8000, 0x100, 0x200, true) > 0.95, "gshare predicts a branch correlated with the previous one");
    CHECK(acc("bimodal", rnd_out, 8000, 0x100, 0x200, true) < 0.60, "bimodal sees only ~50% on that correlated branch");
    CHECK(acc("perceptron", rnd_out, 8000, 0x100, 0x200, true) > 0.95, "perceptron also exploits the correlation");
    CHECK(acc("tournament", rnd_out, 8000, 0x100, 0x200, true) > 0.90, "tournament selects gshare where it is better");
    auto always = [](int) { return true; };
    CHECK(acc("bimodal", always, 1000, 0x100, 0, false) > 0.99 && acc("nt", always, 1000, 0x100, 0, false) < 0.01, "bimodal learns always-taken; static not-taken fails");
}

static TraceInst alu(int rd, int rs1 = -1, uint32_t pc = 0) { return TraceInst{pc, 0, Kind::ALU, (int8_t)rd, (int8_t)rs1, -1, false}; }

static void test_core() {
    const MemCfg mc;
    { std::vector<TraceInst> t; for (int i = 0; i < 4000; ++i) t.push_back(alu(1 + i % 20));
      MemSys m(mc); CoreCfg c; c.width = 4; auto s = run_core(t, c, m);
      CHECK(s.ipc() > 3.8, "independent ALU ops reach ~width IPC on a 4-wide core"); }
    { std::vector<TraceInst> t; for (int i = 0; i < 4000; ++i) t.push_back(alu(1, i ? 1 : -1));
      MemSys m(mc); CoreCfg c; c.width = 4; auto s = run_core(t, c, m);
      CHECK(s.ipc() > 0.95 && s.ipc() < 1.1, "a serial dependency chain limits IPC to ~1 however wide the core"); }
    { std::vector<TraceInst> t; for (int i = 0; i < 4000; ++i) t.push_back(alu(1 + i % 20));
      MemSys m(mc); CoreCfg c; c.inorder = true; c.width = 1; auto s = run_core(t, c, m);
      CHECK(s.ipc() > 0.97 && s.ipc() <= 1.0, "1-wide in-order core sustains IPC ~1"); }
    { auto mk = [] { std::vector<TraceInst> t;                     // 64 independent loads, each to a new cold line
          for (int i = 0; i < 64; ++i) t.push_back(TraceInst{0x100u + 4u * i, 0x40000u + 4096u * i, Kind::LOAD, (int8_t)(1 + i % 20), -1, -1, false});
          return t; };
      MemCfg m2; m2.mshr = 8; MemSys a(m2), b(m2); CoreCfg big, tiny; big.rob = 192; tiny.rob = 2;
      const uint64_t cb = run_core(mk(), big, a).cycles, ct = run_core(mk(), tiny, b).cycles;
      CHECK(cb * 3 < ct, "a large ROB overlaps independent cache misses (memory-level parallelism)"); }
    { std::vector<TraceInst> t;                                     // dependent loads: pointer chase, no overlap possible
      for (int i = 0; i < 64; ++i) t.push_back(TraceInst{0x100, 0x40000u + 4096u * i, Kind::LOAD, 1, 1, -1, false});
      MemCfg m2; MemSys m(m2); CoreCfg c; auto s = run_core(t, c, m);
      CHECK(s.cycles >= 64u * 110u, "dependent misses serialize: ~memory latency per load regardless of ROB size"); }
    { std::vector<TraceInst> t;                                     // always-taken branch: nt predictor mispredicts every time
      for (int i = 0; i < 1000; ++i) t.push_back(TraceInst{0x100, 0x80, Kind::BRANCH, -1, -1, -1, true});
      CoreCfg c; c.bp = "nt"; c.penalty = 10; MemSys m(mc); auto s = run_core(t, c, m);
      CHECK(s.mispredicts == 1000 && s.cycles > 1000u * 10u, "mispredict penalty is charged per wrong prediction"); }
}

int main() {
    test_caches(); test_predictors(); test_core();
    std::printf(failures ? "\n%d TEST(S) FAILED\n" : "\nALL UNIT TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
