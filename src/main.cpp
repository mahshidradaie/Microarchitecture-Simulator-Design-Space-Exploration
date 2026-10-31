// archsim - RISC-V microarchitecture explorer. Usage: archsim --prog file.bin [options]
#include <cstdio>
#include <cstdlib>
#include <map>
#include "core.hpp"
#include "cpu.hpp"

using namespace asim;

static uint32_t parse_size(const std::string &s) {
    char *e; double v = std::strtod(s.c_str(), &e);
    if (*e == 'k' || *e == 'K') v *= 1024; else if (*e == 'm' || *e == 'M') v *= 1024 * 1024;
    return static_cast<uint32_t>(v);
}
static CacheCfg parse_cache(const std::string &s, Repl r) {   // size:ways:line
    CacheCfg c; c.repl = r; size_t a = s.find(':'), b = s.find(':', a + 1);
    c.size = parse_size(s.substr(0, a));
    if (a != std::string::npos) c.ways = static_cast<uint32_t>(std::stoul(s.substr(a + 1, b - a - 1)));
    if (b != std::string::npos) c.line = static_cast<uint32_t>(std::stoul(s.substr(b + 1)));
    return c;
}

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> o;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) throw std::runtime_error("bad argument " + k);
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            o[k.substr(2)] = argv[++i];
        }
        if (!o.count("prog")) throw std::runtime_error("usage: archsim --prog file.bin [--preset inorder|ooo2|ooo4|ooo8] [options]");
        CoreCfg cc; MemCfg mc;
        const std::string preset = o.count("preset") ? o["preset"] : "ooo4";
        if (preset == "inorder") { cc.inorder = true; cc.width = 1; cc.rob = 32; cc.mem_ports = 1; cc.penalty = 3; }
        else if (preset == "ooo2") { cc.width = 2; cc.rob = 64; cc.mem_ports = 1; cc.penalty = 8; }
        else if (preset == "ooo4") { cc.width = 4; cc.rob = 192; cc.mem_ports = 2; cc.penalty = 10; }
        else if (preset == "ooo8") { cc.width = 8; cc.rob = 352; cc.mem_ports = 3; cc.penalty = 12; }
        else throw std::runtime_error("unknown preset " + preset);
        auto num = [&](const char *k, uint32_t &v) { if (o.count(k)) v = static_cast<uint32_t>(std::stoul(o[k])); };
        num("width", cc.width); num("rob", cc.rob); 
        num("penalty", cc.penalty); num("mem-ports", cc.mem_ports); num("mul-lat", cc.mul_lat);
        if (o.count("bp")) cc.bp = o["bp"];
        if (o.count("bp-bits")) cc.bp_bits = static_cast<unsigned>(std::stoul(o["bp-bits"]));
        if (cc.rob == 0 || cc.width == 0) throw std::runtime_error("width and rob must be > 0");
        const Repl r1 = parse_repl(o.count("l1-repl") ? o["l1-repl"] : (o.count("repl") ? o["repl"] : "lru"));
        const Repl r2 = parse_repl(o.count("l2-repl") ? o["l2-repl"] : (o.count("repl") ? o["repl"] : "lru"));
        mc.l1 = parse_cache(o.count("l1") ? o["l1"] : "32k:8:64", r1);
        mc.l2 = parse_cache(o.count("l2") ? o["l2"] : "512k:8:64", r2);
        num("l1-lat", mc.l1_lat); num("l2-lat", mc.l2_lat); num("mem-lat", mc.mem_lat); num("mshr", mc.mshr);
        num("pf-degree", mc.pf_degree);
        if (o.count("pf")) mc.pf = parse_pf(o["pf"]);
        uint64_t maxi = o.count("max-insts") ? std::stoull(o["max-insts"]) : 50000000ull;

        Program p = load_program(o["prog"]);
        FuncResult fr = run_functional(p, maxi);
        if (!fr.halted) throw std::runtime_error("program did not halt within max-insts");
        MemSys mem(mc);
        CoreStats cs = run_core(fr.trace, cc, mem);
        const double ki = cs.insts / 1000.0;
        const MemStats &m = mem.st;
        std::printf("{\"prog\":\"%s\",\"preset\":\"%s\",\"insts\":%llu,\"cycles\":%llu,\"ipc\":%.4f,\"cpi\":%.4f,"
                    "\"branches\":%llu,\"mispredicts\":%llu,\"bp_acc\":%.4f,\"bp_mpki\":%.3f,"
                    "\"l1_miss_rate\":%.4f,\"l1_mpki\":%.3f,\"l2_miss_rate\":%.4f,\"l2_mpki\":%.3f,"
                    "\"amat\":%.3f,\"pf_issued\":%llu,\"pf_useful\":%llu,\"pf_late\":%llu,\"result\":%d}\n",
                    o["prog"].c_str(), preset.c_str(), (unsigned long long)cs.insts, (unsigned long long)cs.cycles,
                    cs.ipc(), cs.cycles ? double(cs.cycles) / cs.insts : 0.0,
                    (unsigned long long)cs.branches, (unsigned long long)cs.mispredicts,
                    cs.branches ? 1.0 - double(cs.mispredicts) / cs.branches : 1.0, cs.mispredicts / ki,
                    m.l1_acc ? double(m.l1_miss) / m.l1_acc : 0.0, m.l1_miss / ki,
                    m.l2_acc ? double(m.l2_miss) / m.l2_acc : 0.0, m.l2_miss / ki,
                    m.loads ? double(m.load_lat_sum) / m.loads : 0.0,
                    (unsigned long long)m.pf_issued, (unsigned long long)m.pf_useful, (unsigned long long)m.pf_late,
                    static_cast<int32_t>(fr.result));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
