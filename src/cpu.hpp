// cpu.hpp - functional RV32I(+MUL) interpreter that records the committed-instruction trace.
// Timing is modelled separately (core.hpp) by replaying this trace: the standard
// execution-driven-functional / trace-driven-timing split.
#pragma once
#include "isa.hpp"

namespace asim {

struct FuncResult {
    std::vector<TraceInst> trace;
    uint32_t result = 0;     // mem[RESULT_ADDR] at exit
    bool halted = false;     // reached ecall
};

inline FuncResult run_functional(const Program &p, uint64_t max_insts) {
    std::vector<uint8_t> mem(MEM_SIZE, 0);
    if (p.code.size() * 4 > p.data_base) throw std::runtime_error("code overlaps data");
    std::memcpy(mem.data(), p.code.data(), p.code.size() * 4);
    if (p.data_base + p.data.size() * 4 > MEM_SIZE) throw std::runtime_error("data too large");
    std::memcpy(mem.data() + p.data_base, p.data.data(), p.data.size() * 4);

    auto rd32 = [&](uint32_t a) {
        if ((a & 3) || a + 4 > MEM_SIZE) throw std::runtime_error("bad load address");
        uint32_t v; std::memcpy(&v, &mem[a], 4); return v; };
    auto wr32 = [&](uint32_t a, uint32_t v) {
        if ((a & 3) || a + 4 > MEM_SIZE) throw std::runtime_error("bad store address");
        std::memcpy(&mem[a], &v, 4); };

    uint32_t x[32] = {0}, pc = 0;
    FuncResult r;
    r.trace.reserve(1 << 20);
    auto R = [](uint32_t n) -> int8_t { return n ? static_cast<int8_t>(n) : static_cast<int8_t>(-1); };

    while (r.trace.size() < max_insts) {
        const uint32_t w = rd32(pc);
        if (w == 0x00000073) { r.halted = true; break; }                 // ecall = halt
        const uint32_t op = w & 0x7f, rd = (w >> 7) & 31, f3 = (w >> 12) & 7;
        const uint32_t rs1 = (w >> 15) & 31, rs2 = (w >> 20) & 31, f7 = w >> 25;
        const int32_t immI = static_cast<int32_t>(w) >> 20;
        const int32_t immS = (static_cast<int32_t>(w & 0xFE000000) >> 20) | ((w >> 7) & 0x1F);
        const int32_t immB = (static_cast<int32_t>(w & 0x80000000) >> 19) | ((w & 0x80) << 4) |
                             ((w >> 20) & 0x7E0) | ((w >> 7) & 0x1E);
        const uint32_t immU = w & 0xFFFFF000;
        const int32_t immJ = (static_cast<int32_t>(w & 0x80000000) >> 11) | (w & 0xFF000) |
                             ((w >> 9) & 0x800) | ((w >> 20) & 0x7FE);
        const uint32_t a = x[rs1], b = x[rs2];
        TraceInst t{pc, 0, Kind::ALU, R(rd), -1, -1, false};
        uint32_t next = pc + 4, res = 0;
        bool wr = false;

        switch (op) {
        case 0x33: {  // R-type
            t.rs1 = R(rs1); t.rs2 = R(rs2); wr = true;
            if (f7 == 1 && f3 == 0) { res = a * b; t.kind = Kind::MUL; break; }
            switch (f3) {
            case 0: res = f7 == 0x20 ? a - b : a + b; break;
            case 1: res = a << (b & 31); break;
            case 2: res = static_cast<int32_t>(a) < static_cast<int32_t>(b); break;
            case 3: res = a < b; break;
            case 4: res = a ^ b; break;
            case 5: res = f7 == 0x20 ? static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31)) : a >> (b & 31); break;
            case 6: res = a | b; break;
            default: res = a & b; break;
            }
            break; }
        case 0x13: {  // OP-IMM
            t.rs1 = R(rs1); wr = true;
            const uint32_t i = static_cast<uint32_t>(immI), sh = i & 31;
            switch (f3) {
            case 0: res = a + i; break;
            case 1: res = a << sh; break;
            case 2: res = static_cast<int32_t>(a) < immI; break;
            case 3: res = a < i; break;
            case 4: res = a ^ i; break;
            case 5: res = (i & 0x400) ? static_cast<uint32_t>(static_cast<int32_t>(a) >> sh) : a >> sh; break;
            case 6: res = a | i; break;
            default: res = a & i; break;
            }
            break; }
        case 0x03: {  // LW
            if (f3 != 2) throw std::runtime_error("only lw supported");
            t.kind = Kind::LOAD; t.rs1 = R(rs1); t.addr = a + static_cast<uint32_t>(immI);
            res = rd32(t.addr); wr = true; break; }
        case 0x23: {  // SW
            if (f3 != 2) throw std::runtime_error("only sw supported");
            t.kind = Kind::STORE; t.rd = -1; t.rs1 = R(rs1); t.rs2 = R(rs2);
            t.addr = a + static_cast<uint32_t>(immS); wr32(t.addr, b); break; }
        case 0x63: {  // BRANCH
            t.kind = Kind::BRANCH; t.rd = -1; t.rs1 = R(rs1); t.rs2 = R(rs2);
            bool tk = false;
            switch (f3) {
            case 0: tk = a == b; break;
            case 1: tk = a != b; break;
            case 4: tk = static_cast<int32_t>(a) < static_cast<int32_t>(b); break;
            case 5: tk = static_cast<int32_t>(a) >= static_cast<int32_t>(b); break;
            case 6: tk = a < b; break;
            case 7: tk = a >= b; break;
            default: throw std::runtime_error("bad branch");
            }
            t.taken = tk; t.addr = pc + static_cast<uint32_t>(immB);
            if (tk) next = t.addr;
            break; }
        case 0x37: res = immU; wr = true; break;                          // LUI
        case 0x17: res = pc + immU; wr = true; break;                     // AUIPC
        case 0x6F: t.kind = Kind::JAL; t.taken = true; t.addr = pc + static_cast<uint32_t>(immJ);
                   res = pc + 4; wr = true; next = t.addr; break;
        case 0x67: t.kind = Kind::JALR; t.taken = true; t.rs1 = R(rs1);
                   t.addr = (a + static_cast<uint32_t>(immI)) & ~1u; res = pc + 4; wr = true; next = t.addr; break;
        default: throw std::runtime_error("unsupported opcode at pc=" + std::to_string(pc));
        }
        if (wr && rd) x[rd] = res;
        x[0] = 0;
        r.trace.push_back(t);
        pc = next;
    }
    r.result = rd32(RESULT_ADDR);
    return r;
}

}  // namespace asim
