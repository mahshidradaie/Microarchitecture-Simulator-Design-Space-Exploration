// isa.hpp - shared types: committed-instruction trace record, program image, constants.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace asim {

enum class Kind : uint8_t { ALU, MUL, LOAD, STORE, BRANCH, JAL, JALR };

// One committed instruction. rd/rs1/rs2 are -1 when absent (x0 counts as absent).
struct TraceInst {
    uint32_t pc;
    uint32_t addr;     // effective address (LOAD/STORE) or target (BRANCH/JAL/JALR)
    Kind     kind;
    int8_t   rd, rs1, rs2;
    bool     taken;
};

constexpr uint32_t MEM_SIZE    = 4u << 20;   // 4 MiB flat memory
constexpr uint32_t DATA_BASE   = 0x10000;
constexpr uint32_t RESULT_ADDR = DATA_BASE;  // benchmarks store their checksum here

struct Program {
    std::vector<uint32_t> code, data;
    uint32_t data_base = DATA_BASE;
};

// File format (little endian): "ASIM", u32 code_words, u32 data_base, u32 data_words, code..., data...
inline Program load_program(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; uint32_t hdr[3];
    f.read(magic, 4); f.read(reinterpret_cast<char *>(hdr), 12);
    if (!f || std::memcmp(magic, "ASIM", 4) != 0) throw std::runtime_error("bad program file");
    Program p; p.data_base = hdr[1];
    p.code.resize(hdr[0]); p.data.resize(hdr[2]);
    f.read(reinterpret_cast<char *>(p.code.data()), 4ull * hdr[0]);
    f.read(reinterpret_cast<char *>(p.data.data()), 4ull * hdr[2]);
    if (!f) throw std::runtime_error("truncated program file");
    return p;
}

}  // namespace asim
