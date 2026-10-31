#!/usr/bin/env python3
"""Minimal RV32I(+MUL) assembler with the pseudo-instructions the benchmarks need
(li, mv, j, nop, beqz, bnez). Produces the program image read by archsim."""
import re, struct

ABI = {"zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4, "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
       **{f"a{i}": 10 + i for i in range(8)}, **{f"s{i}": 16 + i for i in range(2, 12)},
       **{f"t{i}": 25 + i for i in range(3, 7)}}
REG = {**{f"x{i}": i for i in range(32)}, **ABI}

R_OPS = {"add": (0, 0), "sub": (0x20, 0), "sll": (0, 1), "slt": (0, 2), "sltu": (0, 3), "xor": (0, 4),
         "srl": (0, 5), "sra": (0x20, 5), "or": (0, 6), "and": (0, 7), "mul": (1, 0)}
I_OPS = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
SH_OPS = {"slli": (0, 1), "srli": (0, 5), "srai": (0x20, 5)}
B_OPS = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}


def reg(s):
    if s not in REG:
        raise ValueError(f"bad register {s}")
    return REG[s]


def imm(s):
    return int(s, 0)


def enc_r(f7, rs2, rs1, f3, rd, op): return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
def enc_i(i, rs1, f3, rd, op):
    if not -2048 <= i < 2048: raise ValueError(f"12-bit immediate out of range: {i}")
    return ((i & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
def enc_s(i, rs2, rs1, f3, op):
    if not -2048 <= i < 2048: raise ValueError(f"12-bit immediate out of range: {i}")
    return (((i >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((i & 0x1F) << 7) | op
def enc_b(i, rs2, rs1, f3, op):
    return (((i >> 12) & 1) << 31) | (((i >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | \
           (((i >> 1) & 0xF) << 8) | (((i >> 11) & 1) << 7) | op
def enc_u(i20, rd, op): return ((i20 & 0xFFFFF) << 12) | (rd << 7) | op
def enc_j(i, rd, op):
    return (((i >> 20) & 1) << 31) | (((i >> 1) & 0x3FF) << 21) | (((i >> 11) & 1) << 20) | \
           (((i >> 12) & 0xFF) << 12) | (rd << 7) | op


def expand(op, args):
    """Expand pseudo-instructions into real ones (list of (op, args))."""
    if op == "nop": return [("addi", ["zero", "zero", "0"])]
    if op == "mv": return [("addi", [args[0], args[1], "0"])]
    if op == "j": return [("jal", ["zero", args[0]])]
    if op == "beqz": return [("beq", [args[0], "zero", args[1]])]
    if op == "bnez": return [("bne", [args[0], "zero", args[1]])]
    if op == "li":
        v = imm(args[1]) & 0xFFFFFFFF
        if v >= 1 << 31: v -= 1 << 32
        if -2048 <= v < 2048: return [("addi", [args[0], "zero", str(v)])]
        hi = ((v + 0x800) >> 12) & 0xFFFFF
        lo = ((v & 0xFFF) ^ 0x800) - 0x800
        return [("lui", [args[0], str(hi)]), ("addi", [args[0], args[0], str(lo)])]
    return [(op, args)]


def assemble(text):
    items, labels, pc = [], {}, 0
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        while ":" in line:
            lab, line = line.split(":", 1)
            labels[lab.strip()] = pc
            line = line.strip()
        if not line: continue
        parts = line.replace(",", " ").split()
        for op, args in expand(parts[0], parts[1:]):
            items.append((pc, op, args)); pc += 4
    words = []
    for pc, op, a in items:
        if op in R_OPS:
            f7, f3 = R_OPS[op]; w = enc_r(f7, reg(a[2]), reg(a[1]), f3, reg(a[0]), 0x33)
        elif op in I_OPS:
            w = enc_i(imm(a[2]), reg(a[1]), I_OPS[op], reg(a[0]), 0x13)
        elif op in SH_OPS:
            f7, f3 = SH_OPS[op]; w = enc_i((f7 << 5) | (imm(a[2]) & 31), reg(a[1]), f3, reg(a[0]), 0x13)
        elif op in ("lw", "sw"):
            m = re.fullmatch(r"(-?\w+)\((\w+)\)", a[1])
            if not m: raise ValueError(f"bad memory operand {a[1]}")
            off, base = imm(m.group(1)), reg(m.group(2))
            w = enc_i(off, base, 2, reg(a[0]), 0x03) if op == "lw" else enc_s(off, reg(a[0]), base, 2, 0x23)
        elif op in B_OPS:
            off = labels[a[2]] - pc
            w = enc_b(off, reg(a[1]), reg(a[0]), B_OPS[op], 0x63)
        elif op == "lui": w = enc_u(imm(a[1]), reg(a[0]), 0x37)
        elif op == "jal": w = enc_j(labels[a[1]] - pc, reg(a[0]), 0x6F)
        elif op == "jalr": w = enc_i(imm(a[2]) if len(a) > 2 else 0, reg(a[1]), 0, reg(a[0]), 0x67)
        elif op == "ecall": w = 0x73
        else: raise ValueError(f"unknown instruction {op}")
        words.append(w & 0xFFFFFFFF)
    return words


def write_image(path, code, data, data_base=0x10000):
    with open(path, "wb") as f:
        f.write(b"ASIM" + struct.pack("<III", len(code), data_base, len(data)))
        f.write(struct.pack(f"<{len(code)}I", *code))
        f.write(struct.pack(f"<{len(data)}I", *[d & 0xFFFFFFFF for d in data]))
