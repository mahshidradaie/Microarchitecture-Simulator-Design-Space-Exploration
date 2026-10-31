#!/usr/bin/env python3
"""Generates the benchmark programs (RV32 assembly + data images) and a Python reference
result for each one. Writes <outdir>/<name>.bin and <outdir>/expected.json."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from asm import assemble, write_image

BASE = 0x10000
RES = BASE            # result word
ARR = BASE + 0x100    # arrays start here (256-byte aligned)
M32 = 0xFFFFFFFF


def s32(v):
    v &= M32
    return v - (1 << 32) if v >= 1 << 31 else v


CHECKSUM_C = """
    li t0, 0
    li t1, {nn}
    mv t2, a2
    li t6, 0
cs: lw t3, 0(t2)
    add t6, t6, t3
    addi t2, t2, 4
    addi t0, t0, 1
    blt t0, t1, cs
    li t0, {res}
    sw t6, 0(t0)
    ecall
"""


def matmul(order, n=64, seed=1):
    rng = random.Random(seed)
    A = [rng.randint(-8, 8) for _ in range(n * n)]
    B = [rng.randint(-8, 8) for _ in range(n * n)]
    a_base, b_base, c_base = ARR, ARR + 4 * n * n, ARR + 8 * n * n
    data = [0] * ((c_base - BASE) // 4 + n * n)
    data[(a_base - BASE) // 4:(a_base - BASE) // 4 + n * n] = A
    data[(b_base - BASE) // 4:(b_base - BASE) // 4 + n * n] = B
    head = f"    li a0, {a_base}\n    li a1, {b_base}\n    li a2, {c_base}\n    li s3, {n}\n    li s0, 0\n"
    if order == "ijk":
        body = f"""
li_: li s1, 0
lj:  li s2, 0
     li t6, 0
     li t0, {n*4}
     mul t1, s0, t0
     add t1, t1, a0
     slli t2, s1, 2
     add t2, t2, a1
lk:  lw t3, 0(t1)
     lw t4, 0(t2)
     mul t5, t3, t4
     add t6, t6, t5
     addi t1, t1, 4
     addi t2, t2, {n*4}
     addi s2, s2, 1
     blt s2, s3, lk
     mul t1, s0, t0
     add t1, t1, a2
     slli t2, s1, 2
     add t1, t1, t2
     sw t6, 0(t1)
     addi s1, s1, 1
     blt s1, s3, lj
     addi s0, s0, 1
     blt s0, s3, li_
"""
    else:  # ikj: stream along rows of B and C (unit stride)
        body = f"""
li_: li s2, 0
lk:  li t0, {n*4}
     mul t1, s0, t0
     add t3, t1, a0
     slli t4, s2, 2
     add t3, t3, t4
     lw t5, 0(t3)
     add t1, t1, a2
     mul t2, s2, t0
     add t2, t2, a1
     li s1, 0
lj:  lw t3, 0(t2)
     mul t3, t3, t5
     lw t4, 0(t1)
     add t4, t4, t3
     sw t4, 0(t1)
     addi t1, t1, 4
     addi t2, t2, 4
     addi s1, s1, 1
     blt s1, s3, lj
     addi s2, s2, 1
     blt s2, s3, lk
     addi s0, s0, 1
     blt s0, s3, li_
"""
    code = assemble(head + body + CHECKSUM_C.format(nn=n * n, res=RES))
    C = [[sum(A[i * n + k] * B[k * n + j] for k in range(n)) for j in range(n)] for i in range(n)]
    exp = s32(sum(sum(r) for r in C))
    return code, data, exp


def isort(n=768, seed=2):
    rng = random.Random(seed)
    arr = [rng.randint(-100000, 100000) for _ in range(n)]
    data = [0] * ((ARR - BASE) // 4) + arr
    src = f"""
    li a0, {ARR}
    li s3, {n}
    li s0, 1
oi: slli t0, s0, 2
    add t0, t0, a0
    lw t1, 0(t0)
    addi s1, s0, -1
iw: blt s1, zero, dn
    slli t2, s1, 2
    add t2, t2, a0
    lw t3, 0(t2)
    bge t1, t3, dn
    sw t3, 4(t2)
    addi s1, s1, -1
    j iw
dn: addi t2, s1, 1
    slli t2, t2, 2
    add t2, t2, a0
    sw t1, 0(t2)
    addi s0, s0, 1
    blt s0, s3, oi
    li t0, 0
    li t6, 0
    mv t2, a0
ck: lw t3, 0(t2)
    addi t4, t0, 1
    mul t3, t3, t4
    add t6, t6, t3
    addi t2, t2, 4
    addi t0, t0, 1
    blt t0, s3, ck
    li t0, {RES}
    sw t6, 0(t0)
    ecall
"""
    exp = s32(sum(v * (i + 1) for i, v in enumerate(sorted(arr))))
    return assemble(src), data, exp


def stream(n=262144, seed=3):
    rng = random.Random(seed)
    arr = [rng.randint(0, 1000) for _ in range(n)]
    data = [0] * ((ARR - BASE) // 4) + arr
    src = f"""
    li t0, {ARR}
    li t1, {n}
    li t2, 0
    li t6, 0
sl: lw t3, 0(t0)
    add t6, t6, t3
    addi t0, t0, 4
    addi t2, t2, 1
    blt t2, t1, sl
    li t0, {RES}
    sw t6, 0(t0)
    ecall
"""
    return assemble(src), data, s32(sum(arr))


def ptrchase(nodes=12288, laps=5, seed=4):
    rng = random.Random(seed)
    order = list(range(nodes)); rng.shuffle(order)          # one random cycle through all nodes
    vals = [rng.randint(0, 255) for _ in range(nodes)]
    words = (nodes * 64) // 4
    data = [0] * ((ARR - BASE) // 4) + [0] * words
    off = (ARR - BASE) // 4
    nxt = {}
    for k in range(nodes):
        a, b = order[k], order[(k + 1) % nodes]
        data[off + a * 16] = ARR + b * 64                   # node = 64 B: [next, value, ...]
        data[off + a * 16 + 1] = vals[a]
        nxt[a] = b
    steps = nodes * laps
    src = f"""
    li t0, {ARR + order[0] * 64}
    li t1, {steps}
    li t2, 0
    li t6, 0
pl: lw t3, 4(t0)
    add t6, t6, t3
    lw t0, 0(t0)
    addi t2, t2, 1
    blt t2, t1, pl
    li t0, {RES}
    sw t6, 0(t0)
    ecall
"""
    cur, tot = order[0], 0
    for _ in range(steps):
        tot += vals[cur]; cur = nxt[cur]
    return assemble(src), data, s32(tot)


def branchy(iters=60000):
    src = f"""
    li s0, 0
    li s3, {iters}
    li s4, 12345
    li s5, 1103515245
    li s6, 12345
    li t6, 0
bl: mul s4, s4, s5
    add s4, s4, s6
    srli t0, s4, 16
    andi t0, t0, 1
    beqz t0, k1
    addi t6, t6, 3
k1: andi t1, s0, 7
    li t2, 5
    blt t1, t2, k2
    addi t6, t6, 5
k2: beqz t0, k3
    addi t6, t6, 7
k3: addi s0, s0, 1
    blt s0, s3, bl
    li t0, {RES}
    sw t6, 0(t0)
    ecall
"""
    x, tot = 12345, 0
    for i in range(iters):
        x = (x * 1103515245 + 12345) & M32
        bit = (x >> 16) & 1
        if bit: tot += 3
        if (i & 7) >= 5: tot += 5
        if bit: tot += 7
    return assemble(src), [0], s32(tot)


BENCHES = {
    "matmul_ijk": lambda: matmul("ijk"),
    "matmul_ikj": lambda: matmul("ikj"),
    "isort": isort,
    "stream": stream,
    "ptrchase": ptrchase,
    "branchy": branchy,
}


def build_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    expected = {}
    for name, fn in BENCHES.items():
        code, data, exp = fn()
        write_image(os.path.join(outdir, f"{name}.bin"), code, data)
        expected[name] = exp
    with open(os.path.join(outdir, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
    print(f"built {len(expected)} benchmarks in {outdir}")


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else "build/bench")
