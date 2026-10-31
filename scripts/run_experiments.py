#!/usr/bin/env python3
"""Design-space exploration. Runs archsim over parameter sweeps, writes results/results.csv
and figures into docs/. Usage: python3 scripts/run_experiments.py"""
import csv, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM = "build/archsim"
BENCH = "build/bench"
os.makedirs("results", exist_ok=True)
os.makedirs("docs", exist_ok=True)
rows = []


def run(exp, bench, **opts):
    args = [SIM, "--prog", f"{BENCH}/{bench}.bin"]
    for k, v in opts.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    r = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    r.update({"exp": exp, "bench": bench, **{f"opt_{k}": v for k, v in opts.items()}})
    return r


def sweep(jobs):
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        res = list(ex.map(lambda j: run(*j[:2], **j[2]), jobs))
    rows.extend(res)
    return res


def bars(ax, groups, series, values, ylabel, title, fmt="%.2f"):
    w = 0.8 / len(series)
    for si, s in enumerate(series):
        xs = [gi + si * w for gi in range(len(groups))]
        ys = [values[(g, s)] for g in groups]
        b = ax.bar(xs, ys, w, label=s)
        ax.bar_label(b, labels=[fmt % y for y in ys], fontsize=6, padding=1)
    ax.set_xticks([gi + 0.4 - w / 2 for gi in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=7)


BENCHES = ["matmul_ijk", "matmul_ikj", "isort", "stream", "ptrchase", "branchy"]

# ---- E1: branch predictors -------------------------------------------------------------------
preds = ["nt", "btfn", "bimodal", "gshare", "tournament", "perceptron"]
e1 = sweep([("E1_predictors", b, dict(bp=p, preset="ooo4")) for b in ["branchy", "isort", "matmul_ijk"] for p in preds])
v = {(r["bench"], r["opt_bp"]): r["bp_mpki"] for r in e1}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
bars(ax[0], ["branchy", "isort", "matmul_ijk"], preds, v, "branch MPKI (lower is better)", "E1: branch predictor accuracy", "%.1f")
v2 = {(r["bench"], r["opt_bp"]): r["ipc"] for r in e1}
bars(ax[1], ["branchy", "isort", "matmul_ijk"], preds, v2, "IPC (4-wide OoO)", "E1: resulting performance")
fig.tight_layout(); fig.savefig("docs/e1_predictors.png", dpi=150); plt.close(fig)

# ---- E2: core configurations -----------------------------------------------------------------
presets = ["inorder", "ooo2", "ooo4", "ooo8"]
e2 = sweep([("E2_cores", b, dict(preset=p)) for b in BENCHES for p in presets])
v = {(r["bench"], r["opt_preset"]): r["ipc"] for r in e2}
fig, ax = plt.subplots(figsize=(10, 3.6))
bars(ax, BENCHES, presets, v, "IPC", "E2: in-order vs out-of-order cores of increasing width")
fig.tight_layout(); fig.savefig("docs/e2_cores.png", dpi=150); plt.close(fig)

# ---- E3: ROB size and MSHRs (memory-level parallelism) ---------------------------------------
robs = [8, 16, 32, 64, 128, 192, 256, 384, 512]
e3 = sweep([("E3_rob", b, dict(preset="ooo4", rob=r)) for b in ["stream", "ptrchase", "matmul_ijk", "isort"] for r in robs])
mshrs = [1, 2, 4, 8, 16]
e3b = sweep([("E3_mshr", "stream", dict(preset="ooo4", rob=256, mshr=m)) for m in mshrs])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for b in ["stream", "ptrchase", "matmul_ijk", "isort"]:
    ax[0].plot(robs, [next(r["ipc"] for r in e3 if r["bench"] == b and r["opt_rob"] == x) for x in robs], marker="o", label=b)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("ROB entries"); ax[0].set_ylabel("IPC"); ax[0].set_title("E3a: ROB size sweep (4-wide)"); ax[0].legend(fontsize=7)
ax[1].plot(mshrs, [r["ipc"] for r in e3b], marker="o", color="C3")
ax[1].set_xlabel("L1 MSHRs (outstanding misses)"); ax[1].set_ylabel("IPC"); ax[1].set_title("E3b: MSHR sweep, stream, ROB=256")
fig.tight_layout(); fig.savefig("docs/e3_mlp.png", dpi=150); plt.close(fig)

# ---- E4: cache replacement policies ----------------------------------------------------------
repls = ["lru", "plru", "random", "srrip", "brrip"]
e4 = sweep([("E4_repl", b, dict(preset="ooo4", repl=r)) for b in ["ptrchase", "stream", "matmul_ijk", "isort"] for r in repls])
v = {(r["bench"], r["opt_repl"]): r["l2_mpki"] for r in e4}
v2 = {(r["bench"], r["opt_repl"]): r["ipc"] for r in e4}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
bars(ax[0], ["ptrchase", "stream", "matmul_ijk", "isort"], repls, v, "L2 misses per 1000 instructions", "E4: replacement policy vs L2 MPKI (applied to L1 and L2)", "%.1f")
bars(ax[1], ["ptrchase", "stream", "matmul_ijk", "isort"], repls, v2, "IPC", "E4: replacement policy vs IPC", "%.3f")
fig.tight_layout(); fig.savefig("docs/e4_replacement.png", dpi=150); plt.close(fig)

# ---- E5: prefetchers -------------------------------------------------------------------------
pfs = ["none", "nextline", "stride"]
e5 = sweep([("E5_prefetch", b, dict(preset="ooo4", pf=p)) for b in ["stream", "matmul_ijk", "matmul_ikj", "ptrchase", "isort"] for p in pfs])
base = {r["bench"]: r["ipc"] for r in e5 if r["opt_pf"] == "none"}
v = {(r["bench"], r["opt_pf"]): r["ipc"] / base[r["bench"]] for r in e5}
fig, ax = plt.subplots(figsize=(9, 3.6))
bars(ax, ["stream", "matmul_ijk", "matmul_ikj", "ptrchase", "isort"], pfs, v, "speedup over no prefetching", "E5: hardware prefetchers (degree 4)")
fig.tight_layout(); fig.savefig("docs/e5_prefetch.png", dpi=150); plt.close(fig)

# ---- E6: loop interchange vs cache capacity -----------------------------------------------------
l1s = ["2k", "4k", "8k", "16k", "32k", "64k"]
e6 = sweep([("E6_l1size", b, dict(preset="ooo4", l1=f"{s}:4:64")) for b in ["matmul_ijk", "matmul_ikj"] for s in l1s])
l2s = ["8k", "16k", "32k", "64k", "512k"]
e6b = sweep([("E6_l2size", b, dict(preset="ooo4", l1="8k:4:64", l2=f"{s}:8:64")) for b in ["matmul_ijk", "matmul_ikj"] for s in l2s])
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for b in ["matmul_ijk", "matmul_ikj"]:
    ax[0].plot(l1s, [next(r["l1_miss_rate"] for r in e6 if r["bench"] == b and r["opt_l1"] == f"{s}:4:64") for s in l1s], marker="o", label=b)
    ax[1].plot(l1s, [next(r["cycles"] for r in e6 if r["bench"] == b and r["opt_l1"] == f"{s}:4:64") / 1e6 for s in l1s], marker="o", label=b)
    ax[2].plot(l2s, [next(r["cycles"] for r in e6b if r["bench"] == b and r["opt_l2"] == f"{s}:8:64") / 1e6 for s in l2s], marker="o", label=b)
ax[0].set_ylabel("L1 miss rate"); ax[0].set_xlabel("L1 size (4-way)"); ax[0].set_title("E6a: L1 miss rate"); ax[0].legend()
ax[1].set_ylabel("cycles (millions)"); ax[1].set_xlabel("L1 size (4-way), L2 = 512 KB"); ax[1].set_title("E6b: OoO hides L2 hits"); ax[1].legend()
ax[2].set_ylabel("cycles (millions)"); ax[2].set_xlabel("L2 size, L1 = 8 KB"); ax[2].set_title("E6c: misses reaching DRAM"); ax[2].legend()
fig.tight_layout(); fig.savefig("docs/e6_interchange.png", dpi=150); plt.close(fig)

# ---- dump -------------------------------------------------------------------------------------
keys = sorted({k for r in rows for k in r})
with open("results/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} simulations -> results/results.csv, figures in docs/")
