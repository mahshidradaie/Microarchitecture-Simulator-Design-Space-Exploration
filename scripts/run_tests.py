#!/usr/bin/env python3
"""Functional validation: every benchmark must produce the checksum computed by the Python reference."""
import json, subprocess, sys

exp = json.load(open("build/bench/expected.json"))
bad = 0
for name, want in exp.items():
    out = subprocess.run(["build/archsim", "--prog", f"build/bench/{name}.bin"], capture_output=True, text=True)
    if out.returncode:
        print(f"FAIL {name}: {out.stderr.strip()}"); bad += 1; continue
    r = json.loads(out.stdout)
    ok = r["result"] == want
    print(f"{'PASS' if ok else 'FAIL'} {name:12s} insts={r['insts']:>8d} result={r['result']} expected={want}")
    bad += not ok
sys.exit(1 if bad else 0)
