"""Quick end-to-end check of the package.

Rebuilds one instance from the raw OpenStreetMap extract, reruns the five main
algorithms, and compares the weighted service ratio with the stored result in
sim/results/main/. All algorithms are deterministic for a given seed, so the
values should agree to within floating-point tolerance.

Usage:  python3 verify_quick.py [luding|xuwen] [seed]
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sim"))
from runlib import get_instance, run_alg   # noqa: E402
import bridge as BR                        # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "luding"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ref = json.load(open(os.path.join(HERE, "sim", "results", "main", f"{key}_{seed}.json")))["algs"]

t0 = time.time()
I, EA = get_instance(key, seed)            # builds the road graph, the instance, and the travel-time tables
print(f"instance {key} seed {seed}: N={I['N']} stations, built in {time.time() - t0:.0f} s")
ok = True
for alg in ["GEN-EDF", "GEN-LS", "IND", "SEQ", "BRIDGE"]:
    t = time.time()
    sr = BR.metrics(I, run_alg(alg, I, EA))["SR"]
    d = abs(sr - ref[alg]["SR"])
    ok &= d < 1e-9
    print(f"  {alg:8s} service ratio {sr:.6f}  stored {ref[alg]['SR']:.6f}  |diff| {d:.1e}  ({time.time() - t:.1f} s)")
print("MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
