"""补充统计：无人机覆盖时段内基站关断的构成——主动休眠（此后电池阶段仍有节电/满功率时隙）与电池耗尽。
重跑主实验中的 IND、SEQ、BRIDGE（算法确定，结果与 results/main 一致），写入 results/sleep_split.json。"""
import json, os, sys, numpy as np
from multiprocessing import Pool
from runlib import get_instance, run_alg
HERE = os.path.dirname(os.path.abspath(__file__))

def one(args):
    key, seed = args
    I, EA = get_instance(key, seed)
    stored = json.load(open(os.path.join(HERE, "results", "main", f"{key}_{seed}.json")))
    out = dict(key=key, seed=seed)
    for a in ["IND", "SEQ", "BRIDGE"]:
        r = run_alg(a, I, EA); m, x = r["low"]["m"], r["low"]["x"]; dt = I["dt"]
        off = (m == 0) & (x > 0)
        later = np.zeros_like(m, bool)                       # 此后是否仍有电池供电（节电/满功率）时隙
        use = (m == 1) | (m == 2)
        later[:, :-1] = np.flip(np.maximum.accumulate(np.flip(use[:, 1:], 1), 1), 1)
        out[a] = dict(cover_h=float(x.sum() * dt), off_h=float(off.sum() * dt),
                      sleep_h=float((off & later).sum() * dt), dep_h=float((off & ~later).sum() * dt),
                      J=r["J"], J_stored=stored["algs"][a]["J"])
    return out

if __name__ == "__main__":
    tasks = [(k, s) for k in ["luding", "xuwen"] for s in range(20)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 14) as p: R = p.map(one, tasks)
    json.dump(R, open(os.path.join(HERE, "results", "sleep_split.json"), "w"), indent=1)
    for k in ["luding", "xuwen"]:
        for a in ["IND", "SEQ", "BRIDGE"]:
            rr = [r[a] for r in R if r["key"] == k]
            f = lambda m: np.mean([r[m] for r in rr])
            dj = max(abs(r["J"] - r["J_stored"]) / r["J_stored"] for r in rr)
            print(k, a, "cover %.1f off %.1f sleep %.1f dep %.1f  sleep/cover %.1f%%  off/cover %.1f%%  maxrelJdiff %.1e"
                  % (f("cover_h"), f("off_h"), f("sleep_h"), f("dep_h"), 100 * f("sleep_h") / f("cover_h"), 100 * f("off_h") / f("cover_h"), dj))
