"""联合邻域精修的候选排序检验：对主实验全部种子，取主循环（不含精修）的解，枚举其全部 1-swap 候选，
同时用边际收益 G 的估计值与真实下层目标 J 评分，统计秩相关、真实最优候选在 G 排序中的名次与前 κ 召回率。
python3 refine_rank.py [workers] → results/refine_rank.json"""
import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr, kendalltau
from runlib import get_instance
import bridge as BR
HERE = os.path.dirname(os.path.abspath(__file__))

def one(args):
    key, seed = args
    I, EA = get_instance(key, seed)
    r = BR.run_bridge(I, EA, BR.ALGS["w/o refine"])
    routes, ab, low = r["routes"], r["abar"], r["low"]
    G = BR.gains(I, low, {}, ab)
    served = {i for rr in routes for i in rr}
    est, true = [], []
    for out in sorted(served):
        for inn in sorted(set(range(I["N"])) - served):
            rs = BR.routes_from_set(I, EA, (served - {out}) | {inn})
            if len({i for rr in rs for i in rr}) < len(served): continue
            abx = BR.abar_of(I, EA, rs)
            est.append(sum(G[i, int(abx[i])] for i in {i for rr in rs for i in rr} if abx[i] < I["T"]))
            true.append(BR.lower_level(I, abx, {})["J"])
    est, true = np.array(est), np.array(true)
    J0 = low["J"]
    order = np.argsort(-est, kind="stable")
    best = int(np.argmax(true))
    rank_best = int(np.where(order == best)[0][0]) + 1
    improving = true > J0 * (1 + 1e-9)
    top_true = set(np.argsort(-true)[:10].tolist())
    rec = {k: len(top_true & set(order[:k].tolist())) / 10 for k in (10, 30)}
    return dict(key=key, seed=seed, n=len(est), spearman=float(spearmanr(est, true)[0]),
                kendall=float(kendalltau(est, true)[0]), rank_best=rank_best,
                n_improving=int(improving.sum()), best_gain_pct=float(100 * (true.max() - J0) / J0),
                recall10_at10=rec[10], recall10_at30=rec[30])

if __name__ == "__main__":
    tasks = [(k, s) for k in ["luding", "xuwen"] for s in range(20)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 14) as p: R = p.map(one, tasks)
    json.dump(R, open(os.path.join(HERE, "results", "refine_rank.json"), "w"), indent=1)
    for k in ["luding", "xuwen"]:
        rr = [r for r in R if r["key"] == k]
        f = lambda m: np.array([r[m] for r in rr])
        print(k, "n=%.0f spearman %.3f [%.3f,%.3f] kendall %.3f rank_best median %d max %d  best within top30: %d/%d  recall@10 %.2f @30 %.2f"
              % (f("n").mean(), f("spearman").mean(), f("spearman").min(), f("spearman").max(), f("kendall").mean(),
                 np.median(f("rank_best")), f("rank_best").max(), int((f("rank_best") <= 30).sum()), len(rr),
                 f("recall10_at10").mean(), f("recall10_at30").mean()))
