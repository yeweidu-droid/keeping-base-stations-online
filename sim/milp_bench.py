"""最优性基准：在真实场景的小实例上，把 (P0)（含微电网共享预算、无人机班组与时变路网油机路径）
写成 MILP 求全局最优，报告 BRIDGE 的相对间隙。

建模要点：
- 油机路径：每辆车的可行访问序列全部枚举（小实例下数目有限），序列决定各站开始供电时刻。
- 全局最优中，油机到达后基站的班次自然不会被安排（其价值增量恒为 0），因此无需额外限制班次与到达的关系。
- 服务率统一式 s = β + (1-β)φx 中的二次项 β·x 用 McCormick 包络（一个因子为 0/1 时精确）线性化。
- 运行：python3 milp_bench.py [时限秒] [进程数] → results/milp/milp_all.json（两种规模 × 两个场景 × 10 个种子，全部报告）
"""
import sys, os, json, math, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from runlib import get_instance
import bridge as BR
import scenario as SC

RES = os.path.join(HERE, "results")
N_SM = 10      # 小实例站点数
T_SM = 96      # 小实例时域（8 小时）
K_SM, C_SM = 2, 3

def subset_instance(I, EA, idx, K, C, U):
    idx = list(idx)
    S = dict(I)
    S["N"] = len(idx)
    S["site_node"] = [I["site_node"][j] for j in idx]
    S["sites"] = [I["sites"][j] for j in idx]
    for k in ("kappa", "users", "w", "PF", "PE", "rho", "B", "hbat",
              "phi", "R_cell", "f", "h", "base_of", "access", "mg_of"):
        S[k] = I[k][idx]
    S["w"] = I["w"][idx][:, :T_SM]          # 小实例时域截断
    S["mg_of"] = S["mg_of"].astype(int)
    S["mg_members"] = [list(np.where(S["mg_of"] == g)[0]) for g in range(int(S["mg_of"].max()) + 1)]
    S["B_mg"] = np.array([S["B"][np.array(m, int)].sum() for m in S["mg_members"]])
    used = sorted(set(S["base_of"].tolist()))
    remap = {b: j for j, b in enumerate(used)}
    S["base_of"] = np.array([remap[b] for b in S["base_of"]])
    S["U_b"] = [I["U_b"][b] for b in used]
    S["K"], S["C"], S["C_k"] = K, C, None
    S["bases"] = [I["bases"][b] for b in used]
    S["T"] = T_SM
    S.pop("nodes_adj", None)
    EAsub = EA[[0] + [j + 1 for j in idx]][:, :, idx]
    return S, EAsub

def build_milp(I, EA, time_limit=240):
    N, T, dt = I["N"], I["T"], I["dt"]
    K, C = I["K"], I["C"]
    nB = len(I["U_b"])
    # ---- 序列枚举 ----
    seqs = [()]                                        # 0 号为空序列：不出发
    for L in range(1, C + 1):
        for perm in itertools.permutations(range(N), L):
            arr = BR.route_abar(I, EA, list(perm), 0)
            if all(np.isfinite(a) for _, a in arr) and all(a < T for _, a in arr):
                seqs.append(tuple((i, int(a)) for i, a in arr))
    nY = [len(seqs)] * K
    off_y = [0]
    for k in range(1, K): off_y.append(off_y[-1] + nY[k - 1])
    # 序列的到达到达时刻（供 q 的行）
    # ---- 变量下标 ----
    NT = N * T
    i_eE = lambda i, t: i * T + t
    off = {}
    off["y"] = 0; off["eE"] = 0 + sum(nY); off["eF"] = off["eE"] + NT
    off["X"] = off["eF"] + NT; off["q"] = off["X"] + NT
    off["th"] = off["q"] + NT; off["c"] = off["th"] + NT
    # ---- 班次集合（不截断的窗口；截断由目标值自动体现）----
    S = []
    for i in range(N):
        if I["phi"][i] <= 0: continue
        f, h = int(I["f"][i]), int(I["h"][i])
        d = I["uav_start"]
        while True:
            ws, we = d + f, min(d + f + h, T)
            if ws >= T: break
            S.append((i, d, ws, we, d, d + 2 * f + h + I["S"]))
            d += h
    nz = len(S)
    off["z"] = off["c"] + NT
    off["a"] = off["z"] + nz          # a[i,s]：某辆车在时隙 s 到达站点 i（s = 0..T-1）
    nvar = off["a"] + NT
    lb = np.zeros(nvar); ub = np.ones(nvar)
    integrality = np.zeros(nvar)
    for k in range(K): integrality[off_y[k]:off_y[k] + nY[k]] = 1
    integrality[off["eE"]:off["q"]] = 1              # eE, eF, X, q 为 0/1
    integrality[off["a"]:off["a"] + NT] = 1
    integrality[off["z"]:] = 1
    ub[off["c"]:off["c"] + NT] = np.repeat(I["PF"] * dt, T)
    ub[off["th"]:off["th"] + NT] = 1.0
    rows, cols, data, lo, hi = [], [], [], [], []
    r = 0
    def add(coefs, l_, u_):
        nonlocal r
        for j, v in coefs: rows.append(r); cols.append(j); data.append(v)
        lo.append(l_); hi.append(u_); r += 1
    def col(key, i, t): return off[key] + i * T + t
    # 1. 每辆车的路径序列恰取其一
    for k in range(K):
        add([(off_y[k] + s, 1.0) for s in range(nY[k])], 1, 1)
    # 2. 每站至多被一辆车服务
    for i in range(N):
        co = [(off_y[k] + s, 1.0) for k in range(K) for s, sq in enumerate(seqs)
              if sq and any(j == i for j, _ in sq)]
        if co: add(co, 0, 1)
    # 3. 到达-供电的严格编码：a[i,s] 与序列变量之间用**等式**行（右侧为空时也加，保证 a 被完全定义，
    #    否则自由变量会让 MILP 拿到“免费能量”），再由前缀和给出供电指示 q[i,t] = Σ_{s≤t−σ_i} a[i,s]。
    for i in range(N):
        for t in range(T):
            co = [(off_y[k] + s_, 1.0) for k in range(K) for s_, sq in enumerate(seqs)
                  if sq and any(j == i and a_ == t for j, a_ in sq)]
            add([(off["a"] + i * T + t, 1.0)] + [(j, -v) for j, v in co], 0, 0)
        add([(off["a"] + i * T + s, 1.0) for s in range(T)], -np.inf, 1)
        for t in range(T):
            # seqs 中存的是开始供电时刻 ā_i（= 到达 + σ_i），所以供电指示直接取前缀和，不再减 σ
            co = [(off["a"] + i * T + s, 1.0) for s in range(t + 1)]
            add([(off["q"] + i * T + t, 1.0)] + [(j, -v) for j, v in co], 0, 0)
    # 4. 油机供电的时隙必须满功率：eF ≥ q
    for i in range(N):
        for t in range(T): add([(col("eF", i, t), 1.0), (col("q", i, t), -1.0)], 0, np.inf)
    # 5. eE + eF ≤ 1
    for i in range(N):
        for t in range(T): add([(col("eE", i, t), 1.0), (col("eF", i, t), 1.0)], 0, 1)
    # 6. McCormick：θ = (ρ eE + eF)·x
    for i in range(N):
        ri = I["rho"][i]
        for t in range(T):
            jx = col("X", i, t); jth = col("th", i, t); je = col("eE", i, t); jf = col("eF", i, t)
            add([(jth, 1.0), (jx, -1.0)], -np.inf, 0)
            add([(jth, 1.0), (je, -ri), (jf, -1.0)], -np.inf, 0)
            add([(jth, -1.0), (je, ri), (jf, 1.0), (jx, 1.0)], -np.inf, 1.0)
    # 7. 电池抽取量 c
    for i in range(N):
        pe, pf = I["PE"][i] * dt, I["PF"][i] * dt
        for t in range(T):
            je, jf, jq, jc = col("eE", i, t), col("eF", i, t), col("q", i, t), col("c", i, t)
            add([(jc, 1.0), (je, -pe), (jf, -pf)], -np.inf, 0)
            add([(jc, 1.0), (jq, pf)], -np.inf, pf)
            add([(jc, -1.0), (je, pe), (jf, pf), (jq, -pf)], -np.inf, 0)
    # 8. 微电网共享预算
    for gmem, bud in zip(I["mg_members"], I["B_mg"]):
        co = []
        for i in gmem:
            for t in range(T): co.append((col("c", i, t), 1.0))
        if co: add(co, -np.inf, float(bud))
    # 9. 覆盖指示 x_i(t) = Σ_j z_ij 1[t∈W_ij]，对每个时隙都加（右侧为空时 x_i(t)=0，避免自由变量被当成免费覆盖）
    for i in range(N):
        for t in range(T):
            co = [(off["z"] + s, 1.0) for s, (si, d, ws, we, o1, o2) in enumerate(S) if si == i and ws <= t < we]
            add([(col("X", i, t), 1.0)] + [(j, -v) for j, v in co], 0, 0)
    # 10. 无人机容量（按起降点、按基本区间）
    edges = sorted({e for (_, d, ws, we, o1, o2) in S for e in (o1, o2)})
    for b in range(nB):
        for a in range(len(edges) - 1):
            l, rr = edges[a], edges[a + 1]
            co = [(off["z"] + s, 1.0) for s, (si, d, ws, we, o1, o2) in enumerate(S)
                  if I["base_of"][si] == b and o1 <= l and rr <= o2]
            if not co: continue
            co = list(dict.fromkeys(co))
            add(co, -np.inf, float(I["U_b"][b]))
    # ---- 目标 ----
    cvec = np.zeros(nvar)
    for i in range(N):
        w = I["w"][i]; ph = I["phi"][i]; ri = I["rho"][i]
        for t in range(T):
            cvec[col("eE", i, t)] = -w[t] * dt * ri
            cvec[col("eF", i, t)] = -w[t] * dt
            cvec[col("X", i, t)] = -w[t] * dt * ph
            cvec[col("th", i, t)] = w[t] * dt * ph
    A = csr_matrix((data, (rows, cols)), shape=(r, nvar))
    cons = [LinearConstraint(A, np.array(lo), np.array(hi))]
    bnds = Bounds(lb, ub)
    res = milp(c=cvec, constraints=cons, bounds=bnds, integrality=integrality,
               options=dict(time_limit=time_limit, mip_rel_gap=0.0, presolve=True))
    return res, seqs, off, nvar, A, np.array(lo), np.array(hi)

def obj_value(I, off, x_vars, nvar, seqs=None):
    """由 MILP 解重建目标值（用于与 BRIDGE 对齐口径）。"""
    N, T, dt = I["N"], I["T"], I["dt"]
    total = 0.0
    for i in range(N):
        w = I["w"][i]; ph = I["phi"][i]; ri = I["rho"][i]
        X = x_vars[off["X"] + i * T: off["X"] + (i + 1) * T]
        E = x_vars[off["eE"] + i * T: off["eE"] + (i + 1) * T]
        F = x_vars[off["eF"] + i * T: off["eF"] + (i + 1) * T]
        beta = ri * E + F
        total += float((w * (beta + (1 - beta) * ph * X)).sum() * dt)
    return total

CONFIGS = {
    # 名称: (站点数, 油机车数, 每车油机数, 无人机数)
    "A": (10, 1, 2, 1),
    "B": (12, 2, 2, 2),
}

def small_instance(key, seed, cfgname):
    """以离灾害中心最近的站点为圆心取最近的 n 座站；无人机放在服务站点最多的起降点。"""
    n, K, C, U = CONFIGS[cfgname]
    I, EA = get_instance(key, seed)
    lat0 = (I["cfg"]["bbox"][0] + I["cfg"]["bbox"][2]) / 2
    haz = np.array(SC.proj(I["cfg"]["hazard"][0], I["cfg"]["hazard"][1], lat0)).reshape(-1)
    xy = np.array([SC.proj(s["lat"], s["lon"], lat0) for s in I["sites"]]).reshape(-1, 2)
    idx = list(np.argsort(np.hypot(xy[:, 0] - haz[0], xy[:, 1] - haz[1]))[:n])
    Sm, EAm = subset_instance(I, EA, idx, K, C, None)
    cnt = np.bincount(Sm["base_of"][Sm["phi"] > 0], minlength=len(Sm["U_b"])) if (Sm["phi"] > 0).any() else np.zeros(len(Sm["U_b"]), int)
    ub = [0] * len(Sm["U_b"])
    for u in range(U): ub[int(np.argsort(-cnt, kind="stable")[u % len(ub)])] += 1
    Sm["U_b"] = ub
    return Sm, EAm

def one(args):
    key, seed, cfgname, tl = args
    from runlib import run_alg
    Sm, EAm = small_instance(key, seed, cfgname)
    J = {a: run_alg(a, Sm, EAm)["J"] for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]}
    t0 = time.time(); res, seqs, off, nvar, _A, _lo, _hi = build_milp(Sm, EAm, time_limit=tl); t_milp = time.time() - t0
    inc = obj_value(Sm, off, res.x, nvar) if res.x is not None else None
    db = getattr(res, "mip_dual_bound", None)
    ub = -float(db) if db is not None and np.isfinite(db) else None
    if res.status == 0 and inc is not None: ub = inc
    # 间隙分解：把 MILP 解的路径交给 BRIDGE 的下层，得到“路径最优、下层用 BRIDGE”时的值
    J_low_opt = None
    if res.x is not None:
        ab = np.full(Sm["N"], Sm["T"] * 10.0)
        for k in range(Sm["K"]):
            base = k * len(seqs)
            for s_, sq in enumerate(seqs):
                if sq and res.x[base + s_] > 0.5:
                    for i, a in sq: ab[i] = min(ab[i], a)
        J_low_opt = BR.lower_level(Sm, ab, {})["J"]
    J_nogen = BR.lower_level(Sm, np.full(Sm["N"], Sm["T"] * 10.0), {})["J"]
    return dict(key=key, seed=seed, cfg=cfgname, N=Sm["N"], K=Sm["K"], C=Sm["C"], U=int(sum(Sm["U_b"])),
                n_seq=len(seqs), n_var=nvar, status=int(res.status), time=t_milp, J_inc=inc, J_ub=ub,
                exact=bool(res.status == 0), J=J, Wtot=float(Sm["w"].sum() * Sm["dt"]),
                J_low_opt=J_low_opt, J_nogen=J_nogen)

if __name__ == "__main__":
    from multiprocessing import Pool
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    tasks = [(k, s, c, tl) for c in CONFIGS for k in ["luding", "xuwen"] for s in range(10)]
    out = []
    with Pool(nw) as p:
        for r in p.imap_unordered(one, tasks):
            out.append(r)
            g = (100 * (1 - r["J"]["BRIDGE"] / r["J_ub"])) if r["J_ub"] else float("nan")
            print("%s %s seed=%d status=%d %.0fs exact=%s  BRIDGE gap<=%.3f%%" % (r["cfg"], r["key"], r["seed"], r["status"], r["time"], r["exact"], g), flush=True)
    os.makedirs(os.path.join(RES, "milp"), exist_ok=True)
    out.sort(key=lambda r: (r["cfg"], r["key"], r["seed"]))
    json.dump(out, open(os.path.join(RES, "milp", "milp_all.json"), "w"), ensure_ascii=False, indent=1)
