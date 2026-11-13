"""后验最优性上界（命题“后验上界”）与外溢价格界（命题“外溢的价格界”）的数值检验。

上界：J* <= F0(pi) + sum of top-(K*C) single-station marginals Δ0(i @ a_i^min | pi) + U_max，
其中 F0 为无人机覆盖取 0、能量层取线性规划松弛时的目标值，a_i^min 为油机车从车库直达站点 i 的最早开始供电时刻，
U_max 为所有站点全程无基站服务时接力层的最优值。对任意可行路径 pi 成立。

用法：python3 certificate.py [NW] [small]
输出：results/certificate.json
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
import bridge as BR
from runlib import get_instance, run_alg

RES = os.path.join(os.path.dirname(__file__), "results")


def group_items0(I, members):
    return BR._grp_items(I, members, np.zeros((I["N"], I["T"])), "full")


def lp_take(items, budget, elig):
    """线性规划松弛：按效率序取前缀，临界增量取分数。返回 (值, 被取能耗, 临界价格)。"""
    eff, val, cost = items[0], items[1], items[2]
    c = cost * elig; cc = np.cumsum(c)
    full = elig & (cc <= budget + 1e-12)
    v = float((val * full).sum()); e = float((cost * full).sum())
    rest = np.where(elig & ~full)[0]
    price = 0.0
    if len(rest):
        k = rest[0]; frac = max(0.0, budget - e) / cost[k]
        v += frac * val[k]; e += frac * cost[k]; price = float(eff[k])
    return v, e, price


def group_value(I, members, items, budget, ab):
    T, dt = I["T"], I["dt"]
    lim = np.minimum(np.asarray(ab, float).astype(int), T)
    own, tsl = items[4], items[5]
    elig = tsl < lim[own]
    v, _, price = lp_take(items, budget, elig)
    direct = sum(float(I["w"][i, lim[i]:].sum() * dt) for i in members)
    return v + direct, price


def F0(I, ab, cache):
    tot = 0.0
    for g, (mem, bud) in enumerate(zip(I["mg_members"], I["B_mg"])):
        if not mem: continue
        tot += group_value(I, mem, cache[g], float(bud), ab)[0]
    return tot


def earliest_abar_pos(I, EA, grid=3):
    """a[p, i]：站点 i 作为某辆车第 p+1 站时的最早开始供电时刻（FIFO 下的动态规划下界）。"""
    N, T = I["N"], I["T"]; Cmax = max(BR.cap_of(I, k) for k in range(I["K"]))
    Q = EA.shape[1]; off = I.get("t_off", 0)
    arr = np.full((Cmax, N), np.inf)
    starts = [I["truck_starts"][k] if I.get("truck_starts") is not None else (I["truck_start"], 0) for k in range(I["K"])]
    for t, src in starts:
        q = int(np.ceil((t + off) / grid - 1e-9))
        if q < Q: arr[0] = np.minimum(arr[0], np.ceil(EA[src, q, :] - off - 1e-9))
    for p in range(1, Cmax):
        for j in range(N):
            if not np.isfinite(arr[p - 1, j]): continue
            q = int(np.ceil((arr[p - 1, j] + I["sigma_u"] + off) / grid - 1e-9))
            if q >= Q: continue
            arr[p] = np.minimum(arr[p], np.ceil(EA[j + 1, q, :] - off - 1e-9))
    return arr + I["sigma"]


def relax_ub(I, EA, integral=False, time_limit=120, uav=False):
    """放松路径耦合的上界：每站至多选一个“站位” p，开始供电时刻取第 p 站的最早可达时刻，
    每个站位至多 K_p 个站点；能量层保留共享预算线性规划。对任意可行路径 pi 有 F0(pi) <= 该值。"""
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    N, T, dt = I["N"], I["T"], I["dt"]
    apos = earliest_abar_pos(I, EA); Cm = apos.shape[0]
    Kp = [sum(1 for k in range(I["K"]) if BR.cap_of(I, k) > p) for p in range(Cm)]
    W = I["w"] * dt
    tail = np.c_[np.cumsum(W[:, ::-1], axis=1)[:, ::-1], np.zeros(N)]
    # 变量：y[i,p] 与各增量 u
    ny = N * Cm
    yv = np.zeros(ny)
    for i in range(N):
        for p in range(Cm):
            a = apos[p, i]
            if np.isfinite(a) and a < T: yv[i * Cm + p] = tail[i, int(a)]
    items = []
    for g, (mem, bud) in enumerate(zip(I["mg_members"], I["B_mg"])):
        if mem: items.append((g, float(bud), group_items0(I, mem)))
    vals, costs, owns, tsls, kinds, grp = [], [], [], [], [], []
    for g, bud, it in items:
        vals.append(it[1]); costs.append(it[2]); kinds.append(it[3]); owns.append(it[4]); tsls.append(it[5])
        grp.append(np.full(len(it[1]), g))
    vals = np.concatenate(vals); costs = np.concatenate(costs); kinds = np.concatenate(kinds)
    owns = np.concatenate(owns); tsls = np.concatenate(tsls); grp = np.concatenate(grp)
    nu = len(vals); nv = ny + nu
    c = -np.r_[yv, vals]
    R, Cc, V, ub = [], [], [], []
    r = 0
    for i in range(N):                                   # 每站至多一个站位
        for p in range(Cm): R.append(r); Cc.append(i * Cm + p); V.append(1.0)
        ub.append(1.0); r += 1
    for p in range(Cm):                                  # 每个站位至多 K_p 站
        for i in range(N): R.append(r); Cc.append(i * Cm + p); V.append(1.0)
        ub.append(Kp[p]); r += 1
    for g, bud, _ in items:                              # 共享预算
        idx = np.where(grp == g)[0]
        R += [r] * len(idx); Cc += list(ny + idx); V += list(costs[idx]); ub.append(bud); r += 1
    # 电池增量只能用在未接油机的时隙：u + sum_{p: a_ip <= t} y_ip <= 1
    first = np.where(kinds != 1)[0]
    R += list(r + np.arange(len(first))); Cc += list(ny + first); V += [1.0] * len(first)
    for n_, k in enumerate(first):
        i, t = owns[k], tsls[k]
        for p in range(Cm):
            if apos[p, i] <= t: R.append(r + n_); Cc.append(i * Cm + p); V.append(1.0)
    ub += [1.0] * len(first); r += len(first)
    # E→F 不早于 O→E
    key0 = {(owns[k], tsls[k]): k for k in np.where(kinds == 0)[0]}
    second = np.where(kinds == 1)[0]
    for n_, k in enumerate(second):
        R += [r + n_, r + n_]; Cc += [ny + k, ny + key0[(owns[k], tsls[k])]]; V += [1.0, -1.0]
    ub += [0.0] * len(second); r += len(second)
    if uav and sum(I["U_b"]) > 0:
        # 无人机：班次 z_s，有效覆盖 c_it <= sum_{s 覆盖 t} z_s，c_it <= 1 - g_it - beta_batt_it，收益 w*phi*c
        S = BR.build_sorties(I, np.zeros((N, T)), np.full(N, T * 10.0))
        nz = len(S); off_z = nv; off_c = nv + nz
        sites = sorted({s_[0] for s_ in S}); cidx = {}
        for i in sites:
            for t in range(T): cidx[(i, t)] = len(cidx)
        nc = len(cidx); nv2 = nv + nz + nc
        c = np.r_[c, np.zeros(nz), -np.array([W[i, t] * I["phi"][i] for (i, t) in cidx])]
        Tmax = max(s_[5] for s_ in S) + 1
        for b in range(len(I["U_b"])):
            ks = [k for k, s_ in enumerate(S) if I["base_of"][s_[0]] == b]
            if not ks: continue
            for tau in range(Tmax):
                act = [k for k in ks if S[k][4] <= tau < S[k][5]]
                if not act: continue
                R += [r] * len(act); Cc += [off_z + k for k in act]; V += [1.0] * len(act); ub.append(float(I["U_b"][b])); r += 1
        cover = {}
        for k, s_ in enumerate(S):
            for t in range(s_[2], s_[3]): cover.setdefault((s_[0], t), []).append(k)
        for (i, t), ci in cidx.items():
            ks = cover.get((i, t), [])
            R.append(r); Cc.append(off_c + ci); V.append(1.0)
            for k in ks: R.append(r); Cc.append(off_z + k); V.append(-1.0)
            ub.append(0.0); r += 1
        bcoef = np.where(kinds == 2, 1.0, np.where(kinds == 0, I["rho"][owns], 1 - I["rho"][owns]))
        rowof = {}
        for (i, t), ci in cidx.items():
            rowof[(i, t)] = r; R.append(r); Cc.append(off_c + ci); V.append(1.0)
            for p in range(Cm):
                if apos[p, i] <= t: R.append(r); Cc.append(i * Cm + p); V.append(1.0)
            ub.append(1.0); r += 1
        for k in range(nu):
            rr = rowof.get((owns[k], tsls[k]))
            if rr is not None: R.append(rr); Cc.append(ny + k); V.append(bcoef[k])
        nv = nv2
    A = coo_matrix((V, (R, Cc)), shape=(r, nv)).tocsr()
    if integral:
        integ = np.r_[np.ones(ny), np.zeros(nv - ny)]
        res = milp(c, constraints=LinearConstraint(A, -np.inf, np.array(ub)), bounds=Bounds(0, 1),
                   integrality=integ, options=dict(time_limit=time_limit))
        db = getattr(res, "mip_dual_bound", None)
        return float(-db) if db is not None and np.isfinite(db) else float(-res.fun)
    res = linprog(c, A_ub=A, b_ub=np.array(ub), bounds=(0, 1), method="highs")
    return float(-res.fun)


def u_max(I):
    if sum(I["U_b"]) == 0: return 0.0
    beta = np.zeros((I["N"], I["T"]))
    chosen, _, _ = BR.uav_relay(I, beta, np.full(I["N"], I["T"] * 10.0))
    return float(sum(s[6] for s in chosen))


def certificate(I, EA, ab):
    cache = [group_items0(I, mem) if mem else None for mem in I["mg_members"]]
    gof = {i: g for g, mem in enumerate(I["mg_members"]) for i in mem}
    base = F0(I, ab, cache)
    apos = earliest_abar_pos(I, EA)
    gv = {g: group_value(I, mem, cache[g], float(bud), ab)[0]
          for g, (mem, bud) in enumerate(zip(I["mg_members"], I["B_mg"])) if mem}
    Cmax = apos.shape[0]
    marg = np.zeros((Cmax, I["N"]))
    for p in range(Cmax):
        for i in range(I["N"]):
            a = apos[p, i]
            if not np.isfinite(a) or a >= min(ab[i], I["T"]): continue
            ab2 = ab.copy(); ab2[i] = a; g = gof[i]
            marg[p, i] = max(0.0, group_value(I, I["mg_members"][g], cache[g], float(I["B_mg"][g]), ab2)[0] - gv[g])
    KC = sum(BR.cap_of(I, k) for k in range(I["K"]))
    top_direct = float(np.sort(marg[0])[::-1][:KC].sum())
    from scipy.optimize import linear_sum_assignment
    cols = [p for k in range(I["K"]) for p in range(BR.cap_of(I, k))]
    Wm = marg[cols, :].T
    ri, ci = linear_sum_assignment(-Wm)
    top = float(Wm[ri, ci].sum())
    um = u_max(I)
    return dict(F0=base, top=top, top_direct=top_direct, umax=um, UB=base + top + um, KC=KC)


def spill(I, ab):
    """对每个由油机供电的站点 i：S = [U_g(ā)-U_g(∞)] + V_sel，检验 η_ā E_sel <= S <= η_∞ E_sel。"""
    out = []
    T, dt = I["T"], I["dt"]
    for g, (mem, bud) in enumerate(zip(I["mg_members"], I["B_mg"])):
        if not mem or len(mem) < 2: continue
        items = group_items0(I, mem); own, tsl = items[4], items[5]
        lim = np.minimum(np.asarray(ab, float).astype(int), T)
        for i in mem:
            if lim[i] >= T: continue
            others = (own != i) & (tsl < lim[own])
            e_inf = others | (own == i)
            e_a = others | ((own == i) & (tsl < lim[i]))
            v_inf, _, p_inf = lp_take(items, float(bud), e_inf)
            v_a, _, p_a = lp_take(items, float(bud), e_a)
            # 反事实 ∞ 下站点 i 在 [ā,T) 上被选中的增量（含分数）
            eff, val, cost = items[0], items[1], items[2]
            c = cost * e_inf; cc = np.cumsum(c)
            R = (own == i) & (tsl >= lim[i])
            frac = np.clip((float(bud) - (cc - c)) / np.maximum(cost, 1e-15), 0, 1) * e_inf
            Vsel = float((val * frac * R).sum()); Esel = float((cost * frac * R).sum())
            S = (v_a - v_inf) + Vsel
            D = float(I["w"][i, lim[i]:].sum() * dt)
            G = D + (v_a - v_inf)
            out.append(dict(S=S, lo=p_a * Esel, hi=p_inf * Esel, G=G, D=D, Vsel=Vsel))
    return out


def one(args):
    key, seed = args
    I, EA = get_instance(key, seed)
    t0 = time.time()
    rb = run_alg("BRIDGE", I, EA)
    t1 = time.time(); UB = relax_ub(I, EA, uav=True); t_ub = time.time() - t1
    ub0 = relax_ub(I, EA, uav=False) + u_max(I)
    sub = certificate(I, EA, rb["abar"])
    M = json.load(open(os.path.join(RES, "main", "%s_%d.json" % (key, seed))))
    J = {a: M["algs"][a]["J"] for a in ["GEN-EDF", "GEN-LS", "IND", "SEQ", "BRIDGE"]}
    W = float(I["w"].sum() * I["dt"])
    return dict(key=key, seed=seed, W=W, J=J, J_B_rerun=rb["J"], UB=UB, UB_sep=ub0, UB_sub=sub["UB"], sub=sub,
                t_ub=t_ub, spill=spill(I, rb["abar"]), time=time.time() - t0)


def seq_greedy(I, EA):
    """逐车贪心：每辆车在全部可行访问序列中选使 F0（无人机取 0、能量层取线性规划松弛）增量最大者。
    单车子问题精确求解（枚举），对应定理中 alpha = 1 的情形。"""
    import itertools
    N, T, C = I["N"], I["T"], I["C"]
    cache = [group_items0(I, mem) if mem else None for mem in I["mg_members"]]
    ab = np.full(N, T * 10.0); routes = []
    for k in range(I["K"]):
        best, bv = [], F0(I, ab, cache)
        for L in range(1, BR.cap_of(I, k) + 1):
            for perm in itertools.permutations(range(N), L):
                arr = BR.route_abar(I, EA, list(perm), k)
                if not all(np.isfinite(a) and a < T for _, a in arr): continue
                ab2 = ab.copy()
                for i, a in arr: ab2[i] = min(ab2[i], a)
                v = F0(I, ab2, cache)
                if v > bv + 1e-9: bv, best = v, list(perm)
        routes.append(best)
        for i, a in BR.route_abar(I, EA, best, k): ab[i] = min(ab[i], a)
    J = BR.lower_level(I, ab, {})["J"]
    return J, routes


def small_cert(args):
    """在最优性基准的小实例上检验上界不低于已证明的最优值，并运行逐车贪心。"""
    from milp_bench import small_instance
    key, seed, cfg = args
    Sm, EAm = small_instance(key, seed, cfg)
    t0 = time.time(); Jsg, _ = seq_greedy(Sm, EAm); tsg = time.time() - t0
    return dict(key=key, seed=seed, cfg=cfg, UB=relax_ub(Sm, EAm, uav=True), J_SG=Jsg, t_SG=tsg, umax=u_max(Sm))


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    if len(sys.argv) > 2 and sys.argv[2] == "small":
        with Pool(nw) as p:
            sm = list(p.imap_unordered(small_cert, [(k, s, c) for c in ("A", "B") for k in ["luding", "xuwen"] for s in range(10)]))
        C = json.load(open(os.path.join(RES, "certificate.json"))); C["small"] = sm
        json.dump(C, open(os.path.join(RES, "certificate.json"), "w"), default=float)
        print("done small"); sys.exit(0)
    out = []
    with Pool(nw) as p:
        for r in p.imap_unordered(one, [(k, s) for k in ["luding", "xuwen"] for s in range(20)]):
            out.append(r)
            print("%s %d gapB %.3f%% (sep %.3f%%, sub %.2f%%) t_ub %.1fs" % (
                r["key"], r["seed"], 100 * (r["UB"] - r["J"]["BRIDGE"]) / r["UB"],
                100 * (r["UB_sep"] - r["J"]["BRIDGE"]) / r["UB_sep"], 100 * (r["UB_sub"] - r["J"]["BRIDGE"]) / r["UB_sub"],
                r["t_ub"]), flush=True)
        sm = list(p.imap_unordered(small_cert, [(k, s, c) for c in ("A", "B") for k in ["luding", "xuwen"] for s in range(10)]))
    json.dump(dict(main=out, small=sm), open(os.path.join(RES, "certificate.json"), "w"), default=float)
    print("done certificate")
