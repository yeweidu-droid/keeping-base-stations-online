"""BRIDGE 算法、消融变体与对比算法。

状态编码：0=关断 O，1=节电 E，2=满功率 F（电池），3=油机供电。
"""
import math, time, itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

EPS = 1e-9

# ======================================================================
# 站点能量层（算法 1）
# ======================================================================
def energy_alloc(w, x, phi, rho, PF, PE, B, abar, dt, mode="full"):
    """返回长度 T 的状态数组。mode: full | noES（满功率直到耗尽）| nocouple（节电决策不看无人机）
    | thresh（剩余电量低于 30% 时转节电）。"""
    T = len(w); m = np.zeros(T, np.int8); tb = int(min(max(abar, 0), T)); m[tb:] = 3
    if tb <= 0: return m
    if mode == "noES":
        nF = int(math.floor(B / (PF * dt) + 1e-9)); m[:min(tb, nF)] = 2; return m
    if mode == "thresh":
        R = B
        for t in range(tb):
            if R >= 0.3 * B + PF * dt - 1e-12 and R >= PF * dt - 1e-12: m[t] = 2; R -= PF * dt
            elif R >= PE * dt - 1e-12: m[t] = 1; R -= PE * dt
            else: break
        return m
    om = w[:tb] * (1 - phi * x[:tb]) if mode != "nocouple" else w[:tb].copy()
    if PE >= rho * PF or mode == "onoff":    # 定理 2 情形 1：节电被支配（或不允许节电）
        order = np.argsort(-om, kind="stable"); cost = PF * dt
        n = int(math.floor(B / cost + 1e-9))
        sel = [t for t in order[:n] if om[t] > 0]; m[sel] = 2
        return m
    e1 = rho * om / PE; e2 = (1 - rho) * om / (PF - PE)
    effs = np.r_[e1, e2]; kinds = np.r_[np.zeros(tb, np.int8), np.ones(tb, np.int8)]
    ts = np.r_[np.arange(tb), np.arange(tb)]
    costs = np.r_[np.full(tb, PE * dt), np.full(tb, (PF - PE) * dt)]
    order = np.lexsort((kinds, -effs))
    R = B
    for k in order:
        if effs[k] <= 0: break
        t = ts[k]
        if kinds[k] == 0:
            if m[t] == 0 and costs[k] <= R + 1e-12: m[t] = 1; R -= costs[k]
        else:
            if m[t] == 1 and costs[k] <= R + 1e-12: m[t] = 2; R -= costs[k]
    return m

# ======================================================================
# 微电网共享储能（定理 2 的推广）：同组基站共用能量预算，阈值由组级价格唯一确定
# ======================================================================
def mg_on(I, flags=None):
    """是否启用共享储能（场景属性 + 消融开关 mg）。"""
    flags = flags or {}
    return float(I.get("mg_share", 0.0)) > 0 and flags.get("mg", True)

def _grp_items(I, members, x, mode="full"):
    """组内全部时隙的增量，按“收益/能耗”降序（同效率时 O→E 在前）。
    mode: full | onoff（无节电状态，仅满功率/关断）| nocouple（排序时忽略无人机覆盖，收益仍按真实覆盖计）。"""
    T, dt = I["T"], I["dt"]
    eff, val, cost, kind, own, tsl = [], [], [], [], [], []
    for i in members:
        omv = I["w"][i] * (1 - I["phi"][i] * x[i])
        om = omv if mode != "nocouple" else I["w"][i].copy()
        PF, PE, rho = I["PF"][i], I["PE"][i], I["rho"][i]
        if mode in ("noES", "onoff") or PE >= rho * PF:   # 单增量 O→F（kind=2）
            eff.append(om / PF); val.append(omv * dt); cost.append(np.full(T, PF * dt))
            kind.append(np.full(T, 2, np.int8))
            own.append(np.full(T, i)); tsl.append(np.arange(T))
        else:                                          # 双增量 O→E(kind=0) 与 E→F(kind=1)
            eff.append(rho * om / PE); val.append(rho * omv * dt); cost.append(np.full(T, PE * dt))
            kind.append(np.zeros(T, np.int8)); own.append(np.full(T, i)); tsl.append(np.arange(T))
            eff.append((1 - rho) * om / (PF - PE)); val.append((1 - rho) * omv * dt)
            cost.append(np.full(T, (PF - PE) * dt)); kind.append(np.ones(T, np.int8))
            own.append(np.full(T, i)); tsl.append(np.arange(T))
    eff = np.concatenate(eff); val = np.concatenate(val); cost = np.concatenate(cost)
    kind = np.concatenate(kind); own = np.concatenate(own); tsl = np.concatenate(tsl)
    o = np.lexsort((kind, -eff))
    return eff[o], val[o], cost[o], kind[o], own[o], tsl[o]

def _grp_take(items, budget, elig):
    """按效率序取前缀，返回 (取值, 被取下标的布尔数组)。同 (i,t) 的 O→E 必先于 E→F 被取，前置条件自动满足。"""
    cost = items[2]; val = items[1]
    cc = np.cumsum(cost * elig)
    take = elig & (cc <= budget + 1e-9)
    return float((val * take).sum()), take

def group_alloc(I, members, abar, x, budget, mode="full"):
    """共享预算下的组内能量分配。返回 N×T 状态（只填 members，油机后为 3）。"""
    T = I["T"]
    m = np.zeros((I["N"], T), np.int8)
    eff, val, cost, kind, own, tsl = _grp_items(I, members, x, mode)
    lim = np.minimum(np.asarray(abar, float).astype(int), T)
    elig = tsl < lim[own]
    _, take = _grp_take((eff, val, cost, kind, own, tsl), budget, elig)
    kk, ii, tt = kind[take], own[take], tsl[take]
    m[ii[kk == 0], tt[kk == 0]] = 1
    m[ii[kk > 0], tt[kk > 0]] = 2
    for i in members:
        m[i, lim[i]:] = 3
    return m

def rule_alloc_mg(I, members, abar, budget, mode):
    """规则型做法在共享储能下的执行：按时隙顺序从共享池取电，池内电量不足以供电时低优先级站点断电。
    优先级按站点重要度。这对应“现行做法能接入微电网、但不做跨站能量再分配”的情形。"""
    T, dt = I["T"], I["dt"]
    m = np.zeros((I["N"], T), np.int8)
    lim = np.minimum(np.asarray(abar, float).astype(int), T)
    order = sorted(members, key=lambda i: -I["kappa"][i])
    alive = {i: True for i in members}
    R = budget
    for t in range(T):
        for i in order:
            if t >= lim[i]:
                m[i, t] = 3; continue
            if not alive[i]: continue
            if mode == "thresh" and R < 0.3 * budget:
                st, need = 1, I["PE"][i] * dt
            else:
                st, need = 2, I["PF"][i] * dt
            if R >= need - 1e-12:
                R -= need; m[i, t] = st
            else:
                alive[i] = False
    return m

def alloc_all(I, abar, x, flags=None, mode="full"):
    """按微电网分组做共享预算分配；未启用共享时退化为逐站独立分配。"""
    if mg_on(I, flags):
        m = np.zeros((I["N"], I["T"]), np.int8)
        for mem, bud in zip(I["mg_members"], I["B_mg"]):
            if not mem: continue
            if mode in ("full", "onoff", "nocouple"):
                mm = group_alloc(I, mem, abar, x, float(bud), mode)
            else:
                mm = rule_alloc_mg(I, mem, abar, float(bud), mode)
            for i in mem: m[i] = mm[i]
        return m
    return np.array([energy_alloc(I["w"][i], x[i], I["phi"][i], I["rho"][i], I["PF"][i],
                                  I["PE"][i], I["B"][i], abar[i], I["dt"], mode) for i in range(I["N"])])

def beta_of(m, rho):
    b = np.zeros(m.shape)
    b[m >= 2] = 1.0
    if m.ndim == 1: b[m == 1] = rho
    else: b[m == 1] = np.broadcast_to(rho[:, None], m.shape)[m == 1]
    return b

def site_value(w, m, x, phi, rho, dt):
    b = beta_of(m, rho); return float((w * (b + (1 - b) * phi * x)).sum() * dt)

# ======================================================================
# 无人机接力层（定理 3）：区间矩阵线性规划，HiGHS 对偶单纯形给出整数解与对偶价格
# ======================================================================
def build_sorties(I, beta, abar, uav_off=None):
    N, T, dt = I["N"], I["T"], I["dt"]
    S = []
    for i in range(N):
        if I["phi"][i] <= 0: continue
        f, h = int(I["f"][i]), int(I["h"][i])
        lim = int(min(abar[i], T))
        d = I["uav_start"] if uav_off is None else uav_off
        while True:
            ws, we = d + f, min(d + f + h, T)
            if ws >= lim: break
            we2 = min(we, lim)
            xf = I["x_fixed"][i, ws:we2] if I.get("x_fixed") is not None else 0.0
            v = float((I["w"][i, ws:we2] * (1 - beta[i, ws:we2]) * (1 - xf)).sum() * I["phi"][i] * dt)
            if v > 1e-9:
                S.append((i, d, ws, we2, d, d + 2 * f + h + I["S"], v))
            d += h
    return S

def uav_relay(I, beta, abar, U_cap=None, uav_off=None):
    """返回 (选中班次列表, 覆盖 x[N,T], 影子价格 lam[b, τ])。U_cap[b][τ] 可为时变容量（在线用）。"""
    N, T = I["N"], I["T"]
    S = build_sorties(I, beta, abar, uav_off)
    nB = len(I["U_b"]); Tmax = T + 2 * int(I["f"].max()) + int(I["h"].max()) + I["S"] + 2
    lam = np.zeros((nB, Tmax)); x = np.zeros((N, T)); chosen = []
    for b in range(nB):
        idx = [k for k, s in enumerate(S) if I["base_of"][s[0]] == b]
        if not idx: continue
        U_cap = I.get("U_cap") if U_cap is None else U_cap
        cap = np.full(Tmax, float(I["U_b"][b])) if U_cap is None else np.r_[np.asarray(U_cap[b], float), np.full(max(0, Tmax - len(U_cap[b])), float(I["U_b"][b]))][:Tmax]
        if cap.max() <= 0: continue
        rows, cols = [], []
        for c, k in enumerate(idx):
            o1, o2 = S[k][4], min(S[k][5], Tmax)
            rows += list(range(o1, o2)); cols += [c] * (o2 - o1)
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(Tmax, len(idx))).tocsr()
        v = np.array([S[k][6] for k in idx])
        res = linprog(-v, A_ub=A, b_ub=cap, bounds=(0, 1), method="highs-ds")
        z = res.x > 0.5
        lam[b] = -np.asarray(res.ineqlin.marginals)
        for c, k in enumerate(idx):
            if z[c]:
                i, d, ws, we = S[k][0], S[k][1], S[k][2], S[k][3]
                x[i, ws:we] = 1.0; chosen.append(S[k])
    if I.get("x_fixed") is not None: x = np.maximum(x, I["x_fixed"])
    return chosen, x, lam

# ======================================================================
# 下层交替求解（算法 2）
# ======================================================================
def lower_level(I, abar, flags, max_iter=8, eps=1e-6):
    N, T, dt = I["N"], I["T"], I["dt"]
    emode = flags.get("emode", "full")
    use_uav = flags.get("uav", True) and sum(I["U_b"]) > 0
    x = np.zeros((N, T)) if I.get("x_fixed") is None else I["x_fixed"].copy()
    share = mg_on(I, flags)
    m = alloc_all(I, abar, x, flags, emode)
    J = total_value(I, m, x)
    chosen, lam = [], np.zeros((len(I["U_b"]), 1))
    hist = [J]
    if not use_uav:
        return dict(m=m, x=x, chosen=[], lam=lam, J=J, hist=hist)
    for it in range(max_iter if flags.get("bca", True) else 1):
        beta = beta_of(m, I["rho"])
        chosen, x, lam = uav_relay(I, beta, abar)
        if not flags.get("bca", True):
            J = total_value(I, m, x); hist.append(J); break
        if share:
            m2 = alloc_all(I, abar, x, flags, emode)
        else:
            m2 = m.copy()
            for i in range(N):
                cand = energy_alloc(I["w"][i], x[i], I["phi"][i], I["rho"][i], I["PF"][i], I["PE"][i], I["B"][i], abar[i], dt, emode)
                if site_value(I["w"][i], cand, x[i], I["phi"][i], I["rho"][i], dt) >= site_value(I["w"][i], m[i], x[i], I["phi"][i], I["rho"][i], dt) - 1e-12:
                    m2[i] = cand
        J2 = total_value(I, m2, x); hist.append(J2)
        if share and J2 < J - 1e-12:
            J = total_value(I, m, x); break      # 组内联动，退回较优解
        m = m2
        if J2 - J < eps * max(1.0, J): J = J2; break
        J = J2
    return dict(m=m, x=x, chosen=chosen, lam=lam, J=J, hist=hist)

def total_value(I, m, x):
    b = beta_of(m, I["rho"]); return float((I["w"] * (b + (1 - b) * I["phi"][:, None] * x)).sum() * I["dt"])

# ======================================================================
# 油机边际收益（命题 2 的推广：共享预算下含组内能量影子价格的边际收益）
# ======================================================================
def gains_mg(I, low, flags, abar):
    """G_i(ā) = [补上 ā 之后未被无人机覆盖的需求]
              + [站 i 少占的能量在微电网内重新分配后的净价值]
              + [腾出无人机时间 Λ]
    第二项在共享预算下对所有组内站点都非零，是耦合的真正来源。"""
    N, T, dt = I["N"], I["T"], I["dt"]
    x = low["x"]; lam = low["lam"]
    G = np.zeros((N, T + 1))
    lim = np.minimum(np.asarray(abar, float).astype(int), T)
    price = {}
    if flags.get("price", True) and flags.get("uav", True):
        for s in low["chosen"]:
            i, d, ws, we, o1, o2, v = s
            b = I["base_of"][i]
            price.setdefault(i, []).append((ws, float(lam[b, o1:min(o2, lam.shape[1])].sum())))
    for g_mem, bud in zip(I["mg_members"], I["B_mg"]):
        if not g_mem: continue
        items = _grp_items(I, g_mem, x, flags.get("emode", "full"))
        eff, val, cost, kind, own, tsl = items
        static = tsl < lim[own]
        for i in g_mem:
            is_i = (own == i)
            other = static & (~is_i)
            _, take_inf = _grp_take(items, bud, static | is_i)     # 反事实：站 i 全时段靠电池
            base_inf = float((val * take_inf).sum())
            w1 = I["w"][i] * (1 - I["phi"][i] * x[i]) * dt
            tail = np.r_[np.cumsum(w1[::-1])[::-1], 0.0]           # Σ_{t≥a} w(1-φx)Δ
            gi = np.zeros(T + 1)
            for a in range(T + 1):
                v, _ = _grp_take(items, bud, other | (is_i & (tsl < a)))
                gi[a] = v - base_inf
            Lamv = np.zeros(T + 1)
            for ws, pv in price.get(i, []):
                Lamv[:ws + 1] += pv
            G[i] = tail + gi + Lamv
    G[:, T] = 0.0
    return np.maximum(G, 0.0)

def gains(I, low, flags, abar=None):
    if mg_on(I, flags) and flags.get("emode", "full") in ("full", "onoff", "nocouple") and abar is not None \
            and flags.get("gain_mg", True):
        return gains_mg(I, low, flags, abar)
    N, T, dt = I["N"], I["T"], I["dt"]
    emode = flags.get("emode", "full")
    x = low["x"]; lam = low["lam"]
    G = np.zeros((N, T + 1))
    a = np.arange(T + 1)
    # 班次的影子价值
    price = {}
    if flags.get("price", True) and flags.get("uav", True):
        for s in low["chosen"]:
            i, d, ws, we, o1, o2, v = s
            b = I["base_of"][i]
            price.setdefault(i, []).append((ws, float(lam[b, o1:min(o2, lam.shape[1])].sum())))
    for i in range(N):
        w = I["w"][i]; phi = I["phi"][i]; rho = I["rho"][i]; PF = I["PF"][i]; PE = I["PE"][i]; B = I["B"][i]
        xi = x[i]
        base = np.r_[0.0, np.cumsum(w * phi * xi * dt)]             # 全关断时的无人机收益前缀和
        if emode == "noES":
            nF = int(math.floor(B / (PF * dt) + 1e-9))
            fpre = np.r_[0.0, np.cumsum(w * (1 - phi * xi) * dt * (np.arange(T) < nF))]
            VB = base[a] + fpre[a]
        else:
            om = w * (1 - phi * xi) if emode != "nocouple" else w.copy()
            omv = w * (1 - phi * xi)            # 实际收益仍按真实覆盖计
            if PE >= rho * PF or emode == "onoff":
                eff = om / PF; ts = np.arange(T); gain = omv * dt; cost = np.full(T, PF * dt)
                order = np.argsort(-eff, kind="stable")
            else:
                eff = np.r_[rho * om / PE, (1 - rho) * om / (PF - PE)]
                ts = np.r_[np.arange(T), np.arange(T)]
                gain = np.r_[rho * omv * dt, (1 - rho) * omv * dt]
                cost = np.r_[np.full(T, PE * dt), np.full(T, (PF - PE) * dt)]
                kinds = np.r_[np.zeros(T), np.ones(T)]
                order = np.lexsort((kinds, -eff))
            ts, gain, cost, eff = ts[order], gain[order], cost[order], eff[order]
            keep = eff > 0; ts, gain, cost = ts[keep], gain[keep], cost[keep]
            M = ts[None, :] < a[:, None]
            cc = np.cumsum(cost[None, :] * M, axis=1)
            take = M & (cc <= B + 1e-9)
            VB = base[a] + (gain[None, :] * take).sum(1)
        tail = np.r_[np.cumsum(w[::-1] * dt)[::-1], 0.0]
        V = VB + tail
        Vinf = VB[T]
        Lam = np.zeros(T + 1)
        for ws, pv in price.get(i, []):
            Lam[: ws + 1] += pv
        G[i] = V - Vinf + Lam
        G[i, T] = 0.0
    return np.maximum(G, 0.0)

# ======================================================================
# 油机路径：到达时刻、初始路径、改进（算法 3）
# ======================================================================
def route_abar(I, EA, route, k=0, grid=3):
    """按到达时刻递推式计算到达与开始供电时刻（相对时间）。在线子问题通过 truck_starts/t_off 指定每车起点。"""
    T = I["T"]; off = I.get("t_off", 0)
    if I.get("truck_starts") is not None: t, src = I["truck_starts"][k]
    else: t, src = I["truck_start"], 0
    out = []; Q = EA.shape[1]
    for i in route:
        if not np.isfinite(t):
            out.append((i, math.inf)); continue
        q = int(math.ceil((t + off) / grid - 1e-9))
        arr = EA[src, q, i] - off if q < Q else math.inf
        if not np.isfinite(arr):
            out.append((i, math.inf)); t = math.inf; continue
        arr = int(math.ceil(arr - 1e-9)); ab = arr + I["sigma"]
        out.append((i, ab)); t = arr + I["sigma_u"]; src = i + 1
    return out

def cap_of(I, k):
    return I["C_k"][k] if I.get("C_k") is not None else I["C"]

def abar_of(I, EA, routes, fixed=None):
    ab = np.full(I["N"], float(I["T"]) * 10)
    for k, r in enumerate(routes):
        for i, a in route_abar(I, EA, r, k):
            ab[i] = a if np.isfinite(a) else I["T"] * 10
    for i, a in (I.get("fixed_abar") or {}).items(): ab[i] = a
    if fixed:
        for i, a in fixed.items(): ab[i] = a
    return ab

def routes_value(I, EA, routes, G):
    T = I["T"]; v = 0.0
    for k, r in enumerate(routes):
        for i, a in route_abar(I, EA, r, k):
            if np.isfinite(a) and a < T: v += G[i, int(a)]
    return v

def edf_routes(I, EA):
    """现行做法：按重要度降序、满功率耗尽时刻升序排序，逐站分给最早能到达的油机车（追加到车尾）。"""
    N = I["N"]; dep = I["B"] / (I["PF"] * I["dt"])
    fixed = set((I.get("fixed_abar") or {}).keys())
    order = sorted([i for i in range(N) if i not in fixed], key=lambda i: (-I["kappa"][i], dep[i]))
    routes = [[] for _ in range(I["K"])]
    for i in order:
        best, bk = math.inf, None
        for k in range(I["K"]):
            if len(routes[k]) >= cap_of(I, k): continue
            ab = route_abar(I, EA, routes[k] + [i], k)[-1][1]
            if ab < best: best, bk = ab, k
        if bk is not None and best < I["T"]:
            routes[bk].append(i)
        if all(len(routes[k]) >= cap_of(I, k) for k in range(I["K"])): break
    return routes

def route_improve(I, EA, routes, G, eta=0.5, max_sweeps=30):
    K, N = I["K"], I["N"]
    fixed = set((I.get("fixed_abar") or {}).keys())
    routes = [list(r) for r in routes]
    cur = routes_value(I, EA, routes, G)
    def val(rs): return routes_value(I, EA, rs, G)
    # 后悔值插入
    while True:
        routed = {i for r in routes for i in r} | fixed
        if all(len(routes[k]) >= cap_of(I, k) for k in range(K)): break
        best_sc, best_mv = -math.inf, None
        for i in range(N):
            if i in routed: continue
            deltas = []
            for k in range(K):
                if len(routes[k]) >= cap_of(I, k): continue
                for p in range(len(routes[k]) + 1):
                    rs = [list(r) for r in routes]; rs[k].insert(p, i)
                    deltas.append((val(rs) - cur, k, p))
            if not deltas: continue
            deltas.sort(reverse=True)
            d1 = deltas[0][0]; d2 = deltas[1][0] if len(deltas) > 1 else d1
            sc = d1 + eta * (d1 - d2)
            if d1 > 1e-9 and sc > best_sc: best_sc, best_mv = sc, (i, deltas[0][1], deltas[0][2])
        if best_mv is None: break
        i, k, p = best_mv; routes[k].insert(p, i); cur = val(routes)
    # 局部搜索
    for sweep in range(max_sweeps):
        improved = False
        # (1) 路径内全排列
        for k in range(K):
            if len(routes[k]) <= 1: continue
            bestp, bestv = routes[k], cur
            for perm in itertools.permutations(routes[k]):
                rs = [list(r) for r in routes]; rs[k] = list(perm); v = val(rs)
                if v > bestv + 1e-9: bestv, bestp = v, list(perm)
            if bestv > cur + 1e-9: routes[k] = bestp; cur = bestv; improved = True
        routed = [i for r in routes for i in r]; unrouted = [i for i in range(N) if i not in set(routed) | fixed]
        # (2) 以未访问站点替换已访问站点
        best = (cur, None)
        for k in range(K):
            for p in range(len(routes[k])):
                for j in unrouted:
                    rs = [list(r) for r in routes]; rs[k][p] = j; v = val(rs)
                    if v > best[0] + 1e-9: best = (v, rs)
        if best[1] is not None: routes, cur, improved = best[1], best[0], True
        # (3) 跨车移位与交换
        best = (cur, None)
        for a in range(K):
            for pa in range(len(routes[a])):
                for b in range(K):
                    if a == b: continue
                    if len(routes[b]) < cap_of(I, b):
                        for pb in range(len(routes[b]) + 1):
                            rs = [list(r) for r in routes]; i = rs[a].pop(pa); rs[b].insert(pb, i); v = val(rs)
                            if v > best[0] + 1e-9: best = (v, rs)
                    for pb in range(len(routes[b])):
                        rs = [list(r) for r in routes]; rs[a][pa], rs[b][pb] = rs[b][pb], rs[a][pa]; v = val(rs)
                        if v > best[0] + 1e-9: best = (v, rs)
        if best[1] is not None: routes, cur, improved = best[1], best[0], True
        # (4) 有余量时插入未访问站点
        for k in range(K):
            while len(routes[k]) < cap_of(I, k):
                routed = {i for r in routes for i in r} | fixed; bb = (cur, None)
                for j in range(N):
                    if j in routed: continue
                    for p in range(len(routes[k]) + 1):
                        rs = [list(r) for r in routes]; rs[k].insert(p, j); v = val(rs)
                        if v > bb[0] + 1e-9: bb = (v, rs)
                if bb[1] is None: break
                routes, cur, improved = bb[1], bb[0], True
        if not improved: break
    return routes

# ======================================================================
# 联合邻域精修：以真实下层目标为评估，对“服务集合 + 组内分配”做精确邻域搜索
# ======================================================================
def routes_from_set(I, EA, served):
    """把给定站点集合排成路径：按重要度/耗尽时刻贪心分车，车内在到达价值下取最优排列。"""
    T, K, C = I["T"], I["K"], I["C"]
    dep = I["B"] / (I["PF"] * I["dt"])
    order = sorted(served, key=lambda i: (-I["kappa"][i], dep[i]))
    routes = [[] for _ in range(K)]
    for i in order:
        best, bk = math.inf, None
        for k in range(K):
            if len(routes[k]) >= cap_of(I, k): continue
            ab = route_abar(I, EA, routes[k] + [i], k)[-1][1]
            if ab < best: best, bk = ab, k
        if bk is not None and best < T: routes[bk].append(i)
    for k in range(K):
        if len(routes[k]) > 1:
            bp, bv = routes[k], None
            for pm in itertools.permutations(routes[k]):
                v = sum(I["kappa"][i] * a for i, a in route_abar(I, EA, list(pm), k) if np.isfinite(a))
                if bv is None or v < bv: bv, bp = v, list(pm)
            routes[k] = bp
    return routes

def refine_exact(I, EA, routes, flags, low, ab, G, max_eval=60, sweeps=3, verbose=False, gfun=None):
    """1-swap 邻域：按含影子价格的边际收益排序候选，只对前 max_eval 个做真实下层评估。
    gfun(ab, low) 可替换候选排序所用的边际收益（默认与 flags 一致）。"""
    curJ = low["J"]
    cur = [list(r) for r in routes]
    served = {i for r in cur for i in r}
    alls = set(range(I["N"]))
    for sw in range(sweeps):
        scored = []
        for out in sorted(served):
            for inn in sorted(alls - served):
                rs = routes_from_set(I, EA, (served - {out}) | {inn})
                if len({i for r in rs for i in r}) < len(served): continue
                abx = abar_of(I, EA, rs)
                v = sum(G[i, int(abx[i])] for i in {i for r in rs for i in r} if abx[i] < I["T"])
                scored.append((v, rs))
        if not scored: break
        scored.sort(key=lambda x: -x[0])
        best = None
        for v, rs in scored[:max_eval]:
            abx = abar_of(I, EA, rs)
            lx = lower_level(I, abx, flags)
            if lx["J"] > curJ + 1e-9 and (best is None or lx["J"] > best[0]):
                best = (lx["J"], rs, abx, lx)
        if best is None: break
        curJ, cur, ab, low = best[0], [list(r) for r in best[1]], best[2], best[3]
        served = {i for r in cur for i in r}
        G = gains(I, low, flags, ab) if gfun is None else gfun(ab, low)
        if verbose: print("    [exact] sweep %d -> J=%.0f" % (sw, curJ), flush=True)
    return cur, ab, low, curJ

# ======================================================================
# BRIDGE 主循环（算法 4）与各变体
# ======================================================================
def run_bridge(I, EA, flags=None, max_outer=10, log=False):
    flags = dict(flags or {})
    t0 = time.time(); tim = dict(energy_uav=0.0, gain=0.0, route=0.0)
    routes = flags.get("init_routes") or edf_routes(I, EA)
    ab = abar_of(I, EA, routes)
    t = time.time(); low = lower_level(I, ab, flags); tim["energy_uav"] += time.time() - t
    if flags.get("route", True) and flags.get("multistart", True) and not flags.get("route_blind", False) \
            and not (flags.get("emode") == "noES" and not flags.get("uav", True)):
        # 多起点：再以"仅电池收益"的路径优化结果为起点，取下层评估较优者
        t = time.time()
        g = run_bridge(I, EA, dict(route=True, emode="noES", uav=False, price=False, multistart=False,
                                   exact=False, init_routes=flags.get("init_routes")), max_outer=max_outer)
        ab2 = abar_of(I, EA, g["routes"]); low2 = lower_level(I, ab2, flags); tim["route"] += time.time() - t
        if low2["J"] > low["J"]: routes, ab, low = g["routes"], ab2, low2
    hist = [low["J"]]
    if flags.get("route", True):
        for it in range(max_outer):
            t = time.time()
            gflags = flags if not flags.get("route_blind", False) else dict(flags, emode="noES", uav=False, price=False)
            glow = low if not flags.get("route_blind", False) else lower_level(I, ab, gflags)
            G = gains(I, glow, gflags, ab); tim["gain"] += time.time() - t
            t = time.time(); r2 = route_improve(I, EA, routes, G); tim["route"] += time.time() - t
            ab2 = abar_of(I, EA, r2)
            t = time.time(); low2 = lower_level(I, ab2, flags); tim["energy_uav"] += time.time() - t
            if low2["J"] > low["J"] * (1 + 1e-6):
                routes, ab, low = r2, ab2, low2; hist.append(low["J"])
            else:
                break
    # 联合邻域精修：路径层的最后一步用真实下层目标评估（含共享预算下的组内联动）
    if flags.get("exact", True) and flags.get("route", True) and not flags.get("route_blind", False) \
            and not (flags.get("emode") == "noES" and not flags.get("uav", True)):
        t = time.time()
        Gx = gains(I, low, flags, ab)
        routes, ab, low, _ = refine_exact(I, EA, routes, flags, low, ab, Gx,
                                          max_eval=flags.get("exact_eval", 30),
                                          sweeps=flags.get("exact_sweeps", 4))
        if low["J"] > hist[-1]: hist.append(low["J"])
        tim["exact"] = time.time() - t
    return dict(routes=routes, abar=ab, low=low, J=low["J"], hist=hist, time=time.time() - t0, tim=tim)

ALGS = {
    # 对比算法
    "GEN-EDF": dict(route=False, emode="noES", uav=False),
    "GEN-LS":  dict(route=True, emode="noES", uav=False, price=False),
    "SEQ":     dict(route=True, route_blind=True),
    "BRIDGE":  dict(),
    # 消融
    "w/o ES":        dict(emode="onoff"),          # 去掉节电状态，保留按组级价格的满功率/休眠分配
    "w/o EA":        dict(emode="noES"),           # 去掉能量层：满功率运行至组内电量耗尽
    "w/o refine":    dict(exact=False),            # 只用以 G 为目标的主循环，不做联合邻域精修
    "w/o grpprice":  dict(gain_mg=False),          # 边际收益不含组内能量再分配项（按单站电池计算）
    "w/o UAV":       dict(uav=False),
    "w/o coupling":  dict(emode="nocouple"),
    "w/o price":     dict(price=False),
    "w/o BCA":       dict(bca=False),
    "w/o routing":   dict(route=False),
    "w/o MG":        dict(mg=False),
}

BLIND = dict(emode="noES", uav=False, price=False)

def run_seq_refine(I, EA, grp_rank=False, max_eval=30, sweeps=4):
    """SEQ + 联合邻域精修：路径层与 SEQ 相同（GEN-LS），再以真实下层目标做 1-swap 精修。
    grp_rank=False 时候选按单站电池边际收益排序（不含组级能量价格），True 时按 BRIDGE 的边际收益排序。"""
    t0 = time.time()
    g = run_bridge(I, EA, ALGS["GEN-LS"])
    ab = g["abar"]; low = lower_level(I, ab, {})
    if grp_rank:
        gfun = lambda a, lw: gains(I, lw, {}, a)
    else:
        gfun = lambda a, lw: gains(I, lower_level(I, a, BLIND), BLIND, a)
    routes, ab, low, _ = refine_exact(I, EA, g["routes"], {}, low, ab, gfun(ab, low), max_eval=max_eval,
                                      sweeps=sweeps, gfun=gfun)
    return dict(routes=routes, abar=ab, low=low, J=low["J"], hist=[low["J"]], time=time.time() - t0, tim={})

def run_sa(I, EA, budget_s, seed=0, T0_frac=2e-3, Tend_frac=1e-5):
    """在真实下层目标上直接搜索的模拟退火（通用元启发式对照）。
    起点与 BRIDGE 的多起点相同（现行排序路径与 GEN-LS 路径中下层评估较优者），墙钟预算 budget_s 内
    随机执行替换、移位、跨车交换与车内交换四类移动，按 Metropolis 准则接受，返回所见最优解。"""
    t0 = time.time(); rng = np.random.default_rng(seed)
    N, K, T = I["N"], I["K"], I["T"]
    cands = [edf_routes(I, EA), run_bridge(I, EA, ALGS["GEN-LS"])["routes"]]
    def ev(rs):
        ab = abar_of(I, EA, rs); return lower_level(I, ab, {}), ab
    best = None
    for rs in cands:
        lw, ab = ev(rs)
        if best is None or lw["J"] > best[0]: best = (lw["J"], [list(r) for r in rs], ab, lw)
    curJ, cur = best[0], [list(r) for r in best[1]]
    T0 = T0_frac * curJ; n_eval = 0
    while True:
        frac = (time.time() - t0) / budget_s
        if frac >= 1: break
        temp = T0 * (Tend_frac / T0_frac) ** frac
        rs = [list(r) for r in cur]
        served = {i for r in rs for i in r}
        unserved = [j for j in range(N) if j not in served]
        nonempty = [k for k in range(K) if rs[k]]
        mv = rng.integers(4)
        if not nonempty: break
        if mv == 0 and unserved:                                  # 替换：已服务站点换成未服务站点
            k = nonempty[rng.integers(len(nonempty))]; p = rng.integers(len(rs[k]))
            rs[k][p] = unserved[rng.integers(len(unserved))]
        elif mv == 1:                                             # 移位：移到任一有余量的车的任一位置
            a = nonempty[rng.integers(len(nonempty))]; i = rs[a].pop(rng.integers(len(rs[a])))
            ok = [k for k in range(K) if len(rs[k]) < cap_of(I, k)]
            b = ok[rng.integers(len(ok))]; rs[b].insert(rng.integers(len(rs[b]) + 1), i)
        elif mv == 2 and len(nonempty) >= 2:                      # 跨车交换
            a, b = rng.choice(nonempty, 2, replace=False)
            pa, pb = rng.integers(len(rs[a])), rng.integers(len(rs[b]))
            rs[a][pa], rs[b][pb] = rs[b][pb], rs[a][pa]
        else:                                                     # 车内交换
            k = nonempty[rng.integers(len(nonempty))]
            if len(rs[k]) < 2: continue
            p, q = rng.choice(len(rs[k]), 2, replace=False); rs[k][p], rs[k][q] = rs[k][q], rs[k][p]
        lw, ab = ev(rs); n_eval += 1
        d = lw["J"] - curJ
        if d >= 0 or rng.random() < math.exp(d / max(temp, 1e-12)):
            cur, curJ = rs, lw["J"]
            if curJ > best[0]: best = (curJ, [list(r) for r in rs], ab, lw)
    return dict(routes=best[1], abar=best[2], low=best[3], J=best[0], hist=[best[0]], time=time.time() - t0,
                tim={}, n_eval=n_eval)

def state_switches(I, m, abar):
    """电池阶段（油机供电前）各站工作状态的切换次数与平均驻留时长（时隙）。"""
    T = I["T"]; sw = []; dwell = []
    for i in range(I["N"]):
        lim = int(min(abar[i], T)); s = m[i, :lim]
        if lim <= 1: sw.append(0); continue
        ch = np.flatnonzero(np.diff(s) != 0); sw.append(len(ch))
        b = np.r_[0, ch + 1, lim]; dwell += list(np.diff(b))
    return dict(sw_mean=float(np.mean(sw)), dwell_mean=float(np.mean(dwell)) if dwell else float("nan"),
                dwell_short=float(np.mean(np.array(dwell) < 3)) if dwell else float("nan"))

def run_ind(I, EA):
    """IND：三类手段各自决策——油机按现行排序派遣；电量低于 30% 转节电；无人机在站点断电时被动接力。"""
    t0 = time.time()
    N, T, dt = I["N"], I["T"], I["dt"]
    routes = edf_routes(I, EA); ab = abar_of(I, EA, routes)
    m = alloc_all(I, ab, np.zeros((N, T)), {}, "thresh")     # 与 GEN-EDF 相同：接入微电网共享池，但不做跨站再分配
    x = np.zeros((N, T)); chosen = []
    nB = len(I["U_b"]); busy = [[] for _ in range(nB)]
    covered_until = np.zeros(N, int)
    for t in range(I["uav_start"], T):
        for b in range(nB):
            busy[b] = [e for e in busy[b] if e > t]
            free = I["U_b"][b] - len(busy[b])
            if free <= 0: continue
            cand = []
            for i in range(N):
                if I["base_of"][i] != b or I["phi"][i] <= 0: continue
                f, h = int(I["f"][i]), int(I["h"][i]); ta = t + f
                if ta >= T or ta >= ab[i]: continue
                if m[i, ta] == 0 and covered_until[i] <= ta:
                    cand.append((-I["w"][i, ta], i))
            cand.sort()
            for _, i in cand[:free]:
                f, h = int(I["f"][i]), int(I["h"][i]); ws, we = t + f, min(t + f + h, T, int(min(ab[i], T)))
                x[i, ws:we] = 1; covered_until[i] = we
                busy[b].append(t + 2 * f + h + I["S"])
                chosen.append((i, t, ws, we, t, t + 2 * f + h + I["S"], 0.0))
    low = dict(m=m, x=x, chosen=chosen, lam=np.zeros((nB, 1)), J=total_value(I, m, x), hist=[])
    return dict(routes=routes, abar=ab, low=low, J=low["J"], hist=[low["J"]], time=time.time() - t0, tim={})

# ======================================================================
# 评价指标
# ======================================================================
def metrics(I, res):
    m, x = res["low"]["m"], res["low"]["x"]; dt = I["dt"]
    b = beta_of(m, I["rho"]); s = b + (1 - b) * I["phi"][:, None] * x
    W = I["w"]
    tot = float((W * dt).sum())
    out = {}
    out["SR"] = float((W * s).sum() * dt / tot)
    out["WO_kuh"] = float((W * (1 - s)).sum() * dt / 1000.0)            # 千加权用户·小时
    crit = I["kappa"] >= 2
    out["crit_out_h"] = float(((1 - s[crit]).sum(1) * dt).mean()) if crit.any() else 0.0
    out["dark_site_h"] = float((s <= 1e-9).sum() * dt)
    per = (1 - s).sum(1) * dt
    out["per_site_out_h"] = per.tolist()
    out["max_out_h"] = float(per.max())
    sv = s.mean(1); out["jain"] = float(sv.sum() ** 2 / (len(sv) * (sv ** 2).sum()))
    out["curve"] = ((W * s).sum(0) / W.sum(0)).tolist()                  # 各时隙加权服务率
    out["alive_curve"] = (s > 0).sum(0).tolist()
    # 资源
    fl = 0
    for c in res["low"]["chosen"]:
        i = c[0]; fl += (2 * int(I["f"][i]) + int(I["h"][i]))
    out["uav_flight_h"] = fl * dt
    out["uav_cover_h"] = float(x.sum() * dt)
    ab = res["abar"]; served = ab < I["T"]
    out["n_gen"] = int(served.sum())
    out["gen_mean_arrival_h"] = float(ab[served].mean() * dt) if served.any() else float("nan")
    dep = I["B"] / (I["PF"] * dt)
    out["gen_before_depletion"] = int(((ab < dep) & served).sum())
    bat = (m < 3)
    out["es_frac"] = float(((m == 1) & bat).sum() / max(1, bat.sum()))
    out["sleep_under_uav_h"] = float(((m == 0) & (x > 0)).sum() * dt)
    out["time_s"] = res["time"]
    out["J"] = res["J"]
    return out
