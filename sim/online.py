"""在线滚动时域运行（算法 6）：电量遥测存在估计误差，按真实电量执行。"""
import math, numpy as np
import bridge as BR

def demand_error(I, sd, seed, rho_h=0.8):
    """需求预测的对数误差 ξ[N, 小时]：区域公共分量与逐站分量各占一半方差，均为平稳 AR(1)（相邻小时相关 rho_h）。
    规划方看到的需求为 w·exp(sd·ξ)。"""
    N, T, dt = I["N"], I["T"], I["dt"]; H = int(math.ceil(T * dt))
    rng = np.random.default_rng(30_000 + seed)
    def ar(n):
        e = np.zeros((n, H)); e[:, 0] = rng.normal(0, 1, n)
        for h in range(1, H): e[:, h] = rho_h * e[:, h - 1] + math.sqrt(1 - rho_h ** 2) * rng.normal(0, 1, n)
        return e
    xi = (ar(1) + ar(N)) / math.sqrt(2.0)
    return sd * xi

def demand_forecast(I, err, t_r, learn, rho_h=0.8):
    """t_r 时刻的需求预测（N×(T−t_r)）。learn=True 时用当前小时观测到的误差按 AR(1) 条件期望修正后续小时。"""
    T, dt = I["T"], I["dt"]
    hr = np.minimum((np.arange(t_r, T) * dt).astype(int), err.shape[1] - 1)
    e = err[:, hr]
    if learn and t_r > 0:
        h0 = min(int(t_r * dt), err.shape[1] - 1)
        e = e - err[:, [h0]] * rho_h ** np.maximum(hr - h0, 0)[None, :]
    return I["w"][:, t_r:] * np.exp(e)

def make_sub(I, EA, t_r, B_plan, st, w_plan=None):
    T = I["T"]; Tp = T - t_r; N = I["N"]; nB = len(I["U_b"])
    S = dict(I)
    S["T"] = Tp; S["w"] = I["w"][:, t_r:] if w_plan is None else w_plan; S["B"] = np.asarray(B_plan, float).copy(); S["t_off"] = t_r
    # 微电网共享预算：子问题的组预算必须按当前规划电量重算，否则会高估可用能量
    if S.get("mg_members"):
        S["B_mg"] = np.array([S["B"][m].sum() for m in S["mg_members"]])
    S["fixed_abar"] = {i: max(0, a - t_r) for i, a in st["gen_abar"].items()}
    S["truck_starts"] = [(max(t_av - t_r, 0), src) for (t_av, src) in st["truck_pos"]]
    S["C_k"] = list(st["truck_cap"])
    S["uav_start"] = max(0, I["uav_start"] - t_r)
    Tmax = Tp + 40; caps = []
    for b in range(nB):
        c = np.full(Tmax, float(I["U_b"][b]))
        for (o1, o2, bb) in st["uav_busy"]:
            if bb == b: c[max(0, o1 - t_r):max(0, o2 - t_r)] -= 1
        caps.append(np.maximum(c, 0))
    S["U_cap"] = caps
    xf = np.zeros((N, Tp))
    for (i, ws, we) in st["uav_win"]:
        if we > t_r: xf[i, max(0, ws - t_r):we - t_r] = 1
    S["x_fixed"] = xf
    return S

def run_online(I, EA, sigma, seed, policy, flags=None, R=6, z=0.524, grid=3, EA_true=None, refine_eval=0, refine_sweeps=1,
               sd=0.0):
    """policy: open（开环：t=0 按点估计规划一次）| rolling（每 R 个时隙按保守分位数重规划）。
    EA 为规划所用的行程时间表（可含抢通时刻估计误差），EA_true 为油机车实际行驶所依据的真实表。
    refine_eval>0 时每次规划执行一轮评估预算为 refine_eval 的联合邻域精修。"""
    flags = dict(flags or {})
    EAx = EA if EA_true is None else EA_true
    N, T, dt = I["N"], I["T"], I["dt"]; nB = len(I["U_b"])
    rng = np.random.default_rng(10_000 + seed)
    eps = np.clip(rng.normal(0, sigma, N), -0.5, 0.5)       # 电量估计的相对偏差（逐站固定）
    derr = demand_error(I, sd, seed) if sd > 0 else None    # 需求预测误差（规划用，执行与评价按真实需求）
    E = I["B"].copy(); dead = np.zeros(N, bool)
    Eg = np.array(I["B_mg"], float).copy() if I.get("mg_members") else None   # 真实的微电网共享池
    mg_of = I.get("mg_of"); prio = sorted(range(N), key=lambda i: -I["kappa"][i])
    st = dict(gen_abar={}, truck_pos=[(I["truck_start"], 0)] * I["K"], truck_cap=[I["C"]] * I["K"], uav_busy=[], uav_win=[])
    committed = set(); s_rec = np.zeros((N, T)); plan = None; routes_left = None
    noES = flags.get("emode") == "noES"
    t_r = 0; n_plan = 0
    while t_r < T:
        if plan is None or policy == "rolling":
            B_plan = E * (1 + eps) / ((1 + z * sigma) if policy == "rolling" else 1.0); B_plan[dead] = 0.0
            wp = None if derr is None else demand_forecast(I, derr, t_r, learn=(policy == "rolling"))
            S = make_sub(I, EA, t_r, B_plan, st, w_plan=wp)
            f2 = dict(flags)
            f2["exact"] = refine_eval > 0
            f2["exact_eval"], f2["exact_sweeps"] = refine_eval, refine_sweeps
            if routes_left is not None:
                f2["init_routes"] = [[i for i in r if i not in st["gen_abar"]][:S["C_k"][k]] for k, r in enumerate(routes_left)]
            res = BR.run_bridge(S, EA, f2, max_outer=3 if policy == "rolling" else 10)
            plan = dict(t0=t_r, m=res["low"]["m"], chosen=res["low"]["chosen"])
            routes_left = [list(r) for r in res["routes"]]; n_plan += 1
        end = T if policy == "open" else min(T, t_r + R)
        # 固化无人机班次：本周期内起飞的
        for c in plan["chosen"]:
            i, d, ws, we, o1, o2, v = c; d_abs = d + plan["t0"]
            key = (i, d_abs)
            if t_r <= d_abs < end and key not in committed:
                committed.add(key)
                st["uav_busy"].append((o1 + plan["t0"], o2 + plan["t0"], I["base_of"][i]))
                st["uav_win"].append((i, ws + plan["t0"], we + plan["t0"]))
        # 固化油机车：本周期内出发的路段
        for k in range(I["K"]):
            while routes_left[k] and st["truck_cap"][k] > 0:
                t_av, src = st["truck_pos"][k]
                if t_av >= end: break
                i = routes_left[k][0]
                if i in st["gen_abar"]: routes_left[k].pop(0); continue
                q = int(math.ceil(t_av / grid - 1e-9))
                arr = EAx[src, q, i] if q < EAx.shape[1] else math.inf
                routes_left[k].pop(0)
                if not np.isfinite(arr): continue
                arr = int(math.ceil(arr - 1e-9))
                st["gen_abar"][i] = arr + I["sigma"]
                st["truck_pos"][k] = (arr + I["sigma_u"], i + 1); st["truck_cap"][k] -= 1
        # 按真实电量执行
        x = np.zeros((N, end - t_r))
        for (i, ws, we) in st["uav_win"]:
            a, b = max(ws, t_r), min(we, end)
            if a < b: x[i, a - t_r:b - t_r] = 1
        for t in range(t_r, end):
            for i in prio:
                ga = st["gen_abar"].get(i)
                if ga is not None and t >= ga: beta = 1.0
                else:
                    mp = 2 if noES else int(plan["m"][i, t - plan["t0"]])
                    if mp == 3: mp = 2          # 计划认为已供电但实际尚未到达：按满功率耗电
                    if mp in (1, 2) and not dead[i]:
                        need = (I["PF"][i] if mp == 2 else I["PE"][i]) * dt
                        if Eg is None:
                            ok = E[i] >= need - 1e-12
                            if ok: E[i] -= need
                        else:
                            g = mg_of[i]; ok = Eg[g] >= need - 1e-12
                            if ok:
                                Eg[g] -= need; E[i] -= need   # 共享池按站点重要度优先取电；E 同步递减供下一周期估计
                        if ok: beta = 1.0 if mp == 2 else I["rho"][i]
                        else:
                            if Eg is None: E[i] = 0.0
                            dead[i] = True; beta = 0.0
                    else: beta = 0.0
                s_rec[i, t] = beta + (1 - beta) * I["phi"][i] * x[i, t - t_r]
        t_r = end
    W = I["w"]
    return dict(SR=float((W * s_rec).sum() / W.sum()), crit_out_h=float(((1 - s_rec[I["kappa"] >= 2]).sum(1) * dt).mean()),
                n_plan=n_plan, n_gen=len(st["gen_abar"]))
