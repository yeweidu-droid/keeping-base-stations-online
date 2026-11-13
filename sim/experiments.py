"""实验驱动：python experiments.py {main|sens|block|online|runtime} [workers]"""
import os, sys, json, copy, time, math, traceback
import numpy as np
from multiprocessing import Pool
from runlib import get_instance, run_alg, CACHE_VER
import bridge as BR
import online as ON

RES = os.path.join(os.path.dirname(__file__), "results")
SCEN = ["luding", "xuwen"]
MAIN_ALGS = ["GEN-EDF", "GEN-LS", "IND", "SEQ", "BRIDGE"]
ABL = ["w/o ES", "w/o EA", "w/o UAV", "w/o MG", "w/o coupling", "w/o price", "w/o BCA", "w/o routing", "w/o refine", "w/o grpprice"]
# 在线运行方式：(名称, 策略, 算法开关, 精修评估预算, 精修轮数)；开环与离线 BRIDGE 一致，滚动每周期一轮小预算精修
ONLINE_POL = [("GEN-EDF", "open", {"route": False, "emode": "noES", "uav": False}, 0, 1),
              ("BRIDGE-open", "open", {}, 30, 4), ("BRIDGE-rolling", "rolling", {}, 10, 1)]
SENS_ALGS = ["GEN-EDF", "IND", "SEQ", "BRIDGE", "w/o refine", "w/o price"]

def save(exp, name, obj):
    d = os.path.join(RES, exp); os.makedirs(d, exist_ok=True)
    json.dump(obj, open(os.path.join(d, name + ".json"), "w"), ensure_ascii=False)

def done(exp, name):
    """断点续跑：结果文件已存在则跳过（便于分批执行）。"""
    return os.path.exists(os.path.join(RES, exp, name + ".json"))

def set_U(I, U):
    nb = len(I["U_b"]); I["U_b"] = [U // nb + (1 if k < U % nb else 0) for k in range(nb)]

def t_main(args):
    key, seed = args
    if done("main", f"{key}_{seed}"): return args
    I, EA = get_instance(key, seed)
    out = dict(key=key, seed=seed, N=I["N"], n_blocked=I["n_blocked"], ea_time=I.get("ea_time"), algs={})
    for a in MAIN_ALGS + ABL:
        r = run_alg(a, I, EA); M = BR.metrics(I, r)
        if a not in MAIN_ALGS: M.pop("curve"); M.pop("alive_curve")
        M["hist"] = r["hist"]; M["tim"] = r.get("tim", {})
        out["algs"][a] = M
    save("main", f"{key}_{seed}", out)
    return key, seed

def t_sens(args):
    key, seed, param, val = args
    if done("sens", f"{key}_{seed}_{param}_{val}"): return args
    I0, EA = get_instance(key, seed)
    I = copy.deepcopy(I0)
    if param == "U": set_U(I, val)
    elif param == "K": I["K"] = val
    elif param == "bat":
        f = val / 3.5; I["hbat"] = np.clip(I["hbat"] * f, 0.5, 12.0); I["B"] = I["hbat"] * I["PF"]
        I["B_mg"] = np.array([I["B"][m].sum() for m in I["mg_members"]])   # 共享预算必须同步
    elif param == "cap":
        I["phi"] = I["phi_cov"] * np.minimum(1.0, val / np.maximum(I["users"], 1.0))
    elif param == "mg":
        # val = 站间能量转移效率；0 表示不共享（|N_g|=1）。效率按保守口径作用于整个共享池。
        if val <= 0: I["mg_share"] = 0.0
        else: I["B_mg"] = I["B_mg"] * val
    res = {}
    for a in SENS_ALGS:
        r = run_alg(a, I, EA); M = BR.metrics(I, r)
        res[a] = dict(SR=M["SR"], crit=M["crit_out_h"], WO=M["WO_kuh"], time=M["time_s"])
    save("sens", f"{key}_{seed}_{param}_{val}", dict(key=key, seed=seed, param=param, val=val, res=res))
    return args

def t_block(args):
    key, seed, nb = args
    if done("block", f"{key}_{seed}_{nb}"): return args
    I, EA = get_instance(key, seed, overrides={"n_block_nested": nb}, tag=f"_nbn{nb}_{CACHE_VER}")
    res = {}
    for a in SENS_ALGS:
        r = run_alg(a, I, EA); M = BR.metrics(I, r)
        res[a] = dict(SR=M["SR"], crit=M["crit_out_h"], WO=M["WO_kuh"], gen_arr=M["gen_mean_arrival_h"])
    save("block", f"{key}_{seed}_{nb}", dict(key=key, seed=seed, nb=nb, n_blocked=I["n_blocked"], res=res))
    return args

def t_online(args):
    key, seed, sigma = args
    if done("online", f"{key}_{seed}_{sigma}"): return args
    I, EA = get_instance(key, seed)
    res = {}
    for name, pol, fl, rf, sw in ONLINE_POL:
        t = time.time(); r = ON.run_online(I, EA, sigma, seed, pol, fl, refine_eval=rf, refine_sweeps=sw)
        r["time"] = time.time() - t; res[name] = r
    save("online", f"{key}_{seed}_{sigma}", dict(key=key, seed=seed, sigma=sigma, res=res))
    return args

def road_estimate_tables(key, seed, sr):
    """抢通时刻估计误差：规划方使用 r̂_e = r_e·exp(ε_e)，ε_e ~ N(0, sr²)，据此重算时变行程时间表。"""
    from scenario import make_instance, travel_tables
    import pickle
    fn = os.path.join(os.path.dirname(__file__), "cache", f"eahat_{key}_{seed}_{sr}_{CACHE_VER}.pkl")
    if os.path.exists(fn): return pickle.load(open(fn, "rb"))
    I = make_instance(key, seed)
    rng = np.random.default_rng(20_000 + seed)
    adj = I["nodes_adj"]; seen = {}
    for u in adj:
        for v, (tau, r) in list(adj[u].items()):
            if 0 < r < math.inf:
                e = (u, v) if str(u) < str(v) else (v, u)
                if e not in seen: seen[e] = r * math.exp(rng.normal(0, sr))
                adj[u][v] = (tau, seen[e])
    EAh = travel_tables(I)
    pickle.dump(EAh, open(fn, "wb"))
    return EAh

def t_road(args):
    key, seed, sr = args
    if done("road", f"{key}_{seed}_{sr}"): return args
    I, EA = get_instance(key, seed)
    EAh = EA if sr == 0 else road_estimate_tables(key, seed, sr)
    res = {}
    for name, pol, fl, rf, sw in ONLINE_POL:
        t = time.time(); r = ON.run_online(I, EAh, 0.0, seed, pol, fl, EA_true=EA, refine_eval=rf, refine_sweeps=sw)
        r["time"] = time.time() - t; res[name] = r
    save("road", f"{key}_{seed}_{sr}", dict(key=key, seed=seed, sr=sr, res=res))
    return args

def t_runtime(args):
    key, seed, nmax = args
    if done("runtime", f"{key}_{seed}_{nmax}"): return args
    from scenario import make_instance, travel_tables
    import pickle
    fn = os.path.join(os.path.dirname(__file__), "cache", f"inst_{key}_{seed}_n{nmax}_{CACHE_VER}.pkl")
    if os.path.exists(fn): I, EA = pickle.load(open(fn, "rb"))
    else:
        I = make_instance(key, seed, n_max=nmax); t = time.time(); EA = travel_tables(I); I["ea_time"] = time.time() - t
        I.pop("nodes_adj", None); pickle.dump((I, EA), open(fn, "wb"))
    # 油机车与无人机按站点数等比例缩放
    I["K"] = max(1, round(I["cfg"]["K"] * I["N"] / 80)); set_U(I, max(len(I["U_b"]), round(I["cfg"]["U"] * I["N"] / 80)))
    res = {}
    for a in ["IND", "SEQ", "BRIDGE"]:
        r = run_alg(a, I, EA); res[a] = dict(time=r["time"], tim=r.get("tim", {}), SR=BR.metrics(I, r)["SR"], n_outer=len(r["hist"]))
    save("runtime", f"{key}_{seed}_{nmax}", dict(key=key, seed=seed, nmax=nmax, N=I["N"], ea_time=I["ea_time"], res=res))
    return args

MG_ALGS = ["GEN-EDF", "IND", "SEQ", "SEQ+R", "BRIDGE"]

def t_mgsize(args):
    """微电网规模：按目标规模 tgt 重新做（确定性的）最远点划分，其余随机量与默认实例相同。tgt=1 即不共享。"""
    key, seed, tgt = args
    if done("mgsize", f"{key}_{seed}_{tgt}"): return args
    from scenario import make_instance
    I0, EA = get_instance(key, seed)
    J = make_instance(key, seed, overrides={"mg_target": tgt})
    I = dict(I0); I["mg_of"], I["mg_members"], I["B_mg"] = J["mg_of"], J["mg_members"], J["B_mg"]
    res = {}
    for a in MG_ALGS:
        r = BR.run_seq_refine(I, EA) if a == "SEQ+R" else run_alg(a, I, EA)
        M = BR.metrics(I, r); res[a] = dict(SR=M["SR"], WO=M["WO_kuh"], time=M["time_s"])
    save("mgsize", f"{key}_{seed}_{tgt}", dict(key=key, seed=seed, tgt=tgt, sizes=[len(m) for m in I["mg_members"]], res=res))
    return args

NEAR_CFG = {"d1": (1.0, None), "d2c3": (2.0, 3), "d2": (2.0, None)}   # 标签: (组内最大站间距 km, 组规模上限)
def t_mgnear(args):
    """近距离共享（保守配置）：只有相距不超过 dmax 的基站才共享储能，其余随机量与默认实例相同。"""
    key, seed, tag = args
    if done("mgnear", f"{key}_{seed}_{tag}"): return args
    from scenario import make_instance, proj, CFG
    dmax, cap = NEAR_CFG[tag]
    I0, EA = get_instance(key, seed)
    J = make_instance(key, seed, overrides={"mg_dmax_km": dmax, "mg_max_size": cap})
    I = dict(I0); I["mg_of"], I["mg_members"], I["B_mg"] = J["mg_of"], J["mg_members"], J["B_mg"]
    lat0 = (CFG[key]["bbox"][0] + CFG[key]["bbox"][2]) / 2
    xy = np.array([proj(s_["lat"], s_["lon"], lat0) for s_ in I["sites"]]).reshape(-1, 2)
    diam = [float(max(np.hypot(*(xy[a] - xy[b])) for a in m for b in m)) for m in I["mg_members"]]
    res = {}
    for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
        r = run_alg(a, I, EA); M = BR.metrics(I, r); res[a] = dict(SR=M["SR"], WO=M["WO_kuh"], time=M["time_s"])
    save("mgnear", f"{key}_{seed}_{tag}", dict(key=key, seed=seed, tag=tag, dmax_km=dmax, max_size=cap,
                                               sizes=[len(m) for m in I["mg_members"]], diam_km=diam, res=res))
    return args

def t_base(args):
    """更强的对照：SEQ+精修（两种候选排序）与真实目标上的模拟退火（1 倍与 5 倍 BRIDGE 耗时）。"""
    key, seed = args
    if done("base", f"{key}_{seed}"): return args
    I, EA = get_instance(key, seed)
    M0 = json.load(open(os.path.join(RES, "main", f"{key}_{seed}.json")))["algs"]
    tb = M0["BRIDGE"]["time_s"]
    res = {}
    for a in ["SEQ+R", "SEQ+R*", "SA-1x", "SA-5x"]:
        if a == "SEQ+R": r = BR.run_seq_refine(I, EA, grp_rank=False)
        elif a == "SEQ+R*": r = BR.run_seq_refine(I, EA, grp_rank=True)
        else: r = BR.run_sa(I, EA, budget_s=tb * (1 if a == "SA-1x" else 5), seed=seed)
        M = BR.metrics(I, r); res[a] = dict(SR=M["SR"], WO=M["WO_kuh"], J=M["J"], time=M["time_s"], n_eval=r.get("n_eval"))
    for a in ["SEQ", "BRIDGE", "w/o refine"]:
        res[a] = dict(SR=M0[a]["SR"], WO=M0[a]["WO_kuh"], J=M0[a]["J"], time=M0[a]["time_s"])
    save("base", f"{key}_{seed}", dict(key=key, seed=seed, res=res))
    return args

GOLD_H = 6.0
def t_golden(args):
    """黄金时段加权：规划时把灾后前 GOLD_H 小时的需求乘以 (1+γ)，评价按原需求，并单独统计前 GOLD_H 小时。"""
    key, seed, gam = args
    if done("golden", f"{key}_{seed}_{gam}"): return args
    I, EA = get_instance(key, seed)
    H = int(round(GOLD_H / I["dt"]))
    Ip = dict(I); Ip["w"] = I["w"].copy(); Ip["w"][:, :H] *= (1 + gam)
    res = {}
    for a in ["GEN-EDF", "SEQ", "BRIDGE"]:
        r = run_alg(a, Ip, EA); M = BR.metrics(I, r)
        m, x = r["low"]["m"], r["low"]["x"]; b = BR.beta_of(m, I["rho"]); s = b + (1 - b) * I["phi"][:, None] * x
        W = I["w"]
        res[a] = dict(SR=M["SR"], WO=M["WO_kuh"], SR_early=float((W[:, :H] * s[:, :H]).sum() / W[:, :H].sum()),
                      dark_early=float((s[:, :H] <= 1e-9).sum() * I["dt"]), crit=M["crit_out_h"],
                      **BR.state_switches(I, m, r["abar"]))
    save("golden", f"{key}_{seed}_{gam}", dict(key=key, seed=seed, gamma=gam, res=res))
    return args

def t_demand(args):
    """需求预测误差下的在线运行（规划用含误差的需求，执行与评价按真实需求）。"""
    key, seed, sd = args
    if done("demand", f"{key}_{seed}_{sd}"): return args
    I, EA = get_instance(key, seed)
    res = {}
    for name, pol, fl, rf, sw in ONLINE_POL:
        t = time.time(); r = ON.run_online(I, EA, 0.0, seed, pol, fl, refine_eval=rf, refine_sweeps=sw, sd=sd)
        r["time"] = time.time() - t; res[name] = r
    save("demand", f"{key}_{seed}_{sd}", dict(key=key, seed=seed, sd=sd, res=res))
    return args

COMBO = dict(se=0.2, sd=0.4, sr=0.5)
def t_combo(args):
    """三类估计误差同时存在：电量 σe、需求 σd、抢通时刻 σr。"""
    key, seed = args
    if done("combo", f"{key}_{seed}"): return args
    I, EA = get_instance(key, seed)
    EAh = road_estimate_tables(key, seed, COMBO["sr"])
    res = {}
    for name, pol, fl, rf, sw in ONLINE_POL:
        t = time.time()
        r = ON.run_online(I, EAh, COMBO["se"], seed, pol, fl, EA_true=EA, refine_eval=rf, refine_sweeps=sw, sd=COMBO["sd"])
        r["time"] = time.time() - t; res[name] = r
    save("combo", f"{key}_{seed}", dict(key=key, seed=seed, **COMBO, res=res))
    return args

def uav_role(I, r):
    """无人机覆盖的去向：覆盖时长按"该站时域内有无油机""覆盖期间是否在为后续时段留电"拆分，及无人机直接提供的服务量。"""
    m, x, ab, dt = r["low"]["m"], r["low"]["x"], r["abar"], I["dt"]
    b = beta_of_m(I, m); T = I["T"]
    cov = x > 0
    nogen = (ab >= T)[:, None] & cov
    later_bat = np.zeros_like(cov)
    for i in range(I["N"]):
        on = (m[i] >= 1) & (m[i] <= 2) & (np.arange(T) < ab[i])
        later = np.flip(np.maximum.accumulate(np.flip(on)))
        later_bat[i, :-1] = later[1:]
    saving = cov & (m == 0) & later_bat
    direct = I["w"] * (1 - b) * I["phi"][:, None] * x
    crit = I["kappa"] >= 2
    fl = sum(2 * int(I["f"][c[0]]) + int(I["h"][c[0]]) for c in r["low"]["chosen"])
    return dict(J=r["J"], cover_h=float(cov.sum() * dt), flight_h=fl * dt,
                cover_nogen_h=float(nogen.sum() * dt), cover_saving_h=float(saving.sum() * dt),
                direct_kuh=float(direct.sum() * dt / 1000), direct_crit_kuh=float(direct[crit].sum() * dt / 1000),
                n_sites=int(cov.any(1).sum()))

def beta_of_m(I, m): return BR.beta_of(m, I["rho"])

def t_uavrole(args):
    key, seed = args
    if done("uavrole", f"{key}_{seed}"): return args
    I, EA = get_instance(key, seed)
    out = dict(key=key, seed=seed)
    for a in ["IND", "BRIDGE"]:
        out[a] = uav_role(I, run_alg(a, I, EA))
    out["BRIDGE_noUAV_J"] = run_alg("w/o UAV", I, EA)["J"]
    out["W_kuh"] = float(I["w"].sum() * I["dt"] / 1000)
    save("uavrole", f"{key}_{seed}", out)
    return args

def safe(fn):
    def g(a):
        try: return fn(a)
        except Exception: return ("ERR", a, traceback.format_exc())
    return g

TASKS = {
    "main": (t_main, [(k, s) for k in SCEN for s in range(20)]),
    "sens": (t_sens, [(k, s, p, v) for k in SCEN for s in range(10) for p, vals in
                      [("U", [0, 3, 6, 9, 12, 18]), ("K", [1, 2, 3, 4, 5, 6]), ("bat", [2.0, 2.5, 3.5, 4.5, 6.0]),
                       ("cap", [250, 500, 1000, 1e9]), ("mg", [1.0, 0.9, 0.8, 0.7, 0.0])] for v in vals]),
    "road": (t_road, [(k, s, sr) for k in SCEN for s in range(10) for sr in [0.0, 0.25, 0.5, 1.0]]),
    "block": (t_block, [(k, s, nb) for k in SCEN for s in range(10) for nb in [0, 20, 40, 60, 80]]),
    "online": (t_online, [(k, s, sg) for k in SCEN for s in range(10) for sg in [0.0, 0.1, 0.2, 0.3]]),
    "runtime": (t_runtime, [(k, s, n) for k in SCEN for s in range(3) for n in [20, 40, 60, 80, 100, 120]]),
    "mgsize": (t_mgsize, [(k, s, g) for k in SCEN for s in range(10) for g in [1, 2, 3, 5, 10, 20, 40]]),
    "mgnear": (t_mgnear, [(k, s, c) for k in SCEN for s in range(10) for c in NEAR_CFG]),
    "base": (t_base, [(k, s) for k in SCEN for s in range(20)]),
    "golden": (t_golden, [(k, s, g) for k in SCEN for s in range(20) for g in [0, 1, 3, 9]]),
    "demand": (t_demand, [(k, s, sd) for k in SCEN for s in range(10) for sd in [0.2, 0.4, 0.6]]),
    "combo": (t_combo, [(k, s) for k in SCEN for s in range(10)]),
    "uavrole": (t_uavrole, [(k, s) for k in SCEN for s in range(20)]),
}

_FN = None
def _init(name):
    global _FN; _FN = safe(TASKS[name][0])
def _run(a): return _FN(a)

if __name__ == "__main__":
    name = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    fn, tasks = TASKS[name]
    t0 = time.time(); n = 0
    with Pool(nw, initializer=_init, initargs=(name,)) as pool:
        for r in pool.imap_unordered(_run, tasks):
            n += 1
            if isinstance(r, tuple) and r and r[0] == "ERR": print("ERROR", r[1], r[2][-800:], flush=True)
            if n % 10 == 0 or n == len(tasks): print(f"[{name}] {n}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    print("done", name, f"{time.time()-t0:.0f}s")
