"""场景构建：由 OSM 路网与居民点生成基站、需求、电池、功耗、道路损毁、油机车与无人机参数。

时间单位：时隙（5 分钟）；能量单位 kWh；功率单位 kW；需求单位"加权在线用户数"。
"""
import json, math, os, pickle, heapq
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
SLOT_MIN = 5.0
DT_H = SLOT_MIN / 60.0          # 一个时隙的小时数
T_SLOTS = 288                   # 24 小时
UAV_CAP = 1000.0                # 单个无人机小基站载荷可同时服务的在线用户数（默认值，敏感性实验中变化）

# ---------------- 场景配置 ----------------
CFG = {
    "luding": dict(
        name="泸定地震", bbox=(29.40, 101.95, 30.12, 102.42),
        depot=(29.914, 102.234),               # 泸定镇（县城）
        hazard=(29.59, 102.08), hazard_scale_km=15.0,   # 2022-09-05 M6.8 震中（磨西镇附近）
        start_hour=13, n_block=40,
        reopen=[(0.3, 3, 12), (0.4, 12, 30), (0.3, None, None)],  # (比例, 下限h, 上限h)，None=规划期内不能抢通
        slowdown=1.4, K=4, C=4, truck_start_h=0.5,
        U=6, n_bases=3, uav_start_h=0.5, terrain_phi=0.8,
        mg_target=10, mg_share=1.0,
    ),
    "xuwen": dict(
        name="徐闻台风", bbox=(20.20, 109.90, 20.62, 110.60),
        depot=(20.326, 110.175),               # 徐城街道（县城）
        hazard=(20.24, 109.95), hazard_scale_km=20.0,   # 2024-09-06 台风"摩羯"登陆点（角尾乡沿海）
        start_hour=16, n_block=40,
        reopen=[(0.6, 1, 6), (0.3, 6, 18), (0.1, None, None)],
        slowdown=1.3, K=5, C=4, truck_start_h=2.0,
        U=6, n_bases=3, uav_start_h=6.0, terrain_phi=1.0,
        mg_target=10, mg_share=1.0,
    ),
}

SPEED = {  # 灾前通行速度 km/h
    "motorway": 80, "trunk": 60, "primary": 50, "secondary": 40, "tertiary": 30,
    "unclassified": 25, "residential": 20, "service": 15, "track": 12,
}
POP = {"city": 30000, "town": 6000, "suburb": 3000, "quarter": 2000, "village": 1200,
       "neighbourhood": 500, "hamlet": 250, "locality": 80, "isolated_dwelling": 20}
PROFILE = np.array([0.35, 0.25, 0.20, 0.18, 0.18, 0.22, 0.35, 0.55, 0.75, 0.85, 0.90, 0.92,
                    0.95, 0.90, 0.88, 0.90, 0.92, 0.95, 1.00, 1.00, 0.98, 0.90, 0.70, 0.50])

def hav_km(a, b):
    la1, lo1 = map(math.radians, a); la2, lo2 = map(math.radians, b)
    d = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(d))

def proj(lat, lon, lat0):
    return (np.asarray(lon) * 111.32 * math.cos(math.radians(lat0)), np.asarray(lat) * 110.57)

# ---------------- 路网 ----------------
def base_class(hw):
    return hw.replace("_link", "")

def build_graph(key):
    """读 OSM，建无向路网，取最大连通分量，收缩度为 2 的节点。返回 dict（缓存）。"""
    cache = os.path.join(DATA, f"graph_{key}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    cfg = CFG[key]; s, w, n, e = cfg["bbox"]
    d = json.load(open(os.path.join(DATA, f"osm_{key}.json")))
    coord, adj = {}, {}
    def add(u, v, tmin, lkm, cls):
        adj.setdefault(u, {}); adj.setdefault(v, {})
        if v not in adj[u] or adj[u][v][0] > tmin:
            adj[u][v] = (tmin, lkm, cls); adj[v][u] = (tmin, lkm, cls)
    places, amen = [], []
    for el in d["elements"]:
        tg = el.get("tags", {})
        if el["type"] == "way" and "highway" in tg and "geometry" in el:
            cls = base_class(tg["highway"]); sp = SPEED.get(cls)
            if sp is None: continue
            ids, geo = el["nodes"], el["geometry"]
            for a, b, ga, gb in zip(ids[:-1], ids[1:], geo[:-1], geo[1:]):
                pa, pb = (ga["lat"], ga["lon"]), (gb["lat"], gb["lon"])
                coord[a], coord[b] = pa, pb
                lk = hav_km(pa, pb)
                add(a, b, lk / sp * 60.0, lk, cls)
        elif tg.get("place") in POP:
            lat, lon = (el["lat"], el["lon"]) if "lat" in el else (el["center"]["lat"], el["center"]["lon"])
            if s <= lat <= n and w <= lon <= e:
                pop = tg.get("population")
                try: pop = float(str(pop).replace(",", ""))
                except Exception: pop = POP[tg["place"]]
                places.append((lat, lon, tg["place"], pop, tg.get("name", "")))
        elif tg.get("amenity"):
            lat, lon = (el["lat"], el["lon"]) if "lat" in el else (el["center"]["lat"], el["center"]["lon"])
            if s <= lat <= n and w <= lon <= e:
                amen.append((lat, lon, tg["amenity"]))
    # 最大连通分量
    seen, best = set(), []
    for st in adj:
        if st in seen: continue
        comp, stack = [], [st]; seen.add(st)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); stack.append(v)
        if len(comp) > len(best): best = comp
    keep = set(best)
    adj = {u: {v: a for v, a in nb.items() if v in keep} for u, nb in adj.items() if u in keep}
    g = dict(coord={u: coord[u] for u in keep}, adj=adj, places=places, amen=amen)
    pickle.dump(g, open(cache, "wb"))
    return g

def contract(adj, protect):
    """收缩度为 2 且不受保护的节点；边属性 (时间min, 长度km, 最高等级, 中点坐标索引节点)。"""
    adj = {u: dict(nb) for u, nb in adj.items()}
    changed = True
    while changed:
        changed = False
        for u in list(adj.keys()):
            if u in protect or u not in adj: continue
            nb = adj[u]
            if len(nb) != 2: continue
            (a, ea), (b, eb) = list(nb.items())
            if a == b or b in adj[a]:  # 避免平行边，简单跳过
                continue
            t = ea[0] + eb[0]; l = ea[1] + eb[1]
            cls = ea[2] if SPEED.get(ea[2], 0) >= SPEED.get(eb[2], 0) else eb[2]
            del adj[a][u]; del adj[b][u]; del adj[u]
            adj[a][b] = (t, l, cls); adj[b][a] = (t, l, cls)
            changed = True
    return adj

# ---------------- 站点 ----------------
def build_sites(g, cfg, rng, n_max=80):
    lat0 = (cfg["bbox"][0] + cfg["bbox"][2]) / 2
    P = sorted(g["places"], key=lambda p: -p[3])
    xy = np.array([proj(p[0], p[1], lat0) for p in P]).reshape(-1, 2)
    used = np.zeros(len(P), bool); clusters = []
    for k in range(len(P)):
        if used[k]: continue
        dd = np.hypot(*(xy - xy[k]).T)
        mem = np.where((dd < 2.5) & (~used))[0]; used[mem] = True
        pop = sum(P[j][3] for j in mem)
        la = sum(P[j][0] * P[j][3] for j in mem) / pop; lo = sum(P[j][1] * P[j][3] for j in mem) / pop
        clusters.append([la, lo, pop, P[k][4], P[k][2]])
    sites = []
    for la, lo, pop, name, kind in clusters:
        ns = 1 if pop < 6000 else int(min(6, round(pop / 4000)))
        for q in range(ns):
            if ns == 1: dla = dlo = 0.0
            else:
                ang = 2 * math.pi * q / ns
                dla = 0.8 / 110.57 * math.sin(ang); dlo = 0.8 / (111.32 * math.cos(math.radians(la))) * math.cos(ang)
            sites.append(dict(lat=la + dla, lon=lo + dlo, pop=pop / ns, name=name, kind=kind))
    sites.sort(key=lambda s: -s["pop"])
    return sites[:n_max]

def al_hourani_radius_km(h=120.0, fc=2.0e9, pt_dbm=33.0, sens_dbm=-100.0, a=4.88, b=0.43, eta_los=0.1, eta_nlos=21.0):
    """空地信道模型（郊区参数）下，给定高度的最大覆盖半径（km）。"""
    c = 3e8; pl_max = pt_dbm - sens_dbm
    best = 0.0
    for r in np.linspace(0.05, 10, 2000):
        d = math.hypot(h, r * 1000); th = math.degrees(math.atan2(h, r * 1000))
        plos = 1 / (1 + a * math.exp(-b * (th - a)))
        fspl = 20 * math.log10(4 * math.pi * fc * d / c)
        pl = fspl + plos * eta_los + (1 - plos) * eta_nlos
        if pl <= pl_max: best = r
    return best

# ---------------- 时变最短路 ----------------
def ea_dijkstra(nodes_adj, src, t0, reopen):
    """最早到达：沿边 (u,v) 到达时刻 = max(t_u, r_e) + τ_e（允许等待，满足先进先出）。单位：时隙。"""
    dist = {src: t0}; pq = [(t0, src)]; done = set()
    while pq:
        t, u = heapq.heappop(pq)
        if u in done: continue
        done.add(u)
        for v, (tau, r) in nodes_adj[u].items():
            if r == math.inf: continue
            nt = max(t, r) + tau
            if nt < dist.get(v, math.inf):
                dist[v] = nt; heapq.heappush(pq, (nt, v))
    return dist

def partition_near(xy, dmax_km, max_size=None):
    """近距离共享划分：完全连接（complete-linkage）凝聚聚类。每次合并"组内最大站间距"最小的两组，
    直到再合并就会使组内最大站间距超过 dmax_km 或组规模超过 max_size。完全确定，不使用随机数。"""
    N = len(xy)
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    groups = [[i] for i in range(N)]
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if max_size is not None and len(groups[a]) + len(groups[b]) > max_size: continue
                d = D[np.ix_(groups[a], groups[b])].max()
                if d <= dmax_km and (best is None or d < best[0] - 1e-12): best = (d, a, b)
        if best is None: break
        _, a, b = best
        groups[a] = sorted(groups[a] + groups[b]); del groups[b]
    mg_of = np.zeros(N, int)
    for g, mem in enumerate(sorted(groups, key=lambda m: m[0])): mg_of[mem] = g
    return mg_of


def make_instance(key, seed, n_max=80, overrides=None):
    """生成一个场景实例（给定随机种子）。"""
    cfg = dict(CFG[key]); cfg.update(overrides or {})
    rng = np.random.default_rng(seed)
    g = build_graph(key)
    lat0 = (cfg["bbox"][0] + cfg["bbox"][2]) / 2
    sites = build_sites(g, cfg, rng, n_max=n_max)
    N = len(sites)
    # 路网节点投影与吸附
    ids = list(g["coord"].keys())
    nxy = np.array([proj(*g["coord"][u], lat0) for u in ids]).reshape(-1, 2)
    def snap(lat, lon):
        p = np.array(proj(lat, lon, lat0)); dd = np.hypot(*(nxy - p).T); k = int(np.argmin(dd))
        return ids[k], float(dd[k])
    site_node, access_km = [], []
    for s_ in sites:
        u, dk = snap(s_["lat"], s_["lon"]); site_node.append(u); access_km.append(dk)
    depot_node, _ = snap(*cfg["depot"])
    protect = set(site_node) | {depot_node}
    cache = os.path.join(DATA, f"cgraph_{key}_{n_max}.pkl")
    if os.path.exists(cache):
        cadj = pickle.load(open(cache, "rb"))
    else:
        cadj = contract(g["adj"], protect); pickle.dump(cadj, open(cache, "wb"))
    # 道路损毁：按到灾害中心的距离加权采样
    edges = []
    for u, nb in cadj.items():
        for v, (tmin, lkm, cls) in nb.items():
            if str(u) < str(v): edges.append((u, v, tmin, lkm, cls))
    haz = cfg["hazard"]; dep = cfg["depot"]
    wts = []
    for (u, v, tmin, lkm, cls) in edges:
        mid = ((g["coord"][u][0] + g["coord"][v][0]) / 2, (g["coord"][u][1] + g["coord"][v][1]) / 2)
        ok = cls in ("trunk", "primary", "secondary", "tertiary", "unclassified", "track") and hav_km(mid, dep) > 2.0
        wts.append(lkm * math.exp(-hav_km(mid, haz) / cfg["hazard_scale_km"]) if ok else 0.0)
    wts = np.array(wts); wts = wts / wts.sum()
    nb_ = min(cfg["n_block"], int((wts > 0).sum()))
    blk = rng.choice(len(edges), size=nb_, replace=False, p=wts) if nb_ > 0 else []
    reopen = {}
    for bi in blk:
        r = rng.random(); acc = 0.0
        for frac, lo, hi in cfg["reopen"]:
            acc += frac
            if r <= acc:
                reopen[bi] = math.inf if lo is None else rng.uniform(lo, hi) * 60 / SLOT_MIN
                break
    if cfg.get("n_block_nested") is not None:
        # 损毁程度实验：用独立随机流嵌套抽取受损路段（20⊂40⊂60⊂80），其余随机量与默认实例一致
        rng2 = np.random.default_rng(seed + 99991)
        avail = int((wts > 0).sum()); nmax = min(80, avail)
        perm = rng2.choice(len(edges), size=nmax, replace=False, p=wts)
        rs = []
        for bi in perm:
            r = rng2.random(); acc = 0.0
            for frac, lo, hi in cfg["reopen"]:
                acc += frac
                if r <= acc:
                    rs.append(math.inf if lo is None else rng2.uniform(lo, hi) * 60 / SLOT_MIN); break
        nb2 = min(cfg["n_block_nested"], nmax)
        reopen = {perm[j]: rs[j] for j in range(nb2)}
    nodes_adj = {u: {} for u in cadj}
    for k, (u, v, tmin, lkm, cls) in enumerate(edges):
        tau = tmin * cfg["slowdown"] / SLOT_MIN
        r = reopen.get(k, 0.0)
        nodes_adj[u][v] = (tau, r); nodes_adj[v][u] = (tau, r)
    # 站点参数
    xy = np.array([proj(s_["lat"], s_["lon"], lat0) for s_ in sites]).reshape(-1, 2)
    dmat = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) + np.eye(N) * 1e9
    R_cell = np.clip(0.55 * dmat.min(1), 0.8, 6.0)
    R_uav = al_hourani_radius_km()
    phi_cov = 0.85 * np.minimum(1.0, (R_uav / R_cell) ** 2) * cfg["terrain_phi"]
    # 容量上限：单个小基站载荷可同时服务的在线用户数有限，覆盖比例再乘以 min(1, 容量/站点在线用户数)
    users_site = np.array([s_["pop"] for s_ in sites]) * 0.3
    phi = phi_cov * np.minimum(1.0, cfg.get("uav_cap", UAV_CAP) / np.maximum(users_site, 1.0))
    # 重要度：2 km 内有医院/诊所 κ=3；学校/政府/公安/消防 κ=2；县城站点至少 2
    kappa = np.ones(N)
    for i, s_ in enumerate(sites):
        for (la, lo, am) in g["amen"]:
            dk = hav_km((s_["lat"], s_["lon"]), (la, lo))
            if dk < 1.2:
                kappa[i] = max(kappa[i], {"hospital": 3.0, "clinic": 2.5, "townhall": 2.0, "police": 2.0,
                                          "fire_station": 2.0, "school": 1.5}.get(am, 1.0))
        if hav_km((s_["lat"], s_["lon"]), cfg["depot"]) < 2.0: kappa[i] = max(kappa[i], 2.0)
    # 需求：人口 × 在线比例 × 日变化 × 灾后激增 × 噪声
    users = np.array([s_["pop"] for s_ in sites]) * 0.3
    hrs = (cfg["start_hour"] + np.arange(T_SLOTS) * DT_H) % 24
    prof = np.interp(hrs, np.arange(25), np.r_[PROFILE, PROFILE[0]])
    surge = 1 + 0.5 * np.exp(-np.arange(T_SLOTS) * DT_H / 6.0)
    noise = np.exp(rng.normal(0, 0.1, (N, 24)))
    noise_t = noise[:, (np.arange(T_SLOTS) * DT_H).astype(int) % 24]
    w = kappa[:, None] * users[:, None] * prof[None, :] * surge[None, :] * noise_t
    # 基站功耗与电池
    typ = rng.random(N) < 0.6            # True: 4G+5G 站；False: 纯 4G 站
    PF = np.where(typ, rng.uniform(2.0, 3.0, N), rng.uniform(1.0, 1.5, N))
    PE = PF * np.where(typ, rng.uniform(0.35, 0.50, N), rng.uniform(0.65, 0.80, N))
    rho = np.where(typ, rng.uniform(0.60, 0.75, N), rng.uniform(0.50, 0.65, N))
    hbat = np.clip(rng.lognormal(math.log(cfg.get("bat_median_h", 3.5)), 0.4, N), 1.0, 8.0)
    B = hbat * PF
    # 微电网：同组基站共享储能。划分完全确定（最远点采样选中心 + 最近中心归属），不使用随机数，
    # 因此不改变任何既有随机量的抽取序列。
    mg_target = cfg.get("mg_target", 10)
    if cfg.get("mg_dmax_km") is not None:
        # 保守配置：只有相距不超过 mg_dmax_km 的基站才共享储能（可再限制组规模）
        mg_of = partition_near(xy, cfg["mg_dmax_km"], cfg.get("mg_max_size"))
    elif cfg.get("mg_share", 1.0) > 0 and mg_target > 0:
        dep_xy = np.array(proj(cfg["depot"][0], cfg["depot"][1], lat0)).reshape(-1)
        n_mg = max(1, int(math.ceil(N / mg_target)))
        centers = [int(np.argmin(np.hypot(xy[:, 0] - dep_xy[0], xy[:, 1] - dep_xy[1])))]
        while len(centers) < n_mg:
            dc = np.min(np.stack([np.hypot(xy[:, 0] - xy[c, 0], xy[:, 1] - xy[c, 1]) for c in centers]), axis=0)
            centers.append(int(np.argmax(dc)))
        dcg = np.stack([np.hypot(xy[:, 0] - xy[c, 0], xy[:, 1] - xy[c, 1]) for c in centers])
        mg_of = np.argmin(dcg, axis=0).astype(int)
    else:
        mg_of = np.arange(N)
    mg_members = [list(np.where(mg_of == g)[0]) for g in range(int(mg_of.max()) + 1)]
    B_mg = np.array([B[m].sum() for m in mg_members])
    # 无人机起降点：县城 + 若干大镇（最远点优先）
    towns = [i for i in range(N)]
    bases = [cfg["depot"]]
    while len(bases) < cfg["n_bases"]:
        cand = max(towns[:30], key=lambda i: min(hav_km((sites[i]["lat"], sites[i]["lon"]), b) for b in bases) * math.log(sites[i]["pop"] + 10))
        bases.append((sites[cand]["lat"], sites[cand]["lon"]))
    bdist = np.array([[hav_km((s_["lat"], s_["lon"]), b) for b in bases] for s_ in sites])
    base_of = bdist.argmin(1); dist_b = bdist.min(1)
    uav_speed_kmh = 72.0
    E_max, S_swap = 9, 2
    f = np.ceil(dist_b / uav_speed_kmh * 60 / SLOT_MIN).astype(int); f = np.maximum(f, 1)
    h = E_max - 2 * f
    phi = np.where(h >= 2, phi, 0.0); phi_cov = np.where(h >= 2, phi_cov, 0.0)
    U = cfg["U"]; nb = cfg["n_bases"]
    U_b = [U // nb + (1 if k < U % nb else 0) for k in range(nb)]
    inst = dict(
        key=key, name=cfg["name"], seed=seed, N=N, T=T_SLOTS, dt=DT_H, sites=sites,
        w=w, kappa=kappa, users=users, PF=PF, PE=PE, rho=rho, B=B, hbat=hbat, phi=phi, phi_cov=phi_cov, R_cell=R_cell, R_uav=R_uav,
        f=f, h=h, E_max=E_max, S=S_swap, base_of=base_of, U_b=U_b, uav_start=int(round(cfg["uav_start_h"] * 60 / SLOT_MIN)),
        K=cfg["K"], C=cfg["C"], truck_start=int(round(cfg["truck_start_h"] * 60 / SLOT_MIN)), sigma_u=2, sigma=2,
        nodes_adj=nodes_adj, site_node=site_node, depot_node=depot_node, access=np.array(access_km),
        n_blocked=len(reopen), reopen_h=[(v * SLOT_MIN / 60) for v in reopen.values()],
        bases=bases, cfg=cfg,
        mg_of=mg_of, mg_members=mg_members, B_mg=B_mg, mg_share=float(cfg.get("mg_share", 1.0)),
    )
    return inst

def travel_tables(inst, grid=3, horizon=None):
    """EA[s][q][j]：源 s（0=车库，1..N=站点）在时隙 q*grid 出发到站点 j 的最早到达时隙。"""
    T = inst["T"] if horizon is None else horizon
    srcs = [inst["depot_node"]] + list(inst["site_node"])
    acc = np.r_[0.0, inst["access"]] / 15.0 * 60 / SLOT_MIN      # 末段接入按 15 km/h
    Q = T // grid + 1
    EA = np.full((len(srcs), Q, inst["N"]), np.inf)
    for si, s in enumerate(srcs):
        for q in range(Q):
            t0 = q * grid + acc[si]
            dist = ea_dijkstra(inst["nodes_adj"], s, t0, None)
            for j, nd in enumerate(inst["site_node"]):
                EA[si, q, j] = dist.get(nd, np.inf) + acc[j + 1]
    return EA
