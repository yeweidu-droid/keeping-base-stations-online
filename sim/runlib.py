import os, pickle, time, numpy as np
from scenario import make_instance, travel_tables
import bridge as BR
CACHE = os.path.join(os.path.dirname(__file__), "cache"); os.makedirs(CACHE, exist_ok=True)
CACHE_VER = "v3cap"         # 无人机覆盖比例加入载荷容量上限后，实例结构改变，旧缓存失效

def get_instance(key, seed, overrides=None, tag=""):
    fn = os.path.join(CACHE, f"inst_{key}_{seed}{tag}_{CACHE_VER}.pkl")
    if os.path.exists(fn): return pickle.load(open(fn, "rb"))
    I = make_instance(key, seed, overrides=overrides)
    t = time.time(); EA = travel_tables(I); I["ea_time"] = time.time() - t
    I.pop("nodes_adj", None)
    pickle.dump((I, EA), open(fn, "wb"))
    return I, EA

def run_alg(name, I, EA):
    if name == "IND": return BR.run_ind(I, EA)
    if name == "SEQ":
        t0 = time.time()
        g = BR.run_bridge(I, EA, BR.ALGS["GEN-LS"])
        ab = g["abar"]; low = BR.lower_level(I, ab, {})
        return dict(routes=g["routes"], abar=ab, low=low, J=low["J"], hist=[low["J"]], time=time.time() - t0, tim={})
    return BR.run_bridge(I, EA, BR.ALGS[name])
