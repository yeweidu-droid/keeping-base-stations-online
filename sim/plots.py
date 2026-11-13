"""汇总实验结果，生成论文图片（矢量 PDF）。"""
import os, json, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper_zh", "figs"); os.makedirs(FIG, exist_ok=True)

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(f): font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK JP", "PingFang HK", "Songti SC", "Arial Unicode MS"], "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.axisbelow": True, "figure.dpi": 150,
})
# 分类色（参考调色板前 5 槽，已通过相邻对色盲校验），颜色跟随算法，不随排名变化
COL = {"BRIDGE": "#2a78d6", "SEQ": "#eb6834", "IND": "#1baf7a", "GEN-LS": "#eda100", "GEN-EDF": "#e87ba4",
       "BRIDGE-rolling": "#2a78d6", "BRIDGE-open": "#eb6834"}
MK = {"BRIDGE": "o", "SEQ": "s", "IND": "^", "GEN-LS": "D", "GEN-EDF": "v", "BRIDGE-rolling": "o", "BRIDGE-open": "s"}
LAB = {"GEN-EDF": "GEN-EDF", "GEN-LS": "GEN-LS", "IND": "IND", "SEQ": "SEQ", "BRIDGE": "BRIDGE",
       "BRIDGE-open": "BRIDGE（开环）", "BRIDGE-rolling": "BRIDGE（滚动）"}
SCN = {"luding": "泸定地震场景", "xuwen": "徐闻台风场景"}
MAIN = ["GEN-EDF", "GEN-LS", "IND", "SEQ", "BRIDGE"]
INK = "#2b2b29"; INK2 = "#52514e"

def load(exp):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, exp, "*.json")))]

def ms(v): v = np.asarray(v, float); return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0

# ------------------------------------------------------------------ 主实验
def main_results():
    R = load("main"); out = {}
    for key in SCN:
        rs = [r for r in R if r["key"] == key]
        out[key] = {a: {m: [r["algs"][a][m] for r in rs] for m in r0} for a, r0 in rs[0]["algs"].items()} if rs else {}
        out[key]["_n"] = len(rs); out[key]["_meta"] = dict(N=[r["N"] for r in rs], nb=[r["n_blocked"] for r in rs], ea=[r["ea_time"] for r in rs])
    return out

def fig_main_bars(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, key in zip(axes, SCN):
        mu = [ms(D[key][a]["SR"])[0] for a in MAIN]; sd = [ms(D[key][a]["SR"])[1] for a in MAIN]
        x = np.arange(len(MAIN))
        ax.bar(x, mu, width=0.62, color=[COL[a] for a in MAIN], yerr=sd, capsize=2.5,
               error_kw=dict(elinewidth=0.8, ecolor=INK2), edgecolor="white", linewidth=1.0)
        for xi, v in zip(x, mu): ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        lo = min(mu) - 0.08; ax.set_ylim(max(0, lo), min(1.0, max(mu) + 0.06))
        ax.set_xticks(x, MAIN); ax.set_title(SCN[key], color=INK); ax.set_ylabel("加权服务率")
        ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "main_sr.pdf")); plt.close(fig)

def fig_curves(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, key in zip(axes, SCN):
        for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
            c = np.mean(np.array(D[key][a]["curve"]), axis=0); th = np.arange(len(c)) * 5 / 60
            ax.plot(th, c, color=COL[a], lw=1.6 if a == "BRIDGE" else 1.2, label=LAB[a],
                    ls="-" if a in ("BRIDGE", "SEQ") else "--")
        ax.set_xlabel("灾后时间（小时）"); ax.set_title(SCN[key], color=INK); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    axes[0].set_ylabel("逐时隙加权服务率")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "curve.pdf")); plt.close(fig)

def fig_cdf(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, key in zip(axes, SCN):
        for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
            v = np.sort(np.concatenate([np.array(x) for x in D[key][a]["per_site_out_h"]]))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[a], lw=1.6 if a == "BRIDGE" else 1.2,
                    ls="-" if a in ("BRIDGE", "SEQ") else "--", label=LAB[a])
        ax.set_xlabel("单站等效失联时长（小时）"); ax.set_title(SCN[key], color=INK); ax.set_xlim(0, 24)
    axes[0].set_ylabel("站点累计比例")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "cdf.pdf")); plt.close(fig)

ABL = ["w/o EA", "w/o ES", "w/o MG", "w/o UAV", "w/o routing", "SEQ", "w/o grpprice", "w/o refine",
       "w/o price", "w/o coupling", "w/o BCA"]
ABL_LAB = {"w/o EA": "去掉能量层（满功率至耗尽）", "w/o ES": "去掉节电状态（保留休眠分配）", "w/o MG": "不使用共享储能",
           "w/o UAV": "去掉无人机", "w/o routing": "路径不优化", "SEQ": "路径不看下层（SEQ）",
           "w/o grpprice": "边际收益不含组级能量价格", "w/o refine": "去掉联合邻域精修",
           "w/o price": "去掉无人机影子价格", "w/o coupling": "节电不看无人机", "w/o BCA": "去掉交替求解"}

def fig_ablation(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, key in zip(axes, SCN):
        b = np.array(D[key]["BRIDGE"]["SR"])
        mu = [((np.array(D[key][a]["SR"]) - b) * 100).mean() for a in ABL]
        y = np.arange(len(ABL))
        ax.barh(y, mu, color="#2a78d6", height=0.6, edgecolor="white")
        for yi, v in zip(y, mu): ax.text(min(v, 0) - 0.15, yi, f"{v:+.2f}", ha="right", va="center", fontsize=7, color=INK)
        ax.set_yticks(y, [ABL_LAB[a] for a in ABL]); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel("相对完整 BRIDGE 的加权服务率变化（百分点）"); ax.set_title(SCN[key], color=INK)
        ax.set_xlim(min(mu) * 1.45 - 0.6, 0.5); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ablation.pdf")); plt.close(fig)

# ------------------------------------------------------------------ 第三轮补充实验
BASE = [("SEQ", "SEQ（分步优化）"), ("SEQ+R", "SEQ + 精修（候选按单站边际收益排序）"),
        ("SEQ+R*", "SEQ + 精修（候选按组级定价的边际收益排序）"), ("SA-1x", "真实目标上的模拟退火（1 倍耗时）"),
        ("SA-5x", "真实目标上的模拟退火（5 倍耗时）"), ("w/o refine", "BRIDGE 主循环（不精修）"), ("BRIDGE", "BRIDGE")]

MG_T = [1, 5, 10, 20, 40]
def fig_mgsize():
    R = load("mgsize")
    if not R: return
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for c, key in enumerate(SCN):
        rs = [r for r in R if r["key"] == key]; xs = np.arange(len(MG_T))
        ax = axes[2 * c]
        for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
            mu = [np.mean([r["res"][a]["SR"] for r in rs if r["tgt"] == t]) for t in MG_T]
            ax.plot(xs, mu, color=COL[a], marker=MK[a], ms=3.5, lw=1.5 if a == "BRIDGE" else 1.1,
                    ls="-" if a in ("BRIDGE", "SEQ") else "--", label=LAB[a])
        ax.set_xticks(xs, ["不共享"] + [str(t) for t in MG_T[1:]], fontsize=6.5); ax.set_xlabel("微电网平均规模", fontsize=7.5)
        ax.set_title(SCN[key][:2] + "：加权服务率", color=INK)
        ax = axes[2 * c + 1]
        for a, col, lab in [("BRIDGE", COL["BRIDGE"], "BRIDGE − SEQ"), ("SEQ+R", "#8a8984", "SEQ+精修 − SEQ")]:
            mu = [np.mean([(r["res"][a]["SR"] - r["res"]["SEQ"]["SR"]) * 100 for r in rs if r["tgt"] == t]) for t in MG_T]
            ax.plot(xs, mu, color=col, marker="o", ms=3.5, lw=1.4, label=lab)
        ax.set_xticks(xs, ["不共享"] + [str(t) for t in MG_T[1:]], fontsize=6.5); ax.set_xlabel("微电网平均规模", fontsize=7.5)
        ax.set_title(SCN[key][:2] + "：相对 SEQ（百分点）", color=INK); ax.axhline(0, color=INK2, lw=0.6)
        if c == 0: ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("加权服务率")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, "mgsize.pdf")); plt.close(fig)

# ------------------------------------------------------------------ 敏感性
def fig_sens():
    S = load("sens"); B = load("block")
    params = [("U", "无人机数量 $U$", S), ("K", "油机车数量 $K$", S), ("bat", "备用电池续航中位数（小时）", S), ("nb", "受损路段数", B)]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharey="row")
    for r, key in enumerate(SCN):
        for c, (p, lab, src) in enumerate(params):
            ax = axes[r, c]
            recs = [x for x in src if x["key"] == key and (x.get("param") == p if p != "nb" else True)]
            vals = sorted({x["val"] if p != "nb" else x["nb"] for x in recs})
            for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
                mu = [np.mean([x["res"][a]["SR"] for x in recs if (x["val"] if p != "nb" else x["nb"]) == v]) for v in vals]
                if not mu: continue
                ax.plot(vals, mu, color=COL[a], marker=MK[a], ms=3.5, lw=1.5 if a == "BRIDGE" else 1.1,
                        ls="-" if a in ("BRIDGE", "SEQ") else "--", label=LAB[a])
            if key == "luding" and p == "K": ax.axvline(4, color="#c3c2b7", lw=0.8, ls=":")
            if key == "xuwen" and p == "K": ax.axvline(5, color="#c3c2b7", lw=0.8, ls=":")
            if p == "U": ax.axvline(6, color="#c3c2b7", lw=0.8, ls=":")
            if p == "bat": ax.axvline(3.5, color="#c3c2b7", lw=0.8, ls=":")
            if p == "nb": ax.axvline(40, color="#c3c2b7", lw=0.8, ls=":")
            if r == 1: ax.set_xlabel(lab)
            if c == 0: ax.set_ylabel(SCN[key] + "\n加权服务率")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sens.pdf")); plt.close(fig)

def fig_sens2():
    S = load("sens")
    params = [("cap", "无人机载荷容量（在线用户数）", [250, 500, 1000, 1e9], ["250", "500", "1000", "不限"]),
              ("mg", "站间能量转移效率", [1.0, 0.9, 0.8, 0.7, 0.0], ["1.0", "0.9", "0.8", "0.7", "不共享"])]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for c, key in enumerate(SCN):
        for d, (p, lab, vals, ticks) in enumerate(params):
            ax = axes[2 * c + d]
            recs = [x for x in S if x["key"] == key and x.get("param") == p]
            xs = np.arange(len(vals))
            for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
                mu = [np.mean([x["res"][a]["SR"] for x in recs if x["val"] == v]) for v in vals]
                ax.plot(xs, mu, color=COL[a], marker=MK[a], ms=3.5, lw=1.5 if a == "BRIDGE" else 1.1,
                        ls="-" if a in ("BRIDGE", "SEQ") else "--", label=LAB[a])
            ax.set_xticks(xs, ticks, fontsize=6.5); ax.set_xlabel(lab, fontsize=7.5)
            ax.set_title(SCN[key][:2], color=INK)
            if 2 * c + d == 0: ax.set_ylabel("加权服务率")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, "sens2.pdf")); plt.close(fig)

def fig_online():
    O = load("online"); Rd = load("road")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for c, key in enumerate(SCN):
        for d, (src, fld, lab) in enumerate([(O, "sigma", "电量估计误差 $\\sigma_{\\mathrm{e}}$"),
                                             (Rd, "sr", "抢通时刻估计误差 $\\sigma_{\\mathrm{r}}$")]):
            ax = axes[2 * c + d]
            recs = [x for x in src if x["key"] == key]
            if not recs: continue
            sg = sorted({x[fld] for x in recs})
            for a in ["GEN-EDF", "BRIDGE-open", "BRIDGE-rolling"]:
                mu = [np.mean([x["res"][a]["SR"] for x in recs if x[fld] == v]) for v in sg]
                ax.plot(sg, mu, color=COL[a], marker=MK.get(a, "v"), ms=3.5, lw=1.4,
                        ls="-" if a != "BRIDGE-open" else "--", label=LAB[a])
            ax.set_xlabel(lab, fontsize=7.5); ax.set_title(SCN[key][:2], color=INK); ax.set_xticks(sg)
            if 2 * c + d == 0: ax.set_ylabel("加权服务率")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, "online.pdf")); plt.close(fig)

def numbers(D):
    """写作用的关键数字（results/numbers.json）。"""
    from scipy.stats import ttest_rel, wilcoxon, t as tdist
    N = {}
    for key in SCN:
        d = D[key]; out = {}
        def pair(a, b="BRIDGE"):
            x = np.array(d[b]["SR"]) * 100; y = np.array(d[a]["SR"]) * 100; df = x - y; n = len(df)
            ci = tdist.ppf(0.975, n - 1) * df.std(ddof=1) / np.sqrt(n)
            wp = float(wilcoxon(df).pvalue) if np.any(df != 0) else 1.0
            return dict(diff_pp=float(df.mean()), ci95=float(ci), t=float(ttest_rel(x, y).statistic) if df.std() > 0 else 0.0,
                        wilcoxon_p=wp, not_worse=int((df >= -1e-9).sum()), n=n)
        out["pairs"] = {a: pair(a) for a in d if not a.startswith("_") and a != "BRIDGE"}
        out["WO_red_pct"] = {a: float(100 * (1 - np.mean(d["BRIDGE"]["WO_kuh"]) / np.mean(d[a]["WO_kuh"]))) for a in MAIN if a != "BRIDGE"}
        N[key] = out
    def agg(exp, fld):
        R = load(exp); res = {}
        for x in R:
            for a, v in x["res"].items():
                k = (x["key"], str(x.get("param", "")), str(x[fld]), a)
                res.setdefault(k, []).append(v)
        return {"|".join(k): {m: float(np.nanmean([v[m] for v in vs])) for m in vs[0] if isinstance(vs[0][m], (int, float))}
                for k, vs in res.items()}
    N["sens"] = agg("sens", "val"); N["block"] = agg("block", "nb"); N["online"] = agg("online", "sigma")
    N["road"] = agg("road", "sr")
    R = load("runtime"); rt = {}
    for x in R:
        for a, v in x["res"].items():
            rt.setdefault(f'{x["key"]}|{x["N"]}|{a}', []).append(v["time"])
            for kk, vv in v.get("tim", {}).items(): rt.setdefault(f'{x["key"]}|{x["N"]}|{a}|{kk}', []).append(vv)
        rt.setdefault(f'{x["key"]}|{x["N"]}|ea', []).append(x["ea_time"])
    N["runtime"] = {k: float(np.mean(v)) for k, v in rt.items()}
    fp = os.path.join(RES, "refine_rank.json")
    if os.path.exists(fp): N["refine_rank"] = json.load(open(fp))
    json.dump(N, open(os.path.join(RES, "numbers.json"), "w"), ensure_ascii=False, indent=1)

def fig_runtime():
    R = load("runtime")
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    for key, ls in zip(SCN, ["-", "--"]):
        recs = [x for x in R if x["key"] == key]; Ns = sorted({x["N"] for x in recs})
        for a in ["SEQ", "BRIDGE"]:
            mu = [np.mean([x["res"][a]["time"] for x in recs if x["N"] == n]) for n in Ns]
            ax.plot(Ns, mu, color=COL[a], marker=MK[a], ms=3.5, lw=1.3, ls=ls, label=f"{LAB[a]}（{SCN[key][:2]}）")
    ax.set_xlabel("基站数量 $N$"); ax.set_ylabel("单次规划耗时（秒）"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "runtime.pdf")); plt.close(fig)

def summary():
    D = main_results(); S = {}
    for key in SCN:
        d = D[key]; S[key] = {a: {m: ms(d[a][m]) for m in ("SR", "WO_kuh", "crit_out_h", "dark_site_h", "max_out_h", "jain", "time_s", "uav_cover_h", "sleep_under_uav_h", "es_frac", "n_gen", "gen_before_depletion", "gen_mean_arrival_h")}
                                for a in d if not a.startswith("_")}
        S[key]["_meta"] = {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v))) for k, v in d["_meta"].items()}
    return D, S

def fig_ablation_en(D):
    """英文正文消融图：只保留有可见差距的组件。变化不超过 0.06 个百分点的三项不画进图。"""
    keep = ["w/o EA", "w/o ES", "w/o MG", "w/o UAV", "w/o routing", "SEQ", "w/o grpprice", "w/o refine"]
    lab = {"w/o EA": "No energy layer (full power until depletion)",
           "w/o ES": "No energy saving (sleep allocation kept)",
           "w/o MG": "No shared storage", "w/o UAV": "No UAVs",
           "w/o routing": "No route optimization", "SEQ": "Routes ignore lower layers (SEQ)",
           "w/o grpprice": "Marginal gain without group-level price",
           "w/o refine": "No joint neighborhood refinement"}
    title = {"luding": "Luding earthquake", "xuwen": "Xuwen typhoon"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    for ax, key in zip(axes, SCN):
        b = np.array(D[key]["BRIDGE"]["SR"])
        mu = [((np.array(D[key][a]["SR"]) - b) * 100).mean() for a in keep]
        y = np.arange(len(keep))
        ax.barh(y, mu, color="#2a78d6", height=0.62, edgecolor="white")
        for yi, v in zip(y, mu):
            ax.text(min(v, 0) - 0.12, yi, f"{v:+.2f}", ha="right", va="center", fontsize=7, color=INK)
        ax.set_yticks(y, [lab[a] for a in keep])
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Change in weighted service ratio\n(percentage points)")
        ax.set_title(title[key], color=INK)
        ax.set_xlim(min(mu) * 1.45 - 0.6, 0.35)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "paper", "figs", "ablation.pdf"))
    plt.close(fig)

def fig_mgsize_en():
    """英文正文微电网规模图：右侧只画 BRIDGE 相对 SEQ 的增益。"""
    R = load("mgsize")
    if not R:
        return
    title = {"luding": "Luding", "xuwen": "Xuwen"}
    ticks = ["None"] + [str(t) for t in MG_T[1:]]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    for c, key in enumerate(SCN):
        rs = [r for r in R if r["key"] == key]
        xs = np.arange(len(MG_T))
        ax = axes[2 * c]
        for a in ["GEN-EDF", "IND", "SEQ", "BRIDGE"]:
            mu = [np.mean([r["res"][a]["SR"] for r in rs if r["tgt"] == t]) for t in MG_T]
            ax.plot(xs, mu, color=COL[a], marker=MK[a], ms=3.5, lw=1.5 if a == "BRIDGE" else 1.1,
                    ls="-" if a in ("BRIDGE", "SEQ") else "--", label=LAB[a])
        ax.set_xticks(xs, ticks, fontsize=6.5)
        ax.set_xlabel("Average microgrid size", fontsize=7.5)
        ax.set_title(title[key] + ": service ratio", color=INK)
        ax = axes[2 * c + 1]
        mu = [np.mean([(r["res"]["BRIDGE"]["SR"] - r["res"]["SEQ"]["SR"]) * 100 for r in rs if r["tgt"] == t]) for t in MG_T]
        ax.plot(xs, mu, color=COL["BRIDGE"], marker="o", ms=3.5, lw=1.4)
        ax.set_xticks(xs, ticks, fontsize=6.5)
        ax.set_xlabel("Average microgrid size", fontsize=7.5)
        ax.set_title(title[key] + ": gain over SEQ (pp)", color=INK)
        ax.axhline(0, color=INK2, lw=0.6)
    axes[0].set_ylabel("Weighted service ratio")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(HERE, "..", "paper", "figs", "mgsize.pdf"))
    plt.close(fig)

if __name__ == "__main__":
    import sys
    D, S = summary()
    fig_main_bars(D); fig_curves(D); fig_cdf(D); fig_ablation(D)
    if glob.glob(os.path.join(RES, "sens", "*.json")): fig_sens(); fig_sens2()
    if glob.glob(os.path.join(RES, "online", "*.json")): fig_online()
    if glob.glob(os.path.join(RES, "runtime", "*.json")): fig_runtime()
    fig_mgsize()
    numbers(D)
    json.dump({k: {a: {m: list(v) for m, v in S[k][a].items()} if not a.startswith("_") else S[k][a] for a in S[k]} for k in S},
              open(os.path.join(RES, "summary.json"), "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 1 and sys.argv[1] == "en":
        fig_ablation_en(D)
        fig_mgsize_en()
    print("ok")

# ------------------------------------------------------------------ 场景地图
def fig_map():
    import pickle, math
    from scenario import make_instance, build_graph, DATA
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, key in zip(axes, SCN):
        I = make_instance(key, 0); g = build_graph(key); C = g["coord"]
        adj = I["nodes_adj"]
        lat0 = (I["cfg"]["bbox"][0] + I["cfg"]["bbox"][2]) / 2; kx = math.cos(math.radians(lat0))
        segs_ok, segs_bad = [], []
        for u, nb in adj.items():
            for v, (tau, r) in nb.items():
                if str(u) < str(v):
                    seg = [(C[u][1] * kx, C[u][0]), (C[v][1] * kx, C[v][0])]
                    (segs_bad if r > 0 else segs_ok).append(seg)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs_ok, colors="#c3c2b7", linewidths=0.4))
        ax.add_collection(LineCollection(segs_bad, colors="#e34948", linewidths=1.6))
        lat = np.array([s["lat"] for s in I["sites"]]); lon = np.array([s["lon"] for s in I["sites"]]) * kx
        sz = 6 + 40 * np.sqrt(I["users"] / I["users"].max())
        crit = I["kappa"] >= 2
        ax.scatter(lon[~crit], lat[~crit], s=sz[~crit], c="#2a78d6", edgecolors="white", linewidths=0.5, label="基站", zorder=3)
        ax.scatter(lon[crit], lat[crit], s=sz[crit], c="#eb6834", edgecolors="white", linewidths=0.5, label="关键站点（$\\kappa\\geq2$）", zorder=3)
        dep = I["cfg"]["depot"]
        ax.scatter([dep[1] * kx], [dep[0]], marker="s", s=46, c="#008300", edgecolors="white", zorder=4, label="油机车库")
        for k, b in enumerate(I["bases"]):
            ax.scatter([b[1] * kx], [b[0]], marker="^", s=46, c="#4a3aa7", edgecolors="white", zorder=5, label="无人机起降点" if k == 0 else None)
        hz = I["cfg"]["hazard"]
        ax.scatter([hz[1] * kx], [hz[0]], marker="*", s=90, c="#e34948", edgecolors="white", zorder=5, label="震中/登陆点")
        s_, w_, n_, e_ = I["cfg"]["bbox"]; ax.set_xlim(w_ * kx, e_ * kx); ax.set_ylim(s_, n_)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#c3c2b7")
        ax.set_title(f"{SCN[key]}（{I['N']} 座基站，{I['n_blocked']} 段受损道路）", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#e34948", lw=1.6)); l.append("受损路段")
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(FIG, "map.pdf")); plt.close(fig)
