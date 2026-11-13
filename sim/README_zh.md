# 仿真代码（第 6 节实验）

## 复现

```bash
python3 fetch_osm.py            # 抓取 OSM 路网/居民点/设施（mail.ru Overpass 镜像），已缓存于 data/
./run_all.sh                    # 依次运行全部实验（16 进程），结果写入 results/；run_rev3.sh 只跑第三轮补充实验
python3 sleep_split.py          # 补充统计：无人机覆盖时段内基站主动休眠/电池耗尽的构成 → results/sleep_split.json
python3 plots.py                # 由 results/ 生成矢量图，写入 ../paper_zh/figs/；python3 plots.py en 另写英文消融图和微电网规模图到 ../paper/figs/
```

## 文件

| 文件 | 内容 |
|---|---|
| `fetch_osm.py` | 两个场景的 OSM 数据抓取：泸定（2022-09-05 M6.8 地震）、徐闻（2024-09-06 台风"摩羯"） |
| `scenario.py` | 场景构建：路网收缩、居民点聚类布站、重要度、需求曲线、功耗/电池、**微电网划分与共享预算**、道路损毁采样、无人机与油机参数、时变最短路表 |
| `bridge.py` | 算法 1–6：EnergyAlloc（共享预算下的公共阈值贪心）、无人机接力（HiGHS 对偶单纯形，整数解+对偶价格）、LowerLevel、边际收益 G（含组内外溢）、RouteImprove、BRIDGE 主循环、Refine（真实下层目标评估的 1-swap 联合邻域精修）；对比算法 IND 与各消融开关；评价指标 |
| `online.py` | 算法 6 的滚动时域在线运行（电量估计误差、按真实电量执行、固化已执行动作；执行阶段按微电网共享池取电） |
| `runlib.py` | 实例缓存（cache/）与算法分派（含 SEQ） |
| `experiments.py` | 五组实验的任务定义与多进程驱动 |
| `sleep_split.py` | 重跑主实验的 IND/SEQ/BRIDGE（结果与 results/main 逐位一致），把无人机覆盖时段内的基站关断拆成主动休眠（此后仍有电池供电时隙）与电池耗尽，供表 5 使用 |
| `milp_bench.py` | 最优性基准：把 (P0)（微电网共享预算 + 无人机班组 + 时变路网路径）写成 MILP，在真实场景的 10 站×8h 子区域小实例上用 HiGHS 求全局最优 |
| `plots.py` | 读取 results/，生成论文插图（矢量 PDF） |
| `results/` | 每个任务一个 JSON；`summary.json` 为主实验汇总；`block_unpaired_old/` 为早期非配对设计的损毁程度结果（已弃用） |

## 实验规模

- main：2 场景 × 20 种子 × 12 种算法/变体（GEN-EDF、GEN-LS、IND、SEQ、BRIDGE + 7 个消融：w/o ES、w/o UAV、w/o MG、w/o coupling、w/o price、w/o BCA、w/o routing）
- sens：2 × 10 × {U: 0,3,6,9,12,18; K: 1–6; 电池中位数 2,2.5,3.5,4.5,6 h} × 4 算法
- block：2 × 10 × 受损路段数 {0,20,40,60,80}（配对嵌套抽样，其余随机量与默认实例一致）× 4 算法
- online：2 × 10 × σ∈{0,0.1,0.2,0.3} × {GEN-EDF, BRIDGE 开环, BRIDGE 滚动}
- milp：2 × 10 个子区域小实例（10 站 × 8h，1 辆油机车 × 2 台油机，1 架无人机；`python3 milp_bench.py 场景 种子数`）
- runtime：2 × 3 × N∈{20,40,60,80,100,120}（油机车、无人机按 N 等比例缩放）
- base：2 × 20 × {SEQ+R（精修候选按单站边际收益排序）、SEQ+R*（按组级定价的边际收益排序）、真实目标上的模拟退火（1 倍/5 倍 BRIDGE 耗时）}
- mgsize：2 × 10 × 微电网平均规模 {1（不共享）,2,3,5,10,20,40} × {GEN-EDF, IND, SEQ, SEQ+R, BRIDGE}（2、3 为 2026-11-11 补）
- mgnear（2026-11-11 补）：2 × 10 × 近距离共享 {d1: 组内最大站间距 ≤1 km；d2c3: ≤2 km 且每组至多 3 站；d2: ≤2 km} × {GEN-EDF, IND, SEQ, BRIDGE}；划分为完全连接凝聚聚类（scenario.partition_near，覆盖项 mg_dmax_km / mg_max_size），不使用随机数
- golden：2 × 20 × 黄金时段（灾后 6 小时）需求权重放大 γ∈{0,1,3,9}，并统计电池阶段的状态切换次数
- demand：2 × 10 × 需求预测对数误差 σd∈{0.2,0.4,0.6}（区域公共 + 逐站 AR(1)）× 在线运行方式
- combo：2 × 10 × 电量 σe=0.2、需求 σd=0.4、抢通 σr=0.5 同时存在
- uavrole：2 × 20，无人机覆盖去向（IND 与 BRIDGE），结果在 results/uavrole/
- milp 额外记录：MILP 最优路径交给 BRIDGE 下层后的值（间隙分解）与不派油机时的值（油机增益比）
