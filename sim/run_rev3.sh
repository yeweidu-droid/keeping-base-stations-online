#!/usr/bin/env bash
# 第三轮修改补充实验：更强对照、微电网规模、黄金时段加权、需求误差、组合误差、MILP 间隙分解
cd "$(dirname "$0")"
NW=${NW:-16}
for e in base mgsize golden demand combo; do
  python3 experiments.py $e $NW > logs_$e.txt 2>&1
done
python3 milp_bench.py 600 $NW > logs_milp.txt 2>&1
echo REV3DONE > logs_rev3_done.txt
