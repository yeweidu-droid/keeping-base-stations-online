#!/usr/bin/env bash
# 第 6 节全部实验：主实验/消融、敏感性、道路损毁、电量误差在线、抢通误差在线、计算开销、精修排序检验、MILP 基准、
# 后验最优性上界与外溢价格界检验、小实例上的逐车贪心；更强对照、微电网规模、黄金时段加权、需求误差与组合误差
cd "$(dirname "$0")"
NW=${NW:-16}
for e in main sens block online road runtime base mgsize mgnear golden demand combo uavrole; do
  python3 experiments.py $e $NW > logs_$e.txt 2>&1
done
python3 refine_rank.py $NW > logs_refine_rank.txt 2>&1
python3 milp_bench.py 600 $NW > logs_milp.txt 2>&1
python3 certificate.py $NW > logs_certificate.txt 2>&1
python3 certificate.py $NW small >> logs_certificate.txt 2>&1
python3 plots.py
echo ALLDONE > logs_done.txt
