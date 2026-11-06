#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpd_shape.py — 各数据集 σ 分数的 GPD 形状参数 ξ（§III 尾部动机数字）

对每数据集读全部种子缓存，校准(val)段 σ 分数（|y−μ|/σ）在 80 分位 POT 阈值上
做 GPD MLE（floc=0，与 aci_wrap.py 的 EVT 分支同一约定），报逐种子 ξ 与跨种子中位数。
论文引用中位数（种子间聚合约定：median）。
输出 Exp/calib/gpd_shape.csv
"""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
from scipy.stats import genpareto

DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]
SEEDS = ["0", "1", "2"]

rows = []
for D in DS:
    xis = []
    for s in SEEDS:
        f = f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz"
        if not os.path.isfile(f):
            continue
        z = np.load(f)
        s_v = np.abs(z["y_v"] - z["mu_v"]) / z["sig_v"]
        u = float(np.quantile(s_v, 0.8))
        exc = s_v[s_v > u] - u
        xi = float(genpareto.fit(exc, floc=0)[0]) if len(exc) >= 10 else float("nan")
        xis.append(xi)
        print(f"[GPD] {D} s{s}: xi={xi:+.3f}", flush=True)
    med = float(np.nanmedian(xis))
    rows.append((D, len(xis), ";".join(f"{x:+.3f}" for x in xis), med))
    print(f"[GPD] {D}: median xi={med:+.3f}", flush=True)

with open("Exp/calib/gpd_shape.csv", "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["dataset", "n_seeds", "xi_per_seed", "xi_median"])
    w.writerows(rows)
print("→ Exp/calib/gpd_shape.csv")
