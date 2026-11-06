#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""enbpi.py — EnbPI 基线（Xu & Xie 2021, "Conformal Prediction Interval for Dynamic
Time-Series"）：ensemble 预测 + 最近 K 个残差的经验分位区间。

适配 TWLP：3 种子 Prob 集成作 bootstrap ensemble（B=3），残差流按快照时间序，
test 段逐边用最近 K 个已见残差（val 末尾 + test 已见）的对称分位构 90% 区间。
窗口 K ∈ {100, 300, 1000} 敏感性。输出 Exp/calib/enbpi.csv。
"""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
KS = [100, 300, 1000]

rows = []
for D in DS:
    fam = []
    for s in "012":
        f = f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz"
        if os.path.isfile(f):
            fam.append(np.load(f))
    if len(fam) < 2:
        print(f"[ENBPI] {D} 种子不足，跳过"); continue
    nt = min(len(z["y_t"]) for z in fam)
    mu_t = np.mean([z["mu_t"][:nt] for z in fam], 0)     # ensemble 均值
    y_t = fam[0]["y_t"][:nt]
    nv = min(len(z["y_v"]) for z in fam)
    mu_v = np.mean([z["mu_v"][:nv] for z in fam], 0)
    y_v = fam[0]["y_v"][:nv]
    r_v = y_v - mu_v
    r_t = y_t - mu_t
    # 残差流：val（历史窗起点）+ test 逐边滚入
    hist = list(r_v[-max(KS):])                          # 初始窗 = val 末尾
    for K in KS:
        covs, wids = [], []
        hist_k = list(hist[-K:])
        for t in range(nt):
            q5 = np.quantile(hist_k, 0.05); q95 = np.quantile(hist_k, 0.95)
            covs.append(q5 <= r_t[t] <= q95)
            wids.append(q95 - q5)
            hist_k.append(r_t[t])
            if len(hist_k) > K:
                hist_k.pop(0)
        c = np.array(covs); w = np.array(wids)
        r_sorted = np.sort(np.abs(r_t))
        tail_thr = np.quantile(np.abs(r_t), 0.90)
        tail = c[np.abs(r_t) >= tail_thr].mean()
        rows.append((D, f"EnbPI_K{K}", float(c.mean()), float(w.mean()), float(tail)))
        print(f"[ENBPI] {D} K={K}: cov90={c.mean():.3f} width={w.mean():.4g} tail={tail:.3f}", flush=True)

import csv
with open("Exp/calib/enbpi.csv", "w", newline="") as fo:
    wr = csv.writer(fo)
    wr.writerow(["dataset", "method", "cov90", "width", "tail90"])
    wr.writerows(rows)
print("→ Exp/calib/enbpi.csv")
