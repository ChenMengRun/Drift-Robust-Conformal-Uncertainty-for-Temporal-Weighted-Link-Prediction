#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 Xu & Xie 2021：B 个 bootstrap 重采样训练的 ensemble + LOO 残差。
适配：无法重训 B 个模型（GPU 成本），以 3 种子模型近似 bootstrap ensemble（B=3，
如实声明局限）；LOO 残差：对第 i 个校准点，用不含该点的其余校准残差的滑动
分位数近似（EnbPI 的 LOO 精神：残差估计不包含自身）。含有限样本校正
分位 rank = ceil((n+1)(1-alpha)) 的截断实现。
输出 Exp/calib/enbpi_faithful.csv
"""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
K = 300
ALPHA = 0.10

rows = []
for D in DS:
    fam = []
    for s in "012":
        f = f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz"
        if os.path.isfile(f):
            fam.append(np.load(f))
    if len(fam) < 2:
        print(f"[F-ENBPI] {D} 种子不足，跳过"); continue
    nt = min(len(z["y_t"]) for z in fam)
    nv = min(len(z["y_v"]) for z in fam)
    # ensemble 均值（B=3 种子，bootstrap 的近似——局限如实写入正文）
    mu_t = np.mean([z["mu_t"][:nt] for z in fam], 0)
    mu_v = np.mean([z["mu_v"][:nv] for z in fam], 0)
    y_t = fam[0]["y_t"][:nt]
    y_v = fam[0]["y_v"][:nv]
    r_v = y_v - mu_v
    r_t = y_t - mu_t
    covs, wids = [], []
    hist = list(r_v[-K:])
    for t in range(nt):
        n = len(hist)
        # 有限样本校正：rank = ceil((n+1)(1-alpha)) 截断到 n
        qs = np.sort(hist)
        # 双侧 90%：每侧尾 α/2，有限样本校正 ceil((n+1)p)，0-based
        q5 = qs[min(int(np.ceil((n + 1) * ALPHA / 2)) - 1, n - 1)]
        q95 = qs[min(int(np.ceil((n + 1) * (1 - ALPHA / 2))) - 1, n - 1)]
        covs.append(q5 <= r_t[t] <= q95)
        wids.append(q95 - q5)
        # LOO 精神：加入新残差前，用它更新窗口（在线可见标签设定）
        hist.append(r_t[t])
        if len(hist) > K:
            hist.pop(0)
    c = np.array(covs); w = np.array(wids)
    rows.append((D, "EnbPI_faithful_K300", float(c.mean()), float(w.mean())))
    print(f"[F-ENBPI] {D}: cov90={c.mean():.3f} width={w.mean():.4g}", flush=True)

with open("Exp/calib/enbpi_faithful.csv", "w", newline="") as fo:
    wr = csv.writer(fo)
    wr.writerow(["dataset","method","cov90","width"])
    wr.writerows(rows)
print("→ Exp/calib/enbpi_faithful.csv")
