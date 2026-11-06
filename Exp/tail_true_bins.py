#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tail_true_bins.py —— 在线 wrapper（3成员×种子中位数）在真实权重 top-decile 上的覆盖率 → Exp/calib/tail_true_bins.csv
"""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
from scipy.stats import genpareto

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
SEEDS = ["0","1","2"]

def csm(x, w=5):
    pad = np.concatenate([np.full(w-1, x[0]), x])
    return np.convolve(pad, np.ones(w)/w, mode="valid")

rows = []
print(f"{'数据集':11s}{'成员':16s}{'tail90':>10s}")
for D in DS:
    caches = [(s, np.load(f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz"))
              for s in SEEDS if os.path.isfile(f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz")]
    if not caches: continue

    for m in ["sigma+EVT+ACI", "naive+ACI", "ncp_smooth+ACI"]:
        tails = []
        for s, z in caches:
            y_t, mu_t, sig_t = z["y_t"], z["mu_t"], z["sig_t"]
            s_v = np.abs(z["y_v"] - z["mu_v"]) / z["sig_v"]
            s_vn = np.abs(z["y_v"] - z["mu_v"])
            sn_v = csm(s_vn)
            if m == "sigma+EVT+ACI":
                score_t = np.abs(y_t - mu_t) / sig_t
                hist0, upd = s_v, s_v  # 校准与更新都用 σ 分数（更新源=test σ分数）
            elif m == "naive+ACI":
                score_t = np.abs(y_t - mu_t)
                hist0, upd = s_vn, s_vn
            else:
                score_t = csm(np.abs(y_t - mu_t))
                hist0, upd = sn_v, sn_v
            n = len(score_t); start = max(len(hist0)//5, 30)
            block = max(1, min(1000, (n-start)//200 or 1))
            a_t = 0.10
            hist = np.sort(hist0[:start])
            hit = np.zeros(n, dtype=float)
            t = start
            while t < n:
                end = min(t+block, n)
                q = hist[min(int((1-a_t)*len(hist)), len(hist)-1)]
                if m == "sigma+EVT+ACI":
                    u = float(np.quantile(hist0, 0.8))
                    exc = hist0[hist0>u] - u
                    if q >= u and len(exc) >= 10:
                        xi,_,beta = genpareto.fit(exc, floc=0)
                        w_u = float((hist0>u).mean())
                        need = ((1-a_t)-(1-w_u))/w_u
                        if 0 < need < 1:
                            q = max(u+genpareto.ppf(need, xi, scale=beta), q)
                hit[t:end] = (score_t[t:end] <= q).astype(float)
                unc = 1 - hit[t:end].mean()
                a_t = min(max(a_t+0.05*(0.10-unc), 1e-4), 0.5)
                t = end
            thr = np.quantile(y_t[start:], 0.90)
            tails.append(float(hit[start:][y_t[start:] >= thr].mean()))
        med = float(np.median(tails))
        rows.append((D, m, med))
        print(f"{D:11s}{m:16s}{med:10.3f}", flush=True)

with open("Exp/calib/tail_true_bins.csv", "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["dataset", "method", "tail90_true_weight"])
    w.writerows(rows)
