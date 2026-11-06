#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evt_static_tail.py — 静态 σ 区间 vs σ+GPD 区间在真实权重 p95–p99 段的覆盖（§V 静态尾部数字）

静态 split CP（无在线更新）：q = 校准 σ 分数的 (1−α) 分位；区间 μ±q·σ。
GPD 变体：q ≥ u 时抬升为 u+GPD_ppf（u = 校准分数 80 分位，MLE floc=0，与 aci_wrap.py 同约定）。
报 test 段全体覆盖、p95–p99 真实权重段覆盖、平均宽度（含 σ+GPD 的宽度代价比）。
种子间取中位数。输出 Exp/calib/evt_static_tail.csv
"""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
from scipy.stats import genpareto

DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]
SEEDS = ["0", "1", "2"]
ALPHA = 0.10

def cov_width(z, evt):
    s_v = np.abs(z["y_v"] - z["mu_v"]) / z["sig_v"]
    y_t, mu_t, sig_t = z["y_t"], z["mu_t"], z["sig_t"]
    q = float(np.quantile(s_v, 1 - ALPHA))
    if evt:
        u = float(np.quantile(s_v, 0.8))
        exc = s_v[s_v > u] - u
        if len(exc) >= 10:
            xi, _, beta = genpareto.fit(exc, floc=0)
            w_u = float((s_v > u).mean())
            need = ((1 - ALPHA) - (1 - w_u)) / w_u
            if 0 < need < 1:
                q = max(u + float(genpareto.ppf(need, xi, scale=beta)), q)
    half = q * sig_t
    hit = (np.abs(y_t - mu_t) <= half).astype(float)
    return hit, 2 * half

rows = []
for D in DS:
    caches = [np.load(f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz") for s in SEEDS
              if os.path.isfile(f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz")]
    if not caches:
        continue
    res = {}
    for tag, evt in [("sigma", False), ("sigma_evt", True)]:
        covs, tails, wids = [], [], []
        for z in caches:
            y_t = z["y_t"]
            hit, wid = cov_width(z, evt)
            covs.append(float(hit.mean()))
            lo, hi = np.quantile(y_t, [0.95, 0.99])
            m = (y_t >= lo) & (y_t <= hi)
            tails.append(float(hit[m].mean()))
            wids.append(float(wid.mean()))
        res[tag] = (float(np.median(covs)), float(np.median(tails)), float(np.median(wids)))
    cost = res["sigma_evt"][2] / res["sigma"][2] - 1
    rows.append((D, res["sigma"][1], res["sigma_evt"][1],
                 res["sigma"][0], res["sigma_evt"][0], cost))
    print(f"[EVT-TAIL] {D}: p95-99 {res['sigma'][1]:.3f}->{res['sigma_evt'][1]:.3f}  "
          f"cov {res['sigma'][0]:.3f}/{res['sigma_evt'][0]:.3f}  width cost {cost:+.2%}", flush=True)

with open("Exp/calib/evt_static_tail.csv", "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["dataset", "tail_p95_99_sigma", "tail_p95_99_sigma_evt",
                "cov_sigma", "cov_sigma_evt", "width_cost_pct"])
    w.writerows(rows)
print("→ Exp/calib/evt_static_tail.csv")
