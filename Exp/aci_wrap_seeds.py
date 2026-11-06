#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对每数据集读全部可用种子缓存（s0/s1/s2），每种子独立跑各成员的块级 ACI，
报 3 种子中位数 + 跨种子极差（range）。CQR 成员仅 s0（单模型），如实在列标注。
输出 Exp/calib/aci_wrap_seeds.csv（含 median/range 两套列）。
"""
import os, sys, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
from scipy.stats import genpareto
from aci_wrap import aci_scores

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
SEEDS = ["0","1","2"]

rows = []
for D in DS:
    caches = []
    for s in SEEDS:
        f = f"Exp/cache/{D}_SALT_Prob_full_s{s}.npz"
        if os.path.isfile(f):
            caches.append((s, np.load(f)))
    if not caches:
        continue
    zc = np.load(f"Exp/cache/{D}_SALT_CQR_s0_q.npz") if os.path.isfile(f"Exp/cache/{D}_SALT_CQR_s0_q.npz") else None

    def run_members(z, seed):
        out = {}
        s_v = np.abs(z["y_v"] - z["mu_v"]) / z["sig_v"]
        s_t = np.abs(z["y_t"] - z["mu_t"]) / z["sig_t"]
        out["sigma+ACI"] = aci_scores(s_v, s_t, y_t=z["y_t"], mu_t=z["mu_t"], sig_t=z["sig_t"])
        out["sigma+EVT+ACI"] = aci_scores(s_v, s_t, y_t=z["y_t"], mu_t=z["mu_t"], sig_t=z["sig_t"], evt=True)
        s_vn = np.abs(z["y_v"] - z["mu_v"]); s_tn = np.abs(z["y_t"] - z["mu_t"])
        out["naive+ACI"] = aci_scores(s_vn, s_tn)
        def csm(x, w=5):
            pad = np.concatenate([np.full(w-1, x[0]), x])
            return np.convolve(pad, np.ones(w)/w, mode="valid")
        out["ncp_smooth+ACI"] = aci_scores(csm(s_vn), csm(s_tn))
        return out

    # 跨种子收集
    per_seed = []
    for s, z in caches:
        per_seed.append(run_members(z, s))
    members = per_seed[0].keys()

    # CQR（仅 s0，单模型）
    cqr_cov = cqr_wid = None
    if zc is not None:
        sc_v = np.maximum(zc["qlo_v"] - zc["y_v"], zc["y_v"] - zc["qhi_v"])
        sc_t = np.maximum(zc["qlo_t"] - zc["y_t"], zc["y_t"] - zc["qhi_t"])
        bw = zc["qhi_t"] - zc["qlo_t"]
        cqr_cov, cqr_wid, _ = aci_scores(sc_v, sc_t, base_width_t=bw)

    for m in members:
        covs = [r[m][0] for r in per_seed]
        wids = [r[m][1] for r in per_seed]
        rows.append((D, m, len(caches), float(np.median(covs)), float(np.max(covs)-np.min(covs)),
                     float(np.median(wids)), float(np.max(wids)-np.min(wids))))
        print(f"[SEEDS] {D:11s} {m:16s} n={len(caches)} cov_med={np.median(covs):.3f} (range {np.max(covs)-np.min(covs):.3f}) wid_med={np.median(wids):9.3g} (range {np.max(wids)-np.min(wids):9.3g})", flush=True)
    if cqr_cov is not None:
        rows.append((D, "CQR+ACI", 1, cqr_cov, 0.0, cqr_wid, 0.0))
        print(f"[SEEDS] {D:11s} CQR+ACI          n=1(单模型) cov={cqr_cov:.3f} wid={cqr_wid:9.3g}", flush=True)

with open("Exp/calib/aci_wrap_seeds.csv", "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["dataset","method","n_seeds","cov90_median","cov90_range","width_median","width_range"])
    w.writerows(rows)
print("→ Exp/calib/aci_wrap_seeds.csv")
