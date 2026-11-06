#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""block_bootstrap.py — 时间块 bootstrap（替代逐边 iid 重采样）

iid 重采样与论文前提（流非可交换）矛盾；正确抽样单位=时间块。
将 test 段按 50 个快照分块，块内取覆盖均值，B=1000 次重采样块序列 → 覆盖率的
块 bootstrap 95% CI。对 sigma_evt（静态）与 sigma+EVT+ACI（在线）两方法。
输出 Exp/calib/block_bootstrap.csv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
from conformal_twlp import weighted_quantile, drift_weights
from aci_wrap import aci_scores
from scipy.stats import genpareto

B = 1000
NBLK = 50   # 时间块数（快照粒度）
DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]

rows = []
for D in DS:
    zf0 = f"Exp/cache/{D}_SALT_Prob_full_s0.npz"
    z = np.load(zf0) if os.path.isfile(zf0) else np.load(f"Exp/cache/{D}_SALT_Prob_full_s1.npz")
    s_v = np.abs(z["y_v"] - z["mu_v"]) / z["sig_v"]
    s_t = np.abs(z["y_t"] - z["mu_t"]) / z["sig_t"]
    s_vn = np.abs(z["y_v"] - z["mu_v"])
    s_tn = np.abs(z["y_t"] - z["mu_t"])
    # 静态：naive 残差分数 split CP（与 Table 1 Static 列同源）
    q = np.quantile(s_vn, 0.90)
    hit_static = (s_tn <= q).astype(float)
    # 在线：族选择口径——成员从 family_final.csv 动态读取（与 Table 1 同源）
    import csv as _csv
    _fam = {r["dataset"]: r["selected"] for r in _csv.DictReader(open("Exp/calib/family_final.csv"))}
    sel = _fam[D]
    if sel == "sigma":
        _, _, _, hit_on = aci_scores(s_v, s_t, y_t=z["y_t"], mu_t=z["mu_t"],
                                     sig_t=z["sig_t"], evt=True, return_hits=True)
    elif sel == "ncp_smooth":
        def _csm(x, w=5):
            pad = np.concatenate([np.full(w-1, x[0]), x])
            return np.convolve(pad, np.ones(w)/w, mode="valid")
        r_v = np.abs(z["y_v"] - z["mu_v"]); r_t = np.abs(z["y_t"] - z["mu_t"])
        _, _, _, hit_on = aci_scores(_csm(r_v), _csm(r_t), return_hits=True)
    elif sel == "residual":
        r_v = np.abs(z["y_v"] - z["mu_v"]); r_t = np.abs(z["y_t"] - z["mu_t"])
        _, _, _, hit_on = aci_scores(r_v, r_t, return_hits=True)
    else:  # cqr
        zc = np.load(f"Exp/cache/{D}_SALT_CQR_s0_q.npz")
        sc_v = np.maximum(zc["qlo_v"] - zc["y_v"], zc["y_v"] - zc["qhi_v"])
        sc_t = np.maximum(zc["qlo_t"] - zc["y_t"], zc["y_t"] - zc["qhi_t"])
        bw_t = zc["qhi_t"] - zc["qlo_t"]
        _, _, _, hit_on = aci_scores(sc_v, sc_t, base_width_t=bw_t, evt=True, return_hits=True)
    def block_boot(hit):
        blk = np.array_split(hit, NBLK)
        bm = np.array([b.mean() for b in blk])
        rng = np.random.default_rng(0)
        bs = [rng.choice(bm, NBLK, replace=True).mean() for _ in range(B)]
        return bm.mean(), np.percentile(bs, [2.5, 97.5])
    m1, ci1 = block_boot(hit_static)
    m2, ci2 = block_boot(hit_on)
    rows.append((D, "static_sigma_evt", m1, ci1[0], ci1[1]))
    rows.append((D, "online_sigma_evt_aci", m2, ci2[0], ci2[1]))
    print(f"[BLOCK-BOOT] {D}: static {m1:.3f} [{ci1[0]:.3f},{ci1[1]:.3f}] | online {m2:.3f} [{ci2[0]:.3f},{ci2[1]:.3f}]", flush=True)

import csv
with open("Exp/calib/block_bootstrap.csv", "w", newline="") as fo:
    wr = csv.writer(fo)
    wr.writerow(["dataset", "method", "cov90", "ci_lo", "ci_hi"])
    wr.writerows(rows)
print("→ Exp/calib/block_bootstrap.csv")
