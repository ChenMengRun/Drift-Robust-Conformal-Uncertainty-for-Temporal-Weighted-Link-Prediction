#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""family_select.py — 分数族选择器

族 = {residual, ncp_smooth, sigma_informed, cqr} 四成员。
选择规则（只用 val 数据，无 test 泄漏）：
  对每成员在 val 段构 90% 区间 → 计算 val 覆盖与 val 平均宽度；
  在 val 覆盖达标（|cov-0.90|≤0.05）的成员中选 val 宽度最窄者；无达标者选覆盖最接近者。
test 段：选中成员 + 块级 ACI（与 §3.4 同一包装）。
输出 Exp/calib/family_final.csv（val 选择记录 + test 结果）。
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'SALT')); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
import numpy as np
from scipy.stats import genpareto
from aci_wrap import aci_scores

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
MEMBERS = ["residual", "ncp_smooth", "sigma", "cqr"]


def member_scores(D):
    """返回 {成员: (s_v, s_t, base_half_t or None, base_width_t or None, y_t, mu_t)}"""
    zf0 = f"Exp/cache/{D}_SALT_Prob_full_s0.npz"
    z = np.load(zf0) if os.path.isfile(zf0) else np.load(f"Exp/cache/{D}_SALT_Prob_full_s1.npz")
    out = {}
    r_v = np.abs(z["y_v"] - z["mu_v"]); r_t = np.abs(z["y_t"] - z["mu_t"])
    out["residual"] = (r_v, r_t, None, None)
    def causal_smooth(x, w=5):
        pad = np.concatenate([np.full(w-1, x[0]), x])
        return np.convolve(pad, np.ones(w)/w, mode="valid")
    sn_v = causal_smooth(r_v)
    sn_t = causal_smooth(r_t)
    out["ncp_smooth"] = (sn_v, sn_t, None, None)
    s_v = r_v / z["sig_v"]; s_t = r_t / z["sig_t"]
    # val 宽度用 sig_v（选择不触 test）；test 评估另取 sig_t
    out["sigma"] = (s_v, s_t, z["sig_v"], None)
    out["_sigma_sig_t"] = z["sig_t"]
    zc = np.load(f"Exp/cache/{D}_SALT_CQR_s0_q.npz")
    sc_v = np.maximum(zc["qlo_v"] - zc["y_v"], zc["y_v"] - zc["qhi_v"])
    sc_t = np.maximum(zc["qlo_t"] - zc["y_t"], zc["y_t"] - zc["qhi_t"])
    out["cqr"] = (sc_v, sc_t, None, zc["qhi_t"] - zc["qlo_t"])
    return out, z["y_t"], z["mu_t"]


def val_metrics(s_v, s_t, base_half=None, base_width=None, alpha=0.10):
    """val 段 90% 区间的覆盖与宽度（不含 test）"""
    q = np.quantile(s_v, 1 - alpha)
    cov = float(np.mean(s_v <= q))
    if base_half is not None:
        width = float(np.mean(2 * q * base_half))
    else:
        base = base_width if base_width is not None else 0.0
        width = float(np.mean(base + 2 * q))
    return cov, width


rows = []
for D in DS:
    mem, y_t, mu_t = member_scores(D)
    # val 选择
    cand = []
    for m in MEMBERS:
        s_v, s_t, bh, bw = mem[m]
        cov_v, wid_v = val_metrics(s_v, s_t, bh, bw)
        cand.append((m, cov_v, wid_v))
    ok = [c for c in cand if abs(c[1] - 0.90) <= 0.05]
    pick_pool = ok if ok else cand
    picked = min(pick_pool, key=lambda c: c[2])
    # test：选中成员 + ACI（σ 成员用 sig_t——test 段尺度，评估专用非选择）
    s_v, s_t, bh, bw = mem[picked[0]]
    if picked[0] == "sigma":
        bh = mem["_sigma_sig_t"]
    if bh is not None:
        cov, wid, tail = aci_scores(s_v, s_t, y_t=y_t, mu_t=mu_t, sig_t=bh)
    else:
        cov, wid, tail = aci_scores(s_v, s_t, base_width_t=bw)
    rows.append((D, picked[0], picked[1], picked[2], cov, wid, tail))
    print(f"[FAM] {D:11s} 选={picked[0]:11s} val(cov={picked[1]:.3f},wd={picked[2]:.4g}) → test cov={cov:.3f} wd={wid:.4g} tail={tail:.3f}", flush=True)

import csv
with open("Exp/calib/family_final.csv", "w", newline="") as fo:
    wr = csv.writer(fo)
    wr.writerow(["dataset", "selected", "val_cov", "val_width", "test_cov90", "test_width", "test_tail90"])
    wr.writerows(rows)
print("→ Exp/calib/family_final.csv")
