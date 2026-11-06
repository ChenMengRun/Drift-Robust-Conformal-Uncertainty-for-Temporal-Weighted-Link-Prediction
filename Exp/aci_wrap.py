#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aci_wrap.py — 三个对手装同一 ACI 包装 + CQR GPD 对称化（CPU，基于缓存）

方法（每数据集）：
  sigma+ACI / sigma+EVT+ACI     分数=|y-μ|/σ，区间 [μ-qσ, μ+qσ]
  naive+ACI                     分数=|y-μ|，区间 [μ-q, μ+q]
  CQR+ACI / CQR+GPD+ACI         分数=max(qlo-y, y-qhi)，区间 [qlo-q, qhi+q]
输出 Exp/calib/aci_wrap.csv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
from scipy.stats import genpareto

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]


def aci_scores(s_v, s_t, y_t=None, mu_t=None, sig_t=None, base_width_t=None,
               alpha=0.10, gamma=0.05, evt=False, return_hits=False):
    """块级 ACI。三种情形由参数决定：
    - 裸分数（naive/CQR）：half=q，判定 s_t<=q，宽度=base_width_t+2q（若给）否则 2q
    - σ 尺度（sigma）：half=q*sig_t，判定 |y-mu|<=half，宽度=2*mean(half)
    返回 (cov90, width, tail90)"""
    n = len(s_t)
    start = max(len(s_v) // 5, 30)
    block = max(1, min(1000, (n - start) // 200 or 1))
    u = None
    if evt:
        u = float(np.quantile(s_v, 0.8))
        exc = s_v[s_v > u] - u
        xi, _, beta = genpareto.fit(exc, floc=0) if len(exc) >= 10 else (0, 0, 1)
    else:
        xi, beta = 0, 1
    a_t = alpha
    hist = np.sort(s_v[:start])
    cov_all, wid_all = [], []
    t = start
    while t < n:
        end = min(t + block, n)
        q = hist[min(int((1 - a_t) * len(hist)), len(hist) - 1)]
        if evt and u is not None and q >= u and len(s_v[s_v > u]) >= 10:
            w_u = float((s_v > u).mean())
            need = ((1 - a_t) - (1 - w_u)) / w_u
            if 0 < need < 1:
                q = max(u + genpareto.ppf(need, xi, scale=beta), q)
        if sig_t is not None:                     # σ 尺度情形
            half = q * sig_t[t:end]
            cov = (np.abs(y_t[t:end] - mu_t[t:end]) <= half).astype(float)
            wid = 2 * half
        else:                                     # 裸分数情形（naive/CQR）
            cov = (s_t[t:end] <= q).astype(float)
            base = base_width_t[t:end] if base_width_t is not None else np.zeros(end - t)
            wid = base + 2.0 * q
        cov_all.append(cov); wid_all.append(wid)
        unc = 1 - cov.mean()
        a_t = min(max(a_t + gamma * (alpha - unc), 1e-4), 0.5)
        hist = np.sort(np.concatenate([hist, s_t[t:end]]))
        t = end
    c = np.concatenate(cov_all)
    w = np.concatenate(wid_all)
    s_online = s_t[start:]
    tail_thr = np.quantile(s_online, 0.90)
    tail = c[s_online >= tail_thr].mean()
    if return_hits:
        return float(c.mean()), float(w.mean()), float(tail), c
    return float(c.mean()), float(w.mean()), float(tail)


rows = []
for D in DS:
    zf0 = f"Exp/cache/{D}_SALT_Prob_full_s0.npz"
    z = np.load(zf0) if os.path.isfile(zf0) else np.load(f"Exp/cache/{D}_SALT_Prob_full_s1.npz")
    # σ 与 naive 分数
    s_v = np.abs(z["y_v"] - z["mu_v"]) / z["sig_v"]
    s_t = np.abs(z["y_t"] - z["mu_t"]) / z["sig_t"]
    r = aci_scores(s_v, s_t, y_t=z["y_t"], mu_t=z["mu_t"], sig_t=z["sig_t"])
    rows.append((D, "sigma+ACI", *r))
    r = aci_scores(s_v, s_t, y_t=z["y_t"], mu_t=z["mu_t"], sig_t=z["sig_t"], evt=True)
    rows.append((D, "sigma+EVT+ACI", *r))
    s_vn = np.abs(z["y_v"] - z["mu_v"]); s_tn = np.abs(z["y_t"] - z["mu_t"])
    r = aci_scores(s_vn, s_tn)
    rows.append((D, "naive+ACI", *r))
    # ncp_smooth 分数（时序滑窗平滑，NCPNet 精神迁移）+ ACI
    def causal_smooth(x, w=5):
        pad = np.concatenate([np.full(w-1, x[0]), x])
        return np.convolve(pad, np.ones(w)/w, mode="valid")
    k = np.array([1,1,1,1,1])/5
    sn_v = causal_smooth(s_vn)
    sn_t = causal_smooth(s_tn)
    r = aci_scores(sn_v, sn_t)
    rows.append((D, "ncp_smooth+ACI", *r))
    # grouped（Mondrian 按 σ 分组）的在线版：组内 ACI
    from scipy.stats import mstats
    qs = np.quantile(z["sig_v"], [0, .33, .66, 1.0]); qs[0] = -np.inf
    # grouped: only evaluated test edges count toward the denominator
    hit_list, wid_list = [], []
    for g in range(3):
        m_v = (z["sig_v"] > qs[g]) & (z["sig_v"] <= qs[g+1])
        m_t = (z["sig_t"] > qs[g]) & (z["sig_t"] <= qs[g+1])
        if m_v.sum() < 10 or m_t.sum() == 0:
            continue
        cg, wg, tg, h = aci_scores(s_v[m_v], s_t[m_t], y_t=z["y_t"][m_t], mu_t=z["mu_t"][m_t],
                                   sig_t=z["sig_t"][m_t], return_hits=True)
        hit_list.append(h)
        wid_list.append(np.full(len(h), wg))
    if hit_list:
        cov_g = float(np.concatenate(hit_list).mean())
        wid_g = float(np.concatenate(wid_list).mean())
    else:
        cov_g = wid_g = float("nan")
    rows.append((D, "grouped+ACI", cov_g, wid_g, float("nan")))
    # CQR
    zc = np.load(f"Exp/cache/{D}_SALT_CQR_s0_q.npz")
    sc_v = np.maximum(zc["qlo_v"] - zc["y_v"], zc["y_v"] - zc["qhi_v"])
    sc_t = np.maximum(zc["qlo_t"] - zc["y_t"], zc["y_t"] - zc["qhi_t"])
    bw_t = zc["qhi_t"] - zc["qlo_t"]          # CQR 基础区间宽
    r = aci_scores(sc_v, sc_t, base_width_t=bw_t)
    rows.append((D, "CQR+ACI", *r))
    r = aci_scores(sc_v, sc_t, base_width_t=bw_t, evt=True)
    rows.append((D, "CQR+GPD+ACI", *r))

import csv
with open("Exp/calib/aci_wrap.csv", "w", newline="") as fo:
    wr = csv.writer(fo)
    wr.writerow(["dataset", "method", "cov90", "width", "tail90"])
    wr.writerows(rows)
print(f"{'数据集':10s}{'方法':16s}{'cov90':>8s}{'宽度':>11s}{'尾90':>8s}")
for r in rows:
    print(f"{r[0]:10s}{r[1]:16s}{r[2]:8.3f}{r[3]:11.4g}{r[4]:8.3f}", flush=True)
print("→ Exp/calib/aci_wrap.csv")
