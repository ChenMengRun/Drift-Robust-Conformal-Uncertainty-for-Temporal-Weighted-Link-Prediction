#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conformal_twlp.py — 漂移感知共形 TWLP（新立意核心方法实现）

四种非一致性分数 × 可选漂移加权 × 可选 EVT 尾外推，全部 split-conformal 框架：
  naive      score = |y-μ|                     （标准 split CP）
  homo       score = |y-μ|/σ_const             （同方差标准化对照）
  sigma      score = |y-μ|/σ                   （★我们的 σ 感知分数：趋近可交换→收紧覆盖缺口）
  sigma_evt  score = |y-μ|/σ + GPD 尾外推      （★重尾数据集的尾分位数修正）
漂移加权：Barber et al.(2023) 非交换 CP 的加权分位，校准分数按时间几何衰减 w_i=ρ^(位置)。
指标：coverage@{80,90,95}、平均区间宽度、p90-p100 尾桶 coverage。
输入：Exp/cache/{D}_{run}.npz（cache_musig.py 产出）。纯 CPU。
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import genpareto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def weighted_quantile(scores, weights, alpha):
    """Barber 加权分位: q̂ = inf{q: Σw_i·1[s_i≤q] ≥ alpha·Σw}，测试点质量并入分母"""
    idx = np.argsort(scores)
    s, w = scores[idx], weights[idx]
    cw = np.cumsum(w)
    total = cw[-1]
    k = np.searchsorted(cw, alpha * total)
    return s[min(k, len(s) - 1)]


def drift_weights(n, rho):
    """几何时间衰减：越新（越靠近 test）权重越大，w_i = rho^(n-1-i)"""
    return rho ** np.arange(n - 1, -1, -1)


def conformal(mu_v, sig_v, y_v, mu_t, sig_t, y_t, score_kind, rho=1.0,
              evt_quant=0.8, alpha_levels=(0.80, 0.90, 0.95)):
    """返回 {alpha: (coverage, mean_width, tail_cov)}"""
    if score_kind == "naive":
        s_v = np.abs(y_v - mu_v); sc_v = np.ones_like(mu_v); sc_t = np.ones_like(mu_t)
    elif score_kind == "homo":
        c = float(np.sqrt(np.mean((y_v - mu_v) ** 2)))
        s_v = np.abs(y_v - mu_v) / c; sc_v = np.full_like(mu_v, c); sc_t = np.full_like(mu_t, c)
    else:  # sigma / sigma_evt
        s_v = np.abs(y_v - mu_v) / sig_v; sc_v = sig_v; sc_t = sig_t

    w = drift_weights(len(s_v), rho) if rho < 1.0 else np.ones(len(s_v))
    out = {}
    for alpha in alpha_levels:
        a = 1 - alpha
        if score_kind == "sigma_evt":
            # 加权分位 + GPD 尾外推：分数上 tail(≥u) 部分用 GPD 修正
            u = weighted_quantile(s_v, w, evt_quant)
            exc = s_v[s_v > u] - u
            if len(exc) >= 10:
                xi, _, beta = genpareto.fit(exc, floc=0)
                q_emp = weighted_quantile(s_v, w, alpha)
                if q_emp >= u:
                    # 目标: P(s≤q) = 1-a；分解为 P(s≤u) + P(u<s≤q) = (1-w_u) + w_u·GPD(q-u)
                    w_u = float(w[s_v > u].sum() / w.sum())
                    need = ((1 - a) - (1 - w_u)) / w_u   # GPD 的条件累积概率
                    if 0 < need < 1:
                        q = max(u + genpareto.ppf(need, xi, scale=beta), q_emp)
                    else:
                        q = q_emp
                else:
                    q = q_emp
            else:
                q = weighted_quantile(s_v, w, alpha)
        else:
            q = weighted_quantile(s_v, w, alpha)
        half = q * sc_t
        cov = float(np.mean(np.abs(y_t - mu_t) <= half))
        width = float(np.mean(2 * half))
        q90 = np.quantile(y_t, 0.90)
        tail = float(np.mean(np.abs(y_t[y_t >= q90] - mu_t[y_t >= q90]) <= half[y_t >= q90]))
        out[alpha] = (cov, width, tail)
    return out


def aci_online(mu_seq, sig_seq, y_seq, score_kind, alpha=0.10, gamma=0.05, block=1000):
    """在线自适应共形（Gibbs & Candès 2021）：α_{t+1} = α_t + γ(α − 1{未覆盖})。
    大数据集按 block 条边更新一次（块内用当前 α 取分位，块末更新 α），复杂度降 block 倍，
    语义仍为在线博弈更新。返回 (平均覆盖, 平均宽度)。"""
    scale = sig_seq if score_kind != "naive" else np.ones_like(sig_seq)
    errs = np.abs(y_seq - mu_seq) / scale
    start = max(len(errs) // 5, 30)
    n = len(errs)
    block = max(1, min(block, (n - start) // 200 or 1))   # 保证 ≥~200 次更新
    a_t = alpha
    covs, wids = [], []
    t = start
    hist_sorted = np.sort(errs[:start])
    while t < n:
        end = min(t + block, n)
        q_t = hist_sorted[min(int((1 - a_t) * len(hist_sorted)), len(hist_sorted) - 1)]
        half = q_t * scale[t:end]
        covs.append(np.abs(y_seq[t:end] - mu_seq[t:end]) <= half)
        wids.append(2 * half)
        unc = float(np.mean(errs[t:end] > q_t))          # 块内未覆盖率
        a_t = min(max(a_t + gamma * (alpha - unc), 1e-4), 0.5)
        # 块末把本块分数并入校准池（归并排序保持有序）
        hist_sorted = np.sort(np.concatenate([hist_sorted, errs[t:end]]))
        t = end
    return float(np.mean(np.concatenate(covs))), float(np.mean(np.concatenate(wids)))


def conformal_ncp_smooth(mu_v, sig_v, y_v, mu_t, sig_t, y_t, rho=0.999,
                         win=5, alpha_levels=(0.80, 0.90, 0.95)):
    """NCPNet 对照（精神迁移）：分数沿时序邻居滑窗平滑（拓扑邻居需图结构，缓存未存，
    论文中如实标注 adapted）。平滑后按几何权重取加权分位，区间仍按 σ 缩放。"""
    s_v = np.abs(y_v - mu_v) / sig_v
    k = np.ones(win) / win
    s_v_s = np.convolve(s_v, k, mode="same")
    w = drift_weights(len(s_v), rho)
    out = {}
    for alpha in alpha_levels:
        q = weighted_quantile(s_v_s, w, alpha)
        half = q * sig_t
        cov = float(np.mean(np.abs(y_t - mu_t) <= half))
        width = float(np.mean(2 * half))
        q90 = np.quantile(y_t, 0.90)
        tail = float(np.mean(np.abs(y_t[y_t >= q90] - mu_t[y_t >= q90]) <= half[y_t >= q90]))
        out[alpha] = (cov, width, tail)
    return out


def conformal_grouped(mu_v, sig_v, y_v, mu_t, sig_t, y_t, ngroups=5,
                      alpha_levels=(0.80, 0.90, 0.95)):
    """RR-GNN 对照（精神迁移）：按 σ 五分位 Mondrian 分组校准——自适应重加权的
    分组化实现。每组各自取经验分位，测试点按其 σ 落组。"""
    qs = np.quantile(sig_v, np.linspace(0, 1, ngroups + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    out = {}
    for alpha in alpha_levels:
        a = 1 - alpha
        half = np.empty_like(sig_t)
        for g in range(ngroups):
            m_v = (sig_v > qs[g]) & (sig_v <= qs[g + 1])
            m_t = (sig_t > qs[g]) & (sig_t <= qs[g + 1])
            if m_v.sum() < 10 or m_t.sum() == 0:
                if m_t.sum():
                    half[m_t] = np.inf if m_v.sum() == 0 else half[m_t]
                continue
            q = np.quantile(np.abs(y_v[m_v] - mu_v[m_v]) / sig_v[m_v], min(1 - a, 0.9995))
            half[m_t] = q * sig_t[m_t]
        cov = float(np.mean(np.abs(y_t - mu_t) <= half))
        width = float(np.mean(2 * half))
        q90 = np.quantile(y_t, 0.90)
        tail = float(np.mean(np.abs(y_t[y_t >= q90] - mu_t[y_t >= q90]) <= half[y_t >= q90]))
        out[alpha] = (cov, width, tail)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="Exp/cache/*_SALT_Prob_full_s*.npz")
    ap.add_argument("--rho", type=float, default=0.999, help="漂移衰减系数（1.0=关闭加权）")
    ap.add_argument("--no-aci", action="store_true", help="跳过在线 ACI（大数据集太慢时用）")
    ap.add_argument("--out", default="Exp/calib/conformal_results.csv", help="结果 CSV 路径")
    A = ap.parse_args()
    files = sorted(glob.glob(A.pattern))
    print(f"共 {len(files)} 个缓存。分数×加权 消融（rho={A.rho}）\n")
    print(f"{'run':34s}{'分数':10s}{'cov90':>7s}{'wdt90':>9s}{'tail90':>8s}{'cov95':>7s}{'tail95':>8s}")
    rows = []
    for f in files:
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        for kind in ("naive", "homo", "sigma", "sigma_evt"):
            r = conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], kind, rho=A.rho)
            c90, w90, t90 = r[0.90]; c95, w95, t95 = r[0.95]
            print(f"{name:34s}{kind:10s}{c90:7.3f}{w90:9.3g}{t90:8.3f}{c95:7.3f}{t95:8.3f}", flush=True)
            rows.append((name, kind, c90, w90, t90, c95, t95))
        r = conformal_ncp_smooth(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], rho=A.rho)
        c90, w90, t90 = r[0.90]; c95, w95, t95 = r[0.95]
        print(f"{name:34s}{'ncp_smooth':10s}{c90:7.3f}{w90:9.3g}{t90:8.3f}{c95:7.3f}{t95:8.3f}", flush=True)
        rows.append((name, "ncp_smooth", c90, w90, t90, c95, t95))
        r = conformal_grouped(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"])
        c90, w90, t90 = r[0.90]; c95, w95, t95 = r[0.95]
        print(f"{name:34s}{'grouped':10s}{c90:7.3f}{w90:9.3g}{t90:8.3f}{c95:7.3f}{t95:8.3f}", flush=True)
        rows.append((name, "grouped", c90, w90, t90, c95, t95))
        if not A.no_aci:
            for kind in ("naive", "sigma"):
                cov, wid = aci_online(z["mu_t"], z["sig_t"], z["y_t"], kind, alpha=0.10)
                print(f"{name:34s}aci_{kind:6s}{cov:7.3f}{wid:9.3g}{'':8s}{'':7s}{'':8s}", flush=True)
                rows.append((name, f"aci_{kind}", cov, wid, "", "", ""))
    import csv
    with open(A.out, "w", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(["run", "score", "cov90", "width90", "tail90", "cov95", "tail95"])
        wr.writerows(rows)
    print("\n→ Exp/calib/conformal_results.csv")


if __name__ == "__main__":
    main()
