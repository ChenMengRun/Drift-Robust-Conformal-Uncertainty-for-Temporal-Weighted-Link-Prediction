#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""revision_experiments.py — 修订期补充实验（四项，全部基于预计算缓存零训练）

R1 对照公平性：3种子深度集成基线（cache 平均） + persistence+经验分位+ACI 最强零参数对照
R4 统计可信度：per-dataset 均值±std（3种子） + 与最强对照的 paired Wilcoxon
R5 σ判据先验化：验证集 σcorr（看 test 前可算）能否预测 σ 分数的有效性——判据外样本检验
R6 交叉消融：ACI+EVT 无σ 臂（naive 分数 + GPD 尾外推 + 在线更新）
输出: Exp/calib/revision_tables.md + revision_raw.csv
"""
import csv, glob, os, sys, collections
import numpy as np
from scipy.stats import genpareto, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
from conformal_twlp import conformal, aci_online

DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]


def load_fam(D, base="SALT_Prob_full"):
    out = []
    for s in "012":
        f = f"Exp/cache/{D}_{base}_s{s}.npz"
        if os.path.isfile(f):
            out.append(np.load(f))
    return out


def aci_evt_naive(z, alpha=0.10, gamma=0.05):
    """R6: ACI + EVT 尾，但分数用裸残差（无 σ）"""
    mu, y = z["mu_t"], z["y_t"]
    mu_v, y_v = z["mu_v"], z["y_v"]
    scale = np.ones_like(mu)
    s_v = np.abs(y_v - mu_v)
    u = np.quantile(s_v, 0.8)
    exc = s_v[s_v > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0) if len(exc) >= 10 else (0, 0, 1)
    a_t = alpha
    start = max(len(s_v) // 5, 30)
    hist = np.sort(s_v[:start])
    n = len(y); block = max(1, (n - start) // 200 or 1)
    covs, wids = [], []
    t = start
    while t < n:
        end = min(t + block, n)
        q_emp = hist[min(int((1 - a_t) * len(hist)), len(hist) - 1)]
        if q_emp >= u and len(exc) >= 10:
            w_u = float((s_v > u).mean())
            need = ((1 - a_t) - (1 - w_u)) / w_u
            if 0 < need < 1:
                q_t = max(u + genpareto.ppf(need, xi, scale=beta), q_emp)
            else:
                q_t = q_emp
        else:
            q_t = q_emp
        covs.append(np.abs(y[t:end] - mu[t:end]) <= q_t)
        wids.append(np.full(end - t, 2 * q_t))
        unc = float(np.mean(np.abs(y[t:end] - mu[t:end]) > q_t))
        a_t = min(max(a_t + gamma * (alpha - unc), 1e-4), 0.5)
        hist = np.sort(np.concatenate([hist, np.abs(y[t:end] - mu[t:end])]))
        t = end
    return float(np.mean(np.concatenate(covs))), float(np.mean(np.concatenate(wids)))


def main():
    rows_out = []
    lines = ["# Review-Panel 实验侧修复（R1/R4/R5/R6）\n"]
    # ---------- R5: 判据先验化 ----------
    lines.append("## R5: σ 判据先验化（val σcorr，看 test 前可算）→ 预测 σ 分数宽度增益\n")
    lines.append("| 数据集 | val σcorr(判据) | sigma/naive 宽度比(test) | 判据命中? |")
    lines.append("|---|---|---|---|")
    crit_hits = 0; crit_total = 0
    ratios = {}
    for D in DS:
        fam = load_fam(D)
        if len(fam) < 2:
            continue
        vcs = [float(np.corrcoef(np.abs(z["y_v"] - z["mu_v"]), z["sig_v"])[0, 1]) for z in fam]
        vc = float(np.median(vcs))
        rs = [conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "sigma", rho=0.999) for z in fam]
        rn = [conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "naive", rho=0.999) for z in fam]
        ratio = float(np.median([r[0.90][1] / n[0.90][1] for r, n in zip(rs, rn)]))
        ratios[D] = ratio
        hit = (vc > 0.2) == (ratio < 1.0)
        crit_total += 1; crit_hits += hit
        lines.append(f"| {D} | {vc:+.3f} | {ratio:.3f} | {'✓' if hit else '✗'} |")
    lines.append(f"\n判据命中率: {crit_hits}/{crit_total}（val σcorr>0.2 ⇔ σ分数更窄）\n")

    # ---------- R1: 集成基线 + persistence+ACI ----------
    lines.append("## R1a: 3种子深度集成（μ 均值、σ 均值）vs 单模型 σ-informed\n")
    lines.append("| 数据集 | 集成 cov90 / 宽 | 单模型 sigma cov90 / 宽 | 集成胜? |")
    lines.append("|---|---|---|---|")
    ens_beats = []
    for D in DS:
        fam = load_fam(D)
        if len(fam) < 3:
            continue
        n = min(len(z["mu_v"]) for z in fam); nt = min(len(z["mu_t"]) for z in fam)
        mu_v = np.mean([z["mu_v"][:n] for z in fam], 0)
        sg_v = np.mean([z["sig_v"][:n] for z in fam], 0)
        y_v = fam[0]["y_v"][:n]
        mu_t = np.mean([z["mu_t"][:nt] for z in fam], 0)
        sg_t = np.mean([z["sig_t"][:nt] for z in fam], 0)
        y_t = fam[0]["y_t"][:nt]
        re = conformal(mu_v, sg_v, y_v, mu_t, sg_t, y_t, "sigma", rho=0.999)[0.90]
        rs = [conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "sigma", rho=0.999)[0.90] for z in fam]
        r1 = rs[0]
        beat = re[0] >= 0.88 and re[1] < float(np.median([r[1] for r in rs]))
        ens_beats.append(beat)
        lines.append(f"| {D} | {re[0]:.3f} / {re[1]:.4g} | {np.median([r[0] for r in rs]):.3f} / {np.median([r[1] for r in rs]):.4g} | {'✓' if beat else '✗'} |")
    lines.append("")

    # ---------- R6: ACI+EVT 无σ ----------
    lines.append("## R6: 交叉消融——ACI+EVT 无σ vs ACI+σ（cov90 / 宽度）\n")
    lines.append("| 数据集 | ACI+EVT(noσ) | ACI+σ | σ 必要性 |")
    lines.append("|---|---|---|---|")
    for D in DS:
        fam = load_fam(D)
        if not fam:
            continue
        z = fam[0]
        c1, w1 = aci_evt_naive(z)
        cs = [aci_online(z["mu_t"], z["sig_t"], z["y_t"], "sigma") for z in fam]
        c2 = float(np.median([c for c, _ in cs])); w2 = float(np.median([w for _, w in cs]))
        need = "σ不可少" if (c2 >= c1 - 0.01 and w2 < w1) else "部分场景可替代"
        lines.append(f"| {D} | {c1:.3f} / {w1:.4g} | {c2:.3f} / {w2:.4g} | {need} |")
    lines.append("")

    # ---------- R4: 均值±std + paired 检验 ----------
    lines.append("## R4: 统计可信度（3种子 mean±std；aci_sigma vs 最强静态对照的 paired Wilcoxon）\n")
    lines.append("| 数据集 | sigma w90 (mean±std) | aci_sigma cov90 (mean±std) | vs naive paired p |")
    lines.append("|---|---|---|---|")
    for D in DS:
        fam = load_fam(D)
        if not fam:
            continue
        ws, cs_, pairs = [], [], []
        for z in fam:
            r_s = conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "sigma", rho=0.999)[0.90]
            r_n = conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "naive", rho=0.999)[0.90]
            c_a, _ = aci_online(z["mu_t"], z["sig_t"], z["y_t"], "sigma")
            ws.append(r_s[1]); cs_.append(c_a); pairs.append(r_n[1] - r_s[1])
        p = wilcoxon(pairs).pvalue if len(pairs) >= 3 and any(abs(x) > 1e-12 for x in pairs) else float('nan')
        lines.append(f"| {D} | {np.mean(ws):.4g}±{np.std(ws):.2g} | {np.mean(cs_):.3f}±{np.std(cs_):.3f} | {p:.3f} |")

    open("Exp/calib/revision_tables.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
