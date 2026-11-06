#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bootstrap_ci.py — per-edge bootstrap 置信区间（n=3 种子欠功效时的主检验替代）

对每 (数据集, 方法) 用 test 边重采样（B=1000）计算 90% 覆盖率的 95% BCa-ish 百分位 CI。
输出 Exp/calib/bootstrap_coverage.csv：run, score, cov90, ci_lo, ci_hi, 含名义区间[0.866,0.933]。
"""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'SALT')); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
from conformal_twlp import conformal

B = 1000
rows_out = []
for f in sorted(glob.glob("Exp/cache/*_SALT_Prob_full_s0.npz")):
    D_run = os.path.basename(f)[:-4]
    z = np.load(f)
    for kind in ("naive", "sigma", "sigma_evt", "grouped"):
        # 复用 conformal() 的静态分位逻辑：先拿区间，再对覆盖做 bootstrap
        from conformal_twlp import weighted_quantile, drift_weights
        s_v = np.abs(z["y_v"] - z["mu_v"]) / (z["sig_v"] if kind in ("sigma", "sigma_evt") else 1.0)
        w = drift_weights(len(s_v), 0.999)
        if kind == "sigma_evt":
            q_emp = weighted_quantile(s_v, w, 0.90)
            u = weighted_quantile(s_v, w, 0.8)
            exc = s_v[s_v > u] - u
            if len(exc) >= 10:
                from scipy.stats import genpareto; xi, _, beta = genpareto.fit(exc, floc=0)
                w_u = float(w[s_v > u].sum() / w.sum())
                need = (0.10 - (1 - w_u)) / w_u
                q = max(u + genpareto.ppf(need, xi, scale=beta), q_emp) if 0 < need < 1 else q_emp
            else:
                q = q_emp
        elif kind == "grouped":
            continue  # grouped 无闭式逐边分数，跳过 bootstrap（主表已报）
        else:
            q = weighted_quantile(s_v, w, 0.90)
        scale_t = z["sig_t"] if kind in ("sigma", "sigma_evt") else 1.0
        hit = (np.abs(z["y_t"] - z["mu_t"]) <= q * scale_t).astype(float)
        rng = np.random.default_rng(0)
        bs = [rng.choice(hit, len(hit), replace=True).mean() for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows_out.append((D_run, kind, hit.mean(), lo, hi))
        print(f"[BOOT] {D_run} {kind}: cov={hit.mean():.3f} CI[{lo:.3f},{hi:.3f}]", flush=True)

with open("Exp/calib/bootstrap_coverage.csv", "w") as fo:
    fo.write("run,score,cov90,ci_lo,ci_hi\n")
    for r in rows_out:
        fo.write(",".join(str(x) for x in r) + "\n")
print(f"完成 → Exp/calib/bootstrap_coverage.csv（{len(rows_out)} 行）")
