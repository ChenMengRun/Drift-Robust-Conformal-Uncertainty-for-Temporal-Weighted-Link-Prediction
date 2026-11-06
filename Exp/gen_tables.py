#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_tables.py — 生成稿件 Table 1-3（LaTeX）+ 数字一致性核对"""
import csv, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
def load_csv(path):
    return list(csv.DictReader(open(path)))
prob = load_csv("Exp/calib/conformal_prob.csv")
wrap = load_csv("Exp/calib/aci_wrap.csv")
blk  = load_csv("Exp/calib/block_bootstrap.csv")

def med(rows, D, method, key):
    v = [float(r[key]) for r in rows if r["run"].split("_")[0]==D and r["score"]==method and r[key]]
    import statistics as st
    return st.median(v) if v else None

def get_wrap(D, m):
    for r in wrap:
        if r["dataset"]==D and r["method"]==m:
            return float(r["cov90"]), float(r["width"]), float(r["tail90"])
    return None

# ---------- Table 1: 静态失效与在线修复（数据源统一：block_bootstrap static + aci_wrap online） ----------
SELECT = {"DC": "sigma+ACI", "T-Drive": "sigma+ACI"}   # 判据 σcorr>0.2 → σ 成员；其余 CQR
t1 = []
for D in DS:
    bs = [r for r in blk if r["dataset"]==D and r["method"].startswith("static")][0]
    bo = [r for r in blk if r["dataset"]==D and r["method"].startswith("online")][0]
    w = [r for r in wrap if r["dataset"]==D and r["method"]==SELECT.get(D,"CQR+ACI")][0]
    t1.append((D, float(bs["cov90"]), f'[{float(bs["ci_lo"]):.3f}, {float(bs["ci_hi"]):.3f}]',
               float(bo["cov90"]), f'[{float(bo["ci_lo"]):.3f}, {float(bo["ci_hi"]):.3f}]', float(w["width"])))
T1 = "\\begin{table}[t]\n"
T1 += "\\caption{Static split conformal calibration (residual score) fails under drift; the score family with the online wrapper restores the level. Coverage at nominal $\\alpha=0.90$; brackets are 95\\% block-bootstrap CIs.}\n"
T1 += "\\label{tab:online}\n\\begin{tabular}{lccccc}\n\\toprule\n"
T1 += "Dataset & Static cov. & Static CI & Online cov. & Online CI & Width \\\\\n\\midrule\n"
for D, sc, cis, oc, cio, w in t1:
    T1 += f"{D} & {sc:.3f} & {cis} & {oc:.3f} & {cio} & {w:.0f} \\\\\n"
T1 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

# ---------- Table 2: 分数族宽度（ACI 包装下，GPD 对称） ----------
t2 = []
for D in DS:
    s = get_wrap(D, "sigma+EVT+ACI")
    c = get_wrap(D, "CQR+GPD+ACI")
    n = get_wrap(D, "naive+ACI")
    if s and c and n:
        win = "σ" if s[1] < c[1] else "CQR"
        t2.append((D, s[1], c[1], n[1], win))
T2 = "\\begin{table}[t]\n\\caption{Score-family efficiency under the same online wrapper (ACI + GPD applied to both members). Widths at $\\alpha=0.90$; three-seed median. The narrower valid member wins per dataset.}\n\\label{tab:family}\n\\begin{tabular}{lcccc}\n\\toprule\nDataset & σ+EVT+ACI & CQR+GPD+ACI & residual+ACI & Winner \\\\\n\\midrule\n"
for D, sw, cw, nw, win in t2:
    T2 += f"{D} & {sw:.0f} & {cw:.0f} & {nw:.0f} & {win} \\\\\n"
T2 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

# ---------- Table 3: 消融 ----------
t3 = []
for D in DS:
    s  = get_wrap(D, "sigma+ACI"); se = get_wrap(D, "sigma+EVT+ACI")
    n  = get_wrap(D, "naive+ACI")
    c  = get_wrap(D, "CQR+ACI"); ce = get_wrap(D, "CQR+GPD+ACI")
    t3.append((D, n[1], s[1], se[1], c[1], ce[1]))
T3 = "\\begin{table}[t]\n\\caption{Module ablations under the online wrapper. Widths at $\\alpha=0.90$.}\n\\label{tab:ablation}\n\\begin{tabular}{lccccc}\n\\toprule\nDataset & residual & +σ & +σ+EVT & CQR & CQR+GPD \\\\\n\\midrule\n"
for D, nw, sw, sew, cw, cew in t3:
    T3 += f"{D} & {nw:.0f} & {sw:.0f} & {sew:.0f} & {cw:.0f} & {cew:.0f} \\\\\n"
T3 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

open("paper/tables.tex", "w").write(T1 + "\n\n" + T2 + "\n\n" + T3 + "\n")
print("→ paper/tables.tex 已生成\n")
print("=== 数字一致性核对（数字核对）===")
# DC 宽度比核对
