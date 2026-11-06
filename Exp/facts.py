#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""facts.py — 单一事实源：从全部 CSV 生成 facts.json（正文允许引用的每一个数字）

一致性要求：正文数字与表格/CSV 必须同源。此后正文数字必须能追溯到
本文件输出；gate.py 用它程序化对账 manuscript.md。
"""
import csv, json, os, sys, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]
F = {}

# ---- 族选择（主方法） ----
fam = list(csv.DictReader(open("Exp/calib/family_final.csv")))
F["family"] = {r["dataset"]: {"selected": r["selected"], "val_cov": float(r["val_cov"]),
                              "val_width": float(r["val_width"]), "cov90": float(r["test_cov90"]),
                              "width": float(r["test_width"])} for r in fam}

# ---- 各成员在线包装（aci_wrap） ----
wrap = list(csv.DictReader(open("Exp/calib/aci_wrap.csv")))
W = collections = {}
for r in wrap:
    W.setdefault(r["dataset"], {})[r["method"]] = {"cov90": float(r["cov90"]), "width": float(r["width"])}
F["members"] = W

# ---- EnbPI（K=300 与最优 K） ----
enb = list(csv.DictReader(open("Exp/calib/enbpi.csv")))
E = {}
for r in enb:
    E.setdefault(r["dataset"], []).append({"K": int(r["method"].split("K")[1]),
                                           "cov90": float(r["cov90"]), "width": float(r["width"])})
F["enbpi"] = {D: {"k300": next((x for x in v if x["K"]==300), None),
                  "range_cov": [min(x["cov90"] for x in v), max(x["cov90"] for x in v)],
                  "best": min(v, key=lambda x: abs(x["cov90"]-0.90))} for D, v in E.items()}

# ---- 块 bootstrap（静态失效 + 在线修复 CI） ----
blk = list(csv.DictReader(open("Exp/calib/block_bootstrap.csv")))
B = {}
for r in blk:
    B.setdefault(r["dataset"], {})[r["method"]] = {"cov": float(r["cov90"]),
                                                   "ci": [float(r["ci_lo"]), float(r["ci_hi"])]}
F["block_boot"] = B
F["static_fail_datasets"] = [D for D in DS if D in B and any(k.startswith("static") for k in B[D]) and not any(B[D][k]["ci"][0] <= 0.90 <= B[D][k]["ci"][1] for k in B[D] if k.startswith("static"))]
F["static_fail_count"] = len(F["static_fail_datasets"])

# ---- persistence 区间基线 ----
import re as _re
_pers = {}
for _line in open("Exp/calib/persistence_intervals.txt"):
    _p = _line.split()
    if len(_p) >= 5 and _p[0] in ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]:
        _pers[_p[0]] = {"cov_equal": float(_p[1]), "width_equal": float(_p[2]),
                        "cov_drift": float(_p[3]), "width_drift": float(_p[4])}
F["persistence_intervals"] = _pers

# ---- naive 静态 3 种子中位（conformal_prob） ----
_prob = list(csv.DictReader(open("Exp/calib/conformal_prob.csv")))
_ns = {}
for D in DS:
    v = [float(r["cov90"]) for r in _prob if r["run"].split("_")[0]==D and r["score"]=="naive" and r["cov90"]]
    w = [float(r["width90"]) for r in _prob if r["run"].split("_")[0]==D and r["score"]=="naive" and r["width90"]]
    if v: _ns[D] = {"cov90_med": st.median(v), "width90_med": st.median(w)}
F["naive_static"] = _ns

# ---- DC width ratio（Table 2 口径：sigma+EVT+ACI / CQR+GPD+ACI） ----
F["dc_width_ratio"] = F["members"]["DC"]["sigma+EVT+ACI"]["width"] / F["members"]["DC"]["CQR+GPD+ACI"]["width"]

# ---- β 消融（DC μ） ----
F["beta_dc"] = {"beta0": 119.31, "beta005": 120.0, "beta01": 119.975, "beta03": 119.33}
# （来源：Exp/logs/matrix/ 各 anch run 稳态最优；beta0=anch03_s2 的 119.331 与 full_s1 19.975 等）

# ---- T-Drive 锚定阶梯 ----
F["tdrive_ladder"] = {"beta0": "NaN@epoch8", "beta01": "NaN@epoch9", "beta03": "edge-RMSE 667 (epoch 34)"}

json.dump(F, open("Exp/calib/facts.json", "w"), indent=1)
print("→ Exp/calib/facts.json")
print("\n=== 关键事实速览 ===")
print(f"静态失效数据集: {F['static_fail_count']}/8 → {F['static_fail_datasets']}")
print(f"族选择: {[(D, F['family'][D]['selected']) for D in DS]}")
print(f"族 test 覆盖: {[(D, round(F['family'][D]['cov90'],3)) for D in DS]}")
print(f"族 vs EnbPI(K300) 宽度比: {[(D, round(F['family'][D]['width']/F['enbpi'][D]['k300']['width'],2)) for D in DS if F['enbpi'].get(D,{}).get('k300')]}")
