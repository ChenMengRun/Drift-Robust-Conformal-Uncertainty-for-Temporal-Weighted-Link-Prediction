#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_beta_cqr.py — CQR 基线配置（8 数据集 × s0）+ β=0.05 补扫配置（DC/IoT × 3种子）"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n = 0
for D in ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]:
    src = f"SALT/config/{D}/SALT_Prob_full_s0.yaml"
    dst = f"SALT/config/{D}/SALT_CQR_s0.yaml"
    if os.path.isfile(dst): continue
    txt = open(src).read().replace("model: model.salt_prob.SALT_Prob", "model: model.salt_cqr.SALT_CQR")
    open(dst, "w").write(txt); n += 1; print(f"[GEN] {dst}")
for D in ["DC", "IoT"]:
    for s in "012":
        src = f"SALT/config/{D}/SALT_Prob_full_s{s}.yaml"
        dst = f"SALT/config/{D}/SALT_Prob_anch005_s{s}.yaml"
        if os.path.isfile(dst): continue
        txt = open(src).read().replace("beta: 0.0", "beta: 0.05")
        open(dst, "w").write(txt); n += 1; print(f"[GEN] {dst}")
print(f"完成 {n} 个")
