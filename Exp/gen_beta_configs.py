#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_beta_configs.py — β 锚定消融配置生成器（幂等）

对 IoT（概率化有害的反例靶点）和 DC（概率化贡献最大的正例）生成
SALT_Prob_anch{01,03}_s{s}.yaml：唯一差异 = Loss.beta 0.0 → 0.1 / 0.3。
输出目录名与现有 run 天然隔离，ckpt 哨兵自动生效。
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATASETS = ["IoT", "DC"]
BETAS = {"anch01": 0.1, "anch03": 0.3}
SEEDS = [0, 1, 2]

n = 0
for D in DATASETS:
    for s in SEEDS:
        src = f"SALT/config/{D}/SALT_Prob_full_s{s}.yaml"
        if not os.path.isfile(src):
            print(f"[SKIP] 缺模板 {src}"); continue
        txt = open(src).read()
        for tag, beta in BETAS.items():
            dst = f"SALT/config/{D}/SALT_Prob_{tag}_s{s}.yaml"
            if os.path.isfile(dst):
                continue
            out = txt.replace("beta: 0.0", f"beta: {beta}")
            if out == txt and "beta:" not in out:
                print(f"[WARN] {src} 无 beta 字段"); continue
            open(dst, "w").write(out)
            n += 1
            print(f"[GEN] {dst} (beta={beta})")
print(f"完成，新生成 {n} 个配置")
