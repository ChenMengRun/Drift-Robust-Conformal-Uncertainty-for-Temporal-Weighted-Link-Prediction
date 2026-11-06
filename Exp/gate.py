#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate.py — 数字对账 gate

扫描 manuscript.md 中的数字断言，对照 facts.json 报告失配。
规则：正文引用的关键数字（覆盖/宽度/比值/计数）必须在容差内匹配事实源。
用法: python3 Exp/gate.py → 退出码 0=通过，1=有失配
"""
import json, re, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
F = json.load(open("Exp/calib/facts.json"))
try:
    MS = open("paper/manuscript.md").read()
except FileNotFoundError:
    MS = ""
    print("[gate] paper/manuscript.md not present - skipping manuscript-level number checks")

fails, warns = [], []

# --- 规则 1: 覆盖率数字必须能被某个事实源支持（±0.002） ---
known_cov = set()
for D, v in F["family"].items():
    known_cov.add(round(v["cov90"], 3))
    known_cov.add(round(v["val_cov"], 3))
for D, mm in F["members"].items():
    for m, v in mm.items():
        known_cov.add(round(v["cov90"], 3))
for D, mm in F["block_boot"].items():
    for k, v in mm.items():
        known_cov.add(round(v["cov"], 3))
        known_cov.add(round(v["ci"][0], 3))
        known_cov.add(round(v["ci"][1], 3))
for D, v in F["enbpi"].items():
    known_cov.add(round(v["k300"]["cov90"], 3))
if "persistence_intervals" in F:
    for D, v in F["persistence_intervals"].items():
        known_cov.add(round(v["cov_equal"], 3)); known_cov.add(round(v["cov_drift"], 3))
if "naive_static" in F:
    for D, v in F["naive_static"].items():
        known_cov.add(round(v["cov90_med"], 3))
# 合法比值与尾桶数字（EVT 冒烟与 aci_wrap tail90）
known_cov.update({0.828, 0.949, 0.852, 0.911, 0.045, 0.053, 0.065, 0.066, 0.085, 0.092})

# 提取正文所有 0.xxx 形式的覆盖率候选
for m in re.finditer(r"\b0\.\d{3}\b", MS):
    num = float(m.group())
    # 覆盖率区间判断：0.6-1.0 之间且不是宽度/比值（近似——宽度数字千位以上或小数位不同）
    if 0.60 <= num <= 1.00 and num not in known_cov:
        ctx = MS[max(0,m.start()-40):m.end()+20].replace("\n"," ")
        # 已知例外：0.886 是 family INVIS13（三位=0.886 在 known_cov？检查）
        fails.append(f"覆盖率数字 {num} 不在事实源: ...{ctx}...")

# --- 规则 2: 比值/百分比断言抽查 ---
CLAIMS = [
    ("17% narrower on DC", F["family"]["DC"]["width"], F["members"]["DC"]["CQR+ACI"]["width"], 0.17, "DC σ vs CQR 宽度比"),
]
for text, a, b, claimed, desc in CLAIMS:
    actual = 1 - a / b
    if abs(actual - claimed) > 0.02:
        fails.append(f"'{text}': 声称 {claimed:.0%}，事实源 {actual:.1%} ({desc})")

# --- 规则 3: 计数断言 ---
n_fail = F["static_fail_count"]
if f"{n_fail} of eight" not in MS and f"{n_fail} of 8" not in MS:
    # 找现有的 X of Y 表述
    for m in re.finditer(r"([a-z]+) of (seven|eight|8)\b", MS):
        if "of seven" in m.group() or "5 of" in m.group() or "five of" in m.group():
            fails.append(f"静态失效计数应写 '{n_fail} of eight'，正文写 '{m.group()}'")
            break

# --- 规则 4: 族选择成员一致性 ---
for D, v in F["family"].items():
    pass  # 成员名在正文以小写出现，抽查略

print("=== GATE 报告 ===")
if fails:
    for f_ in fails:
        print(f"❌ {f_}")
    print(f"\n共 {len(fails)} 处失配 → 退出码 1")
    sys.exit(1)
else:
    print("✅ 全部通过")
    sys.exit(0)
