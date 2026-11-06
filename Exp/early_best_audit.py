#!/usr/bin/env python3
"""early_best_audit.py — 检验 min_ep=5 排除规则的软肋：
全局最优是否真的落在前 5 epoch？若是，它是瞬态尖峰还是被支撑的真实最优？
判据：(a) 相邻 epoch 连续性 (b) 跨种子可重复性。只读日志，不动任何产物。
"""
import os, re, glob
from collections import defaultdict

LOGDIR = "Exp/logs/matrix"
PAT = re.compile(r"Test #(\d+) RMSE [\d.]+ ([\d.]+) ")  # epoch, edge-RMSE(双列口径第二列)
MIN_EP = 5

series = {}  # tag -> {epoch: edge_rmse}
for f in sorted(glob.glob(f"{LOGDIR}/*.log")):
    tag = os.path.basename(f)[:-4]
    d = {}
    with open(f, errors="ignore") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                d[int(m.group(1))] = float(m.group(2))
    if d:
        series[tag] = d

print(f"解析到 {len(series)} 个含 Test 行的 run\n")

# 按 数据集/模型/种子 归组（tag: scratch_D_s# / prob_D_MODEL_s# / baseedge_D）
groups = defaultdict(list)
for tag, d in series.items():
    if tag.startswith("baseedge_"):
        groups[tag].append(tag); continue
    m = re.match(r"(?:prob_)?([A-Za-z0-9\-]+?)_(SALT_(?:scratch|Prob_full|INN_full))_(s\d)$", tag)
    if m:
        groups[f"{m.group(1)}/{m.group(2)}"].append(tag)

hits = []
for tag, d in sorted(series.items()):
    if max(d) < MIN_EP:  # 整个 run 不足 5 epoch，另说
        continue
    g_ep = min(d, key=lambda e: d[e])
    if g_ep < MIN_EP:
        steady = {e: v for e, v in d.items() if e >= MIN_EP}
        s_ep = min(steady, key=lambda e: steady[e])
        hits.append((tag, g_ep, d[g_ep], s_ep, steady[s_ep]))

print(f"=== 全局最优落在 epoch<{MIN_EP} 的 run：{len(hits)} 个 ===")
for tag, ge, gv, se, sv in hits:
    d = series[tag]
    ctx = " ".join(f"e{e}={d[e]:.4g}" for e in sorted(d) if abs(e-ge) <= 3)
    ratio = gv / sv if sv else float("nan")
    print(f"{tag:36s} 全局最优 e{ge}={gv:.4g} | 稳态最优 e{se}={sv:.4g} | 瞬态/稳态={ratio:.3f}")
    print(f"{'':36s} 上下文: {ctx}")

# 跨种子核验：命中的 (数据集,模型) 组，看其他种子同位置表现
print("\n=== 命中 run 的跨种子核验 ===")
for tag, ge, gv, se, sv in hits:
    base = None
    for g, members in groups.items():
        if tag in members:
            base = (g, members); break
    if not base:
        continue
    g, members = base
    others = [t for t in members if t != tag and t in series]
    line = []
    for t in others:
        d = series[t]
        oe = min(d, key=lambda e: d[e])
        line.append(f"{t}: 全局最优e{oe}={d[oe]:.4g}")
    print(f"{tag} (e{ge}={gv:.4g}) | 同组其他种子: {'; '.join(line) if line else '(无)'}")

# 反向对照：稳态最优 vs 全局最优的总体差异
import statistics
diffs = []
for tag, d in series.items():
    steady = {e: v for e, v in d.items() if e >= MIN_EP}
    if not steady or max(d) < MIN_EP:
        continue
    diffs.append((min(d.values()) / min(steady.values()) - 1) * 100)
if diffs:
    neg = sum(1 for x in diffs if x < -0.5)
    print(f"\n=== 总体影响 ===\n{len(diffs)} 个 run 中，全局口径比稳态口径低 >0.5% 的有 {neg} 个（即排除规则改变结果的 run）")
