#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig1_camera_gen.py — camera-ready Figure 1 草稿生成（口径统一版，2026-11-06）

与已投稿 figure1_camera_ready.png 的差异（仅两处）：
  1. EnbPI 绿叉 = enbpi_faithful.csv（faithful 口径，与正文/README conventions 一致）
  2. 圈选成员 = aci_wrap_seeds.csv 种子中位最窄（与 tex 图注 + Table II Selected 列一致；
     唯一可见变化：LyonSchool 从 Smoothed 改圈 CQR）
其余（家庭四点/坐标范围/刻度/配色/版式）与现役版一致。
输出 figures/figure1_generated.png（不覆盖现役文件）
"""
import csv
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'path'   # 文字转矢量路径：任何环境渲染一致，可直接编辑形状
import matplotlib.pyplot as plt

wrap = list(csv.DictReader(open("Exp/calib/aci_wrap.csv")))
seeds = list(csv.DictReader(open("Exp/calib/aci_wrap_seeds.csv")))
enb = list(csv.DictReader(open("Exp/calib/enbpi_faithful.csv")))
DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]
METHODS = [("naive+ACI", "#8c8c8c", "o", "Residual", "Residual+ACI"),
           ("ncp_smooth+ACI", "#1f77b4", "s", "Smoothed", "Smoothed+ACI"),
           ("sigma+ACI", "#d62728", "o", "$\\sigma$", "$\\sigma$+ACI"),
           ("CQR+ACI", "#9467bd", "D", "CQR", "CQR+ACI")]
# 圈选 = 种子中位最窄（Table II Selected 列）
best = {}
for D in DS:
    rows = [(float(r["width_median"]), r["method"]) for r in seeds
            if r["dataset"] == D and r["method"] in [m[0] for m in METHODS]]
    best[D] = min(rows)[1]
# 图内 "selected: X" 标注偏移（points），避开 y 刻度与其他标记
OFF = {"DC": (-20, 6, "right", "center"), "HMob": (-20, -16, "right", "center"),
       "INVIS13": (17, 2, "left", "center"), "INVIS15": (0, 15, "center", "bottom"),
       "IoT": (0, -19, "center", "top"), "LyonSchool": (0, -19, "center", "top"),
       "T-Drive": (0, -19, "center", "top"), "Thiers13": (-6, 15, "center", "bottom")}
# 绘制层级：naive 最上（4 个面板中 naive 与 σ 宽度几乎重合，保证被选/灰点可读；
# IoT 两者仅差 0.001，属数据巧合，完全重叠无法避免，与现役版一致）
XLIM = {"DC": (144.9, 474.7), "HMob": (44.1, 122.1), "INVIS13": (28.2, 87.3),
        "INVIS15": (24.6, 67.4), "IoT": (22.4, 114.2), "LyonSchool": (9.5, 28.0),
        "T-Drive": (1224.7, 3053.5), "Thiers13": (33.9, 137.8)}
XTICKS = {"DC": [200, 250, 300, 400], "HMob": [50, 100], "INVIS13": [30, 50, 70],
          "INVIS15": [25, 40, 50], "IoT": [50, 100], "LyonSchool": [10, 20],
          "T-Drive": [2000, 2500, 3000], "Thiers13": [50, 100]}

plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
handles, labels = [], []
for ax, D in zip(axes.flat, DS):
    covs = []
    for m, c, mk, short, leg in METHODS:
        r = next((r for r in wrap if r["dataset"] == D and r["method"] == m), None)
        if not r:
            continue
        x, y = float(r["width"]), float(r["cov90"])
        covs.append(y)
        h = ax.scatter(x, y, c=c, marker=mk, s=110, zorder=4 if m == "naive+ACI" else 3,
                       edgecolors='k', linewidths=0.6)
        if D == "DC":
            handles.append(h); labels.append(leg)
        if m == best[D]:
            ax.scatter(x, y, s=400, facecolors='none', edgecolors=c, linewidths=2.6, zorder=5)
            dx, dy, ha, va = OFF[D]
            ax.annotate("selected: " + short, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=10, fontstyle='italic', ha=ha, va=va,
                        color=c, zorder=6,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                                  boxstyle='round,pad=1.2'))
    e = next((r for r in enb if r["dataset"] == D), None)
    if e:
        x, y = float(e["width"]), float(e["cov90"])
        covs.append(y)
        h = ax.scatter(x, y, c="#2ca02c", marker="X", s=140, zorder=3.5, edgecolors='k', linewidths=0.7)
        if D == "DC":
            handles.append(h); labels.append("EnbPI ($K{=}300$)")
    ax.axhline(0.90, color='k', ls='--', lw=0.9, alpha=0.7)
    ymin, ymax = min(covs) - 0.012, max(covs) + 0.010
    ax.set_ylim(min(ymin, 0.896), max(ymax, 0.904))
    ax.set_title(D, fontsize=13)
    ax.set_xscale('log')
    ax.set_xlim(*XLIM[D])
    ax.set_xticks(XTICKS[D])
    ax.set_xticklabels([f"{x:g}" for x in XTICKS[D]])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    if ax in axes[1]:
        ax.set_xlabel("mean interval width (ACI)", fontsize=10.5)

# 图例放进 DC 面板（含 selected 空心圈样式）
leg_handles = handles + [plt.Line2D([], [], color='k', marker='o', markerfacecolor='none',
                                    linestyle='none', markersize=9, markeredgewidth=1.6)]
leg_labels = labels + ["selected"]
axes[0, 0].legend(leg_handles, leg_labels, loc="lower left", fontsize=8.6, frameon=True,
                  framealpha=1.0, edgecolor='0.85', borderpad=0.7, labelspacing=0.5)
fig.suptitle("Coverage–efficiency under the online wrapper "
             "(encircled: narrowest member under seed medians)", fontsize=15, fontweight='bold', y=0.985)
fig.text(0.012, 0.5, "Empirical coverage (nominal $1-\\alpha=0.90$)", va='center',
         rotation=90, fontsize=12)
fig.tight_layout(rect=[0.03, 0.0, 1, 0.955])
fig.savefig("figures/figure1_generated.png", dpi=220)
fig.savefig("figures/figure1_generated.svg")
fig.savefig("figures/figure1_generated.pdf")
print("OK → figures/figure1_generated.{png,svg,pdf}")
