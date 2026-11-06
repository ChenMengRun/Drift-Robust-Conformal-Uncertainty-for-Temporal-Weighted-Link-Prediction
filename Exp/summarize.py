#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize.py —— 汇总实验矩阵：复现 gap 表 + 概率化对比表 + 校准增益表
输入: Exp/pt/<D>/<config>/ 文件名指标、Exp/calib/*/calibration_summary.txt、Exp/paper_results.json
输出: Exp/matrix_summary.md（总表）
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT = os.path.join(ROOT, 'Exp', 'pt')
CALIB = os.path.join(ROOT, 'Exp', 'calib')
PAPER = os.path.join(ROOT, 'Exp', 'paper_results.json')

def ckpt_metrics(d):
    """目录内按文件名 RMSE 最小的 checkpoint → (rmse, mae, n_ckpts)"""
    best = None
    fs = glob.glob(os.path.join(PT, d, '*.pth'))
    for f in fs:
        m = re.match(r'(\d+)_([\d.]+)_([\d.]+)\.pth', os.path.basename(f))
        if not m:
            continue
        rmse, mae = float(m.group(2)), float(m.group(3))
        if best is None or rmse < best[0]:
            best = (rmse, mae)
    return best, len(fs)

def calib_get(pattern, path):
    txt = open(path, encoding='utf-8').read()
    m = re.search(pattern, txt)
    return m.group(1) if m else '-'

def main():
    lines = ['# 实验矩阵汇总', '', '## 1. 复现对齐（scratch vs 论文数字）', '',
             '| 数据集 | 论文RMSE | repro-RMSE | scratch-RMSE(3种子best/mean) | scratch-MAE | ΔRMSE% |',
             '|---|---|---|---|---|---|']
    paper = {}
    if os.path.isfile(PAPER):
        paper = json.load(open(PAPER, encoding='utf-8'))
    for d in sorted(os.listdir(PT)):
        if not os.path.isdir(os.path.join(PT, d)):
            continue
        p_rmse = paper.get(d, {}).get('rmse') or '-'
        repro, _ = ckpt_metrics(os.path.join(d, 'SALT'))
        rows = []
        for s in (0, 1, 2):
            r, _ = ckpt_metrics(os.path.join(d, f'SALT_scratch_s{s}'))
            if r:
                rows.append(r)
        repro_s = f'{repro[0]:.3f}' if repro else '-'
        if rows:
            best = min(r[0] for r in rows); mean = sum(r[0] for r in rows) / len(rows)
            mae_b = min(r[1] for r in rows)
            sc = f'{best:.3f}/{mean:.3f}'
            gap = f'{(best - float(p_rmse)) / float(p_rmse) * 100:+.1f}%' if p_rmse != '-' else '-'
        else:
            sc, mae_b, gap = '-', '-', '-'
        lines.append(f'| {d} | {p_rmse} | {repro_s} | {sc} | {mae_b if isinstance(mae_b, str) else f"{mae_b:.3f}"} | {gap} |')

    lines += ['', '## 2. 概率化 vs 官方（点预测口径 edge-RMSE）', '',
              '| 数据集 | scratch | SALT_Prob(3s best) | SALT_INN(3s best) |', '|---|---|---|---|']
    for d in sorted(os.listdir(PT)):
        if not os.path.isdir(os.path.join(PT, d)):
            continue
        def best3(name):
            vals = []
            for s in (0, 1, 2):
                r, _ = ckpt_metrics(os.path.join(d, f'{name}_full_s{s}'))
                if r:
                    vals.append(r[0])
            return f'{min(vals):.3f}' if vals else '-'
        sc, _ = ckpt_metrics(os.path.join(d, 'SALT_scratch_s0'))
        lines.append(f"| {d} | {f'{sc[0]:.3f}' if sc else '-'} | {best3('SALT_Prob')} | {best3('SALT_INN')} |")

    lines += ['', '## 3. 校准增益（异方差 σ vs 同方差对照）', '',
              '| run | edge-RMSE | CRPS | NLL | ECE | ΔCRPS | ΔNLL |', '|---|---|---|---|---|---|---|']
    for p in sorted(glob.glob(os.path.join(CALIB, '*', 'calibration_summary.txt'))):
        run = os.path.basename(os.path.dirname(p))
        rm = calib_get(r'edge-RMSE=([\d.]+)', p)
        crps = calib_get(r'σ\] CRPS=([\d.]+)', p)
        nll = calib_get(r'σ\] CRPS=[\d.]+\s+NLL=([\d.]+)', p)
        ece = calib_get(r'ECE=([\d.]+)', p)
        dc = calib_get(r'ΔCRPS=([+-][\d.]+)', p)
        dn = calib_get(r'ΔNLL=([+-][\d.]+)', p)
        lines.append(f'| {run} | {rm} | {crps} | {nll} | {ece} | {dc} | {dn} |')

    out = os.path.join(ROOT, 'Exp', 'matrix_summary.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n→ {out}')

if __name__ == '__main__':
    main()
