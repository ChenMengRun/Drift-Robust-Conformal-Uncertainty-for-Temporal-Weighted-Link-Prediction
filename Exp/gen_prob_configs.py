#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_prob_configs.py —— 为 8 个数据集批量生成实验配置（幂等，已有则跳过）

生成三类：
  1. <D>/SALT_scratch_s{0,1,2}.yaml   官方 SALT 从零训练多种子（底=官方 SALT.yaml，
     仅加早停；官方 loss/超参不动——用于量化「scratch vs 文献数字」的 gap 及其方差）
  2. <D>/SALT_Prob_full_s{0,1,2}.yaml  P1 概率化全量（model=salt_prob，nll loss，σ共享TLSSM）
  3. <D>/SALT_INN_full_s{0,1,2}.yaml   P2/Route B 全量（model=salt_inn，WLE→INN out=2，β=3.0）

概率化配置以「各数据集官方 SALT.yaml」为底（保留 win_size/feat_proj_dims/WLE.in_channels
等数据集特定字段），只替换 model/Loss/WLE.model_class/out_channels，并统一加：
  num_epochs=300（覆盖 INVIS15 探针版的 50）、早停、Selection=edge_rmse（公平点预测尺）
种子不经 yaml（run_matrix.sh 用 SALT_SEED 环境变量注入 inn_train.py）。
用法：python Exp/gen_prob_configs.py [--dry-run]
"""
import argparse
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'SALT', 'config')
DATASETS = ['DC', 'HMob', 'INVIS13', 'INVIS15', 'IoT', 'LyonSchool', 'T-Drive', 'Thiers13']

PROB_OVERrides = dict(
    model='model.salt_prob.SALT_Prob',
    Loss={'name': 'nll', 'alpha': 1.0, 'theta': 0.1, 'beta': 0.0},
)
INN_OVERRIDES = dict(
    model='model.salt_inn.SALT_INN',
    Loss={'name': 'nll', 'alpha': 1.0, 'theta': 0.1, 'beta': 3.0},
    WLE={'model_class': 'model.modules.inn', 'out_channels': 2},
)
COMMON = dict(
    num_epochs=300,
    EarlyStop_patience=30,
    EarlyStop_min_delta=1.0e-4,
    Selection='edge_rmse',
    Sel_lambda=0.1,
    Sel_norm=True,
)


def merge(base, overrides):
    out = dict(base)
    for k, v in overrides.items():
        if k == 'WLE' and isinstance(v, dict):
            out['WLE'] = dict(out.get('WLE', {}))
            out['WLE'].update(v)
        else:
            out[k] = v
    out.update(COMMON)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    n_new, n_skip = 0, 0
    for d in DATASETS:
        src = os.path.join(CFG, d, 'SALT.yaml')
        if not os.path.isfile(src):
            print(f'[MISS] {d}: 官方 SALT.yaml 不存在，跳过')
            continue
        with open(src, encoding='utf-8') as f:
            base = yaml.safe_load(f)
        jobs = []
        for s in (0, 1, 2):
            jobs.append((f'SALT_scratch_s{s}.yaml', dict(base, EarlyStop_patience=30,
                                                         EarlyStop_min_delta=1.0e-4)))
            jobs.append((f'SALT_Prob_full_s{s}.yaml', merge(base, PROB_OVERrides)))
            jobs.append((f'SALT_INN_full_s{s}.yaml', merge(base, INN_OVERRIDES)))
        for fname, cfg in jobs:
            dst = os.path.join(CFG, d, fname)
            if os.path.isfile(dst):
                n_skip += 1
                continue
            if args.dry_run:
                print(f'[DRY] {d}/{fname}')
            else:
                with open(dst, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
                print(f'[NEW] {d}/{fname}')
            n_new += 1
    print(f'\n完成: 新建 {n_new} 个配置, 已存在跳过 {n_skip} 个')


if __name__ == '__main__':
    main()
