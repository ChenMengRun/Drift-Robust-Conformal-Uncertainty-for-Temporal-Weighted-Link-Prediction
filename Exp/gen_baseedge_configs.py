#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_baseedge_configs.py —— 解耦对照配置：官方 SALT 结构 + edge 加权损失（SALT_Edge，无概率化）
回答「Prob/INN 的 edge 领先里，损失口径贡献多少、概率化贡献多少」
底=各数据集官方 SALT.yaml；仅换 model=model.salt_edge.SALT_Edge（Loss 段官方原样，
edge 加权在 SALT_Edge.causal_training_loss 内部通过 edge_masks 生效——已确认 inn_train 传入）。
"""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'SALT', 'config')
DS = ['DC', 'HMob', 'INVIS13', 'INVIS15', 'IoT', 'LyonSchool', 'T-Drive', 'Thiers13']

n = 0
for d in DS:
    src = os.path.join(CFG, d, 'SALT.yaml')
    base = yaml.safe_load(open(src, encoding='utf-8'))
    base['model'] = 'model.salt_edge.SALT_Edge'
    base['num_epochs'] = 300
    base['EarlyStop_patience'] = 15
    base['EarlyStop_min_delta'] = 1.0e-4
    base['Selection'] = 'edge_rmse'
    dst = os.path.join(CFG, d, 'SALT_base_edge_auto.yaml')
    yaml.safe_dump(base, open(dst, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)
    print(f'[NEW] {d}/SALT_base_edge_auto.yaml')
    n += 1
print(f'完成 {n} 个')
