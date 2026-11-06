#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""persistence_edge.py — persistence 基线的 edge 口径重算（edge 口径对照）

预测 = 上一快照真值邻接矩阵（零参数零学习），test 段每快照在真实边上算 RMSE，
与矩阵 run 同口径（原始权重空间、仅真实边）。CPU 可跑，numpy 向量化。
"""
import argparse, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'SALT'))
os.chdir(ROOT)
import numpy as np
import torch
from SALT.dataset.dataset import Dataset
from SALT.utils.paramset import Paramset
from SALT.dataset.dataset import get_adj_wei

DATASET_PATH = os.path.join(ROOT, 'Data')
DS_ALL = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]


def to_idx(edges, mapping):
    """边列表 -> (ii, jj, w) 依据 mapping 映射到局部索引；未命中丢弃"""
    lp = mapping
    E = np.asarray(edges, dtype=np.float64)
    gi = np.array([lp.get(int(x), -1) for x in E[:, 0]], dtype=np.int64)
    gj = np.array([lp.get(int(x), -1) for x in E[:, 1]], dtype=np.int64)
    ok = (gi >= 0) & (gj >= 0)
    return gi[ok], gj[ok], E[ok, 2]


def edge_rmse_persist(dataset):
    test_start = dataset.num_snaps - dataset.num_test_snaps
    vals = []
    for t in range(test_start, dataset.num_snaps):
        if dataset.data_level == 2:
            cmap = dataset.node_map_seq_gbl[t]
            pmap = dataset.node_map_seq_gbl[t - 1]
            n = len(cmap)
            gnd = np.zeros((n, n)); prev = np.zeros((n, n))
            ii, jj, w = to_idx(dataset.edge_seq[t], cmap)
            gnd[ii, jj] = w
            shared = {g: cmap[g] for g in cmap if g in pmap}   # 全局id -> t帧局部索引
            pi, pj, pw = to_idx(dataset.edge_seq[t - 1], shared)
            prev[pi, pj] = pw
        else:
            n = dataset.num_nodes
            gnd = get_adj_wei(dataset.edge_seq[t], n, dataset.max_thres)
            prev = get_adj_wei(dataset.edge_seq[t - 1], n, dataset.max_thres)
            E = np.asarray(dataset.edge_seq[t], dtype=np.float64)
            ii = E[:, 0].astype(np.int64); jj = E[:, 1].astype(np.int64)
        if len(ii) == 0:
            continue
        vals.append(float(np.sqrt(np.mean((prev[ii, jj] - gnd[ii, jj]) ** 2))))
    return float(np.mean(vals)), len(vals)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(DS_ALL))
    A = ap.parse_args()
    print(f"{'数据集':11s}{'persistence edge-RMSE':>22s}  {'快照数':>6s}")
    for D in A.datasets.split(','):
        cfg = f"SALT/config/{D}/SALT.yaml"
        if not os.path.isfile(cfg):
            cfg = sorted(glob.glob(f"SALT/config/{D}/SALT_scratch_s0.yaml"))[0]
        p = Paramset(cfg)
        p.device = torch.device('cpu')
        ds = Dataset(p, DATASET_PATH)
        v, n = edge_rmse_persist(ds)
        print(f"{D:11s}{v:22.4f}  {n:6d}")
