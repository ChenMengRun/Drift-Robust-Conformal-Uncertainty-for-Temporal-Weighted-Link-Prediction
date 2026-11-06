#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conformal_persistence.py — 零参数区间基线：persistence 预测 + split-conformal 经验分位

校准分数 = val 段 |y - y_prev|；test 段区间 [ŷ-q, ŷ+q]。与 σ-CP-TWLP 同表对比。
输出每数据集 coverage/宽度（90%）+ 漂移加权版（rho 几何衰减）。CPU。
"""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'SALT'))
os.chdir(ROOT)
import numpy as np
import torch
from SALT.dataset.dataset import Dataset
from SALT.utils.paramset import Paramset
from SALT.dataset.dataset import get_adj_wei

DATASET_PATH = os.path.join(ROOT, 'Data')
DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]


def resid_edges(dataset, t0, t1):
    """逐快照收集 persistence 残差 |y - y_prev|（仅真实边，原始空间）"""
    out = []
    for t in range(t0, t1):
        if dataset.data_level == 2:
            cmap = dataset.node_map_seq_gbl[t]; pmap = dataset.node_map_seq_gbl[t - 1]
            lp = {g: i for i, g in enumerate(cmap)}
            prev = {}
            for e in dataset.edge_seq[t - 1]:
                gi, gj = lp.get(int(e[0])), lp.get(int(e[1]))
                if gi is not None and gj is not None:
                    prev[(gi, gj)] = e[2]
            for e in dataset.edge_seq[t]:
                gi, gj = lp.get(int(e[0])), lp.get(int(e[1]))
                if gi is None or gj is None:
                    continue
                out.append(abs(e[2] - prev.get((gi, gj), 0.0)))
        else:
            prev = {(int(e[0]), int(e[1])): e[2] for e in dataset.edge_seq[t - 1]}
            for e in dataset.edge_seq[t]:
                out.append(abs(e[2] - prev.get((int(e[0]), int(e[1])), 0.0)))
    return np.array(out)


def main():
    print(f"{'数据集':11s}{'cov90等权':>9s}{'宽度等权':>9s}{'cov90漂移':>9s}{'宽度漂移':>9s}")
    for D in DS:
        cfg = f"SALT/config/{D}/SALT.yaml"
        if not os.path.isfile(cfg):
            cfg = sorted(glob.glob(f"SALT/config/{D}/SALT_scratch_s0.yaml"))[0]
        p = Paramset(cfg); p.device = torch.device('cpu')
        ds = Dataset(p, DATASET_PATH)
        ts = ds.num_snaps - ds.num_test_snaps
        vs = ts - ds.num_val_snaps
        r_v = resid_edges(ds, vs, ts)
        r_t = resid_edges(ds, ts, ds.num_snaps)
        q = np.quantile(r_v, 0.90)
        cov = float(np.mean(r_t <= q)); wid = float(2 * q)
        rho = 0.999
        w = rho ** np.arange(len(r_v) - 1, -1, -1)
        idx = np.argsort(r_v); cw = np.cumsum(w[idx])
        k = np.searchsorted(cw, 0.90 * cw[-1]); qs = r_v[idx][min(k, len(r_v) - 1)]
        cov_d = float(np.mean(r_t <= qs)); wid_d = float(2 * qs)
        print(f"{D:11s}{cov:9.3f}{wid:9.4g}{cov_d:9.3f}{wid_d:9.4g}", flush=True)


if __name__ == '__main__':
    main()
