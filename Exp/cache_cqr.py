#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cache_cqr.py — 收集 CQR 基线的 (q_lo, q_hi, y) 缓存（供 ACI 包装/GPD 对称化使用）"""
import glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'SALT')); sys.path.insert(0, os.path.join(ROOT, 'Exp'))
os.chdir(ROOT)
import numpy as np
import torch
from SALT.dataset.dataset import Dataset
from SALT.utils.paramset import Paramset
from SALT.utils.seeds import setup_seed
import calibration as C

DATASET_PATH = os.path.join(ROOT, 'Data')
DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]

for D in DS:
    out = f"Exp/cache/{D}_SALT_CQR_s0_q.npz"
    if os.path.isfile(out):
        print(f"[CQR-CACHE] {D} 已存在"); continue
    run = "SALT_CQR_s0"
    d = f"Exp/pt/{D}/{run}"
    pths = [p for p in glob.glob(d + "/*.pth") if os.path.basename(p).split('_')[0].isdigit()]
    if not pths:
        print(f"[CQR-CACHE] {D} 缺 ckpt"); continue
    ck = max(pths, key=lambda p: int(os.path.basename(p).split('_')[0]))
    cfg = f"SALT/config/{D}/{run}.yaml"
    setup_seed(0)
    params = Paramset(cfg); params.device = torch.device('cuda')
    dataset = Dataset(params, DATASET_PATH)
    net = C.build_model(params).to(params.device); net.eval()
    C.load_weights(net, ck, params)
    ts = dataset.num_snaps - dataset.num_test_snaps
    vs = ts - dataset.num_val_snaps
    def collect(t0, t1):
        qlo, qhi, ys = [], [], []
        with torch.no_grad():
            for tau in range(t0, t1, params.pred_size):
                sup, noise, feat = dataset.get_win_data(tau - params.win_size, tau)
                ql = net.eval_forward_quantiles(sup, feat, noise)
                for k, t in enumerate(range(tau, tau + params.pred_size)):
                    n = ql[k][0].shape[0]
                    el = C.local_edges(dataset, t, n)
                    if not el: continue
                    ii = torch.tensor([e[0] for e in el], dtype=torch.long, device=ql[k][0].device)
                    jj = torch.tensor([e[1] for e in el], dtype=torch.long, device=ql[k][0].device)
                    g = C.gnd_norm_matrix(dataset, t).to(ql[k][0].device)
                    qlo.append(ql[k][0][ii, jj].cpu().numpy() * dataset.max_thres)
                    qhi.append(ql[k][1][ii, jj].cpu().numpy() * dataset.max_thres)
                    ys.append(g[ii, jj].cpu().numpy() * dataset.max_thres)
        return np.concatenate(qlo), np.concatenate(qhi), np.concatenate(ys)
    qlo_v, qhi_v, y_v = collect(vs, ts)
    qlo_t, qhi_t, y_t = collect(ts, dataset.num_snaps)
    np.savez_compressed(out, qlo_v=qlo_v, qhi_v=qhi_v, y_v=y_v, qlo_t=qlo_t, qhi_t=qhi_t, y_t=y_t)
    print(f"[CQR-CACHE] {D} 完成 (val {len(y_v)} / test {len(y_t)})", flush=True)
    del net, dataset; torch.cuda.empty_cache()
print("全部完成")
