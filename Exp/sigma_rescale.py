#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sigma_rescale.py —— σ 温度缩放检验：σ 的排序信息能否经单标量校准兑现为 CRPS 增益？

对给定 ckpt：val 段网格搜索单标量 s（乘到 σ）最小化 val-CRPS → test 段评估缩放后
CRPS/ΔCRPS/cov1σ。若翻正 ⇒ σ 有信息、只差校准层（post-hoc 校准即可兑现）。
用法: python3 Exp/sigma_rescale.py --device cuda --spec D/config_name:ckpt_path ...
"""
import argparse, os, sys, glob

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

def run_one(spec, dev):
    D_run, ck = spec.split(':', 1)
    D, run = D_run.split('/', 1)
    cfg = f"SALT/config/{D}/{run}.yaml"
    setup_seed(0)
    params = Paramset(cfg); params.device = dev
    dataset = Dataset(params, DATASET_PATH)
    net = C.build_model(params).to(dev); net.eval()
    n_load, n_skip, _ = C.load_weights(net, ck, params)
    print(f"[{spec}] 权重加载 {n_load} 跳过 {n_skip}")
    test_start = dataset.num_snaps - dataset.num_test_snaps
    val_start = test_start - dataset.num_val_snaps

    mu_v, sig_v, y_v = C.collect(net, dataset, params, val_start, test_start, True)
    scales = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.05, 2.5, 15)])
    val = [(float(np.mean([C.crps_gaussian(m, s * f, yy) for m, s, yy in zip(mu_v, sig_v, y_v)])), f)
           for f in scales]
    val.sort(); s_val, f_best = val[0]

    mu_t, sig_t, y_t = C.collect(net, dataset, params, test_start, dataset.num_snaps, True)
    crps_raw = float(np.mean([C.crps_gaussian(m, s, yy) for m, s, yy in zip(mu_t, sig_t, y_t)]))
    crps_scl = float(np.mean([C.crps_gaussian(m, s * f_best, yy) for m, s, yy in zip(mu_t, sig_t, y_t)]))
    corr = float(np.corrcoef(np.abs(y_t - mu_t), sig_t)[0, 1])
    cov_raw = float(np.mean(np.abs(y_t - mu_t) <= sig_t))
    cov_scl = float(np.mean(np.abs(y_t - mu_t) <= sig_t * f_best))
    mu_v2, _, y_v2 = C.collect(net, dataset, params, val_start, test_start, False)
    sig_c = float(np.sqrt(np.mean((mu_v2 - y_v2) ** 2)))
    crps_c = float(np.mean([C.crps_gaussian(m, sig_c, yy) for m, yy in zip(mu_t, y_t)]))
    print(f"[{spec}] e={os.path.basename(ck).split('_')[0]}")
    print(f"  σcorr={corr:+.3f} | s*={f_best:.3f} (val-CRPS {s_val:.4f})")
    print(f"  test CRPS: raw={crps_raw:.4f}  scaled={crps_scl:.4f}  const对照={crps_c:.4f}")
    print(f"  ΔCRPS: raw={crps_c-crps_raw:+.4f} → scaled={crps_c-crps_scl:+.4f}   "
          f"cov1σ: {cov_raw:.3f}→{cov_scl:.3f} (名义0.683)")
    del net, dataset; torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--spec', nargs='+', required=True, help='D/run_name:ckpt_path')
    args = ap.parse_args()
    dev = torch.device(args.device)
    for spec in args.spec:
        run_one(spec, dev)
