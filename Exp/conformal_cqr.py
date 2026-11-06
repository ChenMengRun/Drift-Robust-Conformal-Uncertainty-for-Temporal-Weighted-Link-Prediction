#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conformal_cqr.py — CQR 基线的共形评估入表（补齐基线矩阵）

收集 SALT_CQR 的 (q_lo, q_hi, y)（GPU 秒级）→ CQR 分数 s=max(q_lo-y, y-q_hi) 的
split CP（等权+漂移加权）→ cov90/宽度；与 σ-CP（sigma_evt/aci_sigma）同表 PK。
输出 Exp/calib/cqr_baseline.csv。
"""
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
from conformal_twlp import weighted_quantile, drift_weights

DATASET_PATH = os.path.join(ROOT, 'Data')
def conformal_eval(z):
    from conformal_twlp import conformal
    r = conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "sigma_evt", rho=0.999)
    return r[0.90]


DS = ["DC","HMob","INVIS13","INVIS15","IoT","LyonSchool","T-Drive","Thiers13"]

print(f"{'数据集':11s}{'CQR等权':>17s}{'CQR漂移':>17s}{'σ+EVT(ours)':>17s}")
for D in DS:
    run = "SALT_CQR_s0"
    d = f"Exp/pt/{D}/{run}"
    pth = sorted(glob.glob(d + "/*.pth"))
    if not pth:
        print(f"{D:11s}  缺 ckpt"); continue
    ck = max(pth, key=lambda p: int(os.path.basename(p).split('_')[0]) if os.path.basename(p).split('_')[0].isdigit() else -1)
    cfg = f"SALT/config/{D}/{run}.yaml"
    setup_seed(0)
    params = Paramset(cfg); params.device = torch.device('cuda')
    dataset = Dataset(params, DATASET_PATH)
    net = C.build_model(params).to(params.device); net.eval()
    C.load_weights(net, ck, params)
    ts = dataset.num_snaps - dataset.num_test_snaps
    vs = ts - dataset.num_val_snaps
    def collect_qs(t0, t1):
        qlo, qhi, ys = [], [], []
        with torch.no_grad():
            for tau in range(t0, t1, params.pred_size):
                sup, noise, feat = dataset.get_win_data(tau - params.win_size, tau)
                ql = net.eval_forward_quantiles(sup, feat, noise)
                for k, t in enumerate(range(tau, tau + params.pred_size)):
                    mu_t = ql[k][0]
                    n = mu_t.shape[0]
                    el = C.local_edges(dataset, t, n)
                    if not el: continue
                    ii = torch.tensor([e[0] for e in el], dtype=torch.long, device=mu_t.device)
                    jj = torch.tensor([e[1] for e in el], dtype=torch.long, device=mu_t.device)
                    g = C.gnd_norm_matrix(dataset, t).to(mu_t.device)
                    qlo.append(ql[k][0][ii, jj].cpu().numpy() * dataset.max_thres)
                    qhi.append(ql[k][1][ii, jj].cpu().numpy() * dataset.max_thres)
                    ys.append(g[ii, jj].cpu().numpy() * dataset.max_thres)
        return np.concatenate(qlo), np.concatenate(qhi), np.concatenate(ys)
    qlo_v, qhi_v, y_v = collect_qs(vs, ts)
    qlo_t, qhi_t, y_t = collect_qs(ts, dataset.num_snaps)
    s_v = np.maximum(qlo_v - y_v, y_v - qhi_v)
    s_t = np.maximum(qlo_t - y_t, y_t - qhi_t)
    res = {}
    wid_true = float(np.mean(qhi_t - qlo_t))
    for tag, rho in [("CQR等权", 1.0), ("CQR漂移", 0.999)]:
        w = drift_weights(len(s_v), rho) if rho < 1 else np.ones(len(s_v))
        q = weighted_quantile(s_v, w, 0.90)
        cov = float(np.mean(s_t <= q)); wid = wid_true
        res[tag] = (cov, wid)
    zf = f"Exp/cache/{D}_SALT_Prob_full_s0.npz"
    if not os.path.isfile(zf):
        zf = f"Exp/cache/{D}_SALT_Prob_full_s1.npz"
    z = np.load(zf)
    r_ours = conformal_eval(z)
    del net, dataset; torch.cuda.empty_cache()
    print(f"{D:11s}  {res['CQR等权'][0]:.3f}/{res['CQR等权'][1]:8.4g}  {res['CQR漂移'][0]:.3f}/{res['CQR漂移'][1]:8.4g}  {r_ours[0]:.3f}/{r_ours[1]:8.4g}", flush=True)

def conformal_eval(z):
    from conformal_twlp import conformal
    r = conformal(z["mu_v"], z["sig_v"], z["y_v"], z["mu_t"], z["sig_t"], z["y_t"], "sigma_evt", rho=0.999)
    return r[0.90]
