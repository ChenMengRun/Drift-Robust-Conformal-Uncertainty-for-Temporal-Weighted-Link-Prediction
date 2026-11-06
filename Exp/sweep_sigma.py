#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_sigma.py — σ 温度缩放全量扫描（复用 Exp/sigma_rescale.py 的逻辑）

对全部概率模型 run（Prob/INN × 8 数据集 × 3 种子）：
  选 dense 最优稳态 ckpt → val 拟合单标量 s → test 评估 raw/scaled CRPS、ΔCRPS、σcorr、cov1σ
输出: Exp/calib/sigma_sweep.tsv + 汇总 Exp/calib/sigma_sweep_summary.md
用法: python3 Exp/sweep_sigma.py [--device cuda] [--only missing]
"""
import argparse, glob, os, sys
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
DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]
SCALES = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.05, 2.5, 15)])
OUT = "Exp/calib/sigma_sweep.tsv"


def pick_ckpt(d):
    best = (1e18, None, -1)
    for p in glob.glob(os.path.join(d, "*.pth")):
        a = os.path.basename(p).split("_")
        try:
            ep, rm = int(a[0]), float(a[1])
        except (ValueError, IndexError):
            continue
        if ep >= 5 and rm < best[0]:
            best = (rm, p, ep)
    return best


def sweep_one(D, run, ck, dev, rows):
    cfg = f"SALT/config/{D}/{run}.yaml"
    setup_seed(0)
    params = Paramset(cfg); params.device = dev
    dataset = Dataset(params, DATASET_PATH)
    net = C.build_model(params).to(dev); net.eval()
    C.load_weights(net, ck, params)
    ts = dataset.num_snaps - dataset.num_test_snaps
    vs = ts - dataset.num_val_snaps
    mu_v, sig_v, y_v = C.collect(net, dataset, params, vs, ts, True)
    scored = [(float(np.mean([C.crps_gaussian(m, s * f, yy) for m, s, yy in zip(mu_v, sig_v, y_v)])), f) for f in SCALES]
    scored.sort(); _, f_best = scored[0]
    mu_t, sig_t, y_t = C.collect(net, dataset, params, ts, dataset.num_snaps, True)
    crps_raw = float(np.mean([C.crps_gaussian(m, s, yy) for m, s, yy in zip(mu_t, sig_t, y_t)]))
    crps_scl = float(np.mean([C.crps_gaussian(m, s * f_best, yy) for m, s, yy in zip(mu_t, sig_t, y_t)]))
    corr = float(np.corrcoef(np.abs(y_t - mu_t), sig_t)[0, 1]) if sig_t.std() > 0 else float("nan")
    cov_s = float(np.mean(np.abs(y_t - mu_t) <= sig_t * f_best))
    mu_v2, _, y_v2 = C.collect(net, dataset, params, vs, ts, False)
    sig_c = float(np.sqrt(np.mean((mu_v2 - y_v2) ** 2)))
    crps_c = float(np.mean([C.crps_gaussian(m, sig_c, yy) for m, yy in zip(mu_t, y_t)]))
    rows.append((D, run, os.path.basename(ck), corr, f_best, crps_raw, crps_scl, crps_c,
                 crps_c - crps_raw, crps_c - crps_scl, cov_s))
    print(f"[SWEEP] {D}/{run}: σcorr={corr:+.3f} s*={f_best:.2f} Δraw={crps_c-crps_raw:+.4g} Δscl={crps_c-crps_scl:+.4g} cov1σ={cov_s:.3f}", flush=True)
    del net, dataset
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    A = ap.parse_args()
    dev = torch.device(A.device)
    done = set()
    rows = []
    if os.path.isfile(OUT):
        for line in open(OUT):
            f = line.rstrip("\n").split("\t")
            if len(f) > 2 and f[0] != "dataset":
                done.add(f"{f[0]}/{f[1]}"); rows.append(tuple(f[:2]) + tuple(x for x in f[2:]))
        rows = [r for r in rows if True]
    fresh = []
    for D in DS:
        for base in ("SALT_Prob_full", "SALT_INN_full"):
            for s in "012":
                run = f"{base}_s{s}"
                if f"{D}/{run}" in done:
                    continue
                d = f"Exp/pt/{D}/{run}"
                if not os.path.isdir(d):
                    continue
                rm, ck, ep = pick_ckpt(d)
                if ck:
                    fresh.append((D, run, ck))
    print(f"待扫 {len(fresh)} 个 run（已完成 {len(done)}）", flush=True)
    hdr = ("dataset", "run", "ckpt", "sigcorr", "s*", "crps_raw", "crps_scaled",
           "crps_const", "dCRPS_raw", "dCRPS_scaled", "cov1s_scaled")
    allrows = []
    if os.path.isfile(OUT):
        for line in open(OUT):
            f = line.rstrip("\n").split("\t")
            if len(f) == 11 and f[0] != "dataset":
                allrows.append(f)
    with open(OUT, "w") as fo:
        fo.write("\t".join(hdr) + "\n")
        for r in allrows:
            fo.write("\t".join(str(x) for x in r) + "\n")
        for D, run, ck in fresh:
            try:
                out_rows = []
                sweep_one(D, run, ck, dev, out_rows)
                for r in out_rows:
                    fo.write("\t".join(str(x) for x in r) + "\n")
                fo.flush()
            except Exception as e:
                print(f"[ERR] {D}/{run}: {e}", flush=True)
    print(f"[SWEEP] 完成 → {OUT}")


if __name__ == "__main__":
    main()
