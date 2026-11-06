#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cache_musig.py — 共形实验的弹药库：批量缓存全部概率 run 的 (mu, sigma, y) val/test 数组

对每个 (D, run) 选稳态最优 ckpt → GPU 收集 val/test 的边级 (μ,σ,y) → 存
Exp/cache/{D}_{run}.npz。后续共形/校准/EVT 实验全部从缓存读，零 GPU 重复。
用法: python3 Exp/cache_musig.py [--device cuda]
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
OUT = "Exp/cache"


def pick_ckpt(d, min_ep=5):
    best = (1e18, None)
    for p in glob.glob(os.path.join(d, "*.pth")):
        a = os.path.basename(p).split("_")
        try:
            ep, rm = int(a[0]), float(a[1])
        except (ValueError, IndexError):
            continue
        if ep >= min_ep and rm < best[0]:
            best = (rm, p)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-ep", type=int, default=5, help="ckpt 最低 epoch 过滤（早期收敛 run 传 0）")
    A = ap.parse_args()
    dev = torch.device(A.device)
    os.makedirs(OUT, exist_ok=True)
    n_done = 0
    for D in DS:
        for base in ("SALT_Prob_full", "SALT_INN_full", "SALT_Prob_anch01", "SALT_Prob_anch03"):
            for s in "012":
                run = f"{base}_s{s}"
                out = f"{OUT}/{D}_{run}.npz"
                if os.path.isfile(out):
                    n_done += 1
                    continue
                d = f"Exp/pt/{D}/{run}"
                ck = pick_ckpt(d, A.min_ep) if os.path.isdir(d) else None
                if not ck:
                    continue
                cfg = f"SALT/config/{D}/{run}.yaml"
                try:
                    setup_seed(0)
                    params = Paramset(cfg); params.device = dev
                    dataset = Dataset(params, DATASET_PATH)
                    net = C.build_model(params).to(dev); net.eval()
                    C.load_weights(net, ck, params)
                    ts = dataset.num_snaps - dataset.num_test_snaps
                    vs = ts - dataset.num_val_snaps
                    mu_v, sig_v, y_v = C.collect(net, dataset, params, vs, ts, True)
                    mu_t, sig_t, y_t = C.collect(net, dataset, params, ts, dataset.num_snaps, True)
                    np.savez_compressed(out, mu_v=mu_v, sig_v=sig_v, y_v=y_v,
                                        mu_t=mu_t, sig_t=sig_t, y_t=y_t, ckpt=ck)
                    n_done += 1
                    print(f"[CACHE] {D}/{run} <- {os.path.basename(ck)} "
                          f"(val {len(y_v)} / test {len(y_t)} edges)", flush=True)
                    del net, dataset
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ERR] {D}/{run}: {e}", flush=True)
    print(f"[CACHE] 完成，共 {n_done} 个缓存文件 → {OUT}/")


if __name__ == '__main__':
    main()
