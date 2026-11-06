#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reselect_by_crps.py —— 验证：σ 死于选优还是死于方法？

对存量 ckpt（epoch>=5）逐个在 val 段算高斯 CRPS，按 val-CRPS 重选最优 ckpt，
再在 test 段出完整校准指标（CRPS/ΔCRPS vs 同方差/σcorr/cov1σ）。
若重选后 ΔCRPS 转正且 σcorr 回升 ⇒ σ 有信息量、死于选优准则（可修复）；
若 val-CRPS 最优 ckpt 的 σcorr 仍≈0 ⇒ σ 真塌了（需改损失/加锚，走重训）。

只读 Exp/pt，不写 ckpt；输出 Exp/calib/reselect_report.md + 每 run 明细 CSV。
用法: python3 Exp/reselect_by_crps.py --device cuda [--datasets DC,INVIS15 ...]
"""
import argparse, importlib, math, os, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'SALT'))
os.chdir(ROOT)

import numpy as np
import torch
from SALT.dataset.dataset import Dataset
from SALT.utils.paramset import Paramset
from SALT.utils.seeds import setup_seed

sys.path.insert(0, os.path.join(ROOT, 'Exp'))
import calibration as C   # 复用 build_model/load_weights/collect/crps_gaussian（import 时 chdir ROOT，无害）

DATASET_PATH = os.path.join(ROOT, 'Data')
ALL_DS = ["DC", "HMob", "INVIS13", "INVIS15", "IoT", "LyonSchool", "T-Drive", "Thiers13"]
SKIP_RUNS = {"DC/SALT_Prob_full_s0", "DC/SALT_Prob_full_s2"}   # 训练中，不读半成品
MIN_EP = 5


def val_crps_of(net, dataset, params, probabilistic, val_start, test_start):
    mu, sig, y = C.collect(net, dataset, params, val_start, test_start, probabilistic)
    crps = float(np.mean([C.crps_gaussian(m, s, yy) for m, s, yy in zip(mu, sig, y)]))
    corr = float(np.corrcoef(np.abs(y - mu), sig)[0, 1]) if probabilistic and sig.std() > 0 else float('nan')
    return crps, corr, mu, sig, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--datasets', default=','.join(ALL_DS))
    args = ap.parse_args()
    dev = torch.device(args.device)

    report = ["# σ 选优重选报告（σ 选优重选验证）", "",
              f"生成: {os.uname().nodename} {__import__('time').strftime('%F %T')} | device={args.device} | min_ep={MIN_EP}",
              "选优准则: val 段高斯 CRPS 最小（ckpt 文件名 epoch≥5）", ""]
    report.append("| run | 存量ckpt(≥e5) | 最优epoch | valCRPS | →test: CRPS | ΔCRPS | σcorr | cov1σ | 判定 |")
    report.append("|---|---|---|---|---|---|---|---|---|")

    for D in args.datasets.split(','):
        D = D.strip()
        for rundir in sorted(glob.glob(f"Exp/pt/{D}/*/")):
            run = os.path.basename(rundir.rstrip('/'))
            key = f"{D}/{run}"
            if key in SKIP_RUNS:
                continue
            ckpts = []
            for p in glob.glob(os.path.join(rundir, "*.pth")):
                try:
                    ep = int(os.path.basename(p).split('_')[0])
                except ValueError:
                    continue
                if ep >= MIN_EP:
                    ckpts.append((ep, p))
            if len(ckpts) < 2:
                continue   # 无重选余地
            cfg = f"SALT/config/{D}/{run}.yaml"
            if not os.path.isfile(cfg):
                continue
            setup_seed(0)
            params = Paramset(cfg)
            params.device = dev
            dataset = Dataset(params, DATASET_PATH)
            net = C.build_model(params).to(dev)
            probabilistic = hasattr(net, 'eval_forward_dist')
            if not probabilistic:
                continue
            net.eval()
            test_start = dataset.num_snaps - dataset.num_test_snaps
            val_start = test_start - dataset.num_val_snaps

            scored = []
            for ep, p in sorted(ckpts):
                C.load_weights(net, p, params)
                try:
                    crps, corr, *_ = val_crps_of(net, dataset, params, probabilistic, val_start, test_start)
                except Exception as e:
                    print(f"[WARN] {key} e{ep}: {e}")
                    continue
                scored.append((crps, corr, ep, p))
                print(f"[{key}] e{ep}: valCRPS={crps:.4f} σcorr={corr:+.3f}")
            if not scored:
                continue
            scored.sort()
            best_crps, best_corr, best_ep, best_p = scored[0]

            # 最优 ckpt 在 test 段出完整指标
            C.load_weights(net, best_p, params)
            mu_t, sig_t, y_t = C.collect(net, dataset, params, test_start, dataset.num_snaps, True)
            crps_t = float(np.mean([C.crps_gaussian(m, s, yy) for m, s, yy in zip(mu_t, sig_t, y_t)]))
            corr_t = float(np.corrcoef(np.abs(y_t - mu_t), sig_t)[0, 1]) if sig_t.std() > 0 else float('nan')
            cov1 = float(np.mean(np.abs(y_t - mu_t) <= sig_t))
            mu_v, _, y_v = C.collect(net, dataset, params, val_start, test_start, False)
            sig_c = float(np.sqrt(np.mean((mu_v - y_v) ** 2)))
            crps_c = float(np.mean([C.crps_gaussian(m, sig_c, yy) for m, yy in zip(mu_t, y_t)]))
            d_crps = crps_c - crps_t
            verdict = "✅σ有信息(死于选优)" if (d_crps > 0 and corr_t > 0.15) else \
                      ("⚠️部分" if d_crps > 0 else "❌σ真塌(需改损失)")
            report.append(f"| {key} | {len(scored)} | e{best_ep} | {best_crps:.4f} | {crps_t:.4f} | "
                          f"{d_crps:+.4f} | {corr_t:+.3f} | {cov1:.3f} | {verdict} |")
            print(f"[{key}] ⇒ 最优 e{best_ep}: testCRPS={crps_t:.4f} ΔCRPS={d_crps:+.4f} "
                  f"σcorr={corr_t:+.3f} cov1σ={cov1:.3f} {verdict}")
            del net, dataset
            torch.cuda.empty_cache()

    out = os.path.join(ROOT, 'Exp', 'calib', 'reselect_report.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8').write('\n'.join(report) + '\n')
    print(f"\n[RESELECT] 报告 → {out}")


if __name__ == '__main__':
    main()
