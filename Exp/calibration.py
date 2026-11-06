#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibration.py —— SALT / SALT_Prob / SALT_INN 统一校准评估（无卡可跑，CPU 推理）

相对 validate_salt_inn.py::prob_diag 的增量：
  1. 多水平校准曲线 PICP（19 个名义水平）+ ECE，全量&边级
  2. CRPS（高斯解析式，无 scipy 依赖）
  3. σ 十分位分桶：桶内 RMSE vs 桶均 σ（理想 RMSE≈σ）
  4. 真值分位分层（长尾分析）：≤p50 / 50-75 / 75-90 / 90-95 / 95-99 / >p99
     桶内 n / RMSE / MAE / meanσ / coverage95 / NLL
  5. 同方差常数 σ 对照（--const-sigma，σ 由 val 段残差估计，不用 test 信息）
     ——异方差 σ 的增益必须打赢这个对照才有故事
  6. 点预测模型（官方 SALT，无 σ 通道）：退化为 const-σ 模式 + 真值分层
     ——即「复现 gap 分析」工具：官方 Pretrained vs scratch 的长尾分层对比
  7. 输出：summary.txt + 3 张 CSV + 2 张 PNG（matplotlib Agg，英文标签避免字体坑）

用法（cwd = SALT-main）：
  # 概率模型（SALT_Prob / SALT_INN checkpoint）
  python Exp/calibration.py --config SALT/config/INVIS15/SALT_INN_full_s0.yaml \
                            --model Exp/pt/INVIS15/SALT_INN_full_s0/xxx.pth
  # 官方点预测模型（Pretrained 或 scratch checkpoint）
  python Exp/calibration.py --config SALT/config/INVIS15/SALT.yaml \
                            --model Pretrained/INVIS15/INVIS15_pretrain.pth
  # 不传 --model 则加载官方 Pretrained（若存在）
指标空间：主口径=原始权重空间（×max_thres，与 RMSE 同尺度）；NLL 附归一空间。
"""
import argparse
import importlib
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # SALT-main
sys.path.insert(0, ROOT)              # 使 `from SALT.xxx import` 可解析
sys.path.insert(0, os.path.join(ROOT, 'SALT'))   # 使 config 里 `model.salt.SALT` 可解析（与 validate 一致）
os.chdir(ROOT)   # Dataset 相对路径 ./Data

from SALT.dataset.dataset import Dataset, get_adj_wei, get_adj_wei_map   # noqa: E402
from SALT.utils.paramset import Paramset                                  # noqa: E402
from SALT.utils.seeds import setup_seed                                    # noqa: E402

EPSILON = 0.01
DATASET_PATH = os.path.join(ROOT, 'Data')


# ---------------- 高斯工具（标准库实现，无 scipy） ----------------
SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)

def phi(z):
    return math.exp(-0.5 * z * z) / SQRT_2PI

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2))

def crps_gaussian(mu, sigma, y):
    """高斯 CRPS 解析式: σ·[z(2Φ(z)-1) + 2φ(z) - 1/√π], z=(y-μ)/σ"""
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * Phi(z) - 1.0) + 2.0 * phi(z) - 1.0 / math.sqrt(math.pi))

def gauss_nll_full(mu, sigma, y):
    return 0.5 * ((y - mu) ** 2 / sigma ** 2 + 2.0 * math.log(sigma) + math.log(2.0 * math.pi))


# ---------------- 模型 / 权重加载（口径与 validate_salt_inn.py 一致） ----------------
def build_model(params):
    mod_path, cls_name = params.model.rsplit('.', 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(params)

def load_weights(net, ckpt_path, params):
    own = net.state_dict()
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=params.device)
        if isinstance(sd, dict):
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            elif 'model_dict' in sd:
                sd = sd['model_dict']
        tag = ckpt_path
    else:
        pf = os.path.join(ROOT, 'Pretrained', params.data_name, params.data_name + '_pretrain.pth')
        if not os.path.isfile(pf):
            raise FileNotFoundError('未找到官方预训练: %s（点预测 const-sigma 模式不可用）' % pf)
        sd = torch.load(pf, map_location=params.device)['model_dict']
        tag = '官方 Pretrained（形状匹配加载）'
    loaded = {k: v for k, v in sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    net.load_state_dict(loaded, strict=False)
    skipped = len(sd) - len(loaded)
    return len(loaded), skipped, tag


# ---------------- 逐快照收集（边级明细 + 全矩阵流式聚合） ----------------
def gnd_norm_matrix(dataset, t):
    edges = dataset.edge_seq[t]
    if dataset.data_level == 2:
        node_map = dataset.node_map_seq_gbl[t]
        g = get_adj_wei_map(edges, node_map, dataset.num_nodes, dataset.max_thres)
    else:
        g = get_adj_wei(edges, dataset.num_nodes, dataset.max_thres)
    g = g / dataset.max_thres + np.eye(dataset.num_nodes)
    return torch.FloatTensor(g)

def local_edges(dataset, t, n):
    edges = dataset.edge_seq[t]
    out = []
    for e in edges:
        if dataset.data_level == 2:
            si = dataset.node_map_seq_gbl[t].get(int(e[0]))
            di = dataset.node_map_seq_gbl[t].get(int(e[1]))
            if si is None or di is None:
                continue
        else:
            si, di = int(e[0]), int(e[1])
        if si < n and di < n:
            out.append((si, di))
    return out

def collect(net, dataset, params, start, end, probabilistic):
    """返回 (edge_mu, edge_sigma, edge_y) 数组（原始空间）；边收集向量化（每快照一次索引，禁逐边 .item()）"""
    mus, sigs, ys = [], [], []
    with torch.no_grad():
        for tau in range(start, end, params.pred_size):
            sup_list, noise_list, feat_list = dataset.get_win_data(tau - params.win_size, tau)
            if probabilistic:
                dist_list = net.eval_forward_dist(sup_list, feat_list, noise_list)
                mus_t = [d[..., 0] for d in dist_list]
                sigs_t = [torch.exp(d[..., 1]).clamp(min=1e-3, max=1e2) for d in dist_list]
            else:
                pt_list = net(sup_list, feat_list, noise_list, pred_flag=True)
                mus_t = list(pt_list)
                sigs_t = None
            for k, t in enumerate(range(tau, tau + params.pred_size)):
                gnd = gnd_norm_matrix(dataset, t).to(params.device)
                mu = mus_t[k]
                n = mu.shape[0]
                el = local_edges(dataset, t, n)
                if not el:
                    continue
                idx_i = torch.tensor([e[0] for e in el], dtype=torch.long, device=mu.device)
                idx_j = torch.tensor([e[1] for e in el], dtype=torch.long, device=mu.device)
                mus.append((mu[idx_i, idx_j]).cpu().numpy() * dataset.max_thres)
                ys.append((gnd[idx_i, idx_j]).cpu().numpy() * dataset.max_thres)
                if probabilistic:
                    sigs.append((sigs_t[k][idx_i, idx_j]).cpu().numpy() * dataset.max_thres)
    mu_a, y_a = np.concatenate(mus), np.concatenate(ys)
    sig_a = np.concatenate(sigs) if probabilistic else None
    return mu_a, sig_a, y_a


# ---------------- 指标组 ----------------
def calibration_curve(mu, sig, y, levels):
    from statistics import NormalDist
    q = NormalDist()
    rows = []
    for lv in levels:
        z = q.inv_cdf(0.5 + lv / 2.0)
        cov = float(np.mean(np.abs(y - mu) <= z * sig))
        rows.append((lv, cov))
    ece = float(np.mean([abs(c - l) for l, c in rows]))
    return rows, ece

def sigma_decile_bins(mu, sig, y):
    qs = np.quantile(sig, np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        m = (sig >= qs[i]) & (sig <= qs[i + 1] if i == 9 else sig < qs[i + 1])
        if m.sum() == 0:
            continue
        rm = float(np.sqrt(np.mean((mu[m] - y[m]) ** 2)))
        rows.append((i + 1, int(m.sum()), float(sig[m].mean()), rm))
    return rows

def tail_bins(mu, sig, y):
    edges_q = [0, 50, 75, 90, 95, 99, 100]
    qs = np.percentile(y, edges_q)
    rows = []
    for i in range(6):
        lo, hi = qs[i], qs[i + 1]
        m = (y >= lo) & (y <= hi if i == 5 else y < hi)
        if m.sum() == 0:
            continue
        rm = float(np.sqrt(np.mean((mu[m] - y[m]) ** 2)))
        ma = float(np.mean(np.abs(mu[m] - y[m])))
        row = dict(bin=f'p{edges_q[i]}-p{edges_q[i+1]}', n=int(m.sum()),
                   rmse=rm, mae=ma,
                   mean_sigma=float(sig[m].mean()) if sig is not None else float('nan'),
                   cov95=float(np.mean(np.abs(y[m] - mu[m]) <= 1.959964 * sig[m])) if sig is not None else float('nan'),
                   frac=int(m.sum()) / len(y))
        rows.append(row)
    return rows


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0,
                    help='torch CPU 线程数；0=自动(物理核数一半)。仅 CPU 推理生效')
    ap.add_argument('--no-const', action='store_true', help='概率模型时跳过同方差对照')
    args = ap.parse_args()

    if args.device == 'cpu':
        th = args.threads or min(8, max(1, (os.cpu_count() or 4) // 2))
        torch.set_num_threads(th)
        print(f'[CALIB] CPU threads = {th}')

    setup_seed(0)
    params = Paramset(args.config)
    params.device = torch.device(args.device)
    dataset = Dataset(params, DATASET_PATH)

    net = build_model(params).to(params.device)
    n_loaded, n_skip, tag = load_weights(net, args.model, params)
    net.eval()
    probabilistic = hasattr(net, 'eval_forward_dist')
    name = args.tag or (os.path.splitext(os.path.basename(args.model))[0] if args.model else 'pretrained')
    outdir = os.path.join(ROOT, 'Exp', 'calib', f'{params.data_name}_{net.__class__.__name__}_{name}')
    os.makedirs(outdir, exist_ok=True)
    print(f'[CALIB] model={net.__class__.__name__} data={params.data_name} ckpt={tag} '
          f'(loaded={n_loaded} skipped={n_skip}) probabilistic={probabilistic}')
    print(f'[CALIB] 输出目录: {outdir}')

    test_start = dataset.num_snaps - dataset.num_test_snaps
    val_start = test_start - dataset.num_val_snaps

    # μ 的 test 段收集（概率与点预测共用）
    mu_t, sig_t, y_t = collect(net, dataset, params, test_start, dataset.num_snaps, probabilistic)

    lines = []
    lines.append(f'== {net.__class__.__name__} @ {params.data_name} | ckpt: {tag}')
    lines.append(f'权重加载: {n_loaded} 跳过: {n_skip}; 概率模型: {probabilistic}; device: {args.device}')
    lines.append(f'test 边数: {len(y_t)}; 真值范围: [{y_t.min():.2f}, {y_t.max():.2f}]')

    rmse = float(np.sqrt(np.mean((mu_t - y_t) ** 2)))
    mae = float(np.mean(np.abs(mu_t - y_t)))
    lines.append(f'[点预测] edge-RMSE={rmse:.4f}  edge-MAE={mae:.4f}  (原始空间)')

    # ---- 真值分层（点预测也输出——复现 gap 分析工具） ----
    tb = tail_bins(mu_t, sig_t, y_t)
    with open(os.path.join(outdir, 'tail_bins.csv'), 'w') as f:
        f.write('bin,n,rmse,mae,mean_sigma,cov95,frac\n')
        for r in tb:
            f.write(f"{r['bin']},{r['n']},{r['rmse']:.4f},{r['mae']:.4f},"
                    f"{r['mean_sigma']:.4f},{r['cov95']:.4f},{r['frac']:.4f}\n")
    lines.append('\n[真值分层] bin,n,RMSE,MAE,meanσ,cov95 (见 tail_bins.csv)')
    for r in tb:
        lines.append(f"  {r['bin']:>10}  n={r['n']:>6}  RMSE={r['rmse']:8.3f}  MAE={r['mae']:7.3f}  "
                     f"meanσ={r['mean_sigma']:7.3f}  cov95={r['cov95']:.3f}  frac={r['frac']:.3f}")

    # ---- σ 通道分析（概率模型） ----
    if probabilistic:
        levels = [0.05 * i for i in range(2, 20)]   # 0.10 .. 0.95
        rows, ece = calibration_curve(mu_t, sig_t, y_t, levels)
        with open(os.path.join(outdir, 'calib_curve.csv'), 'w') as f:
            f.write('nominal,empirical\n')
            for lv, cov in rows:
                f.write(f'{lv:.2f},{cov:.4f}\n')
        crps = float(np.mean([crps_gaussian(m, s, y) for m, s, y in zip(mu_t, sig_t, y_t)]))
        nll = float(np.mean([gauss_nll_full(m, s, y) for m, s, y in zip(mu_t, sig_t, y_t)]))
        corr = float(np.corrcoef(np.abs(y_t - mu_t), sig_t)[0, 1]) if sig_t.std() > 0 else float('nan')
        lines.append(f'\n[异方差σ] CRPS={crps:.4f}  NLL={nll:.4f}  corr(|err|,σ)={corr:.4f}  ECE={ece:.4f}')
        lines.append('[异方差σ] 校准曲线(名义→经验): ' + '  '.join(f'{l:.2f}→{c:.2f}' for l, c in rows))

        sb = sigma_decile_bins(mu_t, sig_t, y_t)
        with open(os.path.join(outdir, 'sigma_bins.csv'), 'w') as f:
            f.write('decile,n,mean_sigma,rmse\n')
            for d, n, ms, rm in sb:
                f.write(f'{d},{n},{ms:.4f},{rm:.4f}\n')
        lines.append('[σ十分位] decile,n,meanσ,RMSE(理想≈σ) (见 sigma_bins.csv)')
        for d, n, ms, rm in sb:
            lines.append(f'  D{d:>2}  n={n:>6}  σ={ms:7.3f}  RMSE={rm:7.3f}  ratio={rm / max(ms, 1e-9):.2f}')

    # ---- 同方差对照（σ_const 由 val 段估计，不用 test 信息） ----
    if probabilistic and args.no_const:
        lines.append('\n[对照] --no-const: 跳过同方差对照')
    else:
        mu_v, _, y_v = collect(net, dataset, params, val_start, test_start, False)
        resid = mu_v - y_v
        sig_c = float(np.sqrt(np.mean(resid ** 2)))
        crps_c = float(np.mean([crps_gaussian(m, sig_c, y) for m, y in zip(mu_t, y_t)]))
        nll_c = float(np.mean([gauss_nll_full(m, sig_c, y) for m, y in zip(mu_t, y_t)]))
        lv, rows_c, ece_c = None, None, None
        from statistics import NormalDist
        q = NormalDist()
        rows_c = []
        for l in [0.05 * i for i in range(2, 20)]:
            z = q.inv_cdf(0.5 + l / 2.0)
            rows_c.append((l, float(np.mean(np.abs(y_t - mu_t) <= z * sig_c))))
        ece_c = float(np.mean([abs(c - l) for l, c in rows_c]))
        lines.append(f'\n[同方差对照] σ_const(val)={sig_c:.4f}  CRPS={crps_c:.4f}  NLL={nll_c:.4f}  ECE={ece_c:.4f}')
        if probabilistic:
            lines.append(f'[增益] ΔCRPS={crps_c - crps:+.4f} (正=异方差优)  ΔNLL={nll_c - nll:+.4f} (正=异方差优)  '
                         f'ΔECE={ece - ece_c:+.4f} (正=异方差优)')

    # ---- 图（Agg，英文标签） ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if probabilistic:
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
            ax.plot([l for l, _ in rows], [c for _, c in rows], 'o-', ms=4, label='heteroscedastic')
            ax.plot([l for l, _ in rows_c], [c for _, c in rows_c], 's--', ms=4, label='const-sigma')
            ax.set(xlabel='nominal coverage', ylabel='empirical coverage', title=f'{params.data_name} calibration')
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'reliability.png'), dpi=150); plt.close(fig)
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            bins = [r['bin'] for r in tb]; xpos = np.arange(len(tb))
            ax.bar(xpos - 0.18, [r['rmse'] for r in tb], 0.36, label='RMSE')
            ax.bar(xpos + 0.18, [r['mean_sigma'] for r in tb], 0.36, label='mean sigma')
            ax.set_xticks(xpos); ax.set_xticklabels(bins, rotation=30, ha='right')
            ax.set(title=f'{params.data_name} tail bins: RMSE vs sigma', ylabel='weight space')
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'tail.png'), dpi=150); plt.close(fig)
    except Exception as e:
        lines.append(f'[WARN] 绘图失败(不影响CSV): {e}')

    summary = '\n'.join(lines)
    open(os.path.join(outdir, 'calibration_summary.txt'), 'w', encoding='utf-8').write(summary + '\n')
    print(summary)
    print(f'\n[CALIB] 完成 → {outdir}')


if __name__ == '__main__':
    main()
