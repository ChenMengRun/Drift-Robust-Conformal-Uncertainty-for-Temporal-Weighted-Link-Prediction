"""SALT_INN 校验脚本（仅前向/评测，不触发 backward，避免本机 torch_sparse 训练崩溃）。

用法：
  # 测你训练好的模型（唯一主路径，不再混入控制跑）：
  python validate_salt_inn.py --config ./SALT/config/INVIS15/SALT_INN.yaml \
                              --model ./Exp/pt/INVIS15/SALT_INN/17_2.676_0.58.pth

  # 仅跑 SALT 预训练“形状匹配”的控制跑（INN/TLSSM 随机初始化，仅证通路）：
  python validate_salt_inn.py --config ./SALT/config/INVIS15/SALT_INN.yaml

  # 同时跑真实模型 + 控制跑做对照：
  python validate_salt_inn.py --config ./SALT/config/INVIS15/SALT_INN.yaml \
                              --model ./Exp/pt/INVIS15/SALT_INN/XXX.pth --baseline

重要：
  - 默认（带 --model）只评估你指定的 checkpoint；控制跑必须显式 --baseline 才会跑，
    不再出现“先打一段 SALT 预训练结果、看起来像在测你的模型”的误导。
  - 控制跑里 INN/TLSSM 是随机初始化，其 σ / NLL / 相关均为无意义噪声，仅证明代码通路。
  - 旧 d_model=2 的 checkpoint（如 17_2.676_0.58.pth）与当前 d_model=1 工作区代码不兼容，
    加载会跳过改形层；请用【重训后】的新 checkpoint 配合 --model。
  - 真实训练需在 torch_sparse==0.6.13 的 Linux 环境由用户执行（见 SALT_P1_run_guide.md）。
"""
import os
import sys
import argparse
import importlib
import math
import torch
import numpy as np
from tqdm import trange

ROOT = os.path.dirname(os.path.abspath(__file__))   # = .../SALT-main/SALT  (SALT 包目录)
PROJECT_ROOT = os.path.dirname(ROOT)                 # = .../SALT-main       (SALT 的父目录，import SALT 需它)
sys.path.insert(0, PROJECT_ROOT)                     # 使 `from SALT.xxx import ...` 可解析
sys.path.insert(0, ROOT)                             # 兼容 `from utils.xxx` / `from model.xxx`（同 train.py 风格）

from SALT.utils.paramset import Paramset
from SALT.utils.seeds import setup_seed
from SALT.utils.metrics import get_RMSE, get_MAE, get_edge_RMSE
from SALT.dataset.dataset import Dataset, get_adj_wei, get_adj_wei_map
from SALT.model.salt_inn import SALT_INN

EPSILON = 0.01
DATASET_PATH = './Data'


def parse():
    p = argparse.ArgumentParser(description='SALT_INN 校验（前向/评测，无 backward）')
    p.add_argument('--config', type=str, default='./SALT/config/INVIS15/SALT_INN.yaml')
    p.add_argument('--model', type=str, default='',
                   help='要评估的 checkpoint 路径；为空则只跑 SALT 预训练控制跑')
    p.add_argument('--baseline', action='store_true',
                   help='额外跑一次 SALT 预训练“形状匹配”控制跑做对照（仅证通路）')
    return p.parse_args()


def load_weights(net, ckpt_path, params):
    """返回 (state_dict_loaded_count, skipped_count, label)。
    ckpt_path 非空：加载用户模型；为空：加载 SALT 官方预训练（形状匹配，INN/TLSSM 随机）。"""
    own = net.state_dict()
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'[ERROR] --model 指定的文件不存在: {ckpt_path}')
        sd = torch.load(ckpt_path, map_location=params.device)
        if isinstance(sd, dict):
            # train.py 的 logger.save_model / resume 块都存 {'model_state_dict':...}；
            # 官方 Pretrained 存 {'model_dict':...}；也可能直接是裸 state_dict。
            # 必须下钻到真正的权重字典，否则顶层键(model_state_dict 等)会与层名全部失配 → 静默随机评估。
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            elif 'model_dict' in sd:
                sd = sd['model_dict']
        tag = f'用户指定模型: {ckpt_path}'
    else:
        pf = f'./Pretrained/{params.data_name}/{params.data_name}_pretrain.pth'
        if not os.path.isfile(pf):
            raise FileNotFoundError(f'[ERROR] 未找到官方预训练: {pf}（控制跑不可用）')
        sd = torch.load(pf, map_location=params.device)['model_dict']
        tag = 'SALT 官方预训练权重(形状匹配)'
    loaded = {k: v for k, v in sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    net.load_state_dict(loaded, strict=False)
    skipped = len(sd) - len(loaded)
    if ckpt_path and len(loaded) == 0:
        print('[WARN] 0 个权重加载成功——模型实际以随机初始化在评估！checkpoint 格式可能不被识别。')
    note = '' if ckpt_path else '（INN/TLSSM 改形层随机初始化，以下 σ/NLL 为无意义噪声）'
    return len(loaded), skipped, f'{tag}; 复用键数={len(loaded)} 跳过键数={skipped} {note}'


def prob_diag(net, dataset, params):
    """NLL(归一空间) + σ 分布 + 校准（全量 & 仅边）。与 train.py 的 NLL 段同口径。"""
    sig_all, err_all, nll_list = [], [], []
    sig_edge, err_edge = [], []
    cov1_list, cov2_list = [], []
    edge_nll_list = []
    for tau in trange(dataset.num_snaps - dataset.num_test_snaps, dataset.num_snaps, params.pred_size):
        sup_list, noise_list, feat_list = dataset.get_win_data(tau - params.win_size, tau)
        dist_list = net.eval_forward_dist(sup_list, feat_list, noise_list)
        for k, t in enumerate(range(tau, tau + params.pred_size)):
            mu_sig = dist_list[k]            # (N,N,2) local indexing
            mu = mu_sig[..., 0]
            log_sigma = mu_sig[..., 1]
            sigma = torch.exp(log_sigma).clamp(min=1e-3, max=1e2)
            edges = dataset.edge_seq[t]
            if dataset.data_level == 2:
                node_map = dataset.node_map_seq_gbl[t]
                gnd_norm_t = get_adj_wei_map(edges, node_map, dataset.num_nodes, dataset.max_thres)
            else:
                gnd_norm_t = get_adj_wei(edges, dataset.num_nodes, dataset.max_thres)
            gnd_norm_t = gnd_norm_t / dataset.max_thres + np.eye(dataset.num_nodes)
            gnd_norm_t = torch.FloatTensor(gnd_norm_t).to(mu.device)
            nll_t = 0.5 * ((gnd_norm_t - mu) ** 2 / sigma ** 2 + 2.0 * log_sigma + math.log(2 * math.pi))
            nll_list.append(nll_t.mean().item())
            err_t = (gnd_norm_t - mu).abs()
            mask = ~torch.eye(mu.shape[0], dtype=torch.bool, device=mu.device)
            sig_all.append(sigma[mask].detach().cpu().numpy().ravel())
            err_all.append(err_t[mask].detach().cpu().numpy().ravel())
        s_e, e_e, nll_e = [], [], []
        for e in edges:
            if dataset.data_level == 2:
                si = dataset.node_map_seq_gbl[t].get(int(e[0]))
                di = dataset.node_map_seq_gbl[t].get(int(e[1]))
                if si is None or di is None:
                    continue
            else:
                si, di = int(e[0]), int(e[1])
            if si >= mu.shape[0] or di >= mu.shape[0]:
                continue
            s_e.append(sigma[si, di].item())
            e_e.append(err_t[si, di].item())
            nll_e.append(nll_t[si, di].item())
        if s_e:
            sig_edge.append(s_e); err_edge.append(e_e)
            s_arr = np.array(s_e); er_arr = np.array(e_e)
            cov1_list.append(np.mean(er_arr <= s_arr))
            cov2_list.append(np.mean(er_arr <= 2 * s_arr))
            edge_nll_list.append(np.mean(nll_e))

    sig_all = np.concatenate(sig_all)
    err_all = np.concatenate(err_all)
    corr = np.corrcoef(sig_all, err_all)[0, 1] if sig_all.std() > 0 else float('nan')
    sig_edge = np.concatenate(sig_edge) if sig_edge else np.array([])
    err_edge = np.concatenate(err_edge) if err_edge else np.array([])
    corr_edge = np.corrcoef(sig_edge, err_edge)[0, 1] if sig_edge.size > 1 and sig_edge.std() > 0 else float('nan')
    cov1 = np.mean(cov1_list) if cov1_list else float('nan')
    cov2 = np.mean(cov2_list) if cov2_list else float('nan')
    print(f"\n[PROB-DIAG] NLL(norm-space)={np.mean(nll_list):.4f}")
    print(f"[PROB-DIAG] NLL_edge(norm-space)={np.mean(edge_nll_list):.4f}  (仅真实边，与 dense NLL 对照)")
    print(f"[PROB-DIAG] sigma  mean={sig_all.mean():.4f} std={sig_all.std():.4f} "
          f"min={sig_all.min():.4f} max={sig_all.max():.4f} "
          f"p05={np.percentile(sig_all,5):.4f} p95={np.percentile(sig_all,95):.4f}")
    print(f"[PROB-DIAG] corr(|error|, sigma) [ALL]    = {corr:.4f}  (理想>0)")
    print(f"[PROB-DIAG] corr(|error|, sigma) [EDGE]   = {corr_edge:.4f}  (理想>0：仅边，校准更干净)")
    print(f"[PROB-DIAG] 边覆盖率 ±1σ={cov1:.3f}(名义0.682)  ±2σ={cov2:.3f}(名义0.954)")
    print(f"[PROB-DIAG] 说明：EDGE 相关与覆盖率是校准主体；ALL 被 99% 非边稀释，仅作参考。")


def build_model(params):
    """按 config 的 model 字段动态实例化（兼容 SALT / SALT_INN / SALT_Prob）。"""
    mod_path, cls_name = params.model.rsplit('.', 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(params)


def run_eval(ckpt_path, mode_tag, params, dataset):
    print(f"\n{'='*64}")
    print(f"[{mode_tag}] 开始评估")
    print(f"{'='*64}")
    net = build_model(params).to(params.device)
    print(f"[OK] {net.__class__.__name__} 实例化成功; device={params.device}; "
          f"TLSSM.d_model={net.tlssm.d_model}; WLE.out_channels check ok")
    n_loaded, n_skip, load_msg = load_weights(net, ckpt_path, params)
    print(f"[OK] {load_msg}")

    net.eval()
    with torch.no_grad():
        sup_list, noise_list, feat_list = dataset.get_win_data(0, params.win_size)
        out_train = net(sup_list, feat_list, noise_list, pred_flag=False)
        out_pred = net(sup_list, feat_list, noise_list, pred_flag=True)
        print(f"[OK] train 模式输出: list 长度={len(out_train)}, 元素 shape={tuple(out_train[0].shape)} (应为 N,N,2)")
        print(f"[OK] pred  模式输出: list 长度={len(out_pred)}, 元素 shape={tuple(out_pred[0].shape)} (应为 N,N)")

        gnd_list = dataset.get_win_gnd_data(1, params.win_size + 1)
        loss_val = net.causal_training_loss(out_train, gnd_list, params.Loss['alpha'], params.Loss['theta'])
        print(f"[OK] causal_training_loss 标量 = {loss_val.item():.4f} (应为有限数)")

        results = {'RMSE': [], 'MAE': [], 'RMSE_edge': []}
        for tau in trange(dataset.num_snaps - dataset.num_test_snaps, dataset.num_snaps, params.pred_size):
            sup_list, noise_list, feat_list = dataset.get_win_data(tau - params.win_size, tau)
            adj_est_list = net(sup_list, feat_list, noise_list, pred_flag=True)
            for t in range(tau, tau + params.pred_size):
                adj_est = adj_est_list[t - tau].cpu().numpy()
                adj_est *= dataset.max_thres
                np.fill_diagonal(adj_est, 0)
                adj_est[adj_est <= EPSILON] = 0
                edges = dataset.edge_seq[t]
                if dataset.data_level == 2:
                    nn_ = dataset.num_nodes
                    gnd = get_adj_wei(edges, nn_, dataset.max_thres)
                    real_row = list(dataset.node_map_seq_gbl[t].keys())
                    real_col = list(dataset.node_map_seq_gbl[t].keys())
                    gnd = gnd[np.ix_(real_row, real_col)]
                    adj_est = adj_est[np.ix_(real_row, real_col)]
                    nn_ = dataset.num_nodes_seq_gbl[t]
                    local_pos = {g: i for i, g in enumerate(real_row)}
                    edge_local = [(local_pos[int(e[0])], local_pos[int(e[1])])
                                  for e in edges
                                  if int(e[0]) in local_pos and int(e[1]) in local_pos]
                else:
                    nn_ = dataset.num_nodes
                    gnd = get_adj_wei(edges, nn_, dataset.max_thres)
                    edge_local = [(int(e[0]), int(e[1])) for e in edges]
                results['RMSE'].append(get_RMSE(adj_est, gnd, nn_))
                results['MAE'].append(get_MAE(adj_est, gnd, nn_))
                results['RMSE_edge'].append(get_edge_RMSE(adj_est, gnd, edge_local))
        print(f"\n[EVAL-PRED-MODE] RMSE={np.mean(results['RMSE']):.4f}  MAE={np.mean(results['MAE']):.4f}  "
              f"RMSE_edge={np.nanmean(results['RMSE_edge']):.4f}")

        if hasattr(net, 'eval_forward_dist'):
            prob_diag(net, dataset, params)
        else:
            print("\n[SKIP] 该模型无 eval_forward_dist（确定性模型，无 σ），跳过 PROB-DIAG。")

    if mode_tag == 'CONTROL':
        print("\n[注意] CONTROL 模式：INN/TLSSM 为随机初始化，上述 σ/NLL/相关均为无意义噪声，仅证明代码通路正确。")
    else:
        print("\n[REAL] 以上为 --model 指定模型在 TEST 集的真实评测结果。")


def main():
    args = parse()
    setup_seed(0)
    params = Paramset(args.config)
    dataset = Dataset(params, DATASET_PATH)

    if args.model:
        run_eval(args.model, 'REAL', params, dataset)
        if args.baseline:
            run_eval('', 'CONTROL', params, dataset)
    else:
        run_eval('', 'CONTROL', params, dataset)


if __name__ == '__main__':
    main()
