import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from .salt import SALT


class SALT_Prob(SALT):
    """P1 — 异方差高斯概率化长尾建模。

    在 SALT 的 TLSSM 输出端追加一个 prob_head，把每个边 (i,j) 的单值预测
    扩展为 (mu, log_sigma) 双通道，并用 Gaussian NLL 训练。

    相比 SALT 原版 MAE+MSE(Frobenius) 对称 L2 惩罚，NLL 让每条边拥有自己的
    不确定性 sigma：重尾/极端权重不再被对称 L2 过度惩罚，也不对不确定性建模。
    这直接针对 SALT 处理权重长尾的"经验 trick"短板（其用权重调 embedding 分布，
    无概率基础）。

    评估时返回 mu 作为确定性点估计，使 RMSE/MAE 与原 SALT 基线严格可比。
    """

    def __init__(self, args):
        super().__init__(args)
        self.d_model = args.TLSSM['d_model']
        # TLSSM 1-D 隐状态 -> (mu, log_sigma)
        self.prob_head = nn.Linear(self.d_model, 2)

    def forward(self, sup_list, feat_list, noise_list, pred_flag=True):
        N = feat_list[0].shape[0]
        window_size = len(sup_list)
        features = copy.deepcopy(feat_list)
        new_features = None
        for l in range(self.num_feat_proj):
            layer = self.feat_proj[l]
            new_features = []
            for x in features[:window_size + 1]:
                out = F.relu(layer(x))
                new_features.append(out)
            features = new_features

        embedding_inputs = []
        for t in range(window_size):
            concatenated = torch.cat((features[t], noise_list[t]), dim=1)
            embedding_inputs.append(concatenated)

        mpnn_outputs = []
        for t in range(window_size):
            sup = sup_list[t].coalesce()
            edge = sup.indices()
            adj = SparseTensor(row=edge[0], col=edge[1], value=sup.values())
            mpnn_out = self.WLElayer(embedding_inputs[t], adj, edge)
            mpnn_out = mpnn_out.squeeze(-1).contiguous()
            mpnn_out = SparseTensor(row=edge[0], col=edge[1], value=mpnn_out).to_dense().clone()
            mpnn_out = mpnn_out.reshape(N * N, -1).contiguous()
            mpnn_outputs.append(mpnn_out)

        # TLSSM + prob_head
        tlssm_outputs = []
        conv_state, ssm_state = self.tlssm.init_local_and_global_hidden(
            batch_size=N * N, max_seqlen=window_size, dtype=torch.float32)
        for t in range(window_size):
            tlssm_out, conv_state, ssm_state = self.tlssm(
                mpnn_outputs[t].unsqueeze(1), conv_state, ssm_state)
            # tlssm_out: (N*N, 1, d_model)
            params = self.prob_head(tlssm_out)        # (N*N, 1, 2)
            params = params.reshape(N, N, 2)          # (N,N,2) = [mu, log_sigma]
            tlssm_outputs.append(params)

        if pred_flag:
            # 评测用 mu 作点估计，输出与 SALT 原版同形状的 list[(N,N)]
            return [tlssm_outputs[-self.pred_size:][k][..., 0]
                    for k in range(self.pred_size)]
        else:
            return tlssm_outputs  # list[(N,N,2)]

    def causal_training_loss(self, adj_est_list, gnd_list, alpha, theta, beta=0.0, edge_masks=None):
        # adj_est_list[i]: (N,N,2) = (mu, log_sigma)
        # 与 SALT_INN 完全相同的训练目标（edge-masked 高斯 NLL + 可选 β·edge-MSE），
        # 以保证两概率模型差异【仅在于 σ 是否与 μ 共享 TLSSM】——这是 3-way 对比干净的前提。
        # 详见 SALT_INN.causal_training_loss 的注释（为何 edge-mask、β 为何默认 0）。
        win_size = len(adj_est_list)
        nll_fit_sum = 0.0
        nll_full_sum = 0.0
        mse_sum = 0.0
        for i in range(win_size):
            mu = adj_est_list[i][..., 0]
            log_sigma = adj_est_list[i][..., 1]
            sigma = torch.exp(log_sigma).clamp(min=1e-3, max=1e2)
            g = gnd_list[i]
            diff2 = (g - mu) ** 2
            nll_i = 0.5 * (diff2 / sigma ** 2 + 2.0 * log_sigma + math.log(2 * math.pi))
            nll_fit_i = 0.5 * (diff2 / sigma ** 2 + 2.0 * log_sigma)   # 去常数
            decay = (1 - theta) ** (win_size - i - 1)
            if edge_masks is not None:
                m = edge_masks[i]
                n_edges = m.sum().clamp(min=1.0)
                nll_fit_sum += decay * (nll_fit_i * m).sum() / n_edges
                nll_full_sum += decay * (nll_i * m).sum() / n_edges
                mse_sum     += decay * (diff2 * m).sum() / n_edges
            else:
                nll_fit_sum += decay * nll_fit_i.mean()
                nll_full_sum += decay * nll_i.mean()
                mse_sum     += decay * diff2.mean()
        nll_fit = nll_fit_sum / win_size
        nll_full = nll_full_sum / win_size
        mse = mse_sum / win_size
        loss = nll_fit + beta * mse
        self._loss_nll = nll_full      # 供 train.py [loss分拆] 打印（与 SALT_INN 对齐）
        self._loss_mse = mse
        self._loss_total = loss
        return loss

    @torch.no_grad()
    def eval_forward_dist(self, sup_list, feat_list, noise_list):
        """返回未来 pred_size 个快照的 (mu, log_sigma) 分布，供评测算 σ/校准/edge-NLL。

        与 SALT_INN.eval_forward_dist 同约定：forward(pred_flag=False) 产出全窗口
        list[(N,N,2)]，取末 pred_size 个，使 dist_list[k] 与 adj_est_list[k] 指向同目标快照。
        mu/log_sigma 处于归一空间（与训练目标 gnd_norm = g/max_thres+eye 一致）。
        """
        outs = self.forward(sup_list, feat_list, noise_list, pred_flag=False)
        return outs[-self.pred_size:]
