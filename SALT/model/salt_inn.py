import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from .salt import SALT


class SALT_INN(SALT):
    """Route B：以 INN 替换 WLE，输出边权条件分布 (mu, log_sigma)，复用 P1 的 NLL。

    原 SALT 链路：  WLE(out_channels=1) -> dense(N,N,1) -> TLSSM(d_model=1) -> (N,N)
    本变体链路：
        INN(out_channels=2) -> dense(N,N,2) 拆成两条：
          mu_raw  (E,1) -> dense(N,N,1) -> TLSSM(d_model=1) -> mu   （时序动力学，点预测）
          log_sigma_raw (E,1) -> dense(N,N,1) -> 直接作为 log_sigma  （静态边级，不过 TLSSM）

    关键修复（2026-09-27）：σ 通道不再经过共享的 TLSSM。
    之前 μ 与 log_sigma 共用同一套 TLSSM 权重，导致 σ 被 μ 的大小拽着走、
    学不到残差结构，最终 σ 退化为≈常数且 corr(|error|,σ)<0（校准反面）。
    现在 σ 由 INN 的结构感知输出直接给出（每个快照独立），与 μ 的时序演化解耦。
    """

    def __init__(self, args):
        # 强制 INN 输出双通道（mu_raw, log_sigma_raw）
        args.WLE['out_channels'] = 2
        # 仅 μ 过 TLSSM；σ 走 INN 静态输出，故 TLSSM 仍用 d_model=1
        args.TLSSM['d_model'] = 1
        super().__init__(args)

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

        # 边级 INN 输出 (E,2) = (mu_raw, log_sigma_raw)，每个快照独立计算
        mpnn_mu, mpnn_logsig = [], []
        for t in range(window_size):
            sup = sup_list[t].coalesce()
            edge = sup.indices()
            adj = SparseTensor(row=edge[0], col=edge[1], value=sup.values())
            mpnn_out = self.WLElayer(embedding_inputs[t], adj, edge)   # (E, 2)
            mu_raw = mpnn_out[..., 0:1]
            logsig_raw = mpnn_out[..., 1:2]
            # 边级 (E,1) -> 稠密 (N,N,1)
            d_mu = torch.zeros(N, N, 1, device=mpnn_out.device,
                               dtype=mpnn_out.dtype)
            d_mu[edge[0], edge[1]] = mu_raw
            d_logsig = torch.zeros(N, N, 1, device=mpnn_out.device,
                                  dtype=mpnn_out.dtype)
            d_logsig[edge[0], edge[1]] = logsig_raw
            mpnn_mu.append(d_mu.reshape(N * N, 1).contiguous())
            mpnn_logsig.append(d_logsig.reshape(N * N, 1).contiguous())

        # μ 走 TLSSM（d_model=1，时序动力学）；σ 不走 TLSSM（静态边级不确定性）
        tlssm_mu = []
        conv_state, ssm_state = self.tlssm.init_local_and_global_hidden(
            batch_size=N * N, max_seqlen=window_size, dtype=torch.float32)
        for t in range(window_size):
            tlssm_out, conv_state, ssm_state = self.tlssm(
                mpnn_mu[t].unsqueeze(1), conv_state, ssm_state)   # (N*N,1,1)
            tlssm_mu.append(tlssm_out.reshape(N, N, 1))

        # 合并 (N,N,2) = (mu, log_sigma)；σ 取对应快照的静态 log_sigma
        out_all = []
        for t in range(window_size):
            d_logsig_t = mpnn_logsig[t].reshape(N, N, 1)
            out_all.append(torch.cat([tlssm_mu[t], d_logsig_t], dim=-1))

        if pred_flag:
            # 评测用 mu 作点估计，输出与 SALT 原版同形状的 list[(N,N)]
            return [out_all[-self.pred_size:][k][..., 0]
                    for k in range(self.pred_size)]
        else:
            return out_all  # list[(N,N,2)]

    def causal_training_loss(self, adj_est_list, gnd_list, alpha, theta, beta=0.0, edge_masks=None):
        # adj_est_list[i]: (N,N,2) = (mu, log_sigma)；gnd_list[i]: (N,N) 归一空间
        # 复合损失：loss = NLL_fit + β · edge-MSE
        #
        # ★ 关键修复（2026-09-28）：训练 NLL 改为【仅在真实边上】求 mean，而非全 N×N 网格。
        #   原实现在全网格（含 99% 非边，其 GT=0）上求 NLL：模型可把 σ 全局放大，
        #   廉价地"吸收"非边上的 μ 漂移 → NLL 看似下降但 μ 在边上变差 → Test RMSE 随训练恶化
        #   （即此前观察到的 "NLL↓ / RMSE↑" 反相关）。edge-mask 后 NLL 只关心边，
        #   σ 无法在非边上作弊，μ 自然贴合边 → 与评测的 edge-RMSE / edge-NLL 自洽。
        #
        #   - NLL_fit：去掉与参数无关的 0.5·log(2π) 常数后的高斯 NLL（仅优化相关部分）；
        #     含常数的完整 NLL(nll_full) 仅用于 logging / 与 val NLL 口径对照。
        #   - β·edge-MSE：仅在边上约束 μ 贴合目标。注意——β>0 会牺牲校准
        #     （最优处 μ=g 时 NLL 仍推 σ→0，σ 退化为常数，路线 B 的"概率"意义丧失），
        #     故默认 β=0（纯 edge-NLL）即同时获得好 edge-RMSE 与校准 σ；β 仅作实验性杠杆。
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
        # 记录分量供 train.py 打印（beta=0 时 nll_full 即原口径）
        self._loss_nll = nll_full
        self._loss_mse = mse
        self._loss_total = loss
        return loss

    @torch.no_grad()
    def eval_forward_dist(self, sup_list, feat_list, noise_list):
        """返回未来 pred_size 个快照的 (mu, log_sigma) 分布，供 train.py 的 evaluate 计算 NLL。

        与 pred_flag=True 的 RMSE 流水线严格对齐（同样的 [−pred_size:] 取窗），
        因此 dist_list[k] 与 adj_est_list[k] 指向同一目标快照 k。
        mu/log_sigma 处于归一空间（与训练目标 gnd_norm 一致）。
        """
        outs = self.forward(sup_list, feat_list, noise_list, pred_flag=False)  # list[(N,N,2)] x window_size
        return outs[-self.pred_size:]
