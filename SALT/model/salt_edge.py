# SALT_Edge：继承原版确定性 SALT，仅把训练目标从「密集 MAE+MSE」改为
# 「edge 加权 MAE+MSE」，直接对齐评估指标 edge-RMSE。
# 不修改 model/salt.py —— 原版仍可复现 9.88（base SALT 基线）。
import torch
from model.salt import SALT


class SALT_Edge(SALT):
    def causal_training_loss(self, adj_est_list, gnd_list, alpha, theta, beta=0.0, edge_masks=None):
        """edge 加权 MAE+MSE。
        仅在真实边位置 (edge_masks==1) 计算 |adj_est-gnd| 与 (adj_est-gnd)^2，
        对齐评估指标 edge-RMSE，避免梯度淹没在非边零元重建上。
        decay/alpha 加权方式与原密集版一致。
        """
        win_size = len(adj_est_list)
        mae_loss = 0.0
        mse_loss = 0.0
        for i in range(win_size):
            adj_est = adj_est_list[i]
            gnd = gnd_list[i]
            decay = (1 - theta) ** (win_size - i - 1)
            if edge_masks is not None:
                mask = edge_masks[i].to(adj_est.device)
                diff = (adj_est - gnd) * mask
            else:
                diff = adj_est - gnd
            mae_loss += decay * alpha * torch.norm(diff, p=1)
            mse_loss += decay * alpha * torch.norm(diff, p='fro') ** 2
        loss = mae_loss + mse_loss
        self._loss_nll = None
        self._loss_mse = mse_loss
        self._loss_total = loss
        return loss
