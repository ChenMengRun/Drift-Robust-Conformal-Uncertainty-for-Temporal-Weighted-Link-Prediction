import copy
import torch
import torch.nn.functional as F
from model.salt import SALT
from model.salt_edge import SALT_Edge
from model.modules.wle_dense import WeightedLinkEncoderDense


class DenseForwardMixin:
    """用 dense 邻接 + WeightedLinkEncoderDense 复刻 SALT.forward，绕开 torch_sparse。

    与 base SALT.forward 的唯一差异：把每个快照的稀疏邻接改为 dense 矩阵，
    并把 WLElayer 换成 dense 版（消息传递用 dense matmul）。其余（feat_proj、
    TLSSM、reshape、causal_training_loss）完全一致。
    """
    def _swap_wle(self, args):
        w = args.WLE
        self.WLElayer = WeightedLinkEncoderDense(
            in_channels=w['in_channels'],
            hidden_channels=w['hidden_channels'],
            out_channels=w['out_channels'],
            dropout=0.05, edrop=0.4, ln=True, use_xlin=True,
            tailact=True, twolayerlin=True, cndeg=-1, beta=1.0)

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
            # —— dense 邻接替代 SparseTensor ——
            adj = sup.to_dense()                 # [N, N]，元素=真实边权（与 base value=sup.values() 一致）
            mpnn_out = self.WLElayer(embedding_inputs[t], adj, edge)
            mpnn_out = mpnn_out.squeeze(-1).contiguous()   # [E]
            # 把每条边的 mpnn 输出 scatter 回 [N, N]（对应 base 的 SparseTensor(...).to_dense()）
            adj_est = torch.zeros(N, N, device=mpnn_out.device)
            adj_est[edge[0], edge[1]] = mpnn_out            # [E] -> [N, N]
            mpnn_out = adj_est.reshape(N * N, -1).contiguous()   # [N*N, 1]
            mpnn_outputs.append(mpnn_out)

        tlssm_outputs = []
        conv_state, ssm_state = self.tlssm.init_local_and_global_hidden(
            batch_size=N * N, max_seqlen=window_size, dtype=torch.float32)
        for t in range(window_size):
            tlssm_out, conv_state, ssm_state = self.tlssm(
                mpnn_outputs[t].unsqueeze(1), conv_state, ssm_state)
            tlssm_outputs.append(tlssm_out.reshape(N, N))

        if pred_flag:
            return tlssm_outputs[-self.pred_size:]
        else:
            return tlssm_outputs


class SALT_Dense(DenseForwardMixin, SALT):
    """dense 版 base SALT（确定性基线，密集 Frobenius MAE+MSE 训练目标）。"""
    def __init__(self, args):
        super().__init__(args)
        self._swap_wle(args)


class SALT_Edge_Dense(DenseForwardMixin, SALT_Edge):
    """dense 版 edge 加权 SALT（继承 SALT_Edge 的 edge-masked MAE+MSE loss）。"""
    def __init__(self, args):
        super().__init__(args)
        self._swap_wle(args)
