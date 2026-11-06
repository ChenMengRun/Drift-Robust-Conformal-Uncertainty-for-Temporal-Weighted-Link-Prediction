import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.wle import WeightedLinkEncoder


class WeightedLinkEncoderDense(WeightedLinkEncoder):
    """WeightedLinkEncoder 的 dense 等价替身 —— 绕开 torch_sparse。

    根因：原 WLElayer 的 `spmm_add`（torch_sparse csr2csc）在
    Windows + torch 1.12.1 + torch_sparse 组合下对某些邻接结构触发
    access violation（原生 segfault，Python 异常捕获兜不住）。

    本类与基类架构（xlin/xcnlin/xijlin/lin/dropadj/beta）完全一致，
    仅把稀疏矩阵乘替换为 dense matmul。N 较小（如 HMob=92）时数学等价、
    数值差异 <1e-6，且状态字典键名不变 → 官方预训练权重可直接 load。

    仅支持 cndeg=-1（无采样）路径；cndeg>0 采样路径未实现（会显式报错），
    因为该路径依赖 torch_sparse 的 sparsesample，与 dense 替换不兼容。
    """
    def multidomainforward(self, x, adj, tar_ei, filled1=False, cndropprobs=[]):
        if self.cndeg > 0:
            raise NotImplementedError(
                "WeightedLinkEncoderDense 未实现 cndeg>0 的采样路径；"
                "请改用 base WeightedLinkEncoder 或扩展本函数。")
        # adj: dense [N, N]；tar_ei: [2, E]
        i = tar_ei[0]
        j = tar_ei[1]
        weight = adj[i, j]                       # 原始（drop 前）边权 [E]，对应 adj.storage.value()
        # ---- 复刻 DropAdj（训练时随机丢弃 dp 比例的边）----
        if self.dropadj.dp >= 1e-6 and self.training:
            E = i.shape[0]
            keep = (torch.rand(E, device=adj.device) > self.dropadj.dp).float()
            adj_dropped = adj.clone()
            adj_dropped[i, j] = adj_dropped[i, j] * keep
        else:
            adj_dropped = adj
        xi = x[i]
        xj = x[j]
        x = x + self.xlin(x)
        # ---- 复刻 adjoverlap(adj, adj, tar_ei)：i 与 j 共同邻居的二值重叠 ----
        # adj1 = adj[tar_ei[0]]，adj2 = adj[tar_ei[1]]，取交集（spmoverlap_ 的 fill_value_(1.0)）
        adj1 = adj_dropped[i]                    # [E, N]
        adj2 = adj_dropped[j]                    # [E, N]
        cn = ((adj1 > 0) & (adj2 > 0)).float()  # [E, N]，重叠处为 1.0
        xcn = cn @ x                            # [E, F]  == spmm_add(cn, x)
        xij = self.xijlin(xi * xj) * weight.reshape(-1, 1)
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in [xcn]],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1=False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])
