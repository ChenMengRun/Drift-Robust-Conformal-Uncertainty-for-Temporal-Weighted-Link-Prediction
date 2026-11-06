import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_add
from typing import Final, Iterable, List, Tuple

from .wle import DropAdj, adjoverlap


class INNWeightedLinkEncoder(nn.Module):
    """条件分布式 WLE 替代（route B：INN 替换 WLE）。

    接口与 WLE 完全一致: forward(x, adj, tar_ei, filled1=False) -> (E, 2)。
    与 WLE 的唯一区别：输出边权条件高斯参数 (mu, log_sigma)，而非单值点估计。

    这里把"INN"落地为「边权条件分布估计器」：给定结构上下文
    (common-neighbor 聚合 + 端点特征 + 边权) 预测 p(weight | structure) 的
    (mu, sigma)。sigma 通道经 exp 得到正方差，喂给 P1(SALT_Prob) 的异方差
    Gaussian NLL。这样既保留了 WLE 的结构感知聚合，又把单值输出升级为分布，
    与概率化训练主线直接衔接。

    诚实声明：这不是严格的可逆双射网络 (normalizing flow)。其价值在于把 WLE 的
    单值输出升级为概率分布并接到 NLL 训练，而非可反演性本身。若需严格可逆流，
    应另写 RealNVP 式耦合层作用于权重隐变量，但在本回归角色中收益有限，故此处
    用条件高斯参数化实现"INN 思维"（密度估计而非点估计）。
    """
    cndeg: Final[int]

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()
        assert out_channels == 2, \
            "INNWeightedLinkEncoder requires out_channels=2 (mu, log_sigma)"
        self.register_parameter("beta", nn.Parameter(beta * torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        # sigma 通道 bias 初始化为 -1 -> 初始 sigma = exp(-1) ~= 0.37，
        # 避免训练初期 sigma 爆炸导致 NLL 数值不稳。
        with torch.no_grad():
            self.lin[-1].bias[1] = -1.0

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        weight = adj.storage.value()
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj) * weight.reshape(-1, 1)

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs  # (E, 2) = [mu, log_sigma]

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# 别名：使 salt.py 的 dynamic_import(lib_path=..., class_name="WeightedLinkEncoder")
# 无需改动即可加载 INN（只要 WLE.model_class 指向 model.modules.inn）。
class WeightedLinkEncoder(INNWeightedLinkEncoder):
    pass
