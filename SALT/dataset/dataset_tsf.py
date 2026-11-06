"""DatasetTSF：在 base `Dataset` 之上，把 TSF 时序结构特征拼进 `feat_list`（opt-in）。

设计要点
    · **不改 base**：只继承 + 包一层 `get_win_data`，把返回的 `feat_list[t]`
      在特征维上拼接 TSF。base `dataset.py` 一行不动，官方复现不受影响。
    · **索引对齐**：TSF 用全局节点索引（data_level=1 与 =2 的 `feat_tnr` 都是
      全局索引：=2 分支把 local 特征写在 `re_map[i]`=global 行），与 `edge_seq`
      的节点 id 一致，因此两种数据格式都直接可用。
    · **零外部数据**：TSF 只由 `edge_seq` 历史算出，任何数据集都能开。
    · **对照臂**（config `tsf_mode`）：
        real    —— 真 TSF（默认）
        random  —— 同形状/同尺度的 N(0,1) 随机特征（固定种子），用来排除
                   “任何 12 个新特征列都会带来同样增益”这一混淆
        shuffle —— 真 TSF 但每个快照内对节点维做随机置换（破坏节点对应关系，
                   保留 TSF 的边缘分布与时间结构）
"""
import numpy as np
import torch

from dataset.dataset import Dataset
from dataset.tsf_features import build_tsf_sequence, TSF_DIM


class DatasetTSF(Dataset):
    def __init__(self, params, base_dataset_path):
        super().__init__(params, base_dataset_path)

        # 记录"基础特征维度"（不含 TSF），供 trainer 自动推导 feat_proj 首层维度
        if getattr(self, 'data_level', 1) == 2:
            self.base_feat_dim = int(self.feat_lcl_seq[0].shape[1])
        else:
            self.base_feat_dim = int(self.feat_tnr.shape[1])

        mode = str(getattr(params, 'tsf_mode', 'real') or 'real').lower()
        self.tsf_mode = mode

        tsf = build_tsf_sequence(self.edge_seq, self.num_nodes, self.max_thres,
                                 train_end=self.num_train_snaps)
        if mode == 'random':
            _rs = np.random.RandomState(1234)
            tsf = _rs.randn(*tsf.shape).astype(np.float32)
        elif mode == 'shuffle':
            _rs = np.random.RandomState(1234)
            _T, _N, _D = tsf.shape
            _perm = np.argsort(_rs.rand(_T, _N), axis=1)          # 每个快照内独立置换节点
            tsf = np.take_along_axis(tsf, _perm[:, :, None], axis=1).astype(np.float32)
        self.tsf_seq = tsf

        print(f'[TSF] 时序结构特征已构造：mode={mode}  T={self.tsf_seq.shape[0]}  '
              f'节点={self.tsf_seq.shape[1]}  TSF维={self.tsf_seq.shape[2]}  '
              f'基础特征维={self.base_feat_dim} → 拼接后={self.base_feat_dim + TSF_DIM}  '
              f'（训练段 T<{self.num_train_snaps} 逐维 z-score 标准化）')

    def get_win_data(self, start, end):
        sup_list, noise_list, feat_list = super().get_win_data(start, end)
        for idx, t in enumerate(range(start, end)):
            tsf = torch.FloatTensor(self.tsf_seq[t]).to(feat_list[idx].device)
            feat_list[idx] = torch.cat([feat_list[idx], tsf], dim=1)
        return sup_list, noise_list, feat_list
