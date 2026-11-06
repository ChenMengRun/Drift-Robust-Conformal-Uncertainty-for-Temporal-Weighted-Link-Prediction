"""TSF: Temporal Structural Features —— 纯自监督的节点级时序结构特征。

动机（可写进论文的问题陈述）：
    SALT 的节点特征在时间上是**静态**的——
      · HMob / T-Drive：`load_topology()` 只给 `pos_emb`（纯位置编码，零信息量）；
      · 其余数据集：`feat.npy` 是 (num_nodes, F) 的**固定矩阵**，逐快照只做行选取。
    也就是说，模型没有任何"节点级时序上下文"——唯一的时序信号来自邻接序列本身。

    本模块仅从 `edge_seq` 的历史中构造这类上下文，**不需要任何外部数据**，
    因此天然可迁移到所有数据集（这也是"覆盖多数据集"的前提）。

对每个快照 t、每个节点 i，输出 TSF_DIM(=12) 维：
    log1p(度)                       活动规模
    log1p(加权度 × N)               活动强度（×N 抑制节点数差异）
    (d(t) − d(t−1)) / N             度变化节点比例（跨数据集尺度无关）
    (d(t) − d(t−1)) / (d(t)+d(t−1)+1)  相对增长（有界 [−1,1]）
    log1p(过去 {1,2,4,8} 快照平均度) 多尺度活动（4 维）
    邻居集合 Jaccard 距离           邻居更替（churn）
    log1p(邻居平均权 × N)           邻域强度
    是否活跃（本快照有边）          活跃指示
    快照内度排名分位                尺度无关的相对地位

所有分量都做了尺度归一（×N / 分位 / log1p），以保证跨数据集通用。
仅使用 ≤ t 的快照（输入窗口本就给定 ≤ t 的图），无标签泄漏。
"""
import numpy as np

FEATURE_NAMES = [
    "log_deg", "log_wdeg", "deg_delta", "deg_growth",
    "log_act_1", "log_act_2", "log_act_4", "log_act_8",
    "nbr_churn", "log_nbr_wmean", "is_active", "deg_rank",
]
TSF_DIM = len(FEATURE_NAMES)


def _snapshot_stats(edges, num_nodes, max_thres):
    """从单快照边表得到 度 / 归一加权度 / 归一邻居权和 / 邻居集合。"""
    deg = np.zeros(num_nodes, dtype=np.float64)
    wdeg = np.zeros(num_nodes, dtype=np.float64)
    nwsum = np.zeros(num_nodes, dtype=np.float64)
    nbrs = [set() for _ in range(num_nodes)]
    inv = 1.0 / max(float(max_thres), 1e-9)
    for e in edges:
        u = int(e[0]); v = int(e[1]); w = float(e[2])
        w *= inv
        deg[u] += 1.0; deg[v] += 1.0
        wdeg[u] += w;  wdeg[v] += w
        nwsum[u] += w; nwsum[v] += w
        nbrs[u].add(v); nbrs[v].add(u)
    return deg, wdeg, nwsum, nbrs


def build_tsf_sequence(edge_seq, num_nodes, max_thres, scales=(1, 2, 4, 8), train_end=None):
    """返回 float32 数组 [T, num_nodes, TSF_DIM]。

    train_end: 若给出（=训练段快照数），用**仅训练段**的均值/标准差对每一维做
    z-score 标准化并裁剪到 [-10,10]。目的：让 TSF 与 base 特征（sin/cos ∈[-1,1]）
    量纲可比，消除"新特征初始扰动过大"（实测 HMob epoch0 Test 由 13.30 被推到 53.68）。
    统计量只取自训练段 ⇒ 无测试泄漏。
    """
    T = len(edge_seq)
    deg = np.zeros((T, num_nodes)); wdeg = np.zeros((T, num_nodes))
    nw = np.zeros((T, num_nodes)); nbrs = [None] * T
    for t in range(T):
        d, w, n, nb = _snapshot_stats(edge_seq[t], num_nodes, max_thres)
        deg[t] = d; wdeg[t] = w; nw[t] = n; nbrs[t] = nb

    inv_n = 1.0 / max(num_nodes - 1, 1)
    inv_nn = 1.0 / max(num_nodes, 1)
    out = np.zeros((T, num_nodes, TSF_DIM), dtype=np.float32)
    zero = np.zeros(num_nodes, dtype=np.float64)
    for t in range(T):
        d = deg[t]
        dp = deg[t - 1] if t > 0 else zero
        cols = [
            np.log1p(d),
            np.log1p(wdeg[t] * num_nodes),
            (d - dp) * inv_nn,          # 归一为"度发生变化的节点比例"，跨数据集尺度无关
            (d - dp) / (d + dp + 1.0),  # 相对增长，有界 [-1, 1]
        ]
        for k in scales:
            cols.append(np.log1p(deg[max(0, t - k):t + 1].mean(axis=0)))
        if t > 0:
            jac = np.zeros(num_nodes, dtype=np.float64)
            prev = nbrs[t - 1]
            cur = nbrs[t]
            for i in range(num_nodes):
                a = cur[i]; b = prev[i]
                u = a | b
                if u:
                    jac[i] = 1.0 - len(a & b) / len(u)
            cols.append(jac)
        else:
            cols.append(zero.copy())
        cols.append(np.log1p(nw[t] * num_nodes))
        cols.append((d > 0).astype(np.float64))
        cols.append(np.argsort(np.argsort(d)).astype(np.float64) * inv_n)
        out[t] = np.stack(cols, axis=1)

    if train_end is not None and 0 < train_end <= T:
        seg = out[:train_end].reshape(-1, TSF_DIM).astype(np.float64)
        mu = seg.mean(axis=0)
        sd = seg.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out = ((out.astype(np.float64) - mu) / sd)
        out = np.clip(out, -10.0, 10.0)

    out = out.astype(np.float32)
    assert out.shape[-1] == TSF_DIM, (out.shape, TSF_DIM)
    return out
