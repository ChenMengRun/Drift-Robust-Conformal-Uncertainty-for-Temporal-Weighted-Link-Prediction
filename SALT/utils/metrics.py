import numpy as np

def get_RMSE(adj_est, gnd, num_nodes):
    f_norm = np.linalg.norm(gnd-adj_est, ord='fro')**2
    RMSE = np.sqrt(f_norm/(num_nodes*num_nodes))
    return RMSE


def get_MAE(adj_est, gnd, num_nodes):
    MAE = np.sum(np.abs(gnd-adj_est))/(num_nodes*num_nodes)
    return MAE


def get_edge_RMSE(adj_est, gnd, edge_local):
    """仅真实边上的 RMSE（原始权重空间），与 get_RMSE 同口径但避开 99% 非边零元。

    edge_local: list of (i, j) 局部索引（已按 data_level==2 的 node_map 重映射）。
    返回 NaN 若 edge_local 为空（该快照无真边）。
    """
    if not edge_local:
        return float('nan')
    pred = np.array([adj_est[i, j] for (i, j) in edge_local], dtype=np.float64)
    truth = np.array([gnd[i, j] for (i, j) in edge_local], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))