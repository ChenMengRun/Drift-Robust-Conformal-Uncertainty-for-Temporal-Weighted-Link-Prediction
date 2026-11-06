import os
_BARE = os.environ.get('SALT_BARE') == '1'   # =1 时关闭本文件所有稳定性补丁（仅用于对照定位）
# ---- 稳定性打补丁：限定单线程 + 允许重复 OpenMP，规避 numpy/BLAS 原生竞态 access violation ----
# 必须在 import torch / numpy 之前设置，否则不生效。
if not _BARE:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.optim as optim
import math
import time
import faulthandler
faulthandler.enable()   # 关键：原生(C层)崩溃也会打印回溯，定位到具体行/哪个核

from tqdm import trange
from dataset.dataset import *
from dataset.dataset import get_adj_wei_map
from utils.paramset import Paramset
from utils.seeds import setup_seed
from utils.file import dynamic_import_lib
from utils.logger import Logger
from utils.metrics import get_RMSE, get_MAE, get_edge_RMSE

print(os.getcwd())

EPSILON = 0.01 
# 设备选择：尊重外部环境变量（.bat 里 set CUDA_VISIBLE_DEVICES 决定用哪块 GPU）。
# 原先这里硬编码 "5"，会屏蔽掉本机唯一的 GPU（只有 device 0），
# 使得每次训练都静默回退到 CPU —— 实测约 2 倍慢（HMob: CPU 5.0 it/s vs GPU 9.9 it/s）。
# setdefault 保证：bat 设 0 -> 用 GPU；bat 设 -1 -> 强制 CPU（已验证可用）。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
LOG_PATH = './Exp'
DATASET_PATH = './Data'

# ---- 单线程（规避 numpy/BLAS 多线程原生竞态）----
if not _BARE:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# ---- 绕开 dataset.rand_proj 的非连续列视图 + BLAS 点积 ----
# HMob noise_dim=64 → 每次 rand_proj 约 4000 次 strided .dot；每步调用 5 次，
# 数百步累计数百万次 → 在 Windows + 该 conda 环境（torch 1.12.1 + 重复 OpenMP/MKL）
# 下触发 numpy 原生 access violation（dataset.py:369 temp_row -= ...）。
# 改为「连续副本 + 逐元素乘加」，避开 BLAS 与非连续视图路径；数学等价（修改版 Gram-Schmidt）。
def _safe_rand_proj(num_nodes, hid_dim):
    rand_mat = np.random.normal(0, 1.0 / np.sqrt(hid_dim), (num_nodes, hid_dim))
    temp_l = np.sqrt((rand_mat * rand_mat).sum(axis=1))
    for i in range(hid_dim):
        temp_row = np.array(rand_mat[:, i], dtype=rand_mat.dtype, copy=True)
        for j in range(i - 1):
            temp_j = np.array(rand_mat[:, j], dtype=rand_mat.dtype, copy=True)
            temp_product = float((temp_row * temp_j).sum()) / (temp_l[j] ** 2)
            temp_row = temp_row - temp_product * temp_j
        temp_row = temp_row * (temp_l[i] / np.sqrt(float((temp_row * temp_row).sum())))
        rand_mat[:, i] = temp_row
    return rand_mat


if not _BARE:
    import dataset.dataset as _dsmod
    _dsmod.rand_proj = _safe_rand_proj
    print("[patch] dataset.rand_proj -> BLAS-free 实现（规避 numpy 原生 access violation）")


def _hb(msg):
    """训练心跳写日志（SALT_BARE=1 时禁用）。"""
    if _BARE:
        return
    try:
        with open('./Exp/train_trace.log', 'a') as _f:
            _f.write(msg)
    except Exception:
        pass

# ---- 快速稳定性测试：步数上限 + 进程内存(RSS)打印 ----
# 用法：set SALT_MAX_STEPS=80  → 跑满 80 步就正常退出，便于快速验收"是否还崩/内存是否涨"。
_MAX_STEPS = int(os.environ.get('SALT_MAX_STEPS', '0'))   # 0=不限制

def _rss_mb():
    """返回 (当前工作集 MB, 峰值工作集 MB)；失败返回 (None, None)。"""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return info.rss / 1e6, getattr(info, 'peak_wset', info.rss) / 1e6
    except Exception:
        pass
    try:
        import ctypes, ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        _k32 = ctypes.WinDLL('kernel32')
        _k32.GetCurrentProcess.restype = wt.HANDLE
        _psapi = ctypes.WinDLL('psapi')
        _psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
        _psapi.GetProcessMemoryInfo.restype = wt.BOOL
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None, None
        return pmc.WorkingSetSize / 1e6, pmc.PeakWorkingSetSize / 1e6
    except Exception:
        return None, None


def classify_error(e):
    """把异常分类成人类可读的'出错原因'，写入错误日志。"""
    s = str(e).lower()
    if 'out of memory' in s or 'cuda error' in s or 'cudnn' in s:
        return 'CUDA 显存不足 (OOM) 或 GPU 驱动/算子错误'
    if 'nan' in s or 'inf' in s or 'nats' in s or 'not a number' in s or 'non-finite' in s:
        return '出现 NaN/Inf（梯度或 loss 发散）'
    if 'size mismatch' in s or 'shape' in s or 'dimension' in s or 'not same size' in s:
        return '张量形状不匹配（架构变更 / 预训练权重不兼容 / 输入维度错误）'
    if 'file' in s or 'no such' in s or 'not found' in s or 'cannot find' in s or 'no such file' in s:
        return '文件/路径缺失（数据集 .npy 或预训练权重未找到）'
    if 'timeout' in s or 'connection' in s:
        return '网络/连接超时（数据下载或远程资源不可达）'
    if 'keyerror' in s:
        return '配置或字典缺键（yaml 字段缺失 / 数据字段不匹配）'
    if 'valueerror' in s or 'assert' in s:
        return '参数/断言错误（yaml 配置非法或前置校验失败）'
    return '未分类异常（详见下方完整堆栈）'


# ============ 全局异常钩子：任何未捕获异常都写 error_log ============
# 关键：补偿内层 try/except 只覆盖 train()/evaluate() 的漏洞——
# 早期崩溃（如 logger 初始化、数据加载、import）也需写 error_log，
# 否则监督器（train_supervise.py）无法区分"代码崩溃"与"外部杀死"，会无限续训。
import traceback as _tb_mod
def _global_excepthook(exc_type, exc_val, exc_tb):
    if issubclass(exc_type, SystemExit):   # 正常 sys.exit 不写日志，避免误判
        sys.__excepthook__(exc_type, exc_val, exc_tb)
        return
    try:
        _dn = params.data_name
    except Exception:
        _dn = 'UNKNOWN'
    try:
        _cause = classify_error(exc_val)
    except Exception:
        _cause = '未分类异常'
    _log = os.path.join(LOG_PATH, f'error_log_{_dn}.txt')
    try:
        os.makedirs(LOG_PATH, exist_ok=True)
        with open(_log, 'w', encoding='utf-8') as _f:
            _f.write(f"数据集: {_dn}\n")
            _f.write(f"发生时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            _f.write(f"错误分类: {_cause}\n")
            _f.write(f"最后到达 epoch: {globals().get('epoch', 'n/a')}\n\n")
            _f.write("完整堆栈:\n")
            _f.write(''.join(_tb_mod.format_exception(exc_type, exc_val, exc_tb)))
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_val, exc_tb)
sys.excepthook = _global_excepthook


setup_seed(0)
metrics_max = 1e+9
sel_base = {'nll': None, 'rmse': None, 'rmse_edge': None}   # combo 准则的首轮基准（归一化用）
epoch_start = 0

args_parse = argparse.ArgumentParser()
args_parse.add_argument('--config',
                        type=str,
                        default='./SALT/config/HMob/SALT.yaml')
args_parse.add_argument('--reproduction',
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default='false',
                        help='true=加载官方预训练权重后在 Test 上直接评测并退出（复现论文数字，不训练）；'
                             'false=从零训练（跳过预训练阶段，仅用于消融/对照）。'
                             '注意：默认 false——不传本开关即“从零训练”，避免静默只评测。')
args_parse.add_argument('--finetune',
                        action='store_true',
                        help='加载官方预训练 backbone 后【继续训练】（而非测完退出）。'
                             '用于 SALT_INN/SALT_Prob 与官方 SALT 站在同一起跑线对比。'
                             '与 --reproduction 互斥：--finetune 会加载权重但不退出。')
args_parse.add_argument('--resume',
                        action='store_true',
                        help='从 Exp/pt/<DATA>/<MODEL>/resume.ckpt 断点续训'
                             '（恢复模型+优化器+最佳指标+早停计数），息屏/断线/崩溃后重跑即可接着训')
args_parse.add_argument('--freeze_mu',
                        action='store_true',
                        help='SALT_INN 探针：冻结 μ 主干(官方 WLE 投影 + TLSSM)，仅训练 σ 头'
                             '（异方差校准）。不毁 μ 的前提下验证能否得到校准的 σ。')
args = args_parse.parse_args()

params = Paramset(args.config)
# freeze_mu：CLI 优先，否则读 yaml（freeze_mu: true）。仅用于 SALT_INN 探针。
freeze_mu = bool(args.freeze_mu or getattr(params, 'freeze_mu', False))

# ---- 多种子支持：默认仍用 setup_seed(0)（与既有全部结果完全一致）----
# 覆盖优先级：环境变量 SALT_SEED > config 的 `seed:` 键 > 默认 0。
# 必须在 Paramset 之后、Dataset/Model 构造之前生效，才能同时改变
#   模型初始化(torch) 与 数据侧随机源(np.random: rand_proj / noise)。
_seed_raw = os.environ.get('SALT_SEED', None)
if _seed_raw is None or str(_seed_raw).strip() == '':
    _seed_raw = getattr(params, 'seed', None)
if _seed_raw is not None and str(_seed_raw).strip() != '':
    _SEED = int(_seed_raw)
    setup_seed(_SEED)
    print(f'[SEED] setup_seed({_SEED}) 已生效（覆盖默认 0）', flush=True)
else:
    _SEED = 0

logger = Logger(params, LOG_PATH, tensorboard=True)

# ---- 可选：TSF 时序结构特征（config 里 `tsf: true` 打开；默认关，base 复现不受影响）----
_USE_TSF = bool(getattr(params, 'tsf', False))
if _USE_TSF:
    from dataset.dataset_tsf import DatasetTSF
    from dataset.tsf_features import TSF_DIM
    dataset = DatasetTSF(params, DATASET_PATH)
    # 自动推导 feat_proj 首层维度 = 基础特征维 + TSF 维（免去逐数据集手算）
    _fp = list(getattr(params, 'feat_proj_dims', []))
    if _fp:
        _new0 = int(dataset.base_feat_dim) + TSF_DIM
        if int(_fp[0]) != _new0:
            print(f'[TSF] feat_proj_dims[0] 自动调整：{_fp[0]} → {_new0}')
            _fp[0] = _new0
        params.feat_proj_dims = _fp
else:
    dataset = Dataset(params, DATASET_PATH)

Model = dynamic_import_lib(params.model)
net = Model(params).to(params.device)

# SALT_INN（INN 替换 WLE）必须以官方 SALT 主干为起点：feat_proj / tlssm / WLElayer 主体
# 全部来自官方预训练权重，仅末层 μ 通道=官方投影、σ 通道=INN 自身 init。
# 因此 INN 的「加载官方主干 + μ 从官方初始化」不应依赖 --finetune 标志——
# 否则一旦按监督器默认命令（不带 --finetune）启动，μ 初始化被跳过 → 随机 μ → edge-RMSE≈11.9。
is_inn = 'salt_inn' in (getattr(params, 'model', '') or '').lower()

# resume.ckpt 路径提前定义：供“是否加载预训练”判断 + 断点续训共用。
# 关键：若 resume.ckpt 已存在则跳过预训练加载（从断点续训）；否则才加载官方预训练。
resume_ckpt_path = os.path.join(logger.save_dir, 'resume.ckpt')
_resume_exists = args.resume and os.path.exists(resume_ckpt_path)

def init_inn_mu_from_official(net, ckpt_path, device):
    """SALT_INN 微调时，把官方 WLE 的最终投影(out=1)拷进 INN μ 通道(通道0)。

    必须在【每次】--finetune 时执行（无论是否 resume）：否则一旦存在旧的
    resume.ckpt（尤其来自修复前随机 μ 的跑），--resume 会直接加载它并跳过
    官方初始化，导致 μ 卡在随机起点(edge-RMSE≈11.8 而非官方 9.88)。

    关键：必须用 net.get_parameter(k) 取「活」参数并 in-place 写入；
    net.state_dict() 返回的是参数副本，改副本不写回模型（此前为 no-op）。
    """
    if not os.path.exists(ckpt_path):
        return 0
    sd = torch.load(ckpt_path, map_location=device)
    official = sd['model_dict'] if isinstance(sd, dict) and 'model_dict' in sd else sd
    n = 0
    for k, v in official.items():
        if 'WLElayer' not in k:
            continue
        try:
            cur = net.get_parameter(k)
        except AttributeError:
            continue
        # 官方 (1,H)/(1,) → INN (2,H)/(2,)：仅拷通道0(μ)，通道1(σ) 保留 INN 自身 init。
        if v.dim() >= 1 and v.shape[0] == 1 and cur.shape[0] == 2 \
                and tuple(v.shape[1:]) == tuple(cur.shape[1:]):
            with torch.no_grad():
                cur.data[0].copy_(v.reshape(cur.data[0].shape))
            n += 1
    return n


if (args.reproduction or args.finetune or is_inn) and not _resume_exists:
    ckpt = f'./Pretrained/{params.data_name}/{params.data_name}_pretrain.pth'
    if os.path.exists(ckpt):
        model_state = torch.load(ckpt, map_location=params.device)['model_dict']
        own = net.state_dict()
        # 架构变更（如 SALT_INN: WLE→INN 使 out_channels 1→2）会导致最终投影层形状不兼容：
        # WLE 的前置特征提取层(xlin/xcnlin/xijlin/lin隐藏层)形状一致、已正常加载；
        # 仅“最终输出投影” weight(1,H)/bias(1,)→(2,H)/(2,) 形状不符被跳过。
        # 注意：被跳过的是 μ/σ 共用的输出投影层，不是“整个 WLE”。下方对 SALT_INN 额外
        # 把官方投影拷进 μ 通道(通道0)，σ 通道(1)保留 INN 自身 init。
        loaded = {k: v for k, v in model_state.items()
                  if k in own and tuple(own[k].shape) == tuple(v.shape)}
        skipped = [k for k, v in model_state.items()
                   if not (k in own and tuple(own[k].shape) == tuple(v.shape))]
        net.load_state_dict(loaded, strict=False)
        print(f'[load] 复用预训练权重 已加载={len(loaded)} 跳过(形状不符)={len(skipped)}')
        # ---- P2 协议（TSF + finetune）：feat_proj[0] 扩维，新列权重置 0 ----
        # 官方 feat_proj.0.weight 形状 (out, base_dim)；开 TSF 后模型需要 (out, base_dim+TSF_DIM)，
        # 形状过滤会把它跳过 → 这里手动：前 base_dim 列拷官方权重，后 TSF_DIM 列置 0。
        # 结果：【初始输出与官方逐位相同】，之后任何提升都可归因于 TSF 被训练出来。
        if _USE_TSF:
            from dataset.tsf_features import TSF_DIM as _TSFD
            _base = int(getattr(dataset, 'base_feat_dim', 0))
            _w = model_state.get('feat_proj.0.weight')
            _b = model_state.get('feat_proj.0.bias')
            _need = _base + _TSFD
            if _w is not None and _w.shape[1] == _base and int(params.feat_proj_dims[0]) == _need:
                with torch.no_grad():
                    _lin = net.feat_proj[0]
                    _lin.weight.zero_()
                    _lin.weight[:, :_base].copy_(_w.to(_lin.weight.dtype))
                    if _b is not None and _lin.bias is not None:
                        _lin.bias.copy_(_b.to(_lin.bias.dtype))
                print(f'[TSF-P2] feat_proj[0] 扩维 {_base}→{_need}：官方权重填入前 {_base} 列，'
                      f'后 {_TSFD} 列置 0 -> 初始输出与官方等价')
            else:
                print(f'[TSF-P2] 跳过扩维（官方 feat_proj.0.weight='
                      f'{None if _w is None else tuple(_w.shape)}，期望 (out,{_base})）')
        # μ 通道初始化（INN 专属）改在下方【无条件】执行（见 args.finetune 分支），
        # 这样即便走 --resume 也不会因旧 resume.ckpt 存在而跳过官方初始化。
    else:
        print(f'[load] 未找到预训练权重 {ckpt}，将从头训练')

# —— 差分学习率（P2 专用）——
# 官方权重是尖锐最优点：整网 finetune 会在 1 个 epoch 内摧毁它（实测 lr=5e-4 → Test 2.78→4.38；
# 即便 lr=1e-5 也 →3.68；lr=0 才守住 2.81）。故设 backbone_lr_ratio<1（0=完全冻结骨干）时，
# 只让新增的 TSF 特征列（feat_proj[0].weight 后 TSF_DIM 列）以正常 lr 学习，骨干按 lr*ratio 微调。
_lr = float(params.learning_rate)
_bb_ratio_raw = getattr(params, 'backbone_lr_ratio', 1.0)
# 注意：不能用 `or 1.0`——0.0 是 falsy 会被吞掉（曾因此让"冻结骨干"静默退化为整网 finetune）。
_bb_ratio = 1.0 if _bb_ratio_raw is None else float(_bb_ratio_raw)
_tsf_on = bool(getattr(params, 'tsf', False))
if _tsf_on and _bb_ratio == 0.0:
    # 完全冻结整网，只训 feat_proj[0].weight 的新增 TSF 列（官方列梯度置 0）。
    # 理由：官方解是尖锐最优点，连官方那 24 列被推动都会立刻变差（实测 4.39→4.04 仍差）。
    try:
        from dataset.tsf_features import TSF_DIM as _TSFD_C
    except Exception:
        _TSFD_C = 12
    _w = net.feat_proj[0].weight
    _off = int(_w.shape[1]) - int(_TSFD_C)
    for _n, _p in net.named_parameters():
        _p.requires_grad = (_n == 'feat_proj.0.weight')

    def _mask_official(g, _o=_off):
        g = g.clone()
        g[:, :_o] = 0.0
        return g

    _w.register_hook(_mask_official)
    opt = optim.RMSprop([_w], lr=_lr, weight_decay=0.0)
    print(f'[TSF-P2] 冻结骨干 + 仅训 TSF 新列：feat_proj[0].weight[:, {_off}:] '
          f'（{_w.shape[0] * int(_TSFD_C)} 参数，lr={_lr:.2e}）', flush=True)
elif _tsf_on and _bb_ratio != 1.0:
    _bb, _new = [], []
    for _n, _p in net.named_parameters():
        if not _p.requires_grad:
            continue
        (_new if _n == 'feat_proj.0.weight' else _bb).append(_p)
    _new_n = sum(p.numel() for p in _new)
    opt = optim.RMSprop([{'params': _bb, 'lr': _lr * _bb_ratio},
                         {'params': _new, 'lr': _lr}],
                        lr=_lr, weight_decay=1e-5)
    print(f'[TSF-P2] 差分学习率：骨干 lr={_lr * _bb_ratio:.2e}（{len(_bb)} 组参数）'
          f' / feat_proj[0].weight lr={_lr:.2e}（{_new_n} 参数）', flush=True)
else:
    opt = optim.RMSprop(net.parameters(), lr=_lr, weight_decay=1e-5)


# ============ 断点续训支持 ============
# 每个 epoch 结束会把 {模型+优化器+最佳指标+早停计数} 写入
#   Exp/pt/<DATA>/<MODEL>/resume.ckpt
# 用 --resume 启动时从此断点继续；息屏/断线/崩溃后只需重跑同一条命令即可接着训，不丢进度。
epoch_start = 0
best_epoch = -1
epochs_no_improve = 0
metrics_max = 1e9
if args.resume:
    if os.path.exists(resume_ckpt_path):
        _ck = torch.load(resume_ckpt_path, map_location=params.device)
        net.load_state_dict(_ck['model_state_dict'])
        opt.load_state_dict(_ck['optimizer_state_dict'])
        epoch_start = _ck['epoch'] + 1
        best_epoch = _ck.get('best_epoch', -1)
        epochs_no_improve = _ck.get('epochs_no_improve', 0)
        metrics_max = _ck.get('metrics_max', 1e9)
        print(f"[RESUME] 从 epoch={_ck['epoch']} 续训 → 下次从 epoch={epoch_start} 开始"
              f"（已恢复优化器 / 最佳指标={metrics_max:.6f} / 早停计数={epochs_no_improve}）")
    else:
        print(f"[RESUME] 未找到 {resume_ckpt_path}，将从 epoch 0 开始训练")

# ============ SALT_INN：μ 通道从官方 WLE 投影初始化（无条件于 --finetune）============
# 关键修复：此前 μ 初始化被写在 `if args.finetune:` 守卫内，而监督器默认命令
# （train_supervise.py 文档示例、以及上一轮给用户的标准命令都）不带 --finetune，
# 导致 μ 初始化整段被跳过 → 随机 μ → epoch0 edge-RMSE≈11.9
# （特征/TLSSM 已从官方加载、唯独 μ 投影随机，dense≈3.8 是该状态的签名）。
# 现改为：只要是 INN 模型且官方 ckpt 存在，就在「加载官方主干 / resume」之后无条件重拷 μ；
# 但若本次是 --resume 且 resume.ckpt 已存在，则保留其中的 μ（视为已 finetune 的续训，不再重置）。
if is_inn:
    _ft_ckpt = f'./Pretrained/{params.data_name}/{params.data_name}_pretrain.pth'
    if _ft_ckpt and os.path.exists(_ft_ckpt) and not (args.resume and os.path.exists(resume_ckpt_path)):
        _n_mu = init_inn_mu_from_official(net, _ft_ckpt, params.device)
        if _n_mu:
            print(f'[load] INN μ 通道已从官方 WLE 投影初始化 (键={_n_mu})')
        else:
            print(f'[load] [WARN] INN 模型但 init_inn_mu_from_official 返回 0（未拷入 μ）；'
                  f'请检查官方 ckpt 是否含 WLElayer 键: {_ft_ckpt}')
    elif is_inn and not (_ft_ckpt and os.path.exists(_ft_ckpt)):
        print(f'[load] [WARN] INN 模型但找不到官方 ckpt，μ 将保持随机初始化: {_ft_ckpt}')
    else:
        # --resume 且 resume.ckpt 已存在：保留其中 μ（续训），不重置为官方
        print('[load] INN 续训：沿用 resume.ckpt 中的 μ（不重置为官方）')


# ============ SALT_INN 探针：冻结 μ、仅训 σ 头（异方差校准）============
# 动机：此前 full finetune 把 μ 从官方 9.88 训退化到 11.9（死 σ 放大梯度 + β·MSE 拽 μ 越过最优）。
# 本探针把 μ 主干全冻结（WLElayer 特征/TLSSM/μ 投影均锁官方），只让一个轻量 σ 头从「冻结特征」
# 学 p(weight|structure) 的方差，验证“不毁 μ 的前提下能否得到校准的 σ”（edge corr(|error|,σ)>0）。
# 实现：不改动 salt_inn.py / base SALT —— 仅在本文件用 forward hook 把 lin[-1] 第 1 通道(log_sigma)
# 替换为可训练 σ 头输出，μ(通道0) 保持冻结官方值。β 对 σ 头无梯度贡献（μ 已冻结），故用纯 edge-NLL 即可。
if is_inn and freeze_mu:
    h_in = net.WLElayer.lin[-1].in_features
    net.sigma_head = nn.Linear(h_in, 1).to(params.device)
    with torch.no_grad():
        # 从 INN 当前 σ 行(随机 init, bias≈-1→σ≈0.37) 起步，让 σ 头从该初值学校准
        net.sigma_head.weight.copy_(net.WLElayer.lin[-1].weight.data[1:2])
        net.sigma_head.bias.copy_(net.WLElayer.lin[-1].bias.data[1:2])

    # 冻结整个网络，只放行 σ 头
    for _p in net.parameters():
        _p.requires_grad_(False)
    for _p in net.sigma_head.parameters():
        _p.requires_grad_(True)

    # 用 σ 头覆盖 lin[-1] 输出的第 1 通道(log_sigma)；μ(通道0) 保持冻结官方值
    def _sigma_hook(module, inp, out):
        z = inp[0]
        new_logsigma = net.sigma_head(z)          # (E,1)
        out = out.clone()
        out[..., 1:2] = new_logsigma
        return out
    net.WLElayer.lin[-1].register_forward_hook(_sigma_hook)

    # 优化器只更新 σ 头（RMSprop 与主线一致；仅 38 个参数）
    opt = optim.RMSprop(net.sigma_head.parameters(),
                        lr=float(params.learning_rate), weight_decay=1e-5)
    print(f'[freeze_mu] μ 主干已冻结(官方)，仅训 σ 头(输入维={h_in}, 参数量≈{sum(p.numel() for p in net.sigma_head.parameters())})；'
          f'训练目标=edge-NLL（β 对 σ 头无梯度贡献，可忽略）')


def train():
    net.train()
    train_cnt = 0
    loss_list = []
    if not _BARE:
        open('./Exp/train_trace.log', 'w').close()  # 训练心跳日志（诊断崩溃位置）

    for tau in trange(params.win_size, dataset.num_train_snaps - params.pred_size + 1):
        _hb(f'{epoch},{tau},start\n')
        sup_list, noise_list, feat_list= dataset.get_win_data(tau-params.win_size, tau)
        # 归一空间 GT（与模型输出同 LOCAL 索引）。data_level==2 用 node_map 重映射，
        # 否则 get_win_gnd_data 的全局边索引会错位（此前 σ 学烂的潜在根因之一）。
        gnd_list = []
        edge_mask_list = []
        for tt in range(tau - params.win_size + 1, tau + params.pred_size):
            edges = dataset.edge_seq[tt]
            if dataset.data_level == 2:
                g = get_adj_wei_map(edges, dataset.node_map_seq_gbl[tt],
                                    dataset.num_nodes, dataset.max_thres)
                node_map = dataset.node_map_seq_gbl[tt]
                mask = np.zeros((dataset.num_nodes, dataset.num_nodes))
                for e in edges:
                    u, v = int(e[0]), int(e[1])
                    if u in node_map and v in node_map:
                        mask[node_map[u], node_map[v]] = 1.0
                        mask[node_map[v], node_map[u]] = 1.0
            else:
                g = get_adj_wei(edges, dataset.num_nodes, dataset.max_thres)
                mask = np.zeros((dataset.num_nodes, dataset.num_nodes))
                for e in edges:
                    u, v = int(e[0]), int(e[1])
                    mask[u, v] = 1.0
                    mask[v, u] = 1.0
            g = g / dataset.max_thres + np.eye(dataset.num_nodes)
            gnd_list.append(torch.FloatTensor(g).to(params.device))
            edge_mask_list.append(torch.FloatTensor(mask).to(params.device))
        
        adj_est_list = net(sup_list,feat_list,noise_list,pred_flag=False)
        _hb(f'{epoch},{tau},forward_done\n')

        loss = net.causal_training_loss(adj_est_list,
                                        gnd_list,
                                        params.Loss['alpha'],
                                        params.Loss['theta'],
                                        _beta,
                                        edge_masks=edge_mask_list)

        opt.zero_grad()
        loss.backward()
        opt.step()
        _hb(f'{epoch},{tau},backward_done\n')
        loss_list.append(loss.item())
        # NaN/Inf 守卫：发散时立刻停止并给出可定位的原因，而不是静默跑飞。
        if not math.isfinite(loss.item()):
            raise RuntimeError(
                f'训练 loss 出现 NaN/Inf（epoch={epoch}, step={train_cnt}）。'
                f'常见原因：学习率过大导致梯度发散 / 数值下溢 / '
                f'输入含非法值（NaN）。请检查学习率与数据预处理。'
            )
        train_cnt += 1
        if _MAX_STEPS and train_cnt >= _MAX_STEPS:
            print(f'[CAP] 达到 SALT_MAX_STEPS={_MAX_STEPS}，提前正常退出（稳定性测试用）', flush=True)
            sys.exit(0)
        if (not _BARE) and train_cnt % 20 == 0:
            _r, _p = _rss_mb()
            if _r is not None:
                # flush=True 关键：段错误会丢掉未刷新的 stdout 缓冲
                print(f'  [MEM] step {train_cnt}: RSS={_r:.1f} MB  peak={_p:.1f} MB', flush=True)
        if train_cnt % 100 == 0:
            extra = ''
            nll_c = getattr(net, '_loss_nll', None)
            mse_c = getattr(net, '_loss_mse', None)
            if nll_c is not None and mse_c is not None:
                extra = (f'  [loss分拆] NLL(edge)={nll_c.item():.4f}  '
                         f'MSE(β={_beta:.2f})={_beta * mse_c.item():.4f}  β={_beta}')
            print('-Train %d / %d / %d%s' % (train_cnt, dataset.num_train_snaps, np.mean(loss_list), extra))
            
    gen_loss_mean = np.mean(loss_list)
    result = {'G-Loss': gen_loss_mean}
    print('#%d Train G-Loss %f' % (epoch, gen_loss_mean))
    logger.write(result, 'Train', epoch) 


@torch.no_grad()
def evaluate(start, end, mode):
    net.eval()

    results_list_dict = {'RMSE': [],
                         'MAE': [],
                         'NLL': [],
                         'RMSE_edge': [],
                         'NLL_edge': []}
    # σ 校准诊断累积器：收集每条真实边的 |error| 与 σ，供末尾算 corr/coverage
    _calib_err, _calib_sig = [], []

    metrics_dict = {'RMSE': get_RMSE,
                    'MAE': get_MAE}

    for tau in trange(start, end, params.pred_size):
        sup_list, noise_list,feat_list = dataset.get_win_data(tau - params.win_size, tau)

        adj_est_list = net(sup_list, feat_list, noise_list, pred_flag=True)
        # SALT_INN 提供 eval_forward_dist 时额外计算归一空间 NLL（与训练目标一致）。
        # 必须在循环内、用当前窗口的 sup 计算，不能用循环外的残留变量（会 NameError / 错窗口）。
        dist_list = net.eval_forward_dist(sup_list, feat_list, noise_list) \
            if hasattr(net, 'eval_forward_dist') else None

        for t in range(tau, tau + params.pred_size):
            adj_est = adj_est_list[t - tau]
            adj_est = adj_est.cpu().numpy()
            adj_est *= dataset.max_thres

            np.fill_diagonal(adj_est, 0)
            adj_est[adj_est <= EPSILON] = 0

            edges = dataset.edge_seq[t]
            if dataset.data_level == 2:
                num_nodes = dataset.num_nodes
                gnd = get_adj_wei(edges, num_nodes, dataset.max_thres)
                real_row = list(dataset.node_map_seq_gbl[t].keys())
                real_col = list(dataset.node_map_seq_gbl[t].keys())
                gnd = gnd[np.ix_(real_row, real_col)]
                adj_est = adj_est[np.ix_(real_row, real_col)]
                num_nodes = dataset.num_nodes_seq_gbl[t]
                # 局部索引：RMSE_edge 用（与经 real_row 重映射的 gnd/adj_est 对齐）
                local_pos = {g: i for i, g in enumerate(real_row)}
                edge_local = [(local_pos[int(e[0])], local_pos[int(e[1])])
                              for e in edges
                              if int(e[0]) in local_pos and int(e[1]) in local_pos]
                # 全局索引：NLL_edge 用（与 get_adj_wei_map 的 node_map 位置对齐，矩阵为全局尺寸）
                node_map = dataset.node_map_seq_gbl[t]
                edge_global = [(node_map[int(e[0])], node_map[int(e[1])])
                               for e in edges
                               if int(e[0]) in node_map and int(e[1]) in node_map]
            else:
                num_nodes = dataset.num_nodes
                gnd = get_adj_wei(edges, num_nodes, dataset.max_thres)
                edge_local = [(int(e[0]), int(e[1])) for e in edges]
                edge_global = edge_local

            for metric, func in metrics_dict.items():
                result = func(adj_est, gnd, num_nodes)
                results_list_dict[metric].append(result)

            # 仅真实边 RMSE（原始权重空间，与 dense RMSE 同口径，避开 99% 非边零元主导）
            results_list_dict['RMSE_edge'].append(
                get_edge_RMSE(adj_est, gnd, edge_local))

            if dist_list is not None:
                mu_sig = dist_list[t - tau]          # (N,N,2)
                mu = mu_sig[..., 0]
                log_sigma = mu_sig[..., 1]
                sigma = torch.exp(log_sigma).clamp(min=1e-3, max=1e2)
                # 归一空间 GT；data_level==2 时 gnd_norm_t 为全局尺寸、边置于 node_map 位置
                if dataset.data_level == 2:
                    gnd_norm_t = get_adj_wei_map(edges, dataset.node_map_seq_gbl[t],
                                                 dataset.num_nodes, dataset.max_thres)
                else:
                    gnd_norm_t = get_adj_wei(edges, dataset.num_nodes, dataset.max_thres)
                gnd_norm_t = gnd_norm_t / dataset.max_thres + np.eye(dataset.num_nodes)
                gnd_norm_t = torch.FloatTensor(gnd_norm_t).to(mu.device)
                nll_t = 0.5 * ((gnd_norm_t - mu) ** 2 / sigma ** 2
                               + 2.0 * log_sigma + math.log(2 * math.pi))
                results_list_dict['NLL'].append(nll_t.mean().item())
                # 仅真实边 NLL（归一空间，与 dense NLL 同公式，仅取边位置）
                if edge_global:
                    idx_i = torch.tensor([i for (i, j) in edge_global],
                                         dtype=torch.long, device=mu.device)
                    idx_j = torch.tensor([j for (i, j) in edge_global],
                                         dtype=torch.long, device=mu.device)
                    nll_edge = nll_t[idx_i, idx_j].mean().item()
                    results_list_dict['NLL_edge'].append(nll_edge)
                    # —— 校准诊断：收集该窗口真实边的 |error| 与对应 σ ——
                    _calib_err.append((gnd_norm_t[idx_i, idx_j] - mu[idx_i, idx_j]).abs()
                                      .detach().cpu().numpy())
                    _calib_sig.append(sigma[idx_i, idx_j].detach().cpu().numpy())


    # —— σ 校准汇总：edge 级 |error| 与 σ 的 Pearson 相关 + 覆盖率 ——
    # 理想高斯校准：corr>0（σ 随 |error| 变大）、cov1σ≈0.68、cov2σ≈0.95。
    _sigma_corr = float('nan'); _sigma_cov1 = float('nan'); _sigma_cov2 = float('nan')
    if _calib_err and _calib_sig:
        _E = np.concatenate(_calib_err)
        _S = np.concatenate(_calib_sig)
        if len(_E) > 1 and _E.std() > 0 and _S.std() > 0:
            _sigma_corr = float(np.corrcoef(_E, _S)[0, 1])
        _sigma_cov1 = float((_E <= _S).mean())
        _sigma_cov2 = float((_E <= 2.0 * _S).mean())

    results_dict = {}
    for metric, result_list in results_list_dict.items():
        if len(result_list) == 0:
            continue
        results_dict[metric] = np.mean(result_list)
        results_dict[metric + '_std'] = np.std(result_list, ddof=1)

    # 并入 σ 校准诊断（已在循环外汇总到 _sigma_corr / _sigma_cov1 / _sigma_cov2）
    results_dict['sigma_corr'] = _sigma_corr
    results_dict['sigma_cov1'] = _sigma_cov1
    results_dict['sigma_cov2'] = _sigma_cov2

    if mode in ['Val', 'Test']:
        msg = (f'{mode} #{epoch} RMSE {results_dict.get("RMSE", float("nan")):.4f} '
               f'{results_dict.get("RMSE_std", float("nan")):.4f} '
               f'MAE {results_dict.get("MAE", float("nan")):.4f} '
               f'{results_dict.get("MAE_std", float("nan")):.4f}')
        if 'RMSE_edge' in results_dict:
            msg += f' RMSE_edge {results_dict["RMSE_edge"]:.4f}'
        if 'NLL' in results_dict:
            msg += f' NLL {results_dict["NLL"]:.4f}'
        if 'NLL_edge' in results_dict:
            msg += f' NLL_edge {results_dict["NLL_edge"]:.4f}'
        if 'sigma_corr' in results_dict:
            msg += (f' σcorr {results_dict["sigma_corr"]:.3f}'
                    f' cov1σ {results_dict["sigma_cov1"]:.3f}'
                    f' cov2σ {results_dict["sigma_cov2"]:.3f}')
        print(msg)
        log_list = [k for k in ('RMSE', 'MAE', 'NLL') if k in results_dict]
        log_results = {key: results_dict[key] for key in log_list}
        logger.write(log_results, mode, epoch)
    return results_dict


# ============ 早停参数 / 训练前声明 ============
# 早停：从 yaml 读取 EarlyStop_patience；缺省为极大值（即不早停，跑满 num_epochs）。
# 判断依据（选优指标）由 Selection 决定，详见每个 epoch 打印的 [SAVE]/[NOSAVE]。
sel_default = (getattr(params, 'Selection', 'rmse') or 'rmse').lower()
_lam = float(getattr(params, 'Sel_lambda', 0.1))
_use_norm = bool(getattr(params, 'Sel_norm', True))
_patience = int(getattr(params, 'EarlyStop_patience', 1 << 30))
_min_delta = float(getattr(params, 'EarlyStop_min_delta', 0.0))
_beta = float(getattr(params, 'Loss', {}).get('beta', 0.0))   # 复合损失权重；0=纯 NLL


def _criterion_desc(sel):
    if sel == 'nll':
        return 'val_NLL（与训练目标一致，μ 与 σ 同时最优）'
    if sel == 'edge_rmse':
        return 'val_RMSE_edge（仅真实边 RMSE，避开 99% 非边零元主导，点预测公平性尺）'
    if sel == 'combo':
        return f'NLL/基准NLL + {_lam}·RMSE/基准RMSE（NLL 为主、RMSE 为辅，Sel_norm={_use_norm}）'
    if sel == 'edge_combo':
        return f'NLL/基准NLL + {_lam}·RMSE_edge/基准RMSE_edge（NLL 为主、仅边 RMSE 为辅）'
    return '(val_RMSE+val_MAE)/2（仅看预测精度，忽略 σ）'


print(f"\n[CONFIG] 数据集={params.data_name}  总 epoch={params.num_epochs}")
# 运行模式横幅：明确是从零训练 / 加载官方预训练 / 微调，避免“数字对不上官方”这类误判
if args.reproduction and not args.finetune:
    _mode_desc = "reproduction（加载官方预训练权重，Test 上直接评测后退出，不训练——用于复现论文数字）"
elif args.finetune:
    _mode_desc = "finetune（加载官方预训练 backbone 后继续训练，与官方 SALT 同一起跑线）"
else:
    _mode_desc = "scratch（从零训练，跳过预训练阶段——仅用于消融/对照，数字会明显差于官方）"
print(f"[CONFIG] 运行模式 = {_mode_desc}")
print(f"[CONFIG] 模型选择准则 Selection='{sel_default}' → {_criterion_desc(sel_default)}")
_model = getattr(params, 'model', '')
_mname = _model.split('.')[-1] if _model else ''
_is_base_salt = _mname in ('SALT', 'SALT_Dense')          # 密集 Frobenius MAE+MSE
_is_edge_salt = ('Edge' in _mname)                        # edge 加权 MAE+MSE（salt_edge）
if _is_base_salt:
    # 确定性基线：causal_training_loss 内部仍是原 Frobenius MAE+MSE（密集 RMSE 训练目标）
    _loss_desc = "MAE+MSE（密集 RMSE 训练目标，确定性 SALT 基线）"
elif _is_edge_salt:
    _L = getattr(params, 'Loss', {}) or {}
    _loss_desc = (f"edge 加权 MAE+MSE（仅真实边 edge_masks；"
                  f"alpha={_L.get('alpha', '?')}, theta={_L.get('theta', '?')}）")
elif _beta > 0:
    _loss_desc = f"edge-NLL + β·edge-MSE（β={_beta}）"
else:
    _loss_desc = "edge-NLL only（β=0）"
print(f"[CONFIG] 训练损失 = {_loss_desc}")
if is_inn and freeze_mu:
    print(f"[CONFIG] 探针模式 = freeze_mu（μ 主干冻结于官方 9.88 起跑线，仅训 σ 头做异方差校准）")
_patience_str = "∞（不早停，跑满 num_epochs）" if _patience >= (1 << 30) else str(_patience)
print(f"[CONFIG] 早停：连续 {_patience_str} 个 epoch 选优指标无改善(Δ<{_min_delta:.0e}) 即停止训练\n")


# ============ 进程保护：优雅中断 + 哨兵文件（供 train_supervise.py 自动续训）============
import signal as _signal
_INTERRUPTED = {'flag': False, 'stop': False}   # flag=收到终止信号; stop=用户主动停止(Ctrl+C)

def _on_signal(signum, frame):
    _INTERRUPTED['flag'] = True
    # SIGINT(Ctrl+C) 视为用户主动停止；SIGTERM(被外部杀死/睡眠/断电) 视为需续训
    if signum == _signal.SIGINT:
        _INTERRUPTED['stop'] = True

def _write_sentinel(path, **kw):
    """写哨兵文件（DONE/STOP），供监督器判断是否停止。失败静默忽略。"""
    try:
        with open(path, 'w', encoding='utf-8') as _f:
            _f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for _k, _v in kw.items():
                _f.write(f"{_k}={_v}\n")
    except Exception:
        pass



# —— 注册优雅中断处理器（Ctrl+C / 被外部杀死时先落好 resume.ckpt 再退出）——
_signal.signal(_signal.SIGINT, _on_signal)
try:
    _signal.signal(_signal.SIGTERM, _on_signal)
except (ValueError, AttributeError, OSError):
    pass  # Windows 某些环境无 SIGTERM 或不可注册

# ---- 快速探针：SALT_MAX_EPOCHS=N 截断训练轮数（不改 config，便于快速看趋势）----
_MAX_EPOCHS = int(os.environ.get('SALT_MAX_EPOCHS', '0'))
if _MAX_EPOCHS and params.num_epochs > _MAX_EPOCHS:
    print(f'[PROBE] SALT_MAX_EPOCHS={_MAX_EPOCHS}：num_epochs {params.num_epochs} → {_MAX_EPOCHS}', flush=True)
    params.num_epochs = _MAX_EPOCHS

for epoch in range(epoch_start, params.num_epochs):
    if args.reproduction and not args.finetune:
            test_res = evaluate(dataset.num_snaps-dataset.num_test_snaps,
                        dataset.num_snaps,
                        'Test')
            test_rmse, test_mae = test_res['RMSE'], test_res['MAE']
            print(f'TEST RMSE:  {test_rmse},    MAE:    {test_mae}')
            sys.exit(0)
    else:
        # —— 错误捕获：训练/评测中任何异常都被分类并记录到 Exp/error_log_<DATA>.txt ——
        try:
            train()
            val_res = evaluate(dataset.num_snaps-dataset.num_test_snaps-dataset.num_val_snaps,
                              dataset.num_snaps-dataset.num_test_snaps,
                              'Val')
            test_res = evaluate(dataset.num_snaps-dataset.num_test_snaps,
                               dataset.num_snaps,
                               'Test')
        except Exception as e:
            err_log = os.path.join(LOG_PATH, f'error_log_{params.data_name}.txt')
            os.makedirs(LOG_PATH, exist_ok=True)
            _cause = classify_error(e)
            import traceback as _tb
            with open(err_log, 'w', encoding='utf-8') as _f:
                _f.write(f"数据集: {params.data_name}\n")
                _f.write(f"发生时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                _f.write(f"错误分类: {_cause}\n")
                _f.write(f"最后到达 epoch: {epoch}\n\n")
                _f.write("完整堆栈:\n")
                _f.write(_tb.format_exc())
            print(f"\n[TRAINING TERMINATED] 训练因错误中断：{_cause}")
            print(f"              最后到达 epoch={epoch}；完整堆栈已写入：{err_log}")
            print(f"              若本进程未产生该日志而直接消失，说明是被外部信号杀死")
            print(f"              （如 Linux OOM-killer / kill -9 / 断电 / 手动 Ctrl+C），非代码异常。")
            sys.exit(1)

        val_rmse, val_mae = val_res['RMSE'], val_res['MAE']
        test_rmse, test_mae = test_res['RMSE'], test_res['MAE']
        val_edge = val_res.get('RMSE_edge', float('nan'))

        # ---- P2' 诊断：新特征列范数（判断 TSF 到底有没有"接上"）----
        # 关键经验：lr=1e-6 时新列 RMS 只有官方列的 1/200 ≈ 对输出无贡献，
        # 那种情况下任何 Val/Test 变化都不能归因于 TSF。必须盯这个比值。
        if getattr(params, 'tsf', False) and getattr(params, 'backbone_lr_ratio', 1.0) == 0.0:
            try:
                from dataset.tsf_features import TSF_DIM as _TSFD_L
                _w = net.feat_proj[0].weight.detach().float()
                _off = int(_w.shape[1]) - int(_TSFD_L)
                _r_new = _w[:, _off:].pow(2).mean().sqrt().item()
                _r_old = _w[:, :_off].pow(2).mean().sqrt().item()
                print(f'  [TSF-P2] 特征列范数：新列 RMS={_r_new:.3e} / 官方列 RMS={_r_old:.3e}'
                      f'（占比 {100.0 * _r_new / max(_r_old, 1e-12):.2f}%）', flush=True)
            except Exception as _e:
                print(f'  [TSF-P2] 范数诊断失败：{_e}')

        # ---- 模型选择准则（判断依据）----
        # Selection: 'rmse'(A) | 'nll'(B) | 'combo'(C) | 'edge_rmse' | 'edge_combo'
        #   A: 与原版 SALT 完全一致，仅看预测精度（忽略 σ）
        #   B: 与训练目标(NLL)一致，μ 与 σ 同时最优
        #   C: NLL 为主 + λ·denseRMSE 为辅；Sel_norm=true 时按首轮基准归一化
        #   edge_rmse: 仅真实边 RMSE（避开 99% 非边零元主导），点预测公平性尺
        #   edge_combo: NLL 为主 + λ·仅边RMSE 为辅（既约束 μ 又保 σ 诚实）
        sel = sel_default
        lam = _lam
        use_norm = _use_norm
        if sel == 'nll' and 'NLL' in val_res:
            metrics_mean = val_res['NLL']
        elif sel == 'edge_rmse' and 'RMSE_edge' in val_res:
            metrics_mean = val_res['RMSE_edge']
        elif sel == 'combo' and 'NLL' in val_res:
            if use_norm:
                if sel_base['nll'] is None:
                    sel_base['nll'] = val_res['NLL']
                    sel_base['rmse'] = val_rmse
                metrics_mean = (val_res['NLL'] / max(sel_base['nll'], 1e-8)) \
                               + lam * (val_rmse / max(sel_base['rmse'], 1e-8))
            else:
                metrics_mean = val_res['NLL'] + lam * val_rmse
        elif sel == 'edge_combo' and 'NLL' in val_res and 'RMSE_edge' in val_res:
            if use_norm:
                if sel_base['nll'] is None:
                    sel_base['nll'] = val_res['NLL']
                    sel_base['rmse_edge'] = val_edge
                metrics_mean = (val_res['NLL'] / max(sel_base['nll'], 1e-8)) \
                               + lam * (val_edge / max(sel_base['rmse_edge'], 1e-8))
            else:
                metrics_mean = val_res['NLL'] + lam * val_edge
        else:  # 'rmse' 或缺 NLL 时回退到原版准则
            metrics_mean = (val_rmse + val_mae) / 2.0

        # ---- 早停判定 + 模型保存依据 ----
        # 判断依据：选优指标 metrics_mean 越小越好；连续 _patience 个 epoch
        # 相对当前最佳未改善超过 _min_delta，即认为收敛、停止训练。
        improved = (metrics_mean + _min_delta) < metrics_max
        if improved:
            epochs_no_improve = 0
            rel_gain = (metrics_max - metrics_mean) / max(abs(metrics_max), 1e-8) * 100
            metrics_max = metrics_mean
            best_epoch = epoch
            tag = f'{round(test_rmse, 3)}_{round(test_mae, 3)}'
            logger.save_model(net, opt, epoch, tag)
            nll_s = f" NLL={val_res['NLL']:.4f}" if 'NLL' in val_res else ""
            edge_s = f" RMSE_edge={val_edge:.4f}" if 'RMSE_edge' in val_res else ""
            print(f"[SAVE] epoch={epoch} 准则={sel} 选优指标={metrics_mean:.6f} "
                  f"较上一最佳↓{rel_gain:.2f}% → 保存 {epoch}_{tag}.pth")
            print(f"       验证集: RMSE={val_rmse:.4f} MAE={val_mae:.4f}{edge_s}{nll_s}")
        else:
            epochs_no_improve += 1
            print(f"[NOSAVE] epoch={epoch} 准则={sel} 选优指标={metrics_mean:.6f} ≥ 最佳={metrics_max:.6f} "
                  f"(连续 {epochs_no_improve}/{_patience} epoch 无改善)")

        # —— 每 epoch 落盘断点（供 --resume 续训；息屏/断线后重跑即可接着训）——
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_epoch': best_epoch,
            'epochs_no_improve': epochs_no_improve,
            'metrics_max': metrics_max,
        }, resume_ckpt_path)

        # —— 优雅中断检查：收到终止信号则保存好 resume.ckpt 后退出（监督器会续训）——
        if _INTERRUPTED['flag']:
            if _INTERRUPTED['stop']:
                _write_sentinel(os.path.join(logger.save_dir, 'STOP.flag'),
                               reason='SIGINT', epoch=epoch)
                print(f"\n[INTERRUPT] 收到 SIGINT（用户请求停止），已写 STOP.flag → 监督器不会自动续训。")
            else:
                print(f"\n[INTERRUPT] 收到终止信号，resume.ckpt 已保存到 epoch={epoch} → 监督器将自动续训。")
            break

        # —— 跑满 epoch 也写 DONE（正常完成，监督器见此即停）——
        if epoch + 1 >= params.num_epochs:
            _write_sentinel(os.path.join(logger.save_dir, 'DONE.flag'),
                           reason='max_epochs', best_epoch=best_epoch,
                           metrics_max=metrics_max)
            print(f"\n[MAX EPOCHS] 已训满 {params.num_epochs} epoch，训练正常结束。")

        if epochs_no_improve >= _patience:
            print(f"\n[EARLY STOP] 连续 {_patience} 个 epoch 验证选优指标无改善"
                  f"(阈值 min_delta={_min_delta:.0e})。")
            print(f"            最佳 epoch={best_epoch}，选优指标={metrics_max:.6f}（准则={sel}）。")
            print(f"            训练停止，最佳模型已在 epoch={best_epoch} 保存。")
            _write_sentinel(os.path.join(logger.save_dir, 'DONE.flag'),
                           reason='early_stop', best_epoch=best_epoch,
                           metrics_max=metrics_max)
            break