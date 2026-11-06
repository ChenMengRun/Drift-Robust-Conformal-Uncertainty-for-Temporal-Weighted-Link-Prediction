r"""
train_supervise.py — 训练自愈监督器（配合 train.py 的 --resume + 哨兵文件）

用法（在 SALT-main 目录下，且务必脱离 VSCode 终端运行，见下方说明）:
    python SALT/train_supervise.py --config ./SALT/config/INVIS15/SALT_INN.yaml --finetune

行为:
  - 反复启动 `python SALT/inn_train.py --config <cfg> --reproduction <x> --resume`
  - 每次子进程退出后检查哨兵（与 utils/logger.py 同目录: Exp/pt/<data_name>/<config_stem>/）:
      DONE.flag 存在            → 训练已正常完成（早停或跑满），停止监督（成功）
      STOP.flag 存在            → 用户主动停止（Ctrl+C），停止监督
      子进程退出且 error_log 本次被更新 → 疑似代码崩溃；连续崩溃超阈值则停止（防死循环）
      其它（外部杀死 / 睡眠 / 断电）→ 视为可恢复，等待数秒后 --resume 续训
  - 续训次数上限 max_restarts（默认 1000）防止无限循环。

为什么还要脱离 VSCode 终端:
  监督器本身是独立进程；若你把它直接跑在 VSCode 集成终端里，VSCode 关闭 / 自动更新重启
  会连同监督器一起杀死，自愈就失效了。正确做法是用 start 开一个独立窗口跑监督器:
      start "" cmd /c "cd /d D:\桌面\SALT-main && python SALT/train_supervise.py --config ./SALT/config/INVIS15/SALT_INN.yaml --finetune > sup.log 2>&1"
  这样即使关掉 VSCode、甚至睡眠醒来，监督器都会把训练重新拉起，直到 DONE.flag 出现。
"""
import subprocess
import sys
import os
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../SALT
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                 # SALT-main 根目录（含 Exp/ Pretrained/）


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='与 inn_train.py 相同的 --config 路径')
    p.add_argument('--reproduction', default='false',
                   help='透传给 inn_train.py（仅测试官方预训练权重、不训练；一般直接跑 inn_train.py 即可，不必经监督器）')
    p.add_argument('--finetune', action='store_true',
                   help='加载官方预训练 backbone 后继续训练（与官方 SALT 同一起跑线对比）；'
                        '监督器会传 --reproduction false --finetune --resume')
    p.add_argument('--max-restarts', type=int, default=1000, help='最大自动续训次数')
    p.add_argument('--restart-delay', type=int, default=5, help='每次续训前的等待秒数')
    p.add_argument('--crash-limit', type=int, default=3, help='连续代码崩溃多少次后放弃自动续训')
    return p.parse_args()


def compute_pt_dir(config_path):
    """复刻 utils/logger.py: Exp/pt/<data_name>/<config_stem>。
    优先读 yaml 的 data_name；缺失时回退到 config 父目录名（约定路径 .../<DATA>/<MODEL>.yaml）。"""
    config_path = os.path.abspath(config_path)
    stem = os.path.basename(config_path).rsplit('.', 1)[0]
    data_name = None
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        data_name = cfg.get('data_name')
    except Exception:
        pass
    if not data_name:
        data_name = os.path.basename(os.path.dirname(config_path))
    return os.path.join('Exp', 'pt', data_name, stem), data_name


def main():
    args = parse_args()
    # 切到项目根目录，确保子进程的相对路径（SALT/inn_train.py、Exp/pt/...）始终正确，
    # 无论用户在哪个目录调用监督器。
    os.chdir(PROJECT_ROOT)
    cfg_abs = os.path.abspath(args.config)
    pt_dir, data_name = compute_pt_dir(cfg_abs)
    done_flag = os.path.join(pt_dir, 'DONE.flag')
    stop_flag = os.path.join(pt_dir, 'STOP.flag')
    err_log = os.path.join('Exp', f'error_log_{data_name}.txt')

    # 清掉上一次的 STOP/DONE，允许本轮从头完整训练（DONE 残留会让监督器误以为已完成）
    for _f in (stop_flag, done_flag):
        if os.path.exists(_f):
            try:
                os.remove(_f)
            except OSError:
                pass

    if args.finetune:
        # finetune：加载官方 backbone 后训练。必须显式 --reproduction false，
        # 否则 inn_train.py 默认 reproduction=true 会测完即退出（不训练）。
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'inn_train.py'),
               '--config', cfg_abs,
               '--reproduction', 'false',
               '--finetune',
               '--resume']
        mode_desc = 'finetune（加载官方预训练 backbone 后训练）'
    else:
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'inn_train.py'),
               '--config', cfg_abs,
               '--reproduction', args.reproduction,
               '--resume']
        mode_desc = f'reproduction={args.reproduction}'

    restarts = 0
    crash_streak = 0
    print(f"[SUPERVISE] pt_dir = {pt_dir}")
    print(f"[SUPERVISE] 模式: {mode_desc}")
    print(f"[SUPERVISE] 启动训练（被杀后自动续训，上限 {args.max_restarts} 次）...")
    print(f"[SUPERVISE] 命令: {' '.join(cmd)}")

    while restarts <= args.max_restarts:
        err_mtime_before = os.path.getmtime(err_log) if os.path.exists(err_log) else -1

        rc = subprocess.run(cmd).returncode

        if os.path.exists(done_flag):
            print(f"[SUPERVISE] 检测到 DONE.flag → 训练正常完成，监督结束。")
            return
        if os.path.exists(stop_flag):
            print(f"[SUPERVISE] 检测到 STOP.flag（用户 Ctrl+C）→ 停止监督，不续训。")
            return

        # 区分“外部杀死（无新 error_log，可恢复）”与“代码崩溃（error_log 被更新）”
        err_mtime_after = os.path.getmtime(err_log) if os.path.exists(err_log) else -1
        fresh_crash = (err_mtime_after != err_mtime_before)

        if fresh_crash:
            crash_streak += 1
            print(f"[SUPERVISE] 子进程疑似代码崩溃 (rc={rc}, error_log 本次已更新)；"
                  f"连续崩溃 {crash_streak}/{args.crash_limit}")
            if crash_streak >= args.crash_limit:
                print(f"[SUPERVISE] 连续 {args.crash_limit} 次崩溃，疑似代码 bug，停止自动续训。"
                      f"请查看 {err_log}")
                return
        else:
            crash_streak = 0
            restarts += 1
            if restarts > args.max_restarts:
                print(f"[SUPERVISE] 达到续训上限 {args.max_restarts}，停止。")
                return
            print(f"[SUPERVISE] 子进程被外部杀死 (rc={rc}, 无新 error_log) → 视为可恢复，"
                  f"续训第 {restarts} 次（{args.restart_delay}s 后）...")
            time.sleep(args.restart_delay)


if __name__ == '__main__':
    main()
