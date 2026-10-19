# -*- coding: utf-8 -*-
"""
tfm_cli_run.py — TFM 时频电磁法完整处理 命令行 Headless 入口
================================================================
完全离线运行，仅读写本地文件。数据绝不上传云端。

用法 (通过 run_tfm.bat 调):
    run_tfm.bat --cli  --project D:\\MyProject\\myfile  --out D:\\WLF_Work\\example_output\\7_tfm_production
    run_tfm.bat --cli  --no-preprocess --inv            (跳过预处理, 只跑1D反演)
    run_tfm.bat --cli  --occam --occam-exe D:\\Tools\\Occam1D.exe

直接调用:
    python tfm_cli_run.py  -h
"""
from __future__ import annotations
import argparse
import importlib
import re
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

# 保证 import 相对包能工作 (BAT 已 cd 到 tools/)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tfm_engine.preprocess import run_full_preprocess, DEFAULT_CFG
from tfm_engine.inversion  import run_all_inversions, DEFAULT_INV_CFG
from tfm_engine.occam_bridge import materialize_all
from tfm_engine.inv_algorithms import run_all_algo, run_all_algos_compare, ALGO_CHOICES
from tfm_engine.adv_inversion import run_all_advanced, ALGO_ADVANCED
from tfm_engine.lci_inversion import run_all_lci
from tfm_engine.dl_inversion import run_all_dl
from tfm_engine.current_proc import run_all_current


# ================================================================
# 日志: 控制台 + 磁盘文件 (完全离线)
# ================================================================
def _make_logger(log_file: Path) -> Callable[[str], None]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, 'a', encoding='utf-8')

    def _log(msg: str) -> None:
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        fh.write(line + '\n')
        fh.flush()
    return _log


# ================================================================
# 参数合并: CLI HP -> user_cfg dict
# ================================================================
def _merge_preprocess_cfg(args, log_fn: Callable) -> dict:
    cfg = dict(DEFAULT_CFG)
    if args.rhoa_scale is not None:
        cfg['RHOA_SCALE_CAL'] = float(args.rhoa_scale)
        log_fn(f'[HP] 覆盖 RHOA_SCALE_CAL = {cfg["RHOA_SCALE_CAL"]:.2e}')
    if args.g_adc is not None:
        cfg['G_ADC_COUNTS_PER_VOLT'] = int(args.g_adc)
    if args.n_gates is not None:
        cfg['TEM_N_GATES'] = int(args.n_gates)
    if args.t_min_us is not None:
        cfg['T_MIN_US'] = float(args.t_min_us)
    if args.t_max_us is not None:
        cfg['T_MAX_US'] = float(args.t_max_us)
    return cfg


def _merge_inv_cfg(args, log_fn: Callable) -> dict:
    cfg = dict(DEFAULT_INV_CFG)
    if args.lambda_smooth is not None:
        cfg['LAMBDA_SMOOTH'] = float(args.lambda_smooth)
        log_fn(f'[HP] 覆盖 LAMBDA_SMOOTH = {cfg["LAMBDA_SMOOTH"]:.3f}')
    if args.n_layers is not None:
        cfg['N_LAYERS'] = int(args.n_layers)
    if args.z_min_m is not None:
        cfg['Z_MIN_M'] = float(args.z_min_m)
    if args.z_max_m is not None:
        cfg['Z_MAX_M'] = float(args.z_max_m)
    if args.rho_init is not None:
        cfg['RHO_INIT'] = float(args.rho_init)
    if args.rho_bounds is not None:
        lo, hi = [float(x) for x in args.rho_bounds.split(',')]
        cfg['RHO_BOUNDS'] = (lo, hi)
    return cfg


# ================================================================
# 参数解析
# ================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='tfm_cli_run',
        description='TFM 时频电磁法完整处理流水线 (完全离线, Headless CLI)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--project', '-P',
                   default=r'D:\MyProject\myfile',
                   help='数据根目录 (myfile/, 含 1线..5线 + 坐标文件.xls)')
    p.add_argument('--out', '-O',
                   default=r'D:\WLF_Work\example_output\7_tfm_production',
                   help='输出根目录')

    # ------- Step 开关 -------
    p.add_argument('--preprocess', dest='do_pre', action='store_true', default=True,
                   help='运行 Step1-2: 预处理 (Inventory→Cube→ρa→Occam.dat)')
    p.add_argument('--no-preprocess', dest='do_pre', action='store_false',
                   help='跳过预处理')
    p.add_argument('--inv', dest='do_inv', action='store_true', default=True,
                   help='运行 Step3: 1D least_squares 反演')
    p.add_argument('--no-inv', dest='do_inv', action='store_false',
                   help='跳过1D反演')
    p.add_argument('--occam', dest='do_occam', action='store_true', default=True,
                   help='运行 Step4: 生成 Occam1D cfg/model-p/BAT (dry-run)')
    p.add_argument('--no-occam', dest='do_occam', action='store_false',
                   help='跳过Occam桥接')
    p.add_argument('--current', dest='do_current', action='store_true', default=False,
                   help='运行 Step0: 发射电流档处理 (CXXST01 波形/幅值/关断 QC)')

    # ------- 数据体检 / 配置复现 -------
    p.add_argument('--check', dest='do_check', action='store_true', default=False,
                   help='运行 Step0.5 数据体检后退出 (结构对齐/dat 健康; rc=2 有错误)')
    p.add_argument('--check-deep', dest='check_deep', action='store_true', default=False,
                   help='体检时逐个读取全部 dat 文件 (默认每 section 抽检 1 个)')
    p.add_argument('--config', dest='config_path', default=None,
                   help='从参数快照 JSON 复现历史运行 (args_snapshot_*.json 全量恢复; '
                        'cfg_snapshot_*.json 仅恢复预处理超参)')
    p.add_argument('--inv-algo', default='trf',
                   choices=list(ALGO_CHOICES) + list(ALGO_ADVANCED)
                   + ['lci', 'dl', 'pinn', 'mdn', 'tcn', 'all'],
                   help='1D 反演算法: trf(默认)/lm/occam/all 或高级算法 '
                        'occam_adaptive(λ自动寻优+L曲线) / lm_adaptive(Nielsen阻尼) / '
                        'cgls(共轭梯度) / mcmc(后验不确定性) / '
                        'lci(横向约束联合反演,拟2D断面) / '
                        'dl(U-Net 端到端) / '
                        'pinn(物理信息神经网络,物理约束嵌入损失) / '
                        'mdn(混合密度网络,不确定度量化) / '
                        'tcn(时序卷积网络,因果膨胀卷积)')
    p.add_argument('--dl-train', dest='dl_train', action='store_true',
                   default=False,
                   help='仅训练 DL 权重 (U-Net, torch 可用时) 后退出')
    p.add_argument('--dl-samples', type=int, default=4000,
                   help='DL 训练合成样本数 (默认 4000)')
    p.add_argument('--dl-train-adv', dest='dl_train_adv', default=None,
                   choices=['pinn', 'mdn', 'tcn'],
                   help='训练高级 DL 权重 (pinn/mdn/tcn) 后退出')

    # ------- Occam 桥接参数 -------
    p.add_argument('--occam-exe', default='Occam1D.exe',
                   help='Occam1D 商业可执行路径 (找不到自动进入 DRY-RUN)')
    p.add_argument('--lambda-init', type=float, default=1.0,
                   help='★ Occam 单变量首选调参 (0.1→1.0→10)')

    # ------- 预处理超参 -------
    hp_pre = p.add_argument_group('Step1-2 预处理超参数 (★ 单变量少步长)')
    hp_pre.add_argument('--rhoa-scale', type=float, default=None,
                        help='★ 首选调参. ρa 全局校准倍率 (默认 1e6)')
    hp_pre.add_argument('--g-adc', type=int, default=None,
                        help='ADC counts/V (默认 2^22=4194304)')
    hp_pre.add_argument('--n-gates', type=int, default=None,
                        help='时间门数 (默认 25)')
    hp_pre.add_argument('--t-min-us', type=float, default=None,
                        help='最早时间门 us (默认 2)')
    hp_pre.add_argument('--t-max-us', type=float, default=None,
                        help='最晚时间门 us (默认 20000)')

    # ------- 反演超参 -------
    hp_inv = p.add_argument_group('Step3 1D反演超参数 (★ 单变量少步长)')
    hp_inv.add_argument('--lambda-smooth', type=float, default=None,
                        help='★ 首选调参. 层间平滑权重 (默认 0.3)')
    hp_inv.add_argument('--n-layers', type=int, default=None,
                        help='反演层数 (默认 25)')
    hp_inv.add_argument('--z-min-m', type=float, default=None,
                        help='第一层顶界 m (默认 5)')
    hp_inv.add_argument('--z-max-m', type=float, default=None,
                        help='末层底界 m (默认 1500)')
    hp_inv.add_argument('--rho-init', type=float, default=None,
                        help='初始电阻率 Ω·m (默认 100)')
    hp_inv.add_argument('--rho-bounds', type=str, default=None,
                        help='电阻率上下界, 格式 "lo,hi" (默认 "0.1,1e8")')
    hp_inv.add_argument('--n-jobs', type=int, default=1,
                        help='并行进程数 (默认 1=串行; >1 启用 ProcessPool 逐站并行)')
    hp_inv.add_argument('--n-line-jobs', type=int, default=1,
                        help='多测线并行进程数 (1=逐线串行, 默认; >1 按测线 ProcessPool 并行并自动关闭站内并行; '
                             '0=自动 min(测线数,CPU核-1)。轻量算法受 Windows 进程启动开销影响, '
                             '建议仅在大数据/重算法(MCMC/FE正演/外部Occam)时 >1)')
    hp_inv.add_argument('--use-cache', action='store_true', default=False,
                        help='启用逐站缓存断点续跑 (cfg 变更自动失效)')
    hp_inv.add_argument('--mcmc-samples', type=int, default=2000,
                        help='MCMC 采样数 (仅 --inv-algo mcmc 有效, 默认 2000)')
    p.add_argument('--lambda-lat', type=float, default=1.0,
                       help='★ LCI 首选调参. 横向约束权重 (默认 1.0; 越大断面越匀)')
    p.add_argument('--compare', dest='do_compare', action='store_true', default=True,
                   help='运行 Step7: 多算法反演结果对比 QC (断面并排/RMSE统计/拟合曲线)')
    p.add_argument('--no-compare', dest='do_compare', action='store_false',
                   help='跳过多算法对比')
    p.add_argument('--compare-algos', default=None,
                   help='Step7 对比算法白名单 (逗号分隔, 默认扫描全部已有结果)')
    p.add_argument('--report', dest='do_report', action='store_true', default=False,
                   help='流水线结束后一键生成单文件自包含 HTML 报告 (report_gen)')
    p.add_argument('--timelapse', nargs=2, metavar=('EPOCH1_ROOT', 'EPOCH2_ROOT'),
                   default=None,
                   help='多时态对比: 两个期次输出根目录 (共用测线做电阻率变化分析)')
    p.add_argument('--timelapse-labels', default='1,2',
                   help='两期标签, 逗号分隔 (默认 1,2)')

    # ------- Step8: 标准化导出 -------
    p.add_argument('--export', dest='do_export', action='store_true', default=False,
                   help='运行 Step8: 标准化导出 (CSV/XYZ/GRD/GeoTIFF)')
    p.add_argument('--export-formats', default='csv,xyz,grd',
                   help='导出格式, 逗号分隔 (默认 csv,xyz,grd; 可选 geotiff)')
    p.add_argument('--export-algo', default=None,
                   help='导出源算法 (默认与 --inv-algo 一致)')
    p.add_argument('--export-epsg', type=int, default=None,
                   help='GeoTIFF EPSG 代码 (如 4326/4547, 可选)')
    p.add_argument('--export-scale', default='linear', choices=['linear', 'log10'],
                   help='导出数值标度 (默认 linear)')

    # ------- Step9: 2D FEM 正演 -------
    p.add_argument('--fem2d', dest='do_fem2d', action='store_true', default=False,
                   help='运行 Step9: 2D 有限元 DC 正演 (单极-单极拟断面)')
    p.add_argument('--fem2d-algo', default=None,
                   help='FEM 正演源断面算法 (默认与 --inv-algo 一致)')
    p.add_argument('--fem2d-xlim', default='-300,300',
                   help='X 范围 "x0,x1" (默认 -300,300)')
    p.add_argument('--fem2d-zmax', type=float, default=300.0,
                   help='最大深度 m (默认 300)')
    p.add_argument('--fem2d-nx', type=int, default=160,
                   help='横向节点数 (默认 160)')
    p.add_argument('--fem2d-nz', type=int, default=70,
                   help='垂向节点数 (默认 70)')
    p.add_argument('--fem2d-spacings', default='5,10,20,40',
                   help='极距, 逗号分隔 (默认 5,10,20,40)')

    # ------- Step10: VTI 各向异性 -------
    p.add_argument('--vti', dest='do_vti', action='store_true', default=False,
                   help='运行 Step10: VTI 各向异性正演对比 (各向同性 vs VTI)')
    p.add_argument('--vti-lambda', type=float, default=1.5,
                   help='各向异性系数 λ=√(ρv/ρh) (默认 1.5)')
    p.add_argument('--vti-algo', default=None,
                   help='VTI 源断面算法 (默认与 --inv-algo 一致)')

    # ------- Step11: IP 反演 -------
    p.add_argument('--ip-inv', dest='do_ip', action='store_true', default=False,
                   help='运行 Step11: 激发极化 (IP) 反演')
    p.add_argument('--ip-share-tc', dest='ip_share_tc', action='store_true',
                   default=True, help='IP 共享 τ,c (默认 True, 稳健)')
    p.add_argument('--ip-no-share-tc', dest='ip_share_tc', action='store_false',
                   help='IP 每层独立 τ,c (4N 参数, 高自由度)')
    p.add_argument('--ip-lambda', type=float, default=None,
                   help='IP 层间平滑权重 (默认 0.3, 与 1D 反演一致)')
    return p


# ================================================================
# ================================================================
# 参数快照 / 环境指纹 (--config 复现支持)
# ================================================================
_SKIP_SNAPSHOT_KEYS = {'project', 'out', 'config_path', 'do_check', 'check_deep'}


def _environment_fingerprint() -> dict:
    """当前运行环境指纹 (快照内嵌, 复现时对比告警)。"""
    import platform
    import tfm_engine
    fp = {'python': platform.python_version(),
          'platform': platform.platform(),
          'tfm_engine': tfm_engine.__version__,
          'timestamp': datetime.now().isoformat(timespec='seconds')}
    for mod in ('numpy', 'scipy', 'pandas', 'matplotlib'):
        try:
            fp[mod] = importlib.import_module(mod).__version__
        except Exception:
            fp[mod] = 'N/A'
    return fp


def _write_args_snapshot(args, out_root: Path) -> None:
    """把完整 CLI 参数 + 环境指纹写入 out_root/args_snapshot_{ts}.json。"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    payload = {
        '__format__': 'tfm-cli-args-v1',
        'environment': _environment_fingerprint(),
        'args': {k: v for k, v in vars(args).items()
                 if k not in _SKIP_SNAPSHOT_KEYS and _jsonable(v)},
    }
    path = out_root / f'args_snapshot_{ts}.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding='utf-8')
    print(f'[Snapshot] 参数快照: {path}')


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _load_args_snapshot(args) -> None:
    """从快照恢复 CLI 参数。

    支持: args_snapshot_*.json (全量恢复) / cfg_snapshot_*.json (仅预处理超参)。
    恢复时对比环境指纹, 库版本不一致只告警不阻断。
    """
    src = Path(args.config_path)
    if not src.is_file():
        raise FileNotFoundError(f'--config 快照不存在: {src}')
    data = json.loads(src.read_text(encoding='utf-8'))

    fmt = data.get('__format__')
    env = data.get('environment') or {}
    if env:
        cur = _environment_fingerprint()
        diffs = [f'{k}: 快照={env.get(k)} vs 当前={cur.get(k)}'
                 for k in ('python', 'tfm_engine', 'numpy', 'scipy')
                 if env.get(k) not in (None, cur.get(k))]
        if diffs:
            print('[Snapshot] ⚠ 环境指纹差异 (结果可能有差异, 仅告警):')
            for d in diffs:
                print(f'    {d}')

    if fmt == 'tfm-cli-args-v1':
        restored = []
        for k, v in (data.get('args') or {}).items():
            if k in _SKIP_SNAPSHOT_KEYS:
                continue
            if hasattr(args, k):
                setattr(args, k, v)
                restored.append(f'{k}={v}')
        print(f'[Snapshot] 已从 {src.name} 恢复 {len(restored)} 个参数:')
        for r in restored:
            print(f'    {r}')
    else:
        # 旧版 preprocess cfg_snapshot: 键 → CLI 参数映射
        mapping = {'RHOA_SCALE_CAL': 'rhoa_scale',
                   'G_ADC_COUNTS_PER_VOLT': 'g_adc',
                   'TEM_N_GATES': 'n_gates'}
        restored = []
        for ck, ak in mapping.items():
            if ck in data and hasattr(args, ak):
                setattr(args, ak, data[ck])
                restored.append(f'{ak}={data[ck]} (from {ck})')
        print(f'[Snapshot] 已从预处理快照 {src.name} 恢复 {len(restored)} 个超参 '
              f'(旧版快照不含反演参数):')
        for r in restored:
            print(f'    {r}')


# Main
# ================================================================
def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    # ---- --config: 参数快照反向加载 (在其余逻辑之前恢复 args) ----
    if args.config_path:
        _load_args_snapshot(args)

    project_root = Path(args.project)
    out_root     = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- --check: Step0.5 数据体检独立入口 ----
    if args.do_check:
        from tfm_engine.data_check import check_project
        r = check_project(project_root, out_root, deep=args.check_deep,
                          log_fn=lambda s: print(s))
        print(f"[Step0.5] 体检 rc={r['rc']} "
              f"({len(r['errors'])} 错误 / {len(r['warns'])} 告警)")
        return r['rc']

    # ---- 写参数快照 (args + 环境指纹, 供 --config 复现) ----
    _write_args_snapshot(args, out_root)

    # DL 训练独立入口 (与流水线解耦)
    if args.dl_train:
        from tfm_engine.dl_inversion import train_dl, _HAS_TORCH
        print(f'[DL] torch={_HAS_TORCH}, 开始训练 n={args.dl_samples} ...')
        p = train_dl(n_train=args.dl_samples,
                     backend='auto',
                     log_fn=lambda s: print(s))
        print(f'[DL] 权重已保存: {p}')
        return 0

    # 高级 DL 训练独立入口 (PINN/MDN/TCN)
    if args.dl_train_adv:
        from tfm_engine.dl_advanced import train_dl_advanced
        print(f'[DL-adv] 开始训练 {args.dl_train_adv.upper()} ...')
        p = train_dl_advanced(algo=args.dl_train_adv,
                              n_train=args.dl_samples,
                              log_fn=lambda s: print(s))
        print(f'[DL-adv] 权重已保存: {p}')
        return 0

    # 日志
    ts0 = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = out_root / f'cli_run_{ts0}.log'
    log_fn = _make_logger(log_file)

    log_fn('=' * 64)
    log_fn(' TFM CLI Pipeline 启动')
    log_fn(f'  数据根:   {project_root}')
    log_fn(f'  输出根:   {out_root}')
    log_fn(f'  Step1-2 预处理: {args.do_pre}')
    log_fn(f'  Step3   1D反演: {args.do_inv}')
    log_fn(f'  Step11  IP反演: {args.do_ip}')
    log_fn(f'  Step4   Occam : {args.do_occam}')
    log_fn(f'  Step7   Compare QC : {args.do_compare}')
    log_fn(f'  Step8   导出  : {args.do_export}')
    log_fn(f'  Step9   FEM2D : {args.do_fem2d}')
    log_fn(f'  Step10  VTI   : {args.do_vti}')
    log_fn(f'  Step5   报告  : {args.do_report}')
    log_fn(f'  日志文件: {log_file}')
    log_fn('  ★ 完全离线运行, 数据绝不上传云端')
    log_fn('=' * 64)

    # 校验数据根
    if not project_root.is_dir():
        log_fn(f'[FATAL] 数据根不存在: {project_root}')
        return 10

    # 路径快照
    summary = {
        'started_at': ts0,
        'project_root': str(project_root),
        'out_root': str(out_root),
        'steps': {'preprocess': args.do_pre, 'inversion': args.do_inv, 'occam': args.do_occam},
    }

    t_start = time.time()
    rc = 0

    # ---- Step0: 电流档处理 (可选) ----
    if args.do_current:
        try:
            log_fn('\n>>> [Step0] 发射电流档处理 启动')
            cur_summary = run_all_current(project_root, out_root, log_fn=log_fn)
            summary['current'] = cur_summary
            log_fn('<<< [Step0] 电流处理完成')
        except Exception as e:
            log_fn(f'[ERROR] 电流处理失败: {e}')
            log_fn(traceback.format_exc())
            rc = 15

    # ---- Step1-2: 预处理 ----
    if args.do_pre and rc == 0:
        try:
            log_fn('\n>>> [Step1-2] 预处理流水线 启动')
            cfg_pre = _merge_preprocess_cfg(args, log_fn)
            pre_summary = run_full_preprocess(project_root, out_root, cfg_pre, log_fn,
                                              n_line_jobs=args.n_line_jobs)
            summary['preprocess'] = pre_summary
            log_fn('<<< [Step1-2] 预处理完成')
        except Exception as e:
            log_fn(f'[ERROR] 预处理失败: {e}')
            log_fn(traceback.format_exc())
            rc = 20

    # ---- Step3: 1D 反演 (多算法) ----
    if args.do_inv and rc == 0:
        try:
            cfg_inv = _merge_inv_cfg(args, log_fn)
            algo = args.inv_algo
            if algo == 'all':
                log_fn('\n>>> [Step3] 1D 反演 三算法对比 (trf/lm/occam)')
                inv_summary = run_all_algos_compare(out_root, user_cfg=cfg_inv,
                                                    occam_cfg={'LAMBDA_INIT': args.lambda_init},
                                                    n_jobs=args.n_jobs, use_cache=args.use_cache,
                                                    n_line_jobs=args.n_line_jobs,
                                                    log_fn=log_fn)
            elif algo == 'lci':
                log_fn(f'\n>>> [Step3] LCI 横向约束联合反演 启动 '
                       f'λ_lat={args.lambda_lat} n_jobs={args.n_jobs} '
                       f'n_line_jobs={args.n_line_jobs}')
                inv_summary = run_all_lci(out_root,
                                          user_cfg={**cfg_inv, 'LAMBDA_LAT': args.lambda_lat},
                                          n_jobs=args.n_jobs,
                                          n_line_jobs=args.n_line_jobs,
                                          log_fn=log_fn)
            elif algo in ALGO_ADVANCED:
                log_fn(f'\n>>> [Step3] 高级 1D 反演 启动 [algo={algo}] '
                       f'n_jobs={args.n_jobs} n_line_jobs={args.n_line_jobs} cache={args.use_cache}')
                inv_summary = run_all_advanced(out_root, algo=algo, user_cfg=cfg_inv,
                                               n_jobs=args.n_jobs, use_cache=args.use_cache,
                                               n_line_jobs=args.n_line_jobs,
                                               mcmc_samples=args.mcmc_samples,
                                               log_fn=log_fn)
            elif algo == 'dl':
                log_fn('\n>>> [Step3] DL 端到端反演 (U-Net/线性降级)')
                inv_summary = run_all_dl(out_root,
                                         n_layers=cfg_inv.get('N_LAYERS', 25),
                                         log_fn=log_fn)
            elif algo in ('pinn', 'mdn', 'tcn'):
                from tfm_engine.dl_advanced import run_all_advanced_dl
                label = {'pinn': 'PINN', 'mdn': 'MDN', 'tcn': 'TCN'}[algo]
                log_fn(f'\n>>> [Step3] 高级DL反演 [{label}] 启动')
                inv_summary = run_all_advanced_dl(out_root, algo=algo,
                                                   n_layers=cfg_inv.get('N_LAYERS', 25),
                                                   log_fn=log_fn)
            else:
                log_fn(f'\n>>> [Step3] 1D 反演 启动 [algo={algo}] '
                       f'n_jobs={args.n_jobs} n_line_jobs={args.n_line_jobs} cache={args.use_cache}')
                inv_summary = run_all_algo(out_root, algo=algo, user_cfg=cfg_inv,
                                           occam_cfg={'LAMBDA_INIT': args.lambda_init},
                                           n_jobs=args.n_jobs, use_cache=args.use_cache,
                                           n_line_jobs=args.n_line_jobs,
                                           log_fn=log_fn)
            summary['inversion'] = inv_summary
            log_fn('<<< [Step3] 1D反演完成')
        except Exception as e:
            log_fn(f'[ERROR] 1D反演失败: {e}')
            log_fn(traceback.format_exc())
            rc = 30

    # ---- Step11: IP 激发极化反演 (可选, 依赖预处理 occam.dat) ----
    if args.do_ip and rc == 0:
        try:
            from tfm_engine.ip_inversion import run_all_ip_inversions, DEFAULT_IP_CFG
            log_fn('\n>>> [Step11] IP 激发极化反演 启动 '
                   f'share_tc={args.ip_share_tc}')
            ip_cfg = dict(DEFAULT_IP_CFG)
            ip_cfg['SHARE_TAU_C'] = args.ip_share_tc
            if args.ip_lambda is not None:
                ip_cfg['LAMBDA_SMOOTH'] = args.ip_lambda
            ip_summary = run_all_ip_inversions(out_root,
                                               user_cfg=cfg_inv if args.do_inv else None,
                                               ip_cfg=ip_cfg,
                                               log_fn=log_fn)
            summary['ip_inversion'] = ip_summary
            log_fn('<<< [Step11] IP 反演完成')
        except Exception as e:
            log_fn(f'[ERROR] IP 反演失败: {e}')
            log_fn(traceback.format_exc())
            # IP 失败不阻塞主流程
            summary.setdefault('warnings', []).append(f'Step11 ip: {e}')

    # ---- Step4: Occam 桥接 ----
    if args.do_occam and rc == 0:
        try:
            log_fn('\n>>> [Step4] Occam1D 桥接文件 生成 (DRY-RUN 默认)')
            occam_summary = materialize_all(out_root, args.occam_exe, args.lambda_init, log_fn)
            summary['occam_bridge'] = occam_summary
            log_fn('<<< [Step4] Occam 桥接完成')
        except Exception as e:
            log_fn(f'[ERROR] Occam 桥接失败: {e}')
            log_fn(traceback.format_exc())
            rc = 40

    # ---- Step7: 多算法对比 QC (反演后跑, 报告前跑便于消费对比产物) ----
    if args.do_compare and rc == 0:
        try:
            from tfm_engine.model_compare import compare_all_lines
            log_fn('\n>>> [Step7] 多算法反演对比 QC 仪表盘 启动')
            c_algos = None
            if args.compare_algos:
                c_algos = [s.strip() for s in args.compare_algos.split(',')]
            compare_summary = compare_all_lines(out_root, algos=c_algos,
                                                log_fn=log_fn)
            summary['compare'] = {lt: len(rows) for lt, rows in compare_summary.items()}
            log_fn(f'<<< [Step7] 对比完成 ({len(compare_summary)} 测线)')
        except Exception as e:
            log_fn(f'[ERROR] 多算法对比失败: {e}')
            log_fn(traceback.format_exc())
            # 对比失败不阻塞报告/后续步骤, 仅记录
            summary.setdefault('warnings', []).append(f'Step7 compare: {e}')

    # ---- Step8: 标准化导出 (可选) ----
    if args.do_export and rc == 0:
        try:
            from tfm_engine.export_formats import export_all_lines
            e_algo = args.export_algo or args.inv_algo
            if e_algo == 'all':
                e_algo = 'trf'
            e_formats = tuple(s.strip() for s in args.export_formats.split(','))
            log_fn(f'\n>>> [Step8] 标准化导出 启动 algo={e_algo} fmt={e_formats}')
            exp_summary = export_all_lines(out_root, algo=e_algo,
                                           formats=e_formats,
                                           value_scale=args.export_scale,
                                           epsg=args.export_epsg,
                                           log_fn=log_fn)
            summary['export'] = {lt: list(v.keys()) for lt, v in exp_summary.items()}
            log_fn(f'<<< [Step8] 导出完成 ({len(exp_summary)} 测线)')
        except Exception as e:
            log_fn(f'[ERROR] 标准化导出失败: {e}')
            log_fn(traceback.format_exc())
            summary.setdefault('warnings', []).append(f'Step8 export: {e}')

    # ---- Step9: 2D FEM 正演 (可选, 基于反演断面) ----
    if args.do_fem2d and rc == 0:
        try:
            from tfm_engine.fem2d_forward import (polepole_pseudosection,
                                                   rho_from_section,
                                                   plot_pseudosection)
            from tfm_engine.export_formats import load_section
            f_algo = args.fem2d_algo or args.inv_algo
            if f_algo == 'all':
                f_algo = 'trf'
            x0, x1 = [float(v) for v in args.fem2d_xlim.split(',')]
            spacings = [float(v) for v in args.fem2d_spacings.split(',')]
            log_fn(f'\n>>> [Step9] 2D FEM 正演 启动 algo={f_algo} '
                   f'xlim=({x0},{x1}) zmax={args.fem2d_zmax}')
            fem_results = {}
            for line_dir in sorted(out_root.iterdir()):
                if not (line_dir.is_dir() and re.fullmatch(r'[1-5]线', line_dir.name)):
                    continue
                lt = line_dir.name
                try:
                    sec = load_section(out_root, lt, f_algo)
                    rho_fn = rho_from_section(sec['R'], sec['chainage'],
                                              sec['z_centers'])
                    elec_x = np.linspace(x0 + 50, x1 - 50, 8)
                    df = polepole_pseudosection(
                        rho_fn, elec_x,
                        xlim=(x0, x1), zmax=args.fem2d_zmax,
                        nx=args.fem2d_nx, nz=args.fem2d_nz,
                        spacings=spacings, log_fn=log_fn)
                    out_png = line_dir / f'{lt}_fem2d_pseudo.png'
                    plot_pseudosection(df, out_png,
                                       title=f'{lt} 2D FEM 单极-单极拟断面')
                    df.to_csv(line_dir / f'{lt}_fem2d_pseudo.csv',
                              index=False, encoding='utf-8-sig')
                    fem_results[lt] = len(df)
                    log_fn(f'  {lt}: {len(df)} 个 ρa 记录 → {out_png.name}')
                except Exception as e:
                    log_fn(f'  ✗ {lt}: FEM 正演失败: {e}')
                    fem_results[lt] = {'error': str(e)}
            summary['fem2d'] = fem_results
            log_fn(f'<<< [Step9] FEM 正演完成 ({len(fem_results)} 测线)')
        except Exception as e:
            log_fn(f'[ERROR] FEM 正演失败: {e}')
            log_fn(traceback.format_exc())
            summary.setdefault('warnings', []).append(f'Step9 fem2d: {e}')

    # ---- Step10: VTI 各向异性正演对比 (可选) ----
    if args.do_vti and rc == 0:
        try:
            from tfm_engine.anisotropy import compare_iso_vti
            from tfm_engine.fem2d_forward import rho_from_section
            from tfm_engine.export_formats import load_section
            v_algo = args.vti_algo or args.inv_algo
            if v_algo == 'all':
                v_algo = 'trf'
            lam = args.vti_lambda
            log_fn(f'\n>>> [Step10] VTI 各向异性正演 启动 algo={v_algo} λ={lam}')
            vti_results = {}
            for line_dir in sorted(out_root.iterdir()):
                if not (line_dir.is_dir() and re.fullmatch(r'[1-5]线', line_dir.name)):
                    continue
                lt = line_dir.name
                try:
                    sec = load_section(out_root, lt, v_algo)
                    rho_fn = rho_from_section(sec['R'], sec['chainage'],
                                              sec['z_centers'])
                    # 均匀 λ 场
                    lam_fn = lambda x, z, _l=lam: _l
                    elec_x = np.linspace(-250, 250, 8)
                    out_png = line_dir / f'{lt}_vti_compare.png'
                    m, _ = compare_iso_vti(rho_fn, lam_fn, elec_x,
                                           out_png, log_fn=log_fn)
                    m.to_csv(line_dir / f'{lt}_vti_compare.csv',
                             index=False, encoding='utf-8-sig')
                    vti_results[lt] = len(m)
                    log_fn(f'  {lt}: ISO/VTI 对比 {len(m)} 记录 → {out_png.name}')
                except Exception as e:
                    log_fn(f'  ✗ {lt}: VTI 正演失败: {e}')
                    vti_results[lt] = {'error': str(e)}
            summary['vti'] = vti_results
            log_fn(f'<<< [Step10] VTI 正演完成 ({len(vti_results)} 测线)')
        except Exception as e:
            log_fn(f'[ERROR] VTI 正演失败: {e}')
            log_fn(traceback.format_exc())
            summary.setdefault('warnings', []).append(f'Step10 vti: {e}')

    # ---- Step5: HTML 报告 (可选, 依赖 Step7 对比产物) ----
    if args.do_report and rc == 0:
        try:
            from tfm_engine.report_gen import generate_report
            log_fn('\n>>> [Step5] 生成 HTML 报告')
            report_path = generate_report(out_root, log_fn=log_fn)
            summary['report'] = str(report_path)
            log_fn(f'<<< [Step5] 报告完成: {report_path}')
        except Exception as e:
            log_fn(f'[ERROR] 报告生成失败: {e}')
            log_fn(traceback.format_exc())
            rc = 50

    # ---- Step6: 多时态对比 (独立工具, 不依赖本次流水线) ----
    if args.timelapse:
        try:
            from tfm_engine.timelapse import compare_epochs_all_lines
            e1, e2 = args.timelapse
            lab = [s.strip() for s in args.timelapse_labels.split(',')]
            tl_algo = args.inv_algo if args.inv_algo in \
                ('trf', 'lm', 'occam', 'occam_adaptive', 'lm_adaptive',
                 'cgls', 'mcmc', 'lci', 'dl') else 'trf'
            log_fn(f'\n>>> [Step6] 多时态对比 algo={tl_algo} '
                   f'{lab[0]}→{lab[1]}')
            tl_df = compare_epochs_all_lines(e1, e2, algo=tl_algo,
                                             label1=lab[0], label2=lab[1],
                                             log_fn=log_fn)
            summary['timelapse_lines'] = list(tl_df['line']) \
                if len(tl_df) else []
            log_fn('<<< [Step6] 多时态对比完成')
        except Exception as e:
            log_fn(f'[ERROR] 多时态对比失败: {e}')
            log_fn(traceback.format_exc())
            rc = rc or 60

    # ---- 最终汇总 ----
    elapsed = time.time() - t_start
    summary['elapsed_sec'] = round(elapsed, 2)
    summary['return_code'] = rc
    summary_json = out_root / f'cli_summary_{ts0}.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log_fn('\n' + '=' * 64)
    if rc == 0:
        log_fn(f'✓ 全部成功, 耗时 {elapsed:.1f}s')
    else:
        log_fn(f'✗ 失败 (rc={rc}), 耗时 {elapsed:.1f}s')
    log_fn(f'  摘要 JSON: {summary_json}')
    log_fn(f'  完整日志 : {log_file}')
    log_fn('=' * 64)
    return rc


if __name__ == '__main__':
    sys.exit(main())
