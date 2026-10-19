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
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

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
    p.add_argument('--inv-algo', default='trf',
                   choices=list(ALGO_CHOICES) + list(ALGO_ADVANCED)
                   + ['lci', 'dl', 'all'],
                   help='1D 反演算法: trf(默认)/lm/occam/all 或高级算法 '
                        'occam_adaptive(λ自动寻优+L曲线) / lm_adaptive(Nielsen阻尼) / '
                        'cgls(共轭梯度) / mcmc(后验不确定性) / '
                        'lci(横向约束联合反演,拟2D断面) / '
                        'dl(U-Net 端到端, 首次运行自动训练权重)')
    p.add_argument('--dl-train', dest='dl_train', action='store_true',
                   default=False,
                   help='仅训练 DL 权重 (U-Net, torch 可用时) 后退出')
    p.add_argument('--dl-samples', type=int, default=4000,
                   help='DL 训练合成样本数 (默认 4000)')

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
    return p


# ================================================================
# Main
# ================================================================
def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    project_root = Path(args.project)
    out_root     = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # DL 训练独立入口 (与流水线解耦)
    if args.dl_train:
        from tfm_engine.dl_inversion import train_dl, _HAS_TORCH
        print(f'[DL] torch={_HAS_TORCH}, 开始训练 n={args.dl_samples} ...')
        p = train_dl(n_train=args.dl_samples,
                     backend='auto',
                     log_fn=lambda s: print(s))
        print(f'[DL] 权重已保存: {p}')
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
    log_fn(f'  Step4   Occam : {args.do_occam}')
    log_fn(f'  Step7   Compare QC : {args.do_compare}')
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
