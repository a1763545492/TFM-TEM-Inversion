# -*- coding: utf-8 -*-
"""
model_compare.py — 多算法反演结果对比与 QC 仪表盘
================================================================
读取同一测线下不同算法 (TRF/LM/Occam/自适应Occam/自适应LM/CGLS/MCMC/LCI)
的反演产物, 生成三类 QC 成果:

  1. {line}_compare_sections.png : 多算法电阻率断面并排 (共享色标, 可公平比较)
  2. {line}_compare_fits.png     : 逐站拟合曲线 (obs ρa±误差棒 vs 各算法预测)
  3. {line}_compare_stats.csv    : RMSE/成功率/电阻率范围 统计表
  4. {line}_compare_rmse.png     : 各算法逐站 RMSE 折线对比

输入文件约定 (与 inv_algorithms/adv_inversion/lci_inversion 输出一致):
    {out_root}/{line}/{line}_inv_{algo}_R.npy
    {out_root}/{line}/{line}_inv_{algo}_depth.npy
    {out_root}/{line}/{line}_inv_{algo}_summary.csv
    {out_root}/{line}/{line}_occam.dat             (观测数据, 用于拟合曲线)

完全离线, 仅本地文件。
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from .inversion import parse_occam_dat, forward_rhoa, DEFAULT_INV_CFG

# 已知算法及中文名 (顺序即展示顺序)
ALGO_LABELS = {
    'trf': 'TRF', 'lm': 'LM', 'occam': 'Occam',
    'occam_adaptive': 'Occam自适应', 'lm_adaptive': 'LM自适应',
    'cgls': 'CGLS', 'mcmc': 'MCMC', 'lci': 'LCI横向约束',
    'dl': 'DL端到端',
}
ALGO_COLORS = {
    'trf': '#1f77b4', 'lm': '#2ca02c', 'occam': '#d62728',
    'occam_adaptive': '#ff7f0e', 'lm_adaptive': '#9467bd',
    'cgls': '#8c564b', 'mcmc': '#e377c2', 'lci': '#17becf',
    'dl': '#000000',
}


def scan_algos(out_base: Path, line_tag: str) -> list[str]:
    """扫描测线目录下实际存在结果的算法列表。"""
    found = []
    for algo in ALGO_LABELS:
        if (out_base / f'{line_tag}_inv_{algo}_R.npy').is_file():
            found.append(algo)
    return found


def load_algo_result(out_base: Path, line_tag: str, algo: str) -> Optional[dict]:
    """加载单算法结果: R[station,layer](线性Ω·m), z_centers, summary DataFrame。"""
    r_p = out_base / f'{line_tag}_inv_{algo}_R.npy'
    z_p = out_base / f'{line_tag}_inv_{algo}_depth.npy'
    s_p = out_base / f'{line_tag}_inv_{algo}_summary.csv'
    if not r_p.is_file():
        return None
    R = np.load(r_p)
    z = np.load(z_p) if z_p.is_file() else None
    summary = pd.read_csv(s_p) if s_p.is_file() else None
    rmse_med = float('nan')
    if summary is not None and 'rmse_log' in summary.columns:
        vals = pd.to_numeric(summary['rmse_log'], errors='coerce').dropna()
        if len(vals):
            rmse_med = float(vals.median())
    return {'algo': algo, 'R': R, 'z': z, 'summary': summary,
            'rmse_med': rmse_med}


def build_stats(results: dict[str, dict]) -> pd.DataFrame:
    """汇总各算法 QC 统计行。"""
    rows = []
    for algo, res in results.items():
        s = res['summary']
        R = res['R']
        row = {'algo': algo, 'label': ALGO_LABELS.get(algo, algo),
               'stations': R.shape[0], 'layers': R.shape[1],
               'rmse_median_log': res['rmse_med'],
               'success_rate': np.nan,
               'rho_min_med': float(np.median(R.min(axis=1))),
               'rho_max_med': float(np.median(R.max(axis=1)))}
        if s is not None:
            if 'rmse_log' in s.columns:
                v = pd.to_numeric(s['rmse_log'], errors='coerce').dropna()
                if len(v):
                    row.update(rmse_mean_log=float(v.mean()),
                               rmse_p10_log=float(v.quantile(0.1)),
                               rmse_p90_log=float(v.quantile(0.9)))
            if 'success' in s.columns:
                row['success_rate'] = float(pd.Series(s['success']).astype(bool).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = 'rmse_median_log'
    if sort_col in df.columns:
        df = df.sort_values(sort_col, na_position='last').reset_index(drop=True)
    return df


def plot_sections(results: dict[str, dict], out_png: Path, line_tag: str,
                  cfg: dict):
    """多算法断面对比, 共享 log10ρ 色标。"""
    algos = list(results.keys())
    n = len(algos)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    lo = np.log10(cfg['RHO_BOUNDS'][0])
    hi = np.log10(cfg['RHO_BOUNDS'][1])
    vmin = max(lo, min(np.log10(r['R']).min() for r in results.values()))
    vmax = min(hi, max(np.log10(r['R']).max() for r in results.values()))
    last_im = None
    for ax, algo in zip(axes, algos):
        res = results[algo]
        R, z = res['R'], res['z']
        X, Y = np.meshgrid(np.arange(R.shape[0]), np.log10(z), indexing='ij')
        Z = np.log10(np.clip(R, cfg['RHO_BOUNDS'][0], cfg['RHO_BOUNDS'][1]))
        last_im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet_r',
                                vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(f'{ALGO_LABELS.get(algo, algo)}\n'
                     f'RMSE_log10_med={res["rmse_med"]:.3f}', fontsize=11)
        ax.set_xlabel('测点序号')
    axes[0].set_ylabel('深度 (m, log)')
    yt = np.linspace(np.log10(cfg['Z_MIN_M']), np.log10(cfg['Z_MAX_M']), 6)
    axes[0].set_yticks(yt)
    axes[0].set_yticklabels([f'{10**v:.0f}' for v in yt])
    cb = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label('log10(ρ / Ω·m)')
    fig.suptitle(f'{line_tag} 多算法反演断面对比 (共享色标)', y=1.02, fontsize=13)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_rmse_curves(results: dict[str, dict], out_png: Path, line_tag: str):
    """逐站 RMSE_log10 折线: 直接比较各算法在每个测点的拟合优度。"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for algo, res in results.items():
        s = res['summary']
        if s is None or 'rmse_log' not in s.columns:
            continue
        v = pd.to_numeric(s['rmse_log'], errors='coerce')
        ax.plot(np.arange(len(v)), v, marker='o', ms=3, lw=1.2,
                color=ALGO_COLORS.get(algo),
                label=f'{ALGO_LABELS.get(algo, algo)} (中位={np.nanmedian(v):.3f})')
    ax.axhline(0.3, color='gray', ls='--', lw=1, label='良好阈值 0.3')
    ax.set_xlabel('测点序号'); ax.set_ylabel('RMSE_log10')
    ax.set_title(f'{line_tag} 各算法逐站拟合 RMSE 对比')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out_png, dpi=170); plt.close(fig)


def plot_station_fits(results: dict[str, dict], df_obs: pd.DataFrame,
                      out_png: Path, line_tag: str, cfg: dict,
                      max_panels: int = 12):
    """逐站拟合曲线: 观测 ρa±误差棒 vs 各算法正演预测。

    预测 = forward_rhoa(log10(R[station]), t_obs, z_algo, rho_ref)
    (线性扩散正演, 与反演内部一致)。
    """
    stations = sorted(df_obs.Station.unique())
    pick = stations if len(stations) <= max_panels else \
        [stations[int(round(i))] for i in np.linspace(0, len(stations) - 1, max_panels)]
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False)
    rho_ref = cfg.get('RHO_REF_DIFF', 100.0)
    for k, st in enumerate(pick):
        ax = axes[k // ncol][k % ncol]
        sub = df_obs[df_obs.Station == st].sort_values('Time_s')
        t = sub.Time_s.values
        ro = sub.AppRes.values
        eo = sub.StdDev.values if 'StdDev' in sub.columns else None
        good = np.isfinite(ro) & (ro > 0)
        if eo is not None:
            ax.errorbar(t[good], ro[good], yerr=eo[good], fmt='ks', ms=4,
                        capsize=2, label='观测', zorder=5)
        else:
            ax.plot(t[good], ro[good], 'ks', ms=4, label='观测', zorder=5)
        for algo, res in results.items():
            try:
                si = stations.index(st)
            except ValueError:
                continue
            if si >= res['R'].shape[0] or res['z'] is None:
                continue
            theta = np.log10(np.clip(res['R'][si], 1e-3, 1e9))
            pred = forward_rhoa(theta, t, res['z'], rho_ref)
            ax.plot(t, pred, lw=1.4, color=ALGO_COLORS.get(algo),
                    label=ALGO_LABELS.get(algo, algo))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(str(st), fontsize=9)
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(len(pick), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.supxlabel('时间 t (s)'); fig.supylabel('视电阻率 ρa (Ω·m)')
    fig.suptitle(f'{line_tag} 逐站拟合曲线 (观测 vs 各算法预测)', y=1.0)
    fig.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def compare_line(out_root: str | Path, line_tag: str,
                 algos: Optional[list[str]] = None,
                 user_cfg: Optional[dict] = None,
                 log_fn: Optional[Callable[[str], None]] = None,
                 max_fit_panels: int = 12) -> dict:
    """单测线多算法对比, 输出 4 类成果文件, 返回 stats(dict 列表形态)。"""
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_INV_CFG)
    if user_cfg:
        cfg.update(user_cfg)
    out_base = Path(out_root) / line_tag
    available = scan_algos(out_base, line_tag)
    if not available:
        log_fn(f'  [Compare] {line_tag}: 未找到任何算法结果 '
               f'(期望 {line_tag}_inv_<algo>_R.npy)')
        return {}
    if algos:
        use = [a for a in algos if a in available]
        skipped = [a for a in algos if a not in available]
        if skipped:
            log_fn(f'  [Compare] {line_tag}: 跳过缺失算法 {skipped}')
    else:
        use = available
    results = {}
    for algo in use:
        r = load_algo_result(out_base, line_tag, algo)
        if r is not None and r['z'] is not None:
            results[algo] = r
    if not results:
        log_fn(f'  [Compare] {line_tag}: 结果文件不完整 (缺 depth)')
        return {}
    log_fn(f'  [Compare] {line_tag}: 对比算法 {list(results.keys())}')

    stats_df = build_stats(results)
    stats_csv = out_base / f'{line_tag}_compare_stats.csv'
    stats_df.to_csv(stats_csv, index=False, encoding='utf-8-sig')
    log_fn(f'  [Compare] 统计表 → {stats_csv}')

    sec_png = out_base / f'{line_tag}_compare_sections.png'
    plot_sections(results, sec_png, line_tag, cfg)
    log_fn(f'  [Compare] 断面对比 → {sec_png}')

    rmse_png = out_base / f'{line_tag}_compare_rmse.png'
    plot_rmse_curves(results, rmse_png, line_tag)
    log_fn(f'  [Compare] 逐站RMSE → {rmse_png}')

    dat_p = out_base / f'{line_tag}_occam.dat'
    if dat_p.is_file():
        try:
            df_obs = parse_occam_dat(dat_p)
            fit_png = out_base / f'{line_tag}_compare_fits.png'
            plot_station_fits(results, df_obs, fit_png, line_tag, cfg,
                              max_panels=max_fit_panels)
            log_fn(f'  [Compare] 拟合曲线 → {fit_png}')
        except Exception as e:
            log_fn(f'  [Compare] 拟合曲线生成失败(不影响其他成果): {e}')

    return stats_df.to_dict(orient='records')


def compare_all_lines(out_root: str | Path,
                      algos: Optional[list[str]] = None,
                      user_cfg: Optional[dict] = None,
                      log_fn: Optional[Callable[[str], None]] = None,
                      max_fit_panels: int = 12) -> dict:
    """遍历输出根下全部 1线..5线, 汇总全线统计; 另存总表。"""
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    lines = sorted([p.name for p in out_root.iterdir() if p.is_dir()
                    and re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== 多算法对比 QC  测线={lines} ==========')
    all_rows = []
    per_line = {}
    for lt in lines:
        rows = compare_line(out_root, lt, algos, user_cfg, log_fn,
                            max_fit_panels=max_fit_panels)
        per_line[lt] = rows
        for row in rows:
            all_rows.append({'line': lt, **row})
    if all_rows:
        df = pd.DataFrame(all_rows)
        total_csv = out_root / 'all_lines_compare_stats.csv'
        df.to_csv(total_csv, index=False, encoding='utf-8-sig')
        log_fn(f'  [Compare] 全线总表 → {total_csv}')
    return per_line
