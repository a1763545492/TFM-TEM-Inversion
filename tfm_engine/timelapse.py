# -*- coding: utf-8 -*-
"""
timelapse.py — 多时态监测对比 (电阻率变化分析)
================================================================
同一测线在不同观测期次 (epoch) 的反演结果差异分析, 服务于
地下水/注浆/含水层动态监测:

  对齐策略 (无需两期网格严格相同):
    1. 优先按 station_code 对齐公共站; 公共站 < 3 时按里程线性重采样
    2. 深度网格不同则在 log10ρ 域按深度线性插值到基准期网格

  成果 (默认写入 epoch2 根目录下的 _timelapse/):
    {line}_t{label2}-t{label1}_dlog10rho.png   Δlog10ρ 三联断面
    {line}_..._change_pct.png                  变化率(%) 断面 + 变化分布直方图
    {line}_..._changes.csv                     逐站逐层差异长表
    timelapse_summary.csv                      全线变化统计汇总

Δlog10ρ = log10(ρ₂/ρ₁)  (正=电阻率升高, 负=降低)
变化率  = (ρ₂−ρ₁)/ρ₁·100%
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

from .export_formats import load_section


def align_epochs(sec1: dict, sec2: dict) -> dict:
    """把第 2 期对齐到第 1 期网格 (log10ρ 域)。

    返回对齐后的 L1 (n,m), L2 (n,m), chainage, z, stations, align_mode。
    """
    z1 = sec1['z_centers']; z2 = sec2['z_centers']
    L1_all = np.log10(np.maximum(sec1['R'], 1e-9))
    L2_all = np.log10(np.maximum(sec2['R'], 1e-9))

    # ---- 测站对齐 ----
    codes1 = list(sec1['stations']); codes2 = list(sec2['stations'])
    common = [c for c in codes1 if c in codes2]
    if len(common) >= 3:
        idx1 = [codes1.index(c) for c in common]
        idx2 = [codes2.index(c) for c in common]
        L1 = L1_all[idx1]
        # 深度网格对齐 (作用在 L2 站上)
        if z2.shape != z1.shape or not np.allclose(z2, z1, rtol=0.03):
            L2s = np.array([np.interp(z1, z2, L2_all[i]) for i in idx2])
        else:
            L2s = L2_all[idx2]
        chainage = sec1['chainage'][idx1]
        xy = sec1['xy'][idx1]
        stations = common
        mode = 'station_code'
    else:
        # 按里程重采样: 以基准期里程为目标, np.interp 每个深度
        x1 = sec1['chainage']; x2 = sec2['chainage']
        L2_on_x = np.empty_like(L1_all)
        for j in range(L2_all.shape[1]):
            L2_on_x[:, j] = np.interp(x1, x2, L2_all[:, j])
        L1 = L1_all
        if z2.shape != z1.shape or not np.allclose(z2, z1, rtol=0.03):
            L2s = np.array([np.interp(z1, z2, L2_on_x[i])
                            for i in range(L2_on_x.shape[0])])
        else:
            L2s = L2_on_x
        chainage = x1
        xy = sec1['xy']
        stations = codes1
        mode = 'chainage_interp'

    return {'L1': L1, 'L2': L2s, 'z': z1, 'chainage': chainage,
            'xy': xy, 'stations': stations, 'mode': mode}


def plot_dlog_sections(aln: dict, out_png: Path, line_tag: str,
                       label1: str, label2: str,
                       sym_lim: Optional[float] = None):
    """三联: 期1 log10ρ / 期2 log10ρ / Δlog10ρ (红蓝对称色标)。"""
    L1, L2, z = aln['L1'], aln['L2'], aln['z']
    dL = L2 - L1
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
    vmin = min(L1.min(), L2.min()); vmax = max(L1.max(), L2.max())
    x = aln['chainage']
    X, Zg = np.meshgrid(x, z, indexing='ij')
    for ax, L, ttl in [
        (axes[0], L1, f'期次1 {label1}'),
        (axes[1], L2, f'期次2 {label2}'),
    ]:
        im = ax.pcolormesh(X, Zg, L, shading='auto', cmap='jet_r',
                           vmin=vmin, vmax=vmax)
        ax.invert_yaxis(); ax.set_title(ttl)
        ax.set_xlabel('里程 (m)')
        fig.colorbar(im, ax=ax, shrink=0.9, label='log10(ρ/Ω·m)')
    if sym_lim is None:
        sym_lim = float(np.nanpercentile(np.abs(dL), 98))
        sym_lim = max(sym_lim, 0.05)
    im2 = axes[2].pcolormesh(X, Zg, dL, shading='auto', cmap='RdBu_r',
                             vmin=-sym_lim, vmax=sym_lim)
    axes[2].invert_yaxis()
    axes[2].set_title(f'Δlog10ρ = log10(ρ2/ρ1)\n'
                      f'med={np.median(dL):+.3f}  '
                      f'P98|Δ|={np.percentile(np.abs(dL), 98):.3f}')
    axes[2].set_xlabel('里程 (m)')
    fig.colorbar(im2, ax=axes[2], shrink=0.9, label='Δlog10ρ')
    axes[0].set_ylabel('深度 (m)')
    fig.suptitle(f'{line_tag} 多时态电阻率变化对比 (对齐: {aln["mode"]})',
                 y=1.02)
    fig.savefig(out_png, dpi=170, bbox_inches='tight'); plt.close(fig)


def plot_change_pct(aln: dict, out_png: Path, line_tag: str,
                    threshold_pct: float = 20.0):
    """变化率%断面 + 变化幅度分布直方图。"""
    dL = aln['L2'] - aln['L1']
    pct = (10.0 ** dL - 1.0) * 100.0
    pct_clip = np.clip(pct, -100, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={'width_ratios': [3, 1.2]})
    X, Zg = np.meshgrid(aln['chainage'], aln['z'], indexing='ij')
    lim = float(np.nanpercentile(np.abs(pct_clip), 98))
    lim = max(lim, 10.0)
    im = ax1.pcolormesh(X, Zg, pct_clip, shading='auto', cmap='RdBu_r',
                        vmin=-lim, vmax=lim)
    ax1.invert_yaxis(); ax1.set_xlabel('里程 (m)'); ax1.set_ylabel('深度 (m)')
    ax1.set_title(f'{line_tag} 电阻率变化率 (ρ2−ρ1)/ρ1 (%)')
    fig.colorbar(im, ax=ax1, label='%')
    ax2.hist(pct.ravel(), bins=60, color='#2f6db3')
    ax2.axvline(threshold_pct, color='r', ls='--', lw=1,
                label=f'±{threshold_pct:.0f}% 阈值')
    ax2.axvline(-threshold_pct, color='r', ls='--', lw=1)
    ax2.set_xlabel('变化率 (%)'); ax2.set_ylabel('节点数')
    ax2.set_title('变化分布'); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=170); plt.close(fig)


def compare_epochs(epoch1_root: str | Path, epoch2_root: str | Path,
                   line_tag: str, algo: str = 'trf',
                   label1: str = '1', label2: str = '2',
                   out_dir: Optional[str | Path] = None,
                   threshold_log10: float = 0.1,
                   log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """单线两期对比。threshold_log10: 显著变化阈值 (Δlog10ρ, 0.1≈±26%)。"""
    log_fn = log_fn or (lambda s: print(s))
    sec1 = load_section(epoch1_root, line_tag, algo)
    sec2 = load_section(epoch2_root, line_tag, algo)
    aln = align_epochs(sec1, sec2)
    log_fn(f'  [Timelapse] {line_tag}: 对齐方式={aln["mode"]} '
          f'公共网格={aln["L1"].shape}')

    out_dir = Path(out_dir) if out_dir else \
        Path(epoch2_root) / '_timelapse'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{line_tag}_t{label2}-t{label1}_{algo}'

    png1 = out_dir / f'{stem}_dlog10rho.png'
    plot_dlog_sections(aln, png1, line_tag, label1, label2)
    png2 = out_dir / f'{stem}_change_pct.png'
    plot_change_pct(aln, png2, line_tag)
    log_fn(f'  [Timelapse] 差异断面 → {png1}')
    log_fn(f'  [Timelapse] 变化率断面 → {png2}')

    dL = aln['L2'] - aln['L1']
    pct = (10.0 ** dL - 1.0) * 100.0
    n_st, n_l = dL.shape
    rows = []
    for i in range(n_st):
        for j in range(n_l):
            rows.append({'station_code': aln['stations'][i],
                         'chainage_m': aln['chainage'][i],
                         'depth_m': aln['z'][j],
                         'rho1_ohmm': 10.0 ** aln['L1'][i, j],
                         'rho2_ohmm': 10.0 ** aln['L2'][i, j],
                         'dlog10rho': dL[i, j],
                         'change_pct': pct[i, j],
                         'significant': bool(abs(dL[i, j]) >= threshold_log10)})
    df = pd.DataFrame(rows)
    csv_p = out_dir / f'{stem}_changes.csv'
    df.to_csv(csv_p, index=False, encoding='utf-8-sig')

    stats = {
        'line': line_tag, 'algo': algo, 'label1': label1, 'label2': label2,
        'stations': n_st, 'layers': n_l, 'align_mode': aln['mode'],
        'dlog10_median': float(np.median(dL)),
        'dlog10_p05': float(np.percentile(dL, 5)),
        'dlog10_p95': float(np.percentile(dL, 95)),
        'change_pct_median': float(np.median(pct)),
        'frac_up_over_thr': float((dL >= threshold_log10).mean()),
        'frac_down_over_thr': float((dL <= -threshold_log10).mean()),
        'frac_unchanged': float((np.abs(dL) < threshold_log10).mean()),
        'png_dlog10': str(png1), 'png_change_pct': str(png2),
        'csv_changes': str(csv_p),
    }
    return stats


def compare_epochs_all_lines(epoch1_root: str | Path, epoch2_root: str | Path,
                             algo: str = 'trf',
                             label1: str = '1', label2: str = '2',
                             out_dir: Optional[str | Path] = None,
                             log_fn: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """遍历两期根目录下都存在的测线, 输出全线汇总 CSV。"""
    log_fn = log_fn or (lambda s: print(s))
    r1, r2 = Path(epoch1_root), Path(epoch2_root)
    l1 = {p.name for p in r1.iterdir() if p.is_dir()
          and re.fullmatch(r'[1-5]线', p.name)}
    l2 = {p.name for p in r2.iterdir() if p.is_dir()
          and re.fullmatch(r'[1-5]线', p.name)}
    common = sorted(l1 & l2)
    log_fn(f'\n========== 多时态对比 algo={algo} '
          f'期1={label1} 期2={label2} 公共测线={common} ==========')
    out_dir = Path(out_dir) if out_dir else r2 / '_timelapse'
    rows = []
    for lt in common:
        try:
            rows.append(compare_epochs(r1, r2, lt, algo, label1, label2,
                                       out_dir=out_dir, log_fn=log_fn))
        except Exception as e:
            log_fn(f'  ✗ {lt}: 时态对比失败: {e}')
    df = pd.DataFrame(rows)
    if len(df):
        summary_p = out_dir / 'timelapse_summary.csv'
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_p, index=False, encoding='utf-8-sig')
        log_fn(f'  [Timelapse] 全线汇总 → {summary_p}')
    return df
