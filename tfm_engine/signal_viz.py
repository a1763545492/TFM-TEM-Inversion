# -*- coding: utf-8 -*-
"""
tfm_engine.signal_viz (M5)
==========================
TEM curview_kru 风格信号曲线可视化（完全统一图形风格，供 CLI + GUI 共用）：
  ┌─ Ex ─┐┌─ Ey ─┐┌─ Hx ─┐┌─ Hy ─┐
  │原始  ││原始  ││原始  ││原始  │  (time-domain full waveform, t_cutoff竖线)
  │去DC  ││去DC  ││去DC  ││去DC  │
  │陷波  ││陷波  ││陷波  ││陷波  │  (50/100/150Hz notch + bandpass)
  │衰减  ││衰减  ││衰减  ││衰减  │  (关断后 gates loglog, 叠加 median + 误差棒)
  4×4 = 16 子图 FigureCanvas（GUI内嵌 tkinter）+ 批量 PNG 导出
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 110

_CH_ORDER = ['Ex', 'Ey', 'Hx', 'Hy']
_ROW_LABELS = ['原始波形 (raw)', '去DC (DC removed)',
               '陷波+带通 (notch+bandpass)', '衰减曲线 (gated decay)']


def make_decay_4x4figure(station_decay: dict,
                         station_code: str = '',
                         line_tag: str = '',
                         xlim_pre_ms: tuple = (-5.0, 30.0),  # 相对关断点 (ms)
                         ) -> 'matplotlib.figure.Figure':
    """
    构造 4×4 子图信号图（TEM curview_kru）。
    station_decay: preprocess_plus.process_one_cube_decay 返回的 dict
                   或 decay_curves.npz 中单个 station: {ch: {t_full_s, raw, dc_removed,
                   notched, bandpass, t_cutoff_s, gates_t_s, gates_values}}
    """
    fig, axes = plt.subplots(4, 4, figsize=(17, 10), sharex='col',
                             gridspec_kw={'hspace': 0.32, 'wspace': 0.28})
    fig.suptitle(f'TEM Signal Viewer  {line_tag} · Station {station_code}',
                 fontsize=13, y=0.995, weight='bold')
    for ci, ch in enumerate(_CH_ORDER):
        chd = station_decay.get(ch)
        if chd is None:
            for r in range(4):
                axes[r, ci].text(0.5, 0.5, f'{ch}: NO DATA', ha='center', va='center',
                                 transform=axes[r, ci].transAxes, color='gray', fontsize=10)
            axes[0, ci].set_title(ch, fontsize=12, weight='bold', pad=6)
            continue
        axes[0, ci].set_title(ch, fontsize=12, weight='bold', pad=6)
        t_cut = chd['t_cutoff_s']
        t_full_ms = (chd['t_full_s'] - t_cut) * 1000.0  # ms, 0=关断时刻
        lo_ms, hi_ms = xlim_pre_ms
        mask = (t_full_ms >= lo_ms) & (t_full_ms <= hi_ms)
        # Row 0: raw
        ax = axes[0, ci]
        ax.plot(t_full_ms[mask], chd['raw'][mask], linewidth=0.6, color='#1f4e79', alpha=0.9)
        ax.axvline(0.0, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
        # Row 1: DC removed
        ax = axes[1, ci]
        ax.plot(t_full_ms[mask], chd['dc_removed'][mask], linewidth=0.6, color='#2e75b5', alpha=0.9)
        ax.axvline(0.0, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
        # Row 2: notched + bandpass
        ax = axes[2, ci]
        ax.plot(t_full_ms[mask], chd['notched'][mask],  linewidth=0.5, color='#a5a5a5',
                alpha=0.6, label='notch 50/100/150 Hz')
        ax.plot(t_full_ms[mask], chd['bandpass'][mask], linewidth=0.7, color='#c00000',
                alpha=0.9, label='bandpass 2Hz-80kHz')
        ax.axvline(0.0, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
        ax.legend(fontsize=6, loc='upper right', frameon=False)
        # Row 3: gated decay (关断后 t>0, x=log10(t us), y=log10(|V|))
        ax = axes[3, ci]
        gt_s = chd['gates_t_s']
        gv = np.maximum(np.abs(chd['gates_values']), 1e-30)
        ax.loglog(gt_s * 1e6, gv, 'o-', color='#2e75b5', linewidth=1.1,
                  markersize=4.5, markerfacecolor='white', markeredgewidth=1.1)
        ax.set_xlabel('Time after cutoff (μs, log)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.6)
    # 美化行列 label
    for r in range(4):
        ylab = _ROW_LABELS[r]
        axes[r, 0].set_ylabel(ylab, fontsize=9, labelpad=6)
    for ci in range(4):
        axes[3, ci].set_xlabel('Time after cutoff (μs, log)' if ci == 3 else '')
    for ci in range(4):
        axes[0, ci].set_ylabel('Amp (raw)' if ci == 0 else '')
    return fig


def save_station_png(station_decay: dict, out_png: str | Path,
                     station_code: str = '', line_tag: str = '', **kw) -> Path:
    fig = make_decay_4x4figure(station_decay, station_code, line_tag, **kw)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_png


def batch_save_decay_pngs(decay_npz_or_dict: str | Path | dict,
                          out_dir: str | Path,
                          line_tag: str = '',
                          station_subset: Optional[Iterable[str]] = None,
                          log_fn=None) -> int:
    """批量保存每条测线所有 station 的衰减曲线 PNG。返回保存数。"""
    log_fn = log_fn or (lambda s: None)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(decay_npz_or_dict, (str, Path)):
        npz = np.load(Path(decay_npz_or_dict), allow_pickle=True)
        stations = list(npz['stations'].tolist())
        data = npz['data'].item()
    else:
        data = decay_npz_or_dict
        stations = list(data.keys())
    if station_subset is not None:
        sset = set(station_subset)
        stations = [s for s in stations if s in sset]
    cnt = 0
    for st in stations:
        st_safe = str(st).replace('/', '_').replace('\\', '_')
        p = save_station_png(data[st], out_dir / f'{line_tag}_{st_safe}_signals.png',
                             station_code=str(st), line_tag=line_tag)
        cnt += 1
    log_fn(f"  [SIGNAL] {cnt} PNG → {out_dir}")
    return cnt


# ============================================================
# 通道对比图（所有 stations 某通道某阶段叠加 —— TEM verify_kru）
# ============================================================
def plot_channel_overlay(decay_data: dict, channel: str = 'Hx',
                         attr: str = 'bandpass',
                         window_ms: tuple = (-2, 15),
                         max_stations: int = 20,
                         out_png: Optional[str | Path] = None,
                         title: str = '') -> 'matplotlib.figure.Figure':
    """TEM verify_kru: 多站 Hx bandpass 关断窗口叠加。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    keys = list(decay_data.keys())[:max_stations]
    from matplotlib import cm
    cmap = cm.get_cmap('tab20b', max(len(keys), 1))
    for i, st in enumerate(keys):
        chd = decay_data[st].get(channel);
        if chd is None: continue
        t_ms = (chd['t_full_s'] - chd['t_cutoff_s']) * 1000.0
        mask = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
        ax.plot(t_ms[mask], chd[attr][mask], linewidth=0.8,
                color=cmap(i % cmap.N), alpha=0.85, label=str(st))
    ax.axvline(0, color='red', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.set_xlabel('Time from cutoff (ms)'); ax.set_ylabel(f'{channel} {attr}')
    ax.set_title(title or f'{channel} · {attr}  overlay ({len(keys)} stations, TEM verify-kru style)',
                 fontsize=11)
    ax.grid(True, which='major', linestyle=':', linewidth=0.4)
    ax.legend(ncol=3, fontsize=6, loc='upper right', frameon=False)
    fig.tight_layout()
    if out_png:
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150); plt.close(fig)
    return fig


__all__ = ['make_decay_4x4figure', 'save_station_png',
           'batch_save_decay_pngs', 'plot_channel_overlay']
