# -*- coding: utf-8 -*-
"""
tfm_engine.viz_3d (M18)
=========================
拟 2D 断面 / 三维体可视化。

功能:
  1. 增强型 2D 断面（等高线+填充、多算法对比）
  2. 3D 体可视化（plotly 交互式 HTML）
  3. 剖面对比动画帧

3D 体数据格式:
  X: (n_stations,) 测站位置 [m]
  Y: (n_gates,) 时间/深度
  Z: (n_stations, n_gates) ρa 或反演电阻率

完全离线，纯 matplotlib + plotly（可选）。
"""
from __future__ import annotations
from typing import Optional, Callable
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm, PowerNorm

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 增强型 2D 断面
# ============================================================

def plot_2d_section_enhanced(x_stations: np.ndarray,
                             y_depths_or_times: np.ndarray,
                             z_values: np.ndarray,
                             out_png: str,
                             title: str = '2D 断面',
                             x_label: str = '测站位置 (m)',
                             y_label: str = '时间 (ms)',
                             z_label: str = '视电阻率 (Ω·m)',
                             z_range: Optional[tuple] = None,
                             cmap: str = 'jet_r',
                             add_contours: bool = True,
                             add_profiles: Optional[list] = None,
                             log_fn: Optional[Callable] = None) -> None:
    """绘制增强型 2D 断面图（填充+等高线+剖面叠加）。

    Parameters
    ----------
    x_stations : (n_stations,)
    y_depths_or_times : (n_y,)
    z_values : (n_stations, n_y)
    out_png : str
    title, x_label, y_label, z_label : str
    z_range : tuple (min, max) 或 None（自动）
    cmap : str
    add_contours : bool, 是否叠加等高线
    add_profiles : list of int, 叠加剖面线位置的测站索引
    log_fn : callable
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    if z_range is None:
        z_min = max(z_values.min(), 1)
        z_max = z_values.max()
    else:
        z_min, z_max = z_range

    # 填充
    if z_values.min() > 0:
        im = ax.pcolormesh(x_stations, y_depths_or_times, z_values.T,
                           norm=LogNorm(vmin=z_min, vmax=z_max),
                           cmap=cmap, shading='auto')
    else:
        im = ax.pcolormesh(x_stations, y_depths_or_times, z_values.T,
                           vmin=z_min, vmax=z_max,
                           cmap=cmap, shading='auto')

    # 等高线
    if add_contours:
        levels = np.geomspace(z_min, z_max, 10) if z_values.min() > 0 else np.linspace(z_min, z_max, 10)
        try:
            ax.contour(x_stations, y_depths_or_times, z_values.T, levels=levels,
                       colors='white', linewidths=0.3, alpha=0.6)
        except ValueError:
            pass  # 等高线绘制失败（如所有值相同）则跳过

    # 叠加剖面线
    if add_profiles:
        for idx in add_profiles:
            if 0 <= idx < len(x_stations):
                ax.axvline(x=x_stations[idx], color='white', linestyle='--',
                           linewidth=0.5, alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=z_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    if log_fn:
        log_fn(f'  [viz_3d] 2D 断面 → {out_png}')


def plot_section_comparison(x_stations: np.ndarray,
                            y_values: np.ndarray,
                            sections: dict,
                            out_png: str,
                            title: str = '多算法断面对比',
                            y_label: str = '时间 (ms)',
                            z_label: str = 'Ω·m',
                            z_range: Optional[tuple] = None,
                            log_fn: Optional[Callable] = None) -> None:
    """并排绘制多个断面（如 1D 反演、LCI、真频域 的 ρa 对比）。

    Parameters
    ----------
    sections : dict[str, np.ndarray], 键=算法名, 值=(n_stations, n_y)
    """
    n = len(sections)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    if z_range is None:
        z_min = max(min(sec.min() for sec in sections.values()), 1)
        z_max = max(sec.max() for sec in sections.values())
    else:
        z_min, z_max = z_range

    for ax, (name, sec) in zip(axes, sections.items()):
        if sec.min() > 0:
            im = ax.pcolormesh(x_stations, y_values, sec.T,
                               norm=LogNorm(vmin=z_min, vmax=z_max),
                               cmap='jet_r', shading='auto')
        else:
            im = ax.pcolormesh(x_stations, y_values, sec.T,
                               vmin=z_min, vmax=z_max,
                               cmap='jet_r', shading='auto')
        ax.set_xlabel('测站位置 (m)')
        ax.set_ylabel(y_label)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, label=z_label)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    if log_fn:
        log_fn(f'  [viz_3d] 断面对比 → {out_png}')


# ============================================================
# 2. 3D 体可视化（plotly 交互式 HTML）
# ============================================================

def export_3d_volume_html(x_stations: np.ndarray,
                          y_depths: np.ndarray,
                          z_stations: np.ndarray,
                          volume_data: np.ndarray,
                          out_html: str,
                          title: str = '3D 电阻率体',
                          unit: str = 'Ω·m',
                          value_range: Optional[tuple] = None,
                          log_fn: Optional[Callable] = None) -> None:
    """生成交互式 3D 体 HTML（plotly）。

    Parameters
    ----------
    x_stations : (n_x,) 测站位置 [m]
    y_depths : (n_y,) 深度 [m]
    z_stations : (n_z,) 测线位置 [m]（单条测线时 z=0）
    volume_data : (n_x, n_y, n_z) 电阻率值
    out_html : str, 输出 HTML 文件路径
    title : str
    unit : str
    value_range : tuple (min, max) 或 None
    log_fn : callable

    Notes
    -----
    单条测线: z_stations = [0], volume_data shape = (n_stations, n_layers, 1)
    多条测线: z_stations = 各测线位置, volume_data shape = (n_stations_per_line, n_layers, n_lines)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        if log_fn:
            log_fn('  [viz_3d] plotly 未安装，跳过 3D 可视化')
        return

    if value_range is None:
        vmin = max(volume_data.min(), 1)
        vmax = volume_data.max()
    else:
        vmin, vmax = value_range

    # 确保 log 范围内
    volume_safe = np.clip(volume_data, vmin, vmax)
    log_vol = np.log10(volume_safe)
    log_vmin, log_vmax = np.log10(vmin), np.log10(vmax)

    fig = go.Figure(data=go.Volume(
        x=x_stations.flatten(),
        y=y_depths.flatten(),
        z=z_stations.flatten(),
        value=log_vol.flatten(),
        isomin=log_vmin,
        isomax=log_vmax,
        opacity=0.1,
        surface_count=15,
        colorscale='Jet',
        colorbar=dict(title=f'log10({unit})', tickvals=np.linspace(log_vmin, log_vmax, 5),
                      ticktext=[f'{10**v:.0f}' for v in np.linspace(log_vmin, log_vmax, 5)]),
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='测站位置 (m)',
            yaxis_title='深度 (m)',
            zaxis_title='测线位置 (m)',
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=1),
        ),
        width=1000, height=700,
    )

    fig.write_html(out_html)
    if log_fn:
        log_fn(f'  [viz_3d] 3D 体 HTML → {out_html}')


def export_section_interactive(x_stations: np.ndarray,
                               y_times: np.ndarray,
                               z_values: np.ndarray,
                               out_html: str,
                               title: str = '交互式 ρa 断面',
                               log_fn: Optional[Callable] = None) -> None:
    """生成交互式 2D 断面 HTML（plotly，支持缩放/悬停查看数值）。

    Parameters
    ----------
    x_stations : (n_stations,)
    y_times : (n_gates,)
    z_values : (n_stations, n_gates) ρa
    out_html : str
    title : str
    log_fn : callable
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        if log_fn:
            log_fn('  [viz_3d] plotly 未安装，跳过交互式断面')
        return

    z_safe = np.clip(z_values, 1, None)
    log_z = np.log10(z_safe)

    fig = go.Figure(data=go.Heatmap(
        z=log_z.T,
        x=x_stations,
        y=y_times * 1e3,  # ms
        colorscale='Jet',
        colorbar=dict(title='log10(Ω·m)'),
        hovertemplate='x: %{x:.0f}m<br>t: %{y:.2f}ms<br>ρa: %{customdata:.1f}Ω·m<extra></extra>',
        customdata=z_safe.T,
    ))

    fig.update_layout(
        title=title,
        xaxis_title='测站位置 (m)',
        yaxis_title='时间 (ms)',
        yaxis_type='log',
        width=1200, height=600,
    )

    fig.write_html(out_html)
    if log_fn:
        log_fn(f'  [viz_3d] 交互式断面 HTML → {out_html}')


# ============================================================
# 3. 剖面对比动画帧
# ============================================================

def plot_profile_frames(x_stations: np.ndarray,
                        sections_dict: dict,
                        out_dir: str,
                        title_prefix: str = 'profile',
                        log_fn: Optional[Callable] = None) -> list:
    """为每个测站生成剖面动画帧。

    Parameters
    ----------
    x_stations : (n_stations,)
    sections_dict : dict[str, np.ndarray], 键=算法名, 值=(n_stations, n_gates)
    out_dir : str, 输出目录
    title_prefix : str

    Returns
    -------
    frame_paths : list of str
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_stations = len(x_stations)
    frame_paths = []

    for i in range(n_stations):
        fig, axes = plt.subplots(1, len(sections_dict), figsize=(5 * len(sections_dict), 5))
        if len(sections_dict) == 1:
            axes = [axes]

        for ax, (name, sec) in zip(axes, sections_dict.items()):
            z = sec[i, :]
            ax.semilogx(z, np.arange(len(z)), 'b-o', markersize=3)
            ax.set_title(f'{name}\n站 {i}')
            ax.set_xlabel('Ω·m')
            ax.set_ylabel('时间门索引')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{title_prefix} 测站 {i}', fontsize=12)
        fig.tight_layout()
        fpath = out_dir / f'{title_prefix}_frame_{i:04d}.png'
        fig.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(str(fpath))

    if log_fn:
        log_fn(f'  [viz_3d] 生成 {len(frame_paths)} 帧 → {out_dir}')

    return frame_paths


__all__ = [
    'plot_2d_section_enhanced',
    'plot_section_comparison',
    'export_3d_volume_html',
    'export_section_interactive',
    'plot_profile_frames',
]
