# -*- coding: utf-8 -*-
"""
tfm_engine.static_correction (M14)
====================================
静态效应/飞行点校正模块。

静态效应来源:
  - 近地表电性不均匀体（局部良导/高阻体）导致 ρa 曲线整体偏移
  - 电极接地电阻变化、地形效应
  - 表现: ρa 断面出现条带状偏移（相邻测站同一层位电阻率突变）

飞行点来源:
  - 电磁干扰、仪器故障、记录错误
  - 表现: 单点 ρa 值远离相邻点 3σ 以上

校正策略:
  1. 飞行点检测: Hampel 滤波器（中位数 + MAD 阈值）
  2. 静态校正: 稳健趋势去除（LOWESS / 中位数滤波）
     - 对每条测线，沿测站方向对 log10(ρa) 做趋势估计
     - 去除趋势后加回区域背景（保留区域异常，去除局部偏移）
  3. 可选: 空间连续性约束（相邻站模型横向平滑，LCI 前置处理）

统一接口:
  detect_outliers(rhoa_2d, method='hampel', ...) → mask
  remove_static_shift(rhoa_2d, station_x, ...) → rhoa_corrected
  correct_profile(rhoa_2d, station_x, ...) → dict(rhoa_corr, outlier_mask, trend)

完全离线，纯 numpy + scipy。
"""
from __future__ import annotations
from typing import Optional, Callable
import numpy as np
from scipy import stats, signal, interpolate


# ============================================================
# 1. 飞行点检测（Hampel 滤波器）
# ============================================================

def hampel_filter(x: np.ndarray, window: int = 5,
                  n_sigma: float = 3.0) -> np.ndarray:
    """Hampel 滤波器检测异常点。

    Parameters
    ----------
    x : (N,) 输入序列
    window : int, 滑动窗口半宽（总窗口 = 2*window+1）
    n_sigma : float, MAD 阈值倍数

    Returns
    -------
    mask : (N,) bool, True=异常点
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mask = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        window_data = x[lo:hi]
        med = np.median(window_data)
        mad = np.median(np.abs(window_data - med))
        if mad > 1e-12:
            sigma_est = 1.4826 * mad  # MAD → 标准差估计
            if np.abs(x[i] - med) > n_sigma * sigma_est:
                mask[i] = True

    return mask


def detect_outliers_2d(rhoa_2d: np.ndarray,
                       window: int = 3,
                       n_sigma: float = 3.0,
                       axis: int = 1) -> np.ndarray:
    """对 2D ρa 断面逐层（或逐站）检测飞行点。

    Parameters
    ----------
    rhoa_2d : (n_stations, n_gates) 或 (n_stations, n_layers) 视电阻率
    window : int, Hampel 窗口半宽
    n_sigma : float, 阈值
    axis : int, 沿哪个轴检测（1=沿测站方向，默认；0=沿层/门方向）

    Returns
    -------
    mask : (n_stations, n_gates) bool, True=飞行点
    """
    rhoa_2d = np.asarray(rhoa_2d)
    if rhoa_2d.ndim != 2:
        raise ValueError('rhoa_2d 必须是 2D 数组')

    mask = np.zeros_like(rhoa_2d, dtype=bool)

    if axis == 1:  # 沿测站方向（对每一层/门）
        for j in range(rhoa_2d.shape[1]):
            mask[:, j] = hampel_filter(rhoa_2d[:, j], window, n_sigma)
    else:  # 沿层/门方向（对每一站）
        for i in range(rhoa_2d.shape[0]):
            mask[i, :] = hampel_filter(rhoa_2d[i, :], window, n_sigma)

    return mask


# ============================================================
# 2. 静态效应校正（趋势去除）
# ============================================================

def remove_static_shift(rhoa_2d: np.ndarray,
                        station_x: np.ndarray,
                        method: str = 'median_filter',
                        filter_width: int = 5,
                        preserve_regional: bool = True,
                        log_fn: Optional[Callable[[str], None]] = None) -> np.ndarray:
    """去除静态效应导致的条带状偏移。

    原理: 静态偏移是近地表局部异常导致的整体偏移，与深部构造无关。
    对 log10(ρa) 沿测站方向做稳健趋势估计，去除该趋势。

    Parameters
    ----------
    rhoa_2d : (n_stations, n_gates) 视电阻率断面
    station_x : (n_stations,) 测站位置 [m]
    method : str, 'median_filter' / 'lowess' / 'savgol'
    filter_width : int, 滤波器宽度（测站数）
    preserve_regional : bool, 是否保留区域背景（默认 True，只去局部偏移）
    log_fn : callable

    Returns
    -------
    rhoa_corrected : (n_stations, n_gates) 校正后视电阻率
    """
    rhoa_2d = np.asarray(rhoa_2d, dtype=np.float64)
    n_st, n_g = rhoa_2d.shape

    if n_st < 3:
        if log_fn:
            log_fn('  [静态校正] 测站数<3，跳过')
        return rhoa_2d.copy()

    # 对数变换（静态偏移在 log 域是加性）
    log_rhoa = np.log10(np.maximum(rhoa_2d, 1e-3))

    # 估计区域背景（大尺度趋势，保留）
    if preserve_regional:
        # 用大窗口中位数滤波估计背景
        bg_width = max(filter_width * 3, n_st // 3)
        bg_width = min(bg_width, n_st // 2 * 2 + 1)  # 奇数
        if bg_width >= 3:
            background = signal.medfilt(log_rhoa, kernel_size=(bg_width, 1))
        else:
            background = np.mean(log_rhoa, axis=0, keepdims=True)
    else:
        background = np.zeros_like(log_rhoa)

    # 估计局部偏移（小尺度趋势，去除）
    if method == 'median_filter':
        trend = signal.medfilt(log_rhoa, kernel_size=(filter_width, 1))
    elif method == 'savgol':
        trend = signal.savgol_filter(log_rhoa, filter_width, 2, axis=0)
    elif method == 'lowess':
        # LOWESS（局部加权回归）
        from scipy.stats import binned_statistic
        trend = np.zeros_like(log_rhoa)
        for j in range(n_g):
            # 对每一层，沿测站方向做 LOWESS
            # 简化：用移动平均 + 线性插值
            k = min(filter_width, n_st // 2)
            trend[:, j] = signal.savgol_filter(log_rhoa[:, j], 2 * k + 1, 2)
    else:
        raise ValueError(f'未知 method: {method}')

    # 校正：去除局部偏移，保留区域背景
    log_rhoa_corr = log_rhoa - (trend - background)
    rhoa_corrected = 10.0 ** log_rhoa_corr

    if log_fn:
        shift_db = np.mean(np.abs(trend - background))
        log_fn(f'  [静态校正] 方法={method} 平均偏移={shift_db:.3f} log10(Ω·m) '
               f'校正范围=[{rhoa_corrected.min():.1f},{rhoa_corrected.max():.1f}]Ω·m')

    return rhoa_corrected


# ============================================================
# 3. 综合校正流水线
# ============================================================

def correct_profile(rhoa_2d: np.ndarray,
                    station_x: np.ndarray,
                    outlier_method: str = 'hampel',
                    outlier_window: int = 3,
                    outlier_sigma: float = 3.0,
                    static_method: str = 'median_filter',
                    static_width: int = 5,
                    interpolate_outliers: bool = True,
                    log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """综合校正流水线：飞行点检测 → 插值 → 静态校正。

    Parameters
    ----------
    rhoa_2d : (n_stations, n_gates) 原始视电阻率
    station_x : (n_stations,) 测站位置
    outlier_method : str, 'hampel' 或 'zscore'
    outlier_window : int
    outlier_sigma : float
    static_method : str
    static_width : int
    interpolate_outliers : bool, 是否插值替换飞行点
    log_fn : callable

    Returns
    -------
    dict with keys:
      'rhoa_corrected': (n_stations, n_gates) 最终校正结果
      'outlier_mask': (n_stations, n_gates) bool
      'static_shift': (n_stations, n_gates) 估计的静态偏移量 [log10]
      'trend': (n_stations, n_gates) 估计的趋势
    """
    rhoa_2d = np.asarray(rhoa_2d, dtype=np.float64)
    n_st, n_g = rhoa_2d.shape

    # Step 1: 飞行点检测
    if outlier_method == 'hampel':
        outlier_mask = detect_outliers_2d(rhoa_2d, outlier_window, outlier_sigma, axis=1)
    elif outlier_method == 'zscore':
        z = np.abs(stats.zscore(rhoa_2d, axis=0, nan_policy='omit'))
        outlier_mask = z > outlier_sigma
    else:
        raise ValueError(f'未知 outlier_method: {outlier_method}')

    n_outliers = np.sum(outlier_mask)

    # Step 2: 插值替换飞行点
    rhoa_clean = rhoa_2d.copy()
    if interpolate_outliers and n_outliers > 0:
        for j in range(n_g):
            col = rhoa_clean[:, j]
            mask_col = outlier_mask[:, j]
            if np.any(mask_col) and np.any(~mask_col):
                # 线性插值
                x_good = station_x[~mask_col]
                y_good = col[~mask_col]
                f_interp = interpolate.interp1d(x_good, y_good, kind='linear',
                                                bounds_error=False, fill_value='extrapolate')
                col[mask_col] = f_interp(station_x[mask_col])
                rhoa_clean[:, j] = col

    # Step 3: 静态校正
    log_rhoa = np.log10(np.maximum(rhoa_clean, 1e-3))
    bg_width = max(static_width * 3, n_st // 3)
    bg_width = min(bg_width, n_st // 2 * 2 + 1)
    if bg_width >= 3:
        background = signal.medfilt(log_rhoa, kernel_size=(bg_width, 1))
    else:
        background = np.mean(log_rhoa, axis=0, keepdims=True)

    if static_method == 'median_filter':
        trend = signal.medfilt(log_rhoa, kernel_size=(static_width, 1))
    elif static_method == 'savgol':
        trend = signal.savgol_filter(log_rhoa, static_width, 2, axis=0)
    else:
        trend = signal.medfilt(log_rhoa, kernel_size=(static_width, 1))

    static_shift = trend - background
    log_rhoa_corr = log_rhoa - static_shift
    rhoa_corrected = 10.0 ** log_rhoa_corr

    if log_fn:
        log_fn(f'  [综合校正] 飞行点={n_outliers}/{n_st*n_g} ({100*n_outliers/(n_st*n_g):.1f}%) '
               f'平均静态偏移={np.mean(np.abs(static_shift)):.3f} log10(Ω·m)')

    return {
        'rhoa_corrected': rhoa_corrected,
        'outlier_mask': outlier_mask,
        'static_shift': static_shift,
        'trend': trend,
        'background': background,
    }


# ============================================================
# 4. 可视化
# ============================================================

def plot_correction_comparison(rhoa_orig: np.ndarray,
                               rhoa_corr: np.ndarray,
                               outlier_mask: np.ndarray,
                               station_x: np.ndarray,
                               gates_sec: Optional[np.ndarray] = None,
                               out_png: Optional[str] = None,
                               title: str = '静态效应校正对比') -> None:
    """绘制校正前后对比图。

    Parameters
    ----------
    rhoa_orig, rhoa_corr : (n_stations, n_gates)
    outlier_mask : (n_stations, n_gates) bool
    station_x : (n_stations,)
    gates_sec : (n_gates,) 或 None
    out_png : str 或 None
    title : str
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    rcParams['axes.unicode_minus'] = False

    n_st, n_g = rhoa_orig.shape
    if gates_sec is None:
        gates_sec = np.arange(n_g)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 原始断面
    ax = axes[0]
    im = ax.pcolormesh(station_x, gates_sec * 1e3, rhoa_orig.T,
                       norm=matplotlib.colors.LogNorm(vmin=max(rhoa_orig.min(), 1),
                                                      vmax=rhoa_orig.max()),
                       cmap='jet_r', shading='auto')
    ax.set_xlabel('测站位置 (m)')
    ax.set_ylabel('时间 (ms)')
    ax.set_title('原始 ρa 断面')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Ω·m')

    # 校正后断面
    ax = axes[1]
    im = ax.pcolormesh(station_x, gates_sec * 1e3, rhoa_corr.T,
                       norm=matplotlib.colors.LogNorm(vmin=max(rhoa_corr.min(), 1),
                                                      vmax=rhoa_corr.max()),
                       cmap='jet_r', shading='auto')
    ax.set_xlabel('测站位置 (m)')
    ax.set_ylabel('时间 (ms)')
    ax.set_title('校正后 ρa 断面')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Ω·m')

    # 飞行点标记
    ax = axes[2]
    im = ax.pcolormesh(station_x, gates_sec * 1e3, rhoa_corr.T,
                       norm=matplotlib.colors.LogNorm(vmin=max(rhoa_corr.min(), 1),
                                                      vmax=rhoa_corr.max()),
                       cmap='jet_r', shading='auto')
    # 叠加飞行点标记
    for i in range(n_st):
        for j in range(n_g):
            if outlier_mask[i, j]:
                ax.plot(station_x[i], gates_sec[j] * 1e3, 'kx', markersize=4, markeredgewidth=1)
    ax.set_xlabel('测站位置 (m)')
    ax.set_ylabel('时间 (ms)')
    ax.set_title(f'校正后 + 飞行点标记 ({np.sum(outlier_mask)} 点)')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='Ω·m')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=180, bbox_inches='tight')
        print(f'[OUT] 校正对比图 → {out_png}')
    plt.close(fig)


__all__ = [
    'hampel_filter',
    'detect_outliers_2d',
    'remove_static_shift',
    'correct_profile',
    'plot_correction_comparison',
]
