# -*- coding: utf-8 -*-
"""
tfm_engine.resolution (M15)
=============================
深度分辨率矩阵 + 灵敏度核输出。

反演结果的可信度评估:
  - 分辨率矩阵 R = V_k V_k^T （截断奇异值分解）
  - 灵敏度核 J^T J 的对角线（数据对各层的约束强度）
  - 深度分辨率曲线: 每层被数据约束的程度

物理意义:
  - R 接近单位阵 → 该层参数被数据唯一确定
  - R 有非对角大元素 → 层间参数耦合（ trades off）
  - 灵敏度低 → 该层对数据影响小，反演结果不可信

实现:
  - 基于反演最终迭代的雅可比矩阵 J (Ng × N)
  - SVD: J = U S V^T
  - 截断: 保留前 k 个奇异值（使条件数 < 1e3）
  - R = V_k V_k^T, 灵敏度 = diag(J^T J)

统一接口:
  compute_resolution(J, ...) → dict(R, sensitivity, singular_values, ...)
  plot_resolution(R, depth, ...) → PNG

完全离线，纯 numpy。
"""
from __future__ import annotations
from typing import Optional, Callable
import numpy as np


# ============================================================
# 分辨率矩阵计算
# ============================================================

def compute_resolution(J: np.ndarray,
                       depth_centers: np.ndarray,
                       cond_threshold: float = 1e3,
                       log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """计算深度分辨率矩阵和灵敏度核。

    Parameters
    ----------
    J : (n_data, n_layers) 雅可比矩阵 d(ln d)/d(ln m)
    depth_centers : (n_layers,) 层中心深度 [m]
    cond_threshold : float, 奇异值截断条件数阈值
    log_fn : callable

    Returns
    -------
    dict with keys:
      'R': (n_layers, n_layers) 分辨率矩阵
      'sensitivity': (n_layers,) 灵敏度（J^T J 对角线）
      'singular_values': (n_layers,) 奇异值
      'n_effective': int, 有效自由度（保留的奇异值个数）
      'resolution_radius': (n_layers,) 分辨率半径 [m]
      'depth_of_investigation': float, 探测深度 [m]（灵敏度峰值深度）
    """
    J = np.asarray(J, dtype=np.float64)
    n_data, n_layers = J.shape

    # SVD
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    # 截断
    if S[0] > 0:
        cond = S / S[0]
        k = np.sum(cond > 1.0 / cond_threshold)
    else:
        k = 0
    k = max(k, 1)  # 至少保留1个

    V_k = Vt[:k, :].T  # (n_layers, k)
    R = V_k @ V_k.T    # (n_layers, n_layers)

    # 灵敏度: J^T J 对角线（未截断）
    sensitivity = np.sum(J ** 2, axis=0)

    # 分辨率半径: R 矩阵每行的"宽度"（二阶矩）
    resolution_radius = np.zeros(n_layers)
    for i in range(n_layers):
        row = np.abs(R[i, :])
        row_sum = row.sum()
        if row_sum > 1e-12:
            resolution_radius[i] = np.sqrt(np.sum(row * (depth_centers - depth_centers[i]) ** 2) / row_sum)
        else:
            resolution_radius[i] = np.inf

    # 探测深度: 灵敏度峰值对应的深度
    doi = depth_centers[np.argmax(sensitivity)] if len(sensitivity) > 0 else np.nan

    if log_fn:
        log_fn(f'  [分辨率] 有效自由度={k}/{n_layers} 条件数={S[0]/S[k-1] if k>1 else S[0]:.1e} '
               f'探测深度={doi:.1f}m 灵敏度峰值={sensitivity.max():.3e}')

    return {
        'R': R,
        'sensitivity': sensitivity,
        'singular_values': S,
        'n_effective': int(k),
        'resolution_radius': resolution_radius,
        'depth_of_investigation': doi,
        'V': Vt.T,
        'U': U,
    }


def compute_resolution_from_inversion(theta: np.ndarray,
                                      t_g: np.ndarray,
                                      rhoa_obs: np.ndarray,
                                      err_obs: np.ndarray,
                                      z_centers: np.ndarray,
                                      forward_fn: Callable,
                                      rho_ref: float = 100.0,
                                      fd_step: float = 1e-4,
                                      log_fn: Optional[Callable] = None) -> dict:
    """从反演结果直接计算分辨率矩阵。

    自动数值计算雅可比 J，然后调用 compute_resolution。

    Parameters
    ----------
    theta : (n_layers,) log10(ρ) 反演结果
    t_g : (n_gates,) 时间门 [s]
    rhoa_obs : (n_gates,) 观测视电阻率
    err_obs : (n_gates,) 观测误差
    z_centers : (n_layers,) 层中心深度
    forward_fn : callable, forward_fn(theta, t_g, z_centers, rho_ref) → rhoa_pred
    rho_ref : float
    fd_step : float, 数值微分步长
    log_fn : callable

    Returns
    -------
    dict (同 compute_resolution)
    """
    theta = np.asarray(theta)
    n_layers = len(theta)
    n_gates = len(t_g)

    # 权重
    log_err_frac = np.abs(np.log10(err_obs / rhoa_obs + 1e-10))
    w = 1.0 / np.maximum(log_err_frac, 0.05)

    # 数值雅可比
    base = forward_fn(theta, t_g, z_centers, rho_ref)
    log_base = np.log10(np.maximum(base, 1e-3))
    J = np.zeros((n_gates, n_layers))
    for j in range(n_layers):
        tp = theta.copy()
        tp[j] += fd_step
        log_p = np.log10(np.maximum(forward_fn(tp, t_g, z_centers, rho_ref), 1e-3))
        J[:, j] = (log_p - log_base) / fd_step
    J = J * w[:, None]

    return compute_resolution(J, z_centers, log_fn=log_fn)


# ============================================================
# 可视化
# ============================================================

def plot_resolution(res: dict,
                    depth_centers: np.ndarray,
                    out_png: Optional[str] = None,
                    title: str = '深度分辨率矩阵') -> None:
    """绘制分辨率矩阵和灵敏度核。

    Parameters
    ----------
    res : dict, compute_resolution 的输出
    depth_centers : (n_layers,)
    out_png : str
    title : str
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    rcParams['axes.unicode_minus'] = False

    R = res['R']
    sensitivity = res['sensitivity']
    S = res['singular_values']
    n_eff = res['n_effective']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # R 矩阵
    ax = axes[0]
    im = ax.pcolormesh(depth_centers, depth_centers, R, cmap='RdBu_r',
                       vmin=-1, vmax=1, shading='auto')
    ax.set_xlabel('深度 (m)')
    ax.set_ylabel('深度 (m)')
    ax.set_title(f'分辨率矩阵 R (有效自由度={n_eff})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    # 灵敏度
    ax = axes[1]
    ax.semilogy(sensitivity, depth_centers, 'b-o', markersize=4)
    ax.set_xlabel('灵敏度 (J^T J 对角线)')
    ax.set_ylabel('深度 (m)')
    ax.set_title('灵敏度核')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=np.median(sensitivity), color='r', linestyle='--', alpha=0.5,
               label=f'中位数={np.median(sensitivity):.2e}')
    ax.legend()

    # 奇异值
    ax = axes[2]
    ax.semilogy(S, 'bs-', markersize=4)
    ax.axvline(x=n_eff - 0.5, color='r', linestyle='--', alpha=0.5,
               label=f'截断点 (k={n_eff})')
    ax.set_xlabel('奇异值索引')
    ax.set_ylabel('奇异值')
    ax.set_title('SVD 奇异值谱')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=180, bbox_inches='tight')
        print(f'[OUT] 分辨率图 → {out_png}')
    plt.close(fig)


def plot_resolution_summary(res_list: list,
                            depth_centers: np.ndarray,
                            station_labels: Optional[list] = None,
                            out_png: Optional[str] = None) -> None:
    """多测站分辨率汇总图（沿测线方向的分辨率变化）。

    Parameters
    ----------
    res_list : list of dict, 每个测站的 compute_resolution 输出
    depth_centers : (n_layers,)
    station_labels : list of str, 测站标签
    out_png : str
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    rcParams['axes.unicode_minus'] = False

    n_st = len(res_list)
    n_layers = len(depth_centers)

    # 收集数据
    sens_2d = np.array([r['sensitivity'] for r in res_list])  # (n_st, n_layers)
    n_eff_arr = np.array([r['n_effective'] for r in res_list])
    doi_arr = np.array([r['depth_of_investigation'] for r in res_list])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 灵敏度断面
    ax = axes[0]
    im = ax.pcolormesh(np.arange(n_st), depth_centers, sens_2d.T,
                       norm=matplotlib.colors.LogNorm(), cmap='viridis', shading='auto')
    ax.set_xlabel('测站索引')
    ax.set_ylabel('深度 (m)')
    ax.set_title('灵敏度断面')
    ax.set_yscale('log')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    # 有效自由度
    ax = axes[1]
    ax.plot(n_eff_arr, 'bo-')
    ax.set_xlabel('测站索引')
    ax.set_ylabel('有效自由度')
    ax.set_title(f'有效自由度 (均值={n_eff_arr.mean():.1f})')
    ax.grid(True, alpha=0.3)

    # 探测深度
    ax = axes[2]
    ax.plot(doi_arr, 'rs-')
    ax.set_xlabel('测站索引')
    ax.set_ylabel('探测深度 (m)')
    ax.set_title(f'探测深度 (均值={doi_arr.mean():.1f}m)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('测线分辨率汇总', fontsize=14)
    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=180, bbox_inches='tight')
        print(f'[OUT] 分辨率汇总图 → {out_png}')
    plt.close(fig)


__all__ = [
    'compute_resolution',
    'compute_resolution_from_inversion',
    'plot_resolution',
    'plot_resolution_summary',
]
