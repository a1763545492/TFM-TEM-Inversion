# -*- coding: utf-8 -*-
"""
anisotropy.py — VTI 垂直横向各向同性介质支持
================================================================
电阻率张量 (对称轴垂直):

        | σh 0  0 |              ρh: 水平电阻率
    σ = | 0  σh 0 |   ,  D2D = diag(σh, σv)
        | 0  0  σv|              ρv: 垂直电阻率

参数化 (电法各向异性惯例):
    λ   = √(ρv/ρh)   各向异性系数 (λ=1 各向同性, 沉积岩常见 1.0~2.0)
    ρm  = √(ρh·ρv)   几何平均电阻率
    ⇒ ρh = ρm/λ,  ρv = ρm·λ

有限元: 三角单元刚度 Ke = 1/(4A)(σh b·bᵀ + σv c·cᵀ)
(已在 fem2d_forward.assemble 内通过 rho_v_per_tri 支持)。

均匀 VTI 半空间解析校验: 地表供电-测量仅感受几何平均 ρa = √(ρhρv) = ρm
(坐标变换 z' = z√(σv/σh) 可证)。
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from .fem2d_forward import polepole_pseudosection


# ============================================================
# 1) 参数化
# ============================================================
def vti_fields(rho_mean_fn: Callable[[float, float], float],
               lam_fn: Callable[[float, float], float]
               ) -> tuple[Callable, Callable]:
    """由几何平均 ρm 与各向异性系数 λ 场生成 (ρh, ρv) 场函数。"""
    return (lambda x, z: rho_mean_fn(x, z) / max(lam_fn(x, z), 1e-6),
            lambda x, z: rho_mean_fn(x, z) * max(lam_fn(x, z), 1e-6))


def lam_from_rho(rho_h: np.ndarray, rho_v: np.ndarray) -> np.ndarray:
    """由 ρh/ρv 数组计算 λ。"""
    return np.sqrt(np.maximum(rho_v, 1e-9) / np.maximum(rho_h, 1e-9))


# ============================================================
# 2) 正演
# ============================================================
def vti_pseudosection(rho_h_fn: Callable[[float, float], float],
                      rho_v_fn: Callable[[float, float], float],
                      elec_x: Sequence[float],
                      **kwargs) -> pd.DataFrame:
    """VTI 介质单极-单极 ρa 拟断面 (透传网格/间距参数给 FEM 内核)。"""
    return polepole_pseudosection(rho_h_fn, elec_x,
                                  rho_v_fn=rho_v_fn, **kwargs)


def compare_iso_vti(rho_mean_fn: Callable[[float, float], float],
                    lam_fn: Callable[[float, float], float],
                    elec_x: Sequence[float],
                    out_png: str | Path,
                    **fem_kwargs) -> tuple[pd.DataFrame, Path]:
    """同 ρm 下各向同性 vs VTI 正演对比, 返回合并表 + 三联图路径。"""
    out_png = Path(out_png)
    df_iso = polepole_pseudosection(rho_mean_fn, elec_x, **fem_kwargs)
    rho_h, rho_v = vti_fields(rho_mean_fn, lam_fn)
    df_vti = polepole_pseudosection(rho_h, elec_x, rho_v_fn=rho_v,
                                    **fem_kwargs)
    m = df_iso.merge(df_vti, on=['src_x', 'spacing', 'x_mid', 'z_eff'],
                     suffixes=('_iso', '_vti'))
    m['ratio'] = m['rhoa_vti'] / m['rhoa_iso']

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, col, ttl, cmap in [
        (axes[0], 'rhoa_iso', '各向同性 (ρ=ρm)', 'jet_r'),
        (axes[1], 'rhoa_vti', 'VTI (ρh=ρm/λ, ρv=ρm·λ)', 'jet_r'),
    ]:
        sc = ax.scatter(m['x_mid'], m['z_eff'], c=np.log10(m[col]),
                        s=40, cmap=cmap, marker='s')
        ax.invert_yaxis(); ax.set_title(ttl); ax.set_xlabel('x (m)')
        fig.colorbar(sc, ax=ax, shrink=0.85, label='log10 ρa')
    sc2 = axes[2].scatter(m['x_mid'], m['z_eff'], c=m['ratio'],
                          s=40, cmap='RdBu_r', vmin=0.7, vmax=1.3,
                          marker='s')
    axes[2].invert_yaxis(); axes[2].set_title('ρa(VTI)/ρa(ISO)')
    axes[2].set_xlabel('x (m)')
    fig.colorbar(sc2, ax=axes[2], shrink=0.85, label='比值')
    fig.tight_layout(); fig.savefig(out_png, dpi=170); plt.close(fig)
    return m, out_png


# ============================================================
# 3) 反演断面级 VTI 场构造 (供 FE 正演直接消费)
# ============================================================
def vti_section_fields(R_mean: np.ndarray, Lam: np.ndarray,
                       chainage: np.ndarray, z_centers: np.ndarray,
                       x_offset: float = 0.0
                       ) -> tuple[Callable, Callable]:
    """由 ρm 断面 + λ 断面生成 (ρh,ρv) 场 (复用 FEM 断面插值器)。"""
    from .fem2d_forward import rho_from_section
    Rh = R_mean / np.maximum(Lam, 1e-6)
    Rv = R_mean * np.maximum(Lam, 1e-6)
    return (rho_from_section(Rh, chainage, z_centers, x_offset),
            rho_from_section(Rv, chainage, z_centers, x_offset))
