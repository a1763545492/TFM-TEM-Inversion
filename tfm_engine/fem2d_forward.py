# -*- coding: utf-8 -*-
"""
fem2d_forward.py — 2D 有限元直流电阻率正演 (三角剖分 + PEFC 近源加密)
======================================================================
控制方程 (二维剖面, 稳态电流场):

    ∇·(σ(x,z) ∇u) = -I·δ(x-xs, z-zs),   σ = 1/ρ

  · 地表 (z=z0): 空气绝缘, 自然 Neumann 边界 (∂u/∂n=0)
  · 其余外边界: u=0 (远场接地, 需取足够大)
  · 线性三角形单元 (常 σ/单元), 总刚解析组装, scipy.sparse 稀疏求解
  · PEFC = Point-source Enriched grid Concentration:
    用高斯形变函数对水平坐标做**单调加密**, 点源/电极附近网格自动变密,
    深度方向按幂函数向地表加密 (tensor 结构保证无 T 接头、完全协调)

二维全空间/半空间解析解用于校验:
  全空间  u(r) = Iρ/(2π) ln(R/r)
  半空间  u(r) = Iρ/( π) ln(R/r)        (Neumann 镜像)
  → 两点差分消去远场 R:
      ρa(全) = 2π·(V1−V2) / (I·ln(r2/r1))
      ρa(半) =  π·(V1−V2) / (I·ln(r2/r1))

定位: 2D DC 正演内核, 为 2.5D 波数域/感应类正演打基础;
可直接对反演断面 R(n站×n层) 做正演响应计算与布极方案评估。
"""
from __future__ import annotations
from math import pi
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 1) 网格: 结构化三角剖分 + PEFC 点源加密
# ============================================================
def _warp_x(n: int, focus: Sequence[float], width: float,
            strength: float) -> np.ndarray:
    """水平归一化坐标 [0,1], 在 focus(可多个) 附近单调加密。

    密度 1+A·Σexp(-((t-f)/w)²), 先在密网格上积分为 CDF W(t),
    再取逆: s_j = W⁻¹(j/(n-1)) → W' 大处 ds 小 (节点密)。
    """
    tg = np.linspace(0.0, 1.0, 4001)
    g = np.ones_like(tg)
    for f in focus:
        g += strength * np.exp(-((tg - f) / width) ** 2)
    W = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1])
                                         * np.diff(tg))])
    W /= W[-1]
    q = np.linspace(0.0, 1.0, n)
    return np.interp(q, W, tg)


def build_mesh(xlim: tuple[float, float] = (-200.0, 200.0),
               zlim: tuple[float, float] = (0.0, 200.0),
               nx: int = 120, nz: int = 60,
               focus_x: Sequence[float] = (),
               focus_width: float = 0.06,
               focus_strength: float = 3.0,
               surface_power: float = 1.7
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 nodes(N,2)[x,z], tris(M,3)[节点索引], surface_nodes(顶边有序)。"""
    s = _warp_x(nx + 1, focus_x, focus_width, focus_strength)
    xs = xlim[0] + s * (xlim[1] - xlim[0])
    # 深度向地表加密: z = z0 + (z1-z0)·t^p (p>1 → 浅层密)
    tt = np.linspace(0.0, 1.0, nz + 1) ** surface_power
    zs = zlim[0] + tt * (zlim[1] - zlim[0])
    nxx = nx + 1
    nodes = np.array([(x, z) for z in zs for x in xs])
    tris = []
    for iz in range(nz):
        for ix in range(nx):
            n0 = iz * nxx + ix
            n1 = n0 + 1
            n2 = n0 + nxx
            n3 = n2 + 1
            tris.append((n0, n1, n2))
            tris.append((n1, n3, n2))
    tris = np.asarray(tris, dtype=np.int64)
    surface_nodes = np.arange(nxx, dtype=np.int64)
    return nodes, tris, surface_nodes, xs, zs


def tri_geometry(nodes: np.ndarray, tris: np.ndarray):
    """单元几何量: 面积 A, 形函数梯度系数 b,c (逆时针约定)。"""
    p = nodes[tris]                      # (M,3,2)
    x, z = p[:, :, 0], p[:, :, 1]
    # 2A 有符号; 本网格剖分保证为正
    twoA = ((x[:, 1] - x[:, 0]) * (z[:, 2] - z[:, 0])
            - (x[:, 2] - x[:, 0]) * (z[:, 1] - z[:, 0]))
    A = np.abs(twoA) / 2.0
    # b_i = z_j - z_k, c_i = x_k - x_j  (循环)
    b = np.column_stack([z[:, 1] - z[:, 2],
                         z[:, 2] - z[:, 0],
                         z[:, 0] - z[:, 1]])
    c = np.column_stack([x[:, 2] - x[:, 1],
                         x[:, 0] - x[:, 2],
                         x[:, 1] - x[:, 0]])
    return A, b, c


# ============================================================
# 2) 总刚组装 ∫ σ ∇φi·∇φj dA
# ============================================================
def assemble(nodes: np.ndarray, tris: np.ndarray,
             rho_per_tri: np.ndarray,
             rho_v_per_tri: Optional[np.ndarray] = None) -> csc_matrix:
    """线性三角单元: Ke = 1/(4A)(σh b·bᵀ + σv c·cᵀ)。

    rho_v_per_tri=None 时各向同性 σh=σv=1/ρ。
    VTI: ρh 水平电阻率, ρv 垂直电阻率 (∂N/∂x∝b, ∂N/∂z∝c)。
    """
    A, b, c = tri_geometry(nodes, tris)
    sig_h = 1.0 / np.maximum(rho_per_tri, 1e-9)
    sig_v = (1.0 / np.maximum(rho_v_per_tri, 1e-9)
             if rho_v_per_tri is not None else sig_h)
    coef_h = sig_h / (4.0 * A)             # (M,)
    coef_v = sig_v / (4.0 * A)
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            rows.append(tris[:, i])
            cols.append(tris[:, j])
            vals.append(coef_h * b[:, i] * b[:, j]
                        + coef_v * c[:, i] * c[:, j])
    n = nodes.shape[0]
    K = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    return csc_matrix(K)


def _locate_triangle(nodes: np.ndarray, tris: np.ndarray,
                     pt: tuple[float, float]) -> tuple[int, np.ndarray]:
    """点所在单元索引 + 面积坐标 (向量化搜索, 单元数千规模够用)。"""
    p = nodes[tris]                                  # (M,3,2)
    x, z = p[:, :, 0], p[:, :, 1]
    A_all = ((x[:, 1] - x[:, 0]) * (z[:, 2] - z[:, 0])
             - (x[:, 2] - x[:, 0]) * (z[:, 1] - z[:, 0]))
    px, pz = pt
    L0 = ((x[:, 1] - px) * (z[:, 2] - pz)
          - (x[:, 2] - px) * (z[:, 1] - pz)) / A_all
    L1 = ((x[:, 2] - px) * (z[:, 0] - pz)
          - (x[:, 0] - px) * (z[:, 2] - pz)) / A_all
    L2 = 1.0 - L0 - L1
    inside = (L0 >= -1e-8) & (L1 >= -1e-8) & (L2 >= -1e-8)
    idx = np.where(inside)[0]
    if not len(idx):
        raise ValueError(f'点源 {pt} 落在网格外')
    k = int(idx[0])
    return k, np.array([L0[k], L1[k], L2[k]])


def source_vector(nodes: np.ndarray, tris: np.ndarray,
                  pt: tuple[float, float], current: float = 1.0) -> np.ndarray:
    """点源按面积坐标分配到单元三节点 (源在节点上即单点承载)。"""
    n = nodes.shape[0]
    rhs = np.zeros(n)
    k, L = _locate_triangle(nodes, tris, pt)
    for loc in range(3):
        rhs[tris[k, loc]] += current * L[loc]
    return rhs


def grounded_nodes(nodes: np.ndarray, surface_nodes: np.ndarray,
                   xlim, zlim, tol: float = 1e-9) -> np.ndarray:
    """左/右/底边接地 (u=0); 顶边保留 Neumann。"""
    x, z = nodes[:, 0], nodes[:, 1]
    m = ((np.abs(x - xlim[0]) < tol) | (np.abs(x - xlim[1]) < tol)
         | (np.abs(z - zlim[1]) < tol))
    return np.where(m)[0]


def solve(nodes: np.ndarray, tris: np.ndarray, rho_per_tri: np.ndarray,
          sources: Sequence[tuple[float, float, float]],
          surface_nodes: np.ndarray, xlim, zlim,
          rho_v_per_tri: Optional[np.ndarray] = None) -> np.ndarray:
    """多源正演: K 一次 LU 分解, 返回 u(N, n_src)。rho_v 非 None 即 VTI。"""
    K = assemble(nodes, tris, rho_per_tri, rho_v_per_tri)
    gnd = grounded_nodes(nodes, surface_nodes, xlim, zlim)
    B = np.column_stack([
        source_vector(nodes, tris, (sx, sz), I) for sx, sz, I in sources
    ]) if len(sources) > 1 else source_vector(
        nodes, tris, (sources[0][0], sources[0][1]), sources[0][2])[:, None]
    # 齐次 Dirichlet: 大对角元法
    K = K.tolil()
    K[gnd, :] = 0.0
    K[gnd, gnd] = 1.0
    B[gnd, :] = 0.0
    lu = splu(csc_matrix(K))
    U = np.column_stack([lu.solve(B[:, j]) for j in range(B.shape[1])])
    return U


# ============================================================
# 3) 电阻率模型: 反演断面 → 任意 (x,z) 的 ρ
# ============================================================
def rho_from_section(R: np.ndarray, chainage: np.ndarray,
                     z_centers: np.ndarray,
                     x_offset: float = 0.0) -> Callable[[float, float], float]:
    """R(n站,n层) 断面 → ρ(x,z)。log10ρ 域双线性阶梯插值, 站间/层间连续线性。

    x_offset: FE 域 x=0 与测线里程零点的偏移。
    """
    L = np.log10(np.maximum(R, 1e-9))

    def rho_fn(x: float, z: float) -> float:
        # 沿里程对每个深度层插值, 再沿深度插值
        col = np.array([np.interp(x - x_offset, chainage, L[:, j],
                                  left=L[0, j], right=L[-1, j])
                        for j in range(L.shape[1])])
        return float(10.0 ** np.interp(z, z_centers, col))
    return rho_fn


def element_rho(nodes: np.ndarray, tris: np.ndarray,
                rho_fn: Callable[[float, float], float]) -> np.ndarray:
    cen = nodes[tris].mean(axis=1)
    return np.array([rho_fn(x, z) for x, z in cen])


# ============================================================
# 4) 视电阻率 (二维半空间/全空间, 两点差分消远场)
# ============================================================
def apparent_rho_halfspace(V1: float, V2: float, r1: float, r2: float,
                           I: float = 1.0) -> float:
    """半空间地表两点电位差 → ρa = π(V1−V2)/(I·ln(r2/r1))。"""
    return pi * (V1 - V2) / (I * np.log(r2 / r1))


def apparent_rho_fullspace(V1: float, V2: float, r1: float, r2: float,
                           I: float = 1.0) -> float:
    return 2.0 * pi * (V1 - V2) / (I * np.log(r2 / r1))


def polepole_pseudosection(rho_fn: Callable[[float, float], float],
                           elec_x: Sequence[float],
                           xlim=(-300.0, 300.0), zmax: float = 300.0,
                           nx: int = 160, nz: int = 70,
                           spacings: Sequence[float] = (5.0, 10.0, 20.0, 40.0),
                           I: float = 1.0,
                           rho_v_fn: Optional[Callable[[float, float], float]] = None,
                           log_fn: Optional[Callable[[str], None]] = None
                           ) -> pd.DataFrame:
    """单极-单极装置 ρa 拟断面 (2D 半空间)。

    对每个供电极 xs, 间距 a ∈ spacings 取地表 r1=a, r2=2a 两点电位差,
    ρa = π(Va−V2a)/(I ln2); 记录点 (xs+1.5a, 0.5a)。
    """
    log_fn = log_fn or (lambda s: None)
    elec_x = np.asarray(elec_x, dtype=float)
    nodes, tris, surf, _, _ = build_mesh(
        xlim, (0.0, zmax), nx, nz,
        focus_x=(elec_x - xlim[0]) / (xlim[1] - xlim[0]))
    rho_e = element_rho(nodes, tris, rho_fn)
    rho_ve = element_rho(nodes, tris, rho_v_fn) if rho_v_fn else None
    # 源点落在地表网格节点上 (取最近顶边节点坐标)
    top = nodes[surf]
    src_pts = []
    for sx in elec_x:
        j = int(np.argmin(np.abs(top[:, 0] - sx)))
        src_pts.append((float(top[j, 0]), 0.0, I))
    log_fn(f'[FEM2D] 节点 {nodes.shape[0]}, 单元 {tris.shape[0]}, '
          f'供电极 {len(src_pts)}{", VTI" if rho_v_fn else ""}')
    U = solve(nodes, tris, rho_e, src_pts, surf, xlim, (0.0, zmax),
              rho_v_per_tri=rho_ve)

    rows = []
    x_sorted_all = top[:, 0]
    order = np.argsort(x_sorted_all)
    x_top = x_sorted_all[order]
    for k, (sx, _, _) in enumerate(src_pts):
        v = U[surf, k][order]
        for a in spacings:
            i1 = np.argmin(np.abs(x_top - (sx + a)))
            i2 = np.argmin(np.abs(x_top - (sx + 2 * a)))
            r1 = abs(x_top[i1] - sx); r2 = abs(x_top[i2] - sx)
            if r1 <= 0 or r2 <= r1:
                continue
            if x_top[i2] >= xlim[1] - 2 * a:
                continue
            rhoa = apparent_rho_halfspace(v[i1], v[i2], r1, r2, I)
            rows.append({'src_x': sx, 'spacing': a,
                         'x_mid': sx + 1.5 * a, 'z_eff': 0.5 * a,
                         'r1': r1, 'r2': r2, 'rhoa': rhoa})
    return pd.DataFrame(rows)


def plot_pseudosection(df: pd.DataFrame, out_png: str | Path,
                       title: str = '2D FEM 单极-单极 ρa 拟断面') -> Path:
    """散点式拟断面 (色点 = ρa, log 色标)。"""
    out_png = Path(out_png)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sc = ax.scatter(df['x_mid'], df['z_eff'], c=np.log10(df['rhoa']),
                    s=46, cmap='jet_r', marker='s')
    ax.invert_yaxis()
    ax.set_xlabel('水平距离 (m)'); ax.set_ylabel('有效深度 (m)')
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, shrink=0.9)
    cb.set_label('log10(ρa/Ω·m)')
    fig.tight_layout(); fig.savefig(out_png, dpi=170); plt.close(fig)
    return out_png


def plot_mesh(nodes: np.ndarray, tris: np.ndarray, out_png: str | Path,
              rho_per_tri: Optional[np.ndarray] = None) -> Path:
    """网格/模型检查图。"""
    out_png = Path(out_png)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if rho_per_tri is not None:
        pc = ax.tripcolor(nodes[:, 0], nodes[:, 1], tris,
                          facecolors=np.log10(rho_per_tri), cmap='jet_r')
        fig.colorbar(pc, ax=ax, label='log10(ρ/Ω·m)')
    else:
        ax.triplot(nodes[:, 0], nodes[:, 1], tris, lw=0.3, color='#444')
    ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box')
    ax.set_title('2D FEM 三角网格 (PEFC 近源加密)')
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png
