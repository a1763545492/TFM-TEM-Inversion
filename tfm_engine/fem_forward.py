# -*- coding: utf-8 -*-
"""
tfm_engine.fem_forward (M13)
==============================
FEM 频率域电磁法 1D 正演 — 与 TEM 真频域共享层状介质递推核，
但直接在频率点计算场分量，无需频→时转换。

物理基础:
  - 频率域麦克斯韦方程，层状介质反射系数递推（同 _refl_coef_freq）
  - 中心回线装置：垂直磁场 Hz(ω)，水平电场 Ex(ω)（若需要）
  - 支持 Pelton Cole-Cole 频率域复电阻率（IP 效应天然兼容）

与 TEM 真频域的关系:
  - TEM: Hz_sec(ω) → Key2012 DLF → dBz/dt(t)
  - FEM: Hz_total(ω) = Hz_primary(ω) + Hz_sec(ω)，直接输出 |Hz| 和相位
  - 视电阻率: Cagniard ρa = |Zxy|²/(ω·μ0)，Zxy = Ex/Hy（平面波近似）
    或中心回线定义: ρa = (2πf·μ0·|Hz_sec| / |Hz_primary|) 的变换形式

统一接口:
  fem_forward_1d(model, tx_area, tx_current, freq_hz) → (freq_hz, Hz_total)
  fem_apparent_resistivity(Hz_total, freq_hz, tx_area, tx_current) → rhoa

完全离线，纯 numpy。
"""
from __future__ import annotations
from typing import Optional, Callable
import numpy as np

from .forward import (_refl_coef_freq, _GUPT47_BASE, _GUPT47_J1,
                      _MU0, LayeredModel)


# ============================================================
# 频率域 1D FEM 正演
# ============================================================

def fem_hz_total(omega: np.ndarray, model: LayeredModel,
                 tx_area_m2: float, tx_current_a: float) -> np.ndarray:
    """复频率域总垂直磁场 Hz_total(iω) = Hz_primary + Hz_sec.

    Hz_primary(ω) = I / (2·a)  （中心回线一次场，与 ω 无关，实数）
    Hz_sec(ω)     = (I·a/2) ∫ λ·R*·J1(λa) dλ

    Parameters
    ----------
    omega : (Nω,) angular frequency [rad/s]
    model : LayeredModel
    tx_area_m2 : float, 发射回线面积 [m²]
    tx_current_a : float, 发射电流 [A]

    Returns
    -------
    Hz_total : (Nω,) complex, 总垂直磁场 [A/m]
    """
    a = np.sqrt(tx_area_m2 / np.pi)
    if a <= 0:
        raise ValueError('tx_area 必须为正')

    # 一次场（均匀空间，与频率无关）
    hz_primary = tx_current_a / (2.0 * a)

    # 二次场（复频率域）
    lam = _GUPT47_BASE / a
    if model.has_ip:
        rho_eff = model.complex_rho(omega)
        rho_air = np.full((1, omega.size), np.inf)
        rho_full = np.concatenate([rho_air, rho_eff], axis=0)
    else:
        rho_full = np.concatenate([[np.inf], model.rho_ohm_m])
    thick_full = np.concatenate([[0.0], model.thickness_m])
    R_star = _refl_coef_freq(lam, omega, rho_full, thick_full, _MU0)
    f_lam = R_star * lam[None, :]
    integral = f_lam @ _GUPT47_J1 / a
    hz_sec = (tx_current_a * a / 2.0) * integral

    return hz_primary + hz_sec


def fem_forward_1d(model: LayeredModel,
                   tx_area_m2: float,
                   tx_current_a: float,
                   freq_hz: np.ndarray,
                   include_ip: bool = True,
                   log_fn: Optional[Callable[[str], None]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FEM 频率域 1D 正演主接口。

    Parameters
    ----------
    model : LayeredModel
    tx_area_m2 : float
    tx_current_a : float
    freq_hz : (Nf,) 频率点 [Hz]
    include_ip : bool, 是否包含 IP 效应（默认 True）
    log_fn : callable, 日志回调

    Returns
    -------
    freq_hz : (Nf,) 频率 [Hz]
    hz_mag : (Nf,) 磁场幅值 [A/m]
    hz_phase_deg : (Nf,) 相位 [度]
    """
    freq_hz = np.atleast_1d(freq_hz).astype(np.float64)
    if np.any(freq_hz <= 0):
        raise ValueError('freq_hz 必须全为正数')

    omega = 2.0 * np.pi * freq_hz

    if not include_ip and model.has_ip:
        model = LayeredModel(model.thickness_m, model.rho_ohm_m, None, None, None)

    hz = fem_hz_total(omega, model, tx_area_m2, tx_current_a)

    hz_mag = np.abs(hz)
    hz_phase_deg = np.degrees(np.angle(hz))

    if log_fn:
        log_fn(f'  [FEM-1D] Nf={len(freq_hz)} f_range=[{freq_hz.min():.1f},{freq_hz.max():.1f}]Hz '
               f'Hz_med={np.median(hz_mag):.3e}A/m phase_range=[{hz_phase_deg.min():.1f},{hz_phase_deg.max():.1f}]deg')

    return freq_hz, hz_mag, hz_phase_deg


def fem_apparent_resistivity(hz_mag: np.ndarray,
                             freq_hz: np.ndarray,
                             tx_area_m2: float,
                             tx_current_a: float,
                             method: str = 'central_loop') -> np.ndarray:
    """由 FEM 磁场幅值计算视电阻率。

    method='central_loop' : 中心回线定义
        均匀半空间: |Hz_sec| / |Hz_primary| = (ωμ0 a²) / (2ρ)  对晚期/低频成立
        → ρa = (ωμ0 a² · |Hz_primary|) / (2 · |Hz_sec|)
        但 FEM 通常观测总场，需分离二次场（假设一次场已知）

    method='cagniard' : Cagniard 平面波近似（需 Ex/Hy，此处用等效阻抗近似）
        ρa = |Z|² / (ω·μ0)，Z ≈ iωμ0 · (a/2) · (Hz_total / Hz_primary) / (1 + Hz_sec/Hz_primary)
        简化: 对导电大地，|Hz_total| ≈ |Hz_primary| · (1 + iωμ0 a² / (4ρ) + ...)
        → 用迭代反解 ρa

    本实现采用中心回线迭代反解（物理自洽）:
        给定 |Hz_total| 测量值，假设 ρa，正演计算 |Hz_total(ρa)|，
        用 Brent 法找使 |Hz_total(ρa)| = |Hz_total| 的 ρa。

    Parameters
    ----------
    hz_mag : (Nf,) 观测磁场幅值 [A/m]
    freq_hz : (Nf,) 频率 [Hz]
    tx_area_m2 : float
    tx_current_a : float
    method : str, 'central_loop' 或 'cagniard'

    Returns
    -------
    rhoa : (Nf,) 视电阻率 [Ω·m]
    """
    from scipy.optimize import brentq

    freq_hz = np.atleast_1d(freq_hz)
    hz_mag = np.atleast_1d(hz_mag)
    a = np.sqrt(tx_area_m2 / np.pi)
    hz_primary = tx_current_a / (2.0 * a)
    omega = 2.0 * np.pi * freq_hz

    rhoa = np.zeros_like(freq_hz)

    def _residual(rho, om, hz_obs):
        """正演 |Hz_total(ρ)| - hz_obs = 0"""
        model = LayeredModel(np.array([np.inf]), np.array([rho]))
        hz_pred = np.abs(fem_hz_total(np.array([om]), model, tx_area_m2, tx_current_a))[0]
        return hz_pred - hz_obs

    for i, (f, om, h_obs) in enumerate(zip(freq_hz, omega, hz_mag)):
        # 初始猜测：均匀半空间低频近似
        # |Hz_sec| ≈ |Hz_primary| · ωμ0 a² / (2ρ)
        # 若 h_obs ≈ |Hz_primary|（弱导电），则 ρ 很大
        # 若 h_obs >> |Hz_primary|（强导电），则 ρ 很小
        ratio = h_obs / hz_primary
        if ratio < 1.01:
            rho_guess = 1e4  # 高阻
        elif ratio > 2.0:
            rho_guess = 1.0  # 低阻
        else:
            # 线性插值猜测
            rho_guess = 1e4 * np.exp(-np.log(1e4) * (ratio - 1.0) / 1.0)

        try:
            # 在 [0.1, 1e6] 范围内搜索
            rhoa[i] = brentq(_residual, 0.1, 1e6, args=(om, h_obs), xtol=1e-3, rtol=1e-3)
        except ValueError:
            rhoa[i] = rho_guess  # 失败时用猜测值

    return rhoa


# ============================================================
# FEM 频率域正演合成数据生成（用于 NN 训练样本）
# ============================================================

def fem_synthetic_dataset(n_models: int = 1000,
                          n_freq: int = 20,
                          freq_range: tuple = (1.0, 10000.0),
                          n_layers_max: int = 3,
                          rho_range: tuple = (1.0, 1000.0),
                          thick_range: tuple = (10.0, 500.0),
                          tx_area_m2: float = 1e6,
                          tx_current_a: float = 30.0,
                          seed: int = 42) -> dict:
    """批量生成 FEM 频率域合成样本（用于 NN 训练或算法验证）。

    Returns
    -------
    dict with keys:
      'freq_hz': (n_freq,)
      'models': list of LayeredModel
      'hz_mag': (n_models, n_freq)
      'hz_phase': (n_models, n_freq)
      'rho_true': (n_models, n_layers_max) 真实电阻率
      'thick_true': (n_models, n_layers_max) 真实厚度
    """
    rng = np.random.default_rng(seed)
    freq_hz = np.geomspace(freq_range[0], freq_range[1], n_freq)

    models = []
    hz_mag_all = []
    hz_phase_all = []
    rho_true = np.full((n_models, n_layers_max), np.nan)
    thick_true = np.full((n_models, n_layers_max), np.nan)

    for i in range(n_models):
        n_l = rng.integers(1, n_layers_max + 1)
        thick = np.concatenate([
            rng.uniform(thick_range[0], thick_range[1], n_l - 1),
            [np.inf]
        ])
        rho = rng.uniform(rho_range[0], rho_range[1], n_l)
        model = LayeredModel(thick, rho)

        _, hz_mag, hz_phase = fem_forward_1d(model, tx_area_m2, tx_current_a, freq_hz)

        models.append(model)
        hz_mag_all.append(hz_mag)
        hz_phase_all.append(hz_phase)
        rho_true[i, :n_l] = rho
        thick_true[i, :n_l] = thick

    return {
        'freq_hz': freq_hz,
        'models': models,
        'hz_mag': np.array(hz_mag_all),
        'hz_phase': np.array(hz_phase_all),
        'rho_true': rho_true,
        'thick_true': thick_true,
    }


__all__ = [
    'fem_hz_total',
    'fem_forward_1d',
    'fem_apparent_resistivity',
    'fem_synthetic_dataset',
]
