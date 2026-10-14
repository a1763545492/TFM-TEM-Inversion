# -*- coding: utf-8 -*-
"""
tfm_engine.debye_inversion
==========================
IP 谱反演扩展（与 ip_inversion.py 层状 ρ0/m/τ/c 反演互补）：
  invert_double_cole_cole : 双 Cole-Cole 项拟合单站 ρa(t)
        ρ(t) = ρ0·[1 − m1·g(t;τ1,c1) − m2·g(t;τ2,c2)],  g=1−1/(1+(t/τ)^c)
        适用: 同一测点含两种弛豫机制 (如细粒导电矿物 + 粗粒硫化物)
  invert_debye_spectrum   : Debye 分解谱 (NNLS 非负最小二乘)
        ρ(t) = ρ0·(1 − Σ_k m_k·(1−exp(−t/τ_k)))
        τ_k 对数等距 (默认 15 个, 1µs~10s), m_k≥0 表征弛豫强度谱
        迭代: 固定 ρ0 → NNLS 求 m_k → 回推 ρ0, 共 n_iter 轮

完全离线, 纯 numpy + scipy (nnls)。
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scipy.optimize import least_squares, nnls

__all__ = ['invert_double_cole_cole', 'invert_debye_spectrum',
           'DEFAULT_2CC_CFG', 'DEFAULT_DEBYE_CFG']


def _logit(x, lo, hi):
    return np.log((x - lo) / (hi - x))


def _ilogit(y, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-np.clip(y, -20, 20)))


DEFAULT_2CC_CFG = dict(
    M_LIMITS=(1e-3, 0.8),
    TAU_LIMITS_LOG=(np.log10(1e-6), np.log10(30.0)),
    C_LIMITS=(0.10, 0.95),
    M_SUM_MAX=0.95,         # m1+m2 物理上限 (惩罚)
    MAX_NFEV=300,
    INIT=(0.10, 0.05),      # (m1_init, m2_init)
)


DEFAULT_DEBYE_CFG = dict(
    N_TAU=15,
    TAU_MIN=1e-6,
    TAU_MAX=10.0,
    N_ITER=3,               # ρ0 ↔ m_k 交替轮数
    D_CLIP=(0.0, 0.95),     # 目标 d = 1−ρ/ρ0 的截断区间
)


# ============================================================
# 双 Cole-Cole 项
# ============================================================
def invert_double_cole_cole(t_obs: np.ndarray, rhoa_obs: np.ndarray,
                            err_obs: Optional[np.ndarray] = None,
                            share_c: bool = False,
                            user_cfg: Optional[dict] = None,
                            log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """双 Cole-Cole 拟合单站 ρa(t)。
    参数向量 p (share_c=True → 6 参数; False → 7 参数):
        [log10ρ0, logit m1, logit m2, log10τ1, log10τ2, logit c1, (logit c2)]
    返回 dict(rho0, m1, m2, tau1, tau2, c1, c2, m_total, rmse_log, ...).
    """
    cfg = dict(DEFAULT_2CC_CFG)
    if user_cfg: cfg.update(user_cfg)
    log_fn = log_fn or (lambda s: None)
    t = np.asarray(t_obs, dtype=np.float64)
    rho = np.asarray(rhoa_obs, dtype=np.float64)
    good = np.isfinite(t) & (t > 0) & np.isfinite(rho) & (rho > 0)
    t, rho = t[good], rho[good]
    if t.size < 5:
        return {'status': 'FAIL', 'error': f'only {t.size} valid obs'}
    if err_obs is not None:
        e = np.asarray(err_obs, dtype=np.float64)[good]
        w = 1.0 / np.maximum(np.abs(np.log10(np.maximum(e, 1e-4) / rho + 1e-3)), 0.05)
    else:
        w = np.ones_like(rho)

    lo_m, hi_m = cfg['M_LIMITS']
    lo_c, hi_c = cfg['C_LIMITS']
    lo_t, hi_t = cfg['TAU_LIMITS_LOG']
    n_params = 6 if share_c else 7

    def unpack(p):
        rho0 = 10.0 ** p[0]
        m1 = _ilogit(p[1], lo_m, hi_m)
        m2 = _ilogit(p[2], lo_m, hi_m)
        tau1 = 10.0 ** np.clip(p[3], lo_t, hi_t)
        tau2 = 10.0 ** np.clip(p[4], lo_t, hi_t)
        c1 = _ilogit(p[5], lo_c, hi_c)
        c2 = c1 if share_c else _ilogit(p[6], lo_c, hi_c)
        return rho0, m1, m2, tau1, tau2, c1, c2

    def model_rho(p):
        rho0, m1, m2, tau1, tau2, c1, c2 = unpack(p)
        x1 = np.clip(t / tau1, 1e-4, 1e5)
        x2 = np.clip(t / tau2, 1e-4, 1e5)
        g1 = 1.0 - 1.0 / (1.0 + x1 ** c1)
        g2 = 1.0 - 1.0 / (1.0 + x2 ** c2)
        return rho0 * (1.0 - m1 * g1 - m2 * g2), m1, m2

    def resid(p):
        pred, m1, m2 = model_rho(p)
        r_data = (np.log10(np.maximum(pred, 1e-4)) - np.log10(rho)) * w
        r_pen = np.array([max(0.0, m1 + m2 - cfg['M_SUM_MAX']) * 10.0])
        return np.concatenate([r_data, r_pen])

    m1_0, m2_0 = cfg['INIT']
    t_med = float(np.median(t))
    p0 = np.zeros(n_params)
    p0[0] = np.log10(np.median(rho))
    p0[1] = _logit(m1_0, lo_m, hi_m)
    p0[2] = _logit(m2_0, lo_m, hi_m)
    p0[3] = np.log10(max(t_med * 0.5, 10.0 ** lo_t))
    p0[4] = np.log10(min(t_med * 5.0, 10.0 ** hi_t))
    p0[5] = _logit(0.5, lo_c, hi_c)
    if not share_c:
        p0[6] = _logit(0.5, lo_c, hi_c)

    try:
        res = least_squares(resid, p0, method='trf', max_nfev=cfg['MAX_NFEV'],
                            x_scale='jac')
        rho0, m1, m2, tau1, tau2, c1, c2 = unpack(res.x)
        pred, _, _ = model_rho(res.x)
        rmse = float(np.sqrt(np.mean((np.log10(np.maximum(pred, 1e-4))
                                      - np.log10(rho)) ** 2)))
        out = {'status': 'OK', 'nfev': int(res.nfev), 'rmse_log': rmse,
               'rho0': float(rho0), 'm1': float(m1), 'm2': float(m2),
               'tau1': float(tau1), 'tau2': float(tau2),
               'c1': float(c1), 'c2': float(c2),
               'm_total': float(m1 + m2), 'share_c': bool(share_c),
               'rhoa_fit': pred}
        log_fn(f"  [2CC] OK m1={m1:.3f} m2={m2:.3f} "
               f"τ1={tau1:.2e} τ2={tau2:.2e} RMSE={rmse:.3f}")
        return out
    except Exception as exc:
        log_fn(f"  [2CC] FAIL: {exc}")
        return {'status': 'FAIL', 'error': str(exc)}


# ============================================================
# Debye 分解谱 (NNLS)
# ============================================================
def invert_debye_spectrum(t_obs: np.ndarray, rhoa_obs: np.ndarray,
                          user_cfg: Optional[dict] = None,
                          log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """Debye 分解单站 ρa(t) → 弛豫谱 m(τ)。
    ρ(t) = ρ0·(1 − Σ_k m_k·(1−exp(−t/τ_k))),  m_k ≥ 0 (nnls)。
    返回 dict(rho0, tau_k, m_k, m_total, rmse_log, ...).
    """
    cfg = dict(DEFAULT_DEBYE_CFG)
    if user_cfg: cfg.update(user_cfg)
    log_fn = log_fn or (lambda s: None)
    t = np.asarray(t_obs, dtype=np.float64)
    rho = np.asarray(rhoa_obs, dtype=np.float64)
    good = np.isfinite(t) & (t > 0) & np.isfinite(rho) & (rho > 0)
    t, rho = t[good], rho[good]
    if t.size < 5:
        return {'status': 'FAIL', 'error': f'only {t.size} valid obs'}

    tau_k = np.geomspace(cfg['TAU_MIN'], cfg['TAU_MAX'], cfg['N_TAU'])
    A = 1.0 - np.exp(-t[:, None] / tau_k[None, :])     # (Ng, N_TAU)
    d_lo, d_hi = cfg['D_CLIP']
    rho0 = float(np.median(rho))
    m_k = np.zeros(cfg['N_TAU'])
    for _ in range(cfg['N_ITER']):
        d = np.clip(1.0 - rho / rho0, d_lo, d_hi)
        m_k, _ = nnls(A, d)
        pred_ratio = 1.0 - A @ m_k                     # ρ_pred/ρ0
        mask = (pred_ratio > 0.1) & np.isfinite(rho)
        if mask.sum() > 3:
            rho0 = float(np.exp(np.mean(np.log(rho[mask])
                                        - np.log(pred_ratio[mask]))))
        else:
            break
    pred = rho0 * (1.0 - A @ m_k)
    rmse = float(np.sqrt(np.mean((np.log10(np.maximum(pred, 1e-4))
                                  - np.log10(rho)) ** 2)))
    out = {'status': 'OK', 'rho0': rho0, 'tau_k': tau_k, 'm_k': m_k,
           'm_total': float(m_k.sum()), 'rmse_log': rmse,
           'n_tau': cfg['N_TAU'], 'rhoa_fit': pred}
    log_fn(f"  [Debye] OK ρ0={rho0:.1f} m_total={out['m_total']:.3f} "
           f"N_τ={cfg['N_TAU']} RMSE={rmse:.3f}")
    return out
