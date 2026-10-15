# -*- coding: utf-8 -*-
"""
tfm_engine.ip_inversion (M4)
============================
TEM 体系 1D TEM 激电 (IP/极化率) 反演（完全按 标准IP做法.m / IP反演.m 设计）：
  模型参数：每层 4 个 Pelton Cole-Cole 参数
       θ_n = [log10 ρ0_n, logit m_n, log10 τ_n, logit c_n]   (共 4N)
       或 共享 τ,c：每层仅 [log10 ρ0_n, logit m_n] + 2 全局 = 2N+2（稳健推荐）
  残差设计 (双残差 + 平滑，与 1D ρ 反演统一 least_squares 架构)：
       r_data = concatenate([
           log10(rhoa_pred/rhoa_obs) / 0.05,          ← Ng 项
           log10(|Vt_pred|/|Vt_obs|) / max(|log10err|, 0.05)  ← Ng 项
       ])
       r_smooth = sqrt(λ_smooth) · diff(θ_per_layer[1..N])    ← 4(N-1) 项
  bounds: ρ0∈(0.1,1e8)  m∈(0.001,0.8)  τ∈(1e-6,30)  c∈(0.1,0.95)
  算法: scipy least_squares(TRF, max_nfev=300)

完全离线, 纯 numpy + scipy。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from scipy.optimize import least_squares
from .forward import LayeredModel, tem_forward_1d, apparent_resistivity_dbzdt, _MU0

DEFAULT_IP_CFG = dict(
    SHARE_TAU_C=True,           # True=全局共享τ,c (2N+2参数,稳); False=每层4参数 (4N)
    LAMBDA_SMOOTH=0.3,
    ERROR_RHOA_FRAC=0.05,
    ERROR_VT_FRAC=0.08,
    RHO0_BOUNDS_LOG=(np.log10(0.1), np.log10(1e8)),
    M_LIMITS=(1e-3, 0.8),
    TAU_LIMITS_LOG=(np.log10(1e-6), np.log10(30.0)),
    C_LIMITS=(0.10, 0.95),
    MAX_NFEV=300,
    INIT_m=0.05, INIT_tau=1e-3, INIT_c=0.5,
)


def _logit(x, lo, hi):  return np.log((x - lo) / (hi - x))
def _ilogit(y, lo, hi): return lo + (hi - lo) / (1.0 + np.exp(-np.clip(y, -20, 20)))


def _theta_to_model(theta: np.ndarray, N: int, share_tau_c: bool,
                    cfg: dict) -> LayeredModel:
    lo_m, hi_m = cfg['M_LIMITS']
    lo_t_log, hi_t_log = cfg['TAU_LIMITS_LOG']
    lo_c,  hi_c  = cfg['C_LIMITS']
    rho0 = 10.0 ** theta[0:N]
    if share_tau_c:
        m  = _ilogit(theta[N:2*N], lo_m, hi_m)
        tau_global = 10.0 ** np.clip(theta[2*N],   lo_t_log, hi_t_log)
        c_global   = _ilogit(theta[2*N+1], lo_c, hi_c)
        tau = np.full(N, tau_global); c = np.full(N, c_global)
    else:
        m  = _ilogit(theta[N:2*N], lo_m, hi_m)
        tau_log = theta[2*N:3*N]
        c_logit = theta[3*N:4*N]
        tau = 10.0 ** np.clip(tau_log, lo_t_log, hi_t_log)
        c = _ilogit(c_logit, lo_c, hi_c)
    thickness = np.full(N, 20.0); thickness[-1] = np.inf
    return LayeredModel(thickness, rho0, m, tau, c)


def invert_one_station_ip(rhoa_obs: np.ndarray, vt_obs: np.ndarray,
                          gates_sec: np.ndarray,
                          tx_area: float, tx_current: float,
                          n_layers: int = 25,
                          user_cfg: Optional[dict] = None,
                          log_fn: Optional[Callable[[str], None]] = None,
                          vt_only_mode: bool = False
                          ) -> dict:
    """单站 IP 4参数反演。返回 model + 残差。

    vt_only_mode=True: 仅用 Vt 残差 (TEM 标准IP做法 工业做法 — 直接拟合关断后
    衰减曲线, 保留 IP 弛豫尾部). rhoa_obs 仅用于初值估计, 不进入残差。
    vt_only_mode=False (默认): 双残差 rhoa+Vt (旧路径, 数据已去 IP 时退化解)。
    """
    cfg = dict(DEFAULT_IP_CFG); (user_cfg is None) or cfg.update(user_cfg)
    log_fn = log_fn or (lambda s: None)
    rhoa_obs = np.maximum(np.asarray(rhoa_obs, dtype=np.float64), 1e-4)
    vt_obs   = np.maximum(np.abs(np.asarray(vt_obs, dtype=np.float64)), 1e-30)
    Ng = len(gates_sec); N = n_layers
    share = bool(cfg['SHARE_TAU_C'])
    n_params = (2*N + 2) if share else (4*N)

    def unpack_theta(theta):
        m = _theta_to_model(theta, N, share, cfg)
        _, vt_pred = tem_forward_1d(m, tx_area, tx_current, gates_sec, include_ip=True)
        rhoa_pred = apparent_resistivity_dbzdt(vt_pred, gates_sec, tx_area, tx_current)
        rhoa_pred = np.maximum(rhoa_pred, 1e-4)
        vt_pred   = np.maximum(np.abs(vt_pred), 1e-30)
        return vt_pred, rhoa_pred

    def resid(theta):
        vt_pred, rhoa_pred = unpack_theta(theta)
        if vt_only_mode:
            # 纯 Vt 残差 (TEM 工业做法): rhoa_obs 信息不可靠(已去IP), 只用 Vt
            r_data = np.log10(vt_pred / vt_obs) / cfg['ERROR_VT_FRAC']
        else:
            r1 = np.log10(rhoa_pred / rhoa_obs) / cfg['ERROR_RHOA_FRAC']
            r2 = np.log10(vt_pred / vt_obs) / cfg['ERROR_VT_FRAC']
            r_data = np.concatenate([r1, r2])
        # 层间平滑：对 ρ0 每层 diff; 对 m 每层 diff; (τ,c 全局时跳过)
        sm = []
        lam = np.sqrt(cfg['LAMBDA_SMOOTH'])
        sm.append(lam * np.diff(theta[0:N]))
        sm.append(lam * 0.7 * np.diff(theta[N:2*N]))  # m 平滑略小权重
        if not share:
            sm.append(lam * 0.5 * np.diff(theta[2*N:3*N]))
            sm.append(lam * 0.5 * np.diff(theta[3*N:4*N]))
        return np.concatenate([r_data] + [s for s in sm if len(s) > 0])

    # 初值：ρ0 = ρa 几何均值分配（浅层小、深层大），m=0.05, τ=1e-3, c=0.5
    rho_geo = np.exp(np.mean(np.log(rhoa_obs)))
    rho_init = np.logspace(np.log10(max(0.2, rho_geo/5)),
                           np.log10(min(5e7, rho_geo*5)), N)
    theta0 = np.zeros(n_params)
    theta0[0:N] = np.log10(rho_init)
    theta0[N:2*N] = _logit(np.full(N, cfg['INIT_m']), *cfg['M_LIMITS'])
    if share:
        theta0[2*N]   = np.log10(cfg['INIT_tau'])
        theta0[2*N+1] = _logit(cfg['INIT_c'], *cfg['C_LIMITS'])
    else:
        theta0[2*N:3*N] = np.log10(cfg['INIT_tau'])
        theta0[3*N:4*N] = _logit(cfg['INIT_c'], *cfg['C_LIMITS'])
    # bounds: ±inf (因为所有参数已通过 log/logit 自然约束)
    try:
        res = least_squares(resid, theta0, method='trf', max_nfev=cfg['MAX_NFEV'],
                            x_scale='jac', verbose=0)
        model_f = _theta_to_model(res.x, N, share, cfg)
        vt_fit, rhoa_fit = unpack_theta(res.x)
        rmse_r = np.sqrt(np.mean((np.log10(rhoa_fit / rhoa_obs))**2))
        rmse_v = np.sqrt(np.mean((np.log10(vt_fit / vt_obs))**2))
        out = {'status': 'OK', 'nfev': res.nfev,
               'rmse_log10_rhoa': float(rmse_r),
               'rmse_log10_vt':   float(rmse_v),
               'vt_only_mode':    vt_only_mode,
               'model': model_f,
               'rhoa_fit': rhoa_fit, 'vt_fit': vt_fit,
               'theta_final': res.x.copy()}
        tag = '[Vt-only]' if vt_only_mode else '[IP]'
        log_fn(f"  {tag} OK nfev={res.nfev} RMSE_ρa={rmse_r:.3f} RMSE_Vt={rmse_v:.3f}")
        return out
    except Exception as exc:
        log_fn(f"  [IP] FAIL: {exc}")
        return {'status': 'FAIL', 'error': str(exc)}


# ============================================================
# 全线入口 → 与 run_all_inversions 输出结构一致 (R[st,4N]/summary/断面图)
# ============================================================
def run_all_ip_inversions(out_root: str | Path,
                          user_cfg: Optional[dict] = None,
                          ip_cfg: Optional[dict] = None,
                          log_fn: Optional[Callable[[str], None]] = None,
                          prefer_vt_only: bool = True,
                          ) -> dict:
    from pathlib import Path
    import pandas as pd
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .inversion import depth_layer_centers, DEFAULT_INV_CFG, parse_occam_dat  # 公开API

    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_IP_CFG); (ip_cfg is None) or cfg.update(ip_cfg)
    out_root = Path(out_root)
    import re as _re
    summary_total = []
    line_dirs = sorted([p for p in out_root.iterdir()
                        if p.is_dir() and _re.fullmatch(r'[1-5]线', p.name)])
    for line_tag in [p.name for p in line_dirs]:
        ldir = out_root / line_tag
        occam = ldir / f'{line_tag}_occam.dat'
        if not occam.exists():
            log_fn(f"  [IP] skip {line_tag}: Occam.dat 不存在"); continue
        # 解析 Occam.dat：stations × gates（ρa/Vt 双观测）
        try:
            df_raw = parse_occam_dat(occam)
        except Exception as exc:
            log_fn(f"  [IP] parse Occam fail: {exc}"); continue
        if df_raw.empty:
            log_fn(f"  [IP] skip {line_tag}: Occam.dat 为空"); continue
        df = df_raw.rename(columns={
            'Station': 'station', 'Time_s': 't_sec',
            'AppRes': 'rhoa_ohm_m', 'StdDev': 'stddev_ohm_m'
        }).copy()
        stations = list(dict.fromkeys(df['station'].tolist()))
        tx_area_guess = 4.45e6 if int(line_tag.rstrip('线')) % 2 == 1 else 1.72e6
        I_guess = 37.0 if line_tag == '5线' else 30.0
        user_inv_cfg = dict(DEFAULT_INV_CFG)
        if isinstance(user_cfg, dict): user_inv_cfg.update(user_cfg)
        z_axis = depth_layer_centers(user_inv_cfg['N_LAYERS'],
                                     user_inv_cfg['Z_MIN_M'],
                                     user_inv_cfg['Z_MAX_M'])
        N_L = len(z_axis)
        R_m, R_tau, R_c = [np.full((len(stations), N_L), np.nan) for _ in range(3)]
        R_rho = np.full((len(stations), N_L), np.nan)
        line_summ = []
        gates_unique = np.sort(df['t_sec'].unique())[:25]

        # === TEM 标准IP做法 工业做法: 优先加载 decay_curves.npz 中的真实 Vt_obs ===
        # decay NPZ 含完整 IP 弛豫尾部 (AATZero+陷波+带通+MAD+gating 后的衰减曲线)
        # prefer_vt_only=False 时强制走旧双残差路径 (即使 NPZ 存在也不用)
        npz_path = ldir / f'{line_tag}_decay_curves.npz'
        decay_data = None
        vt_only_mode = False
        if prefer_vt_only and npz_path.exists():
            try:
                with np.load(npz_path, allow_pickle=True) as _nz:
                    decay_data = _nz['data'].item()  # dict[station][channel] -> dict
                vt_only_mode = True
                log_fn(f"  [IP] ✓ 加载 decay NPZ: {npz_path.name} ({len(decay_data)} 站)")
                log_fn(f"  [IP] 启用 vt_only_mode=True (TEM 标准IP做法 工业做法)")
            except Exception as exc:
                log_fn(f"  [IP] 加载 decay NPZ 失败, 回退双残差: {exc}")
                decay_data = None
        else:
            reason = "用户在 GUI 关闭了 prefer_vt_only" if not prefer_vt_only else "decay NPZ 不存在"
            log_fn(f"  [IP] 使用双残差模式 (Occam ρa + Vt 反推) — 原因: {reason}")

        log_fn(f"\n=== IP反演 {line_tag} N_LAYERS={N_L} share_τc={cfg['SHARE_TAU_C']} λ={cfg['LAMBDA_SMOOTH']} vt_only={vt_only_mode} ===")
        for si, st in enumerate(stations):
            sub = df[df['station'] == st].sort_values('t_sec')
            if len(sub) < 5: continue
            rhoa = sub['rhoa_ohm_m'].to_numpy(dtype=np.float64)
            t_s  = sub['t_sec'].to_numpy(dtype=np.float64)

            if vt_only_mode and decay_data and st in decay_data:
                # === 新路径: 直接用 NPZ 中的真实 Vt_obs (含 IP 弛豫) ===
                ch_dict = decay_data[st]
                # 优先 Ex (电场衰减), 回退 Hx, 再回退任意
                ch = ch_dict.get('Ex') or ch_dict.get('Hx') or next(iter(ch_dict.values()))
                vt_obs_real = np.maximum(np.abs(ch['gates_values']), 1e-30)
                gates_real  = ch['gates_t_s']
                # rhoa 仍传, 仅用于初值估计 rho_geo
                rhoa_for_init = rhoa[:min(25, len(rhoa))]
                r = invert_one_station_ip(
                    rhoa_for_init, vt_obs_real, gates_real,
                    tx_area_guess, I_guess, n_layers=N_L,
                    user_cfg=ip_cfg, log_fn=log_fn,
                    vt_only_mode=True,
                )
            else:
                # === 旧路径: Occam ρa + Vt 反推 ===
                from .preprocess import DEFAULT_CFG as _DCFG
                cal = _DCFG['RHOA_SCALE_CAL']
                t_safe = np.maximum(t_s, 1e-12)
                rhoa_safe = np.maximum(rhoa, 1e-4)
                vt_est = (cal / 20.0 / np.pi) ** 1.5 / (t_safe ** 2.25) / (rhoa_safe ** 1.5) * (I_guess * tx_area_guess)
                vt_est = np.maximum(np.abs(vt_est), 1e-30)
                r = invert_one_station_ip(
                    rhoa[:25], vt_est[:25], gates_unique[:min(25,len(t_s))],
                    tx_area_guess, I_guess, n_layers=N_L,
                    user_cfg=ip_cfg, log_fn=log_fn,
                    vt_only_mode=False,
                )
            if r['status'] != 'OK': continue
            mdl: LayeredModel = r['model']
            R_rho[si, :len(mdl.rho_ohm_m)] = mdl.rho_ohm_m
            R_m  [si, :len(mdl.rho_ohm_m)] = mdl.chargeability
            if cfg['SHARE_TAU_C']:
                R_tau[si, :] = mdl.tau_s[0]; R_c[si, :] = mdl.c_exp[0]
            else:
                R_tau[si, :len(mdl.rho_ohm_m)] = mdl.tau_s
                R_c  [si, :len(mdl.rho_ohm_m)] = mdl.c_exp
            line_summ.append({'station': st,
                              'RMSE_rhoa_log': r['rmse_log10_rhoa'],
                              'RMSE_vt_log':   r['rmse_log10_vt'],
                              'nfev': r['nfev'],
                              'vt_only': int(vt_only_mode)})
        if not line_summ: continue
        # 保存
        np.save(ldir / f'{line_tag}_ip_R_rho.npy', R_rho)
        np.save(ldir / f'{line_tag}_ip_R_m.npy',   R_m)
        np.save(ldir / f'{line_tag}_ip_R_tau.npy', R_tau)
        np.save(ldir / f'{line_tag}_ip_R_c.npy',   R_c)
        np.save(ldir / f'{line_tag}_ip_depth.npy', z_axis)
        pd.DataFrame(line_summ).to_csv(ldir / f'{line_tag}_ip_inversion_summary.csv',
                                       index=False, encoding='utf-8-sig')
        summary_total.extend([{**r, 'line': line_tag} for r in line_summ])
        # 3 张断面：ρ / m / τ
        _plot_section(R_rho, z_axis, stations, 'Resistivity IP (Ω·m)', 'jet',
                      ldir / f'{line_tag}_ip_rho_section.png')
        _plot_section(R_m,   z_axis, stations, 'Chargeability m (Pelton)', 'plasma',
                      ldir / f'{line_tag}_ip_m_section.png', vmin=0, vmax=0.5)
        _plot_section(np.log10(R_tau), z_axis, stations, 'log10 τ (s)', 'cividis',
                      ldir / f'{line_tag}_ip_tau_section.png')
        n_ok = len(line_summ)
        log_fn(f"  ✓ [{line_tag}] IP反演: {n_ok}/{len(stations)} stations  "
               f"m median={np.nanmedian(R_m):.3f}  tau median={np.nanmedian(R_tau):.2e}s")
    return {'n_lines': len(line_dirs), 'n_stations_ok': len(summary_total),
            'summary_rows': summary_total}


def _plot_section(R: np.ndarray, z: np.ndarray, stations: list, title: str, cmap: str,
                  out_png: Path, vmin=None, vmax=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(stations)), 5.5))
    X = np.arange(len(stations) + 1)
    Z = np.append(z, z[-1]*1.5)  # 末层加宽
    if vmin is None:
        mask = (np.isfinite(R)) & (R != 0)
        vals = R[mask] if mask.sum() > 5 else np.asarray(R).ravel()
        try:
            vals_finite = vals[np.isfinite(vals)]
            if len(vals_finite) >= 5:
                vmin, vmax = np.nanpercentile(vals_finite, [2, 98])
        except Exception:
            pass
    pm = ax.pcolormesh(X, Z, R.T, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.set_yscale('log')
    ax.set_ylim(Z.max(), Z.min())
    ax.set_xticks(np.arange(len(stations)) + 0.5, minor=False)
    ax.set_xticklabels(stations, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Depth (m, log)'); ax.set_xlabel('Station')
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(pm, ax=ax, pad=0.02)
    cbar.set_label(title, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


__all__ = ['DEFAULT_IP_CFG', 'invert_one_station_ip', 'run_all_ip_inversions']
