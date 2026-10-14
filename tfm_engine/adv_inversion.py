# -*- coding: utf-8 -*-
"""
tfm_engine.adv_inversion
========================
高级 1D 电阻率反演算法集：
  occam_adaptive : λ 自动寻优 + L曲线记录 (Constable 1987 自适应版)
  lm_adaptive    : Nielsen 自适应阻尼 Levenberg-Marquardt
  cgls           : 模型空间共轭梯度最小二乘 (Rodi-Mackie 风格光滑反演)
  mcmc           : Metropolis-Hastings 后验采样, 输出均值/标准差/95% HPD

统一接口与 inv_algorithms 一致, 完全离线, 纯 numpy + scipy.
"""
from __future__ import annotations
import json, hashlib, re
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import least_squares

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from .inversion import (forward_rhoa, DEFAULT_INV_CFG, depth_layer_centers,
                        parse_occam_dat, MU0)

ALGO_ADVANCED = ('occam_adaptive', 'lm_adaptive', 'cgls', 'mcmc')

# ============================================================
# 公共残差 / 雅可比 (与 inv_algorithms 一致, 避免循环 import)
# ============================================================
def _weights(rhoa_obs: np.ndarray, err_obs: np.ndarray, eps: float) -> np.ndarray:
    log_err_frac = np.abs(np.log10(err_obs / rhoa_obs + eps))
    return 1.0 / np.maximum(log_err_frac, 0.05)


def _data_resid(theta: np.ndarray, t_g, r_g, w_g, z_c, rho_ref) -> np.ndarray:
    pred = np.maximum(forward_rhoa(theta, t_g, z_c, rho_ref), 1e-3)
    return (np.log10(pred) - np.log10(np.maximum(r_g, 1e-3))) * w_g


def _numeric_jacobian(theta, t_g, z_c, rho_ref, w_g, fd_step=1e-4) -> np.ndarray:
    base = forward_rhoa(theta, t_g, z_c, rho_ref)
    log_base = np.log10(np.maximum(base, 1e-3))
    J = np.zeros((t_g.size, theta.size))
    for j in range(theta.size):
        tp = theta.copy(); tp[j] += fd_step
        log_p = np.log10(np.maximum(forward_rhoa(tp, t_g, z_c, rho_ref), 1e-3))
        J[:, j] = (log_p - log_base) / fd_step
    return J * w_g[:, None]


def _smooth_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    """一阶粗糙度 D (n-1, n) 和 DtD (n, n)"""
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1.0; D[i, i + 1] = 1.0
    return D, D.T @ D


# ============================================================
# 算法 1: 自适应 Occam (λ 自动寻优 + L曲线)
# ============================================================
def _invert_occam_adaptive(t_g, r_g, w_g, z_c, cfg):
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    th = np.full(n_l, np.log10(cfg['RHO_INIT']))
    target = 1.0
    tol_misfit = 0.05
    max_iter = 20
    lam_min, lam_max = 1e-6, 1e4
    D, DtD = _smooth_matrix(n_l)
    nfev = 0
    l_curve = []  # (lambda, chi2, roughness)

    def misfit(thv):
        r = _data_resid(thv, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        return float(np.mean(r ** 2))

    chi2 = misfit(th); nfev += 1
    lam = float(cfg.get('LAMBDA_INIT', 1.0))
    for it in range(max_iter):
        if abs(chi2 - target) < tol_misfit:
            break
        Jw = _numeric_jacobian(th, t_g, z_c, cfg['RHO_REF_DIFF'], w_g)
        nfev += n_l
        r = _data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        JtJ = Jw.T @ Jw
        Jtr = Jw.T @ r
        # λ 对数网格搜索 12 点, 取 χ²≤target 中最大 λ (最光滑)
        best = None
        lams = np.geomspace(max(lam_min, 1e-6), lam_max, 12)
        for lam_try in lams:
            A = JtJ + lam_try * DtD
            b = -Jtr - lam_try * (DtD @ th)
            try:
                dm = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            th_new = np.clip(th + dm, lo, hi)
            chi2_new = misfit(th_new); nfev += 1
            rough = float(np.linalg.norm(D @ th_new))
            l_curve.append((float(lam_try), float(chi2_new), rough))
            score = (0 if chi2_new <= target else 1, abs(chi2_new - target), -lam_try)
            if best is None or score < best[0]:
                best = (score, lam_try, th_new, chi2_new)
        if best is None:
            break
        _, lam, th_new, chi2_new = best
        step = float(np.max(np.abs(th_new - th)))
        th, chi2 = th_new, chi2_new
        if step < 1e-4:
            break
    return th, {'success': abs(chi2 - target) < 2 * tol_misfit or len(l_curve) > 0,
                'nfev': nfev, 'cost': chi2, 'lambda_final': lam,
                'l_curve': l_curve, 'n_iter': it + 1}


# ============================================================
# 算法 2: 自适应阻尼 LM (Nielsen 策略)
# ============================================================
def _invert_lm_adaptive(t_g, r_g, w_g, z_c, cfg):
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    th = np.full(n_l, np.log10(cfg['RHO_INIT']))
    lam = 1e-3
    lam_factor_up = 4.0
    lam_factor_down = 0.5
    max_iter = 50
    tol_step = 1e-4
    nfev = 0

    def full_cost(thv):
        rd = _data_resid(thv, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        rs = np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(thv)
        return float(np.sum(rd ** 2) + np.sum(rs ** 2))

    cost = full_cost(th); nfev += 1
    for it in range(max_iter):
        Jw = _numeric_jacobian(th, t_g, z_c, cfg['RHO_REF_DIFF'], w_g)
        nfev += n_l
        rd = _data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        rs = np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(th)
        # 增广雅可比
        D, _ = _smooth_matrix(n_l)
        D *= np.sqrt(cfg['LAMBDA_SMOOTH'])
        Ja = np.vstack([Jw, D])
        r = np.concatenate([rd, rs])
        # Nielsen: 用 Ja^T Ja 对角线缩放
        JtJ = Ja.T @ Ja
        diag_JtJ = np.diag(JtJ)
        diag_JtJ = np.where(diag_JtJ < 1e-12, 1e-12, diag_JtJ)
        H = JtJ + lam * np.diag(diag_JtJ)
        g = Ja.T @ r
        try:
            dm = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            lam *= lam_factor_up
            continue
        th_new = np.clip(th + dm, lo, hi)
        cost_new = full_cost(th_new); nfev += 1
        if cost_new < cost:
            th = th_new
            cost = cost_new
            lam *= lam_factor_down
            if np.max(np.abs(dm)) < tol_step:
                break
        else:
            lam *= lam_factor_up
            if lam > 1e12:
                break
    return th, {'success': True, 'nfev': nfev, 'cost': cost,
                'lambda_final': lam, 'n_iter': it + 1}


# ============================================================
# 算法 3: CGLS (模型空间共轭梯度, Rodi-Mackie 风格)
# ============================================================
def _invert_cgls(t_g, r_g, w_g, z_c, cfg):
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    th = np.full(n_l, np.log10(cfg['RHO_INIT']))
    lam = cfg['LAMBDA_SMOOTH']
    max_iter = 50
    tol = 1e-4
    nfev = 0

    D, DtD = _smooth_matrix(n_l)
    for it in range(max_iter):
        Jw = _numeric_jacobian(th, t_g, z_c, cfg['RHO_REF_DIFF'], w_g)
        nfev += n_l
        rd = _data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        # 右端项 = -J^T rd (负梯度方向)
        b = -Jw.T @ rd
        # 算子 A = J^T J + λ DtD
        def A_op(x):
            return Jw.T @ (Jw @ x) + lam * (DtD @ x)
        dm = np.zeros(n_l)
        r_cg = b - A_op(dm)
        p = r_cg.copy()
        rsold = float(r_cg @ r_cg)
        if rsold < 1e-20:
            break
        for cg_it in range(min(n_l, 200)):
            Ap = A_op(p)
            alpha = rsold / max(float(p @ Ap), 1e-30)
            dm += alpha * p
            r_cg -= alpha * Ap
            rsnew = float(r_cg @ r_cg)
            if np.sqrt(rsnew) < tol:
                break
            p = r_cg + (rsnew / rsold) * p
            rsold = rsnew
        th_new = np.clip(th + dm, lo, hi)
        if np.max(np.abs(th_new - th)) < tol:
            th = th_new
            break
        th = th_new
    cost = float(np.mean(_data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF']) ** 2))
    return th, {'success': True, 'nfev': nfev, 'cost': cost, 'n_iter': it + 1}


# ============================================================
# 算法 4: MCMC (Metropolis-Hastings, 纯 numpy)
# ============================================================
def _invert_mcmc(t_g, r_g, w_g, z_c, cfg, n_samples=2000, burn_in=1000):
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    # 先用 TRF 快速拿 MAP 初值
    def resid(th):
        return np.concatenate([
            _data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF']),
            np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(th)
        ])
    th0 = np.full(n_l, np.log10(cfg['RHO_INIT']))
    try:
        res = least_squares(resid, th0, bounds=(np.full(n_l, lo), np.full(n_l, hi)),
                            method='trf', max_nfev=200, x_scale='jac')
        th_map = res.x
    except Exception:
        th_map = th0

    def log_likelihood(thv):
        rd = _data_resid(thv, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        rs = np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(thv)
        return -0.5 * float(np.sum(rd ** 2) + np.sum(rs ** 2))

    # proposal 标准差 = (hi-lo)/15
    sigma_prop = (hi - lo) / 15.0
    theta_curr = th_map.copy()
    logL_curr = log_likelihood(theta_curr)
    samples = []
    n_accept = 0
    for i in range(n_samples + burn_in):
        theta_prop = theta_curr + np.random.normal(0, sigma_prop, n_l)
        theta_prop = np.clip(theta_prop, lo, hi)
        logL_prop = log_likelihood(theta_prop)
        alpha = min(1.0, np.exp(logL_prop - logL_curr))
        if np.random.rand() < alpha:
            theta_curr = theta_prop.copy()
            logL_curr = logL_prop
            n_accept += 1
        if i >= burn_in:
            samples.append(theta_curr.copy())
    samples = np.array(samples)  # (S, n_l)
    th_mean = samples.mean(axis=0)
    th_std = samples.std(axis=0)
    hpd_lo = np.percentile(samples, 2.5, axis=0)
    hpd_hi = np.percentile(samples, 97.5, axis=0)
    return th_mean, {
        'success': True, 'nfev': n_samples + burn_in,
        'accept_rate': n_accept / (n_samples + burn_in),
        'theta_std': th_std, 'theta_hpd_lo': hpd_lo, 'theta_hpd_hi': hpd_hi,
        'samples': samples, 'cost': -2.0 * logL_curr / t_g.size,
    }


# ============================================================
# 统一单站入口
# ============================================================
def invert_one_station_advanced(t_obs, rhoa_obs, err_obs, z_centers, cfg,
                                algo: str = 'occam_adaptive',
                                occam_cfg: Optional[dict] = None,
                                mcmc_samples: int = 2000,
                                mcmc_burn: int = 1000):
    n_l = z_centers.size
    th0 = np.full(n_l, np.log10(cfg['RHO_INIT']))
    good = np.isfinite(rhoa_obs) & (rhoa_obs > 0) & np.isfinite(err_obs) & (err_obs > 0)
    if good.sum() < 3:
        return th0, {'success': False, 'reason': f'only {good.sum()} valid obs',
                     'rmse_log': np.nan, 'nfev': 0}
    t_g, r_g, e_g = t_obs[good], rhoa_obs[good], err_obs[good]
    w_g = _weights(r_g, e_g, cfg['LOG_EPS'])

    if algo == 'occam_adaptive':
        th, info = _invert_occam_adaptive(t_g, r_g, w_g, z_centers, cfg)
    elif algo == 'lm_adaptive':
        th, info = _invert_lm_adaptive(t_g, r_g, w_g, z_centers, cfg)
    elif algo == 'cgls':
        th, info = _invert_cgls(t_g, r_g, w_g, z_centers, cfg)
    elif algo == 'mcmc':
        th, info = _invert_mcmc(t_g, r_g, w_g, z_centers, cfg,
                                n_samples=mcmc_samples, burn_in=mcmc_burn)
    else:
        raise ValueError(f'未知算法 {algo!r}, 可选 {ALGO_ADVANCED}')

    pred = forward_rhoa(th, t_g, z_centers, cfg['RHO_REF_DIFF'])
    info['rmse_log'] = float(np.sqrt(np.mean(
        (np.log10(np.maximum(pred, 1e-3)) - np.log10(r_g)) ** 2)))
    rho_lin = 10.0 ** th
    info.update(rho_min=float(rho_lin.min()), rho_med=float(np.median(rho_lin)),
                rho_max=float(rho_lin.max()))
    return th, info


# ============================================================
# 测线/全线入口 (带并行 + 缓存)
# ============================================================
def _cfg_hash(cfg: dict, algo: str) -> str:
    """简化的 cfg 确定性哈希"""
    s = json.dumps({k: str(v) for k, v in sorted(cfg.items())}, sort_keys=True) + '|' + algo
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _cache_path(out_base: Path, st: str, algo: str, cfg_hash: str) -> Path:
    return out_base / '_inv_cache' / f'{st}_{algo}_{cfg_hash}.npz'


def _slim_info(info: dict) -> dict:
    """剥离大数组, 防止污染 CSV / 缓存 (samples/l_curve 等大对象只进诊断图)"""
    slim = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray) and v.size > 64:
            continue
        if isinstance(v, list) and len(v) > 200:
            continue
        slim[k] = v
    return slim


def _adv_station_worker(args: dict):
    """模块级 worker (Windows spawn 可 pickle)。args 全为纯数据。"""
    idx = args['idx']; st = args['st']
    cp = args.get('cache_path')
    if cp:
        cp = Path(cp)
        if cp.is_file():
            try:
                d = np.load(cp, allow_pickle=True)
                return idx, st, d['th'], dict(d['info'].item()), True
            except Exception:
                pass
    th, info = invert_one_station_advanced(
        args['t_obs'], args['rhoa_obs'], args['err_obs'],
        args['z_c'], args['cfg'], args['algo'],
        mcmc_samples=args.get('mcmc_samples', 2000),
        mcmc_burn=args.get('mcmc_burn', 1000))
    if cp:
        try:
            np.savez_compressed(cp, th=th, info=_slim_info(info))
        except Exception:
            pass
    return idx, st, th, info, False


def invert_line_advanced(line_tag: str, out_root: str | Path, algo: str = 'occam_adaptive',
                         user_cfg: Optional[dict] = None,
                         n_jobs: int = 1, use_cache: bool = False,
                         mcmc_samples: int = 2000, mcmc_burn: int = 1000,
                         log_fn: Optional[Callable[[str], None]] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_INV_CFG)
    if user_cfg: cfg.update(user_cfg)
    out_base = Path(out_root) / line_tag
    dat = out_base / f'{line_tag}_occam.dat'
    if not dat.is_file():
        log_fn(f'  [SKIP] {line_tag}: 无 {dat.name}')
        return {}
    log_fn(f'\n=== 1D 高级反演[{algo}] {line_tag}  N_LAYERS={cfg["N_LAYERS"]} ===')
    df = parse_occam_dat(dat)
    stations = sorted(df.Station.unique())
    z_c = depth_layer_centers(cfg['N_LAYERS'], cfg['Z_MIN_M'], cfg['Z_MAX_M'])
    cfg_hash = _cfg_hash(cfg, algo)
    cache_dir = out_base / '_inv_cache'
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    R_log = np.zeros((len(stations), cfg['N_LAYERS']))
    summary = []
    cache_hits = 0

    # 组装纯数据参数包 (spawn 可 pickle)
    args_list = []
    for idx, st in enumerate(stations):
        sub = df[df.Station == st].sort_values('Time_s')
        cp = _cache_path(out_base, st, algo, cfg_hash) if use_cache else None
        args_list.append({
            'idx': idx, 'st': st,
            't_obs': sub.Time_s.values, 'rhoa_obs': sub.AppRes.values,
            'err_obs': sub.StdDev.values,
            'z_c': z_c, 'cfg': cfg, 'algo': algo,
            'cache_path': str(cp) if cp else None,
            'mcmc_samples': mcmc_samples, 'mcmc_burn': mcmc_burn,
        })

    def _collect(idx, st, th, info, hit):
        nonlocal cache_hits
        R_log[idx] = th
        info['Station'] = st
        summary.append(info)
        if hit: cache_hits += 1
        log_fn(f'  [{st}] {algo} RMSE_log={info.get("rmse_log", np.nan):.3f} '
               f'nfev={info.get("nfev", 0)} cache={hit}')

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as exe:
            futures = [exe.submit(_adv_station_worker, a) for a in args_list]
            for fut in as_completed(futures):
                _collect(*fut.result())
    else:
        for a in args_list:
            _collect(*_adv_station_worker(a))

    # 汇总 CSV 前再剥离一次大对象 (ndarray/list)
    slim_summary = []
    for info in summary:
        slim_summary.append(_slim_info(info))
    R_lin = 10.0 ** R_log
    np.save(out_base / f'{line_tag}_inv_{algo}_R.npy', R_lin)
    np.save(out_base / f'{line_tag}_inv_{algo}_depth.npy', z_c)
    pd.DataFrame(slim_summary).to_csv(out_base / f'{line_tag}_inv_{algo}_summary.csv',
                                      index=False, encoding='utf-8-sig')
    rmses = [s['rmse_log'] for s in summary if np.isfinite(s.get('rmse_log', np.nan))]
    rmse_med = float(np.median(rmses)) if rmses else np.nan

    # 断面 PNG
    if len(stations) >= 2:
        fig, ax = plt.subplots(figsize=(max(8, len(stations) * 0.4 + 2), 6))
        X, Y = np.meshgrid(np.arange(len(stations)), np.log10(z_c), indexing='ij')
        Z = np.log10(np.clip(R_lin, cfg['RHO_BOUNDS'][0], cfg['RHO_BOUNDS'][1]))
        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet_r')
        ax.set_xlabel(f'测点 ({len(stations)})'); ax.set_ylabel('深度 (m, log)')
        yt = np.linspace(np.log10(cfg['Z_MIN_M']), np.log10(cfg['Z_MAX_M']), 6)
        ax.set_yticks(yt); ax.set_yticklabels([f'{10**v:.0f}' for v in yt])
        step = max(1, len(stations) // 10)
        ax.set_xticks(np.arange(0, len(stations), step))
        ax.set_xticklabels([stations[i] for i in range(0, len(stations), step)], rotation=45)
        ax.invert_yaxis()
        ax.set_title(f'{line_tag} 1D反演[{algo}] ρ断面  RMSE_log10_med={rmse_med:.3f}')
        fig.colorbar(im, ax=ax).set_label('log10(ρ/Ω·m)')
        fig.tight_layout()
        png = out_base / f'{line_tag}_inv_{algo}_section.png'
        fig.savefig(png, dpi=180); plt.close(fig)

    # 诊断图：L曲线 / MCMC 后验
    if algo == 'occam_adaptive':
        _plot_lcurve(summary, out_base / f'{line_tag}_inv_{algo}_lcurve.png')
    elif algo == 'mcmc':
        _plot_mcmc_posterior(summary, z_c, out_base / f'{line_tag}_inv_{algo}_posterior.png')

    log_fn(f'  [SUMMARY] {line_tag}[{algo}]: {len(stations)}站 '
           f'RMSE_log10_med={rmse_med:.3f} cache_hits={cache_hits}/{len(stations)}')
    return {'stations': len(stations), 'rmse_median_log': rmse_med, 'algo': algo,
            'cache_hits': cache_hits}


def _plot_lcurve(summary_list, out_png: Path):
    """取第一站绘制 L 曲线示意"""
    info0 = summary_list[0] if summary_list else {}
    l_curve = info0.get('l_curve')
    if not l_curve:
        return
    l_curve = np.array(l_curve)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(l_curve[:, 1], l_curve[:, 2], c=np.log10(l_curve[:, 0]), cmap='viridis', s=30)
    ax.set_xlabel('χ² (数据拟合误差)'); ax.set_ylabel('Ω = ||Dθ||² (模型粗糙度)')
    ax.set_title('L曲线 (λ 自动寻优轨迹)')
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('log10(λ)')
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _plot_mcmc_posterior(summary_list, z_c, out_png: Path):
    """MCMC 后验均值 ± 95% HPD 断面示意 (取前 8 层)"""
    info0 = summary_list[0] if summary_list else {}
    samples = info0.get('samples')
    if samples is None or samples.ndim != 2:
        return
    hpd_lo = info0.get('theta_hpd_lo')
    hpd_hi = info0.get('theta_hpd_hi')
    th_mean = samples.mean(axis=0)
    n_show = min(8, len(z_c))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill_betweenx(z_c[:n_show], 10**hpd_lo[:n_show], 10**hpd_hi[:n_show], alpha=0.3, label='95% HPD')
    ax.plot(10**th_mean[:n_show], z_c[:n_show], 'k-', label='后验均值')
    ax.invert_yaxis(); ax.set_yscale('log')
    ax.set_xlabel('ρ (Ω·m)'); ax.set_ylabel('深度 (m)')
    ax.set_title('MCMC 后验电阻率剖面 (首站)')
    ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def run_all_advanced(out_root: str | Path, algo: str = 'occam_adaptive',
                     user_cfg: Optional[dict] = None,
                     n_jobs: int = 1, use_cache: bool = False,
                     mcmc_samples: int = 2000,
                     log_fn: Optional[Callable[[str], None]] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    lines = sorted([p.name for p in out_root.iterdir() if p.is_dir()
                    and re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== 高级反演 algo={algo} 测线={lines} ==========')
    return {lt: invert_line_advanced(lt, out_root, algo, user_cfg,
                                     n_jobs, use_cache,
                                     mcmc_samples=mcmc_samples, log_fn=log_fn)
            for lt in lines}
