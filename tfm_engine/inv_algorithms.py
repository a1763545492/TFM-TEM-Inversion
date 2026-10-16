# -*- coding: utf-8 -*-
"""
tfm_engine.inv_algorithms (M9)
==============================
1D 电阻率反演 多算法统一接口（对应 WLF 体系里可选的反演算法支路）:

  algo='trf'   : scipy least_squares(TRF, bounds) — 现有 inversion.py 算法 (默认, 稳健)
  algo='lm'    : Levenberg-Marquardt (scipy method='lm') — 经典阻尼最小二乘, 无界(软裁剪)
  algo='occam' : 纯 Python Occam 光滑反演 (Constable 1987):
                   外环迭代线性化, 每步在 λ 候选集上搜索使 χ²/N → TARGET_MISFIT,
                   正则项为一阶粗糙度 ||Dθ||², 与商业 Occam1D 同原理.

统一输出 (每测线 × 每算法):
  {X线}_inv_{algo}_R.npy        [station, N_LAYERS] 真电阻率 (Ω·m, 线性)
  {X线}_inv_{algo}_depth.npy    [N_LAYERS] 层中心深度 (m)
  {X线}_inv_{algo}_summary.csv  逐站 RMSE / nfev / ρ 范围
  {X线}_inv_{algo}_section.png  反演断面图
多算法对比 (run_all_algos_compare):
  {X线}_inv_compare.png         3 算法断面并排 + RMSE 对比

与 inversion.py 的 forward_rhoa (扩散深度近似) 完全共用, 保证算法间可比性。
完全离线, 纯 numpy + scipy。
"""
from __future__ import annotations
import json, re, hashlib
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

from .inversion import (parse_occam_dat, depth_layer_centers, forward_rhoa,
                        DEFAULT_INV_CFG, MU0, interp_weight_matrix, analytic_jacobian)

ALGO_CHOICES = ('trf', 'lm', 'occam')

DEFAULT_OCCAM_CFG = dict(
    TARGET_MISFIT=1.0,       # 目标 χ²/N
    LAMBDA_INIT=1.0,         # 初始正则化因子 (调参纪律: 单变量 0.1→1→10)
    MAX_ITER=20,
    TOL_MISFIT=0.05,         # |χ²/N - target| 收敛容差
    MIN_STEP=1e-4,           # θ 步长收敛阈值
)


# ============================================================
# 通用: 加权残差 + 数值雅可比
# ============================================================
def _weights(rhoa_obs: np.ndarray, err_obs: np.ndarray, eps: float) -> np.ndarray:
    log_err_frac = np.abs(np.log10(err_obs / rhoa_obs + eps))
    return 1.0 / np.maximum(log_err_frac, 0.05)


def _data_resid(theta: np.ndarray, t_g, r_g, w_g, z_c, rho_ref) -> np.ndarray:
    pred = np.maximum(forward_rhoa(theta, t_g, z_c, rho_ref), 1e-3)
    return (np.log10(pred) - np.log10(np.maximum(r_g, 1e-3))) * w_g


def _numeric_jacobian(theta, t_g, z_c, rho_ref, w_g, fd_step=1e-4) -> np.ndarray:
    """d(log10 rhoa_pred * w)/dθ, 前向差分. (Ng, N) — 仅用于解析雅可比校验/回退。"""
    base = forward_rhoa(theta, t_g, z_c, rho_ref)
    log_base = np.log10(np.maximum(base, 1e-3))
    J = np.zeros((t_g.size, theta.size))
    for j in range(theta.size):
        tp = theta.copy(); tp[j] += fd_step
        log_p = np.log10(np.maximum(forward_rhoa(tp, t_g, z_c, rho_ref), 1e-3))
        J[:, j] = (log_p - log_base) / fd_step
    return J * w_g[:, None]


def _full_jacobian(theta, t_g, z_c, rho_ref, w_g, lam_smooth, mode='analytic'):
    """完整残差 [数据残差; sqrt(λ)·一阶差分] 的雅可比 (Ng+N-1, N)。

    analytic : 插值权重矩阵直接构造, 0 次额外正演 (模型对 θ 线性, 精确)
    numeric  : 前向差分, N+1 次正演, 仅供梯度校验
    """
    n_l = theta.size
    if mode == 'numeric':
        Jd = _numeric_jacobian(theta, t_g, z_c, rho_ref, w_g)
    else:
        Jd = analytic_jacobian(theta, t_g, z_c, rho_ref, w_g)
    Dm = np.zeros((n_l - 1, n_l))
    Dm[np.arange(n_l - 1), np.arange(n_l - 1)] = -np.sqrt(lam_smooth)
    Dm[np.arange(n_l - 1), np.arange(1, n_l)] = np.sqrt(lam_smooth)
    return np.vstack([Jd, Dm])


# ============================================================
# 算法 1: TRF (委托给现有实现逻辑, 局部重写以保持算法间残差一致)
# ============================================================
def _invert_trf(t_g, r_g, w_g, z_c, cfg) -> tuple[np.ndarray, dict]:
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    th0 = np.full(n_l, np.log10(cfg['RHO_INIT']))
    jac_mode = cfg.get('JAC_MODE', 'analytic')

    def resid(th):
        return np.concatenate([_data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF']),
                               np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(th)])

    def jac(th):
        return _full_jacobian(th, t_g, z_c, cfg['RHO_REF_DIFF'], w_g,
                              cfg['LAMBDA_SMOOTH'], jac_mode)

    res = least_squares(resid, th0, jac=jac,
                        bounds=(np.full(n_l, lo), np.full(n_l, hi)),
                        method='trf', max_nfev=cfg['MAX_NFEV'], x_scale='jac')
    return res.x, {'success': bool(res.success), 'nfev': int(res.nfev), 'cost': float(res.cost)}


# ============================================================
# 算法 2: Levenberg-Marquardt (无界, 解后裁剪)
# ============================================================
def _invert_lm(t_g, r_g, w_g, z_c, cfg) -> tuple[np.ndarray, dict]:
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    th0 = np.full(n_l, np.log10(cfg['RHO_INIT']))
    jac_mode = cfg.get('JAC_MODE', 'analytic')

    def resid(th):
        thc = np.clip(th, lo, hi)
        return np.concatenate([_data_resid(thc, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF']),
                               np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(thc)])

    def jac(th):
        thc = np.clip(th, lo, hi)
        return _full_jacobian(thc, t_g, z_c, cfg['RHO_REF_DIFF'], w_g,
                              cfg['LAMBDA_SMOOTH'], jac_mode)

    res = least_squares(resid, th0, jac=jac, method='lm', max_nfev=cfg['MAX_NFEV'] * 5)
    return np.clip(res.x, lo, hi), {'success': bool(res.success), 'nfev': int(res.nfev),
                                    'cost': float(res.cost)}


# ============================================================
# 算法 3: 纯 Python Occam (Constable 1987, 一阶粗糙度)
# ============================================================
def _invert_occam(t_g, r_g, w_g, z_c, cfg, occfg) -> tuple[np.ndarray, dict]:
    n_l = z_c.size
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])
    th = np.full(n_l, np.log10(cfg['RHO_INIT']))
    n_d = t_g.size
    # 一阶粗糙度矩阵 D (n_l-1, n_l)
    D = np.zeros((n_l - 1, n_l))
    for i in range(n_l - 1):
        D[i, i] = -1.0; D[i, i + 1] = 1.0
    DtD = D.T @ D
    lam = float(occfg['LAMBDA_INIT'])
    target = float(occfg['TARGET_MISFIT'])

    def misfit(thv):
        r = _data_resid(thv, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        return float(np.mean(r ** 2))

    chi2 = misfit(th)
    nfev = 1
    n_iter = 0
    jac_mode = cfg.get('JAC_MODE', 'analytic')
    for it in range(int(occfg['MAX_ITER'])):
        n_iter = it + 1
        if abs(chi2 - target) < float(occfg['TOL_MISFIT']):
            break
        if jac_mode == 'numeric':
            Jw = _numeric_jacobian(th, t_g, z_c, cfg['RHO_REF_DIFF'], w_g)
            nfev += n_l
        else:
            Jw = analytic_jacobian(th, t_g, z_c, cfg['RHO_REF_DIFF'], w_g)
            nfev += 1
        r = _data_resid(th, t_g, r_g, w_g, z_c, cfg['RHO_REF_DIFF'])
        JtJ = Jw.T @ Jw
        Jtr = Jw.T @ r
        # λ 候选搜索: 使新 misfit 逼近 target (Occam 核心: 最大 λ 使 χ²≤target)
        best = None
        for lam_try in lam * np.array([5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]):
            A = JtJ + lam_try * DtD
            b = -Jtr - lam_try * (DtD @ th)
            try:
                dm = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            th_new = np.clip(th + dm, lo, hi)
            chi2_new = misfit(th_new)
            nfev += 1
            # 偏好: χ²_new ≤ target 中最大 λ (最光滑); 否则 χ² 最小
            score = (0 if chi2_new <= target else 1,
                     abs(chi2_new - target), lam_try)
            if best is None or score < best[0]:
                best = (score, lam_try, th_new, chi2_new)
        if best is None:
            break
        _, lam, th_new, chi2_new = best
        step = float(np.max(np.abs(th_new - th)))
        th, chi2 = th_new, chi2_new
        if step < float(occfg['MIN_STEP']):
            break
    return th, {'success': bool(abs(chi2 - target) < 2 * float(occfg['TOL_MISFIT']) or n_iter > 0),
                'nfev': int(nfev), 'cost': chi2, 'n_iter': n_iter, 'lambda_final': lam}


# ============================================================
# 统一入口: 单站 → 测线 → 全线
# ============================================================
def invert_one_station_algo(t_obs, rhoa_obs, err_obs, z_centers, cfg,
                            algo: str = 'trf',
                            occam_cfg: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    n_l = z_centers.size
    th0 = np.full(n_l, np.log10(cfg['RHO_INIT']))
    good = np.isfinite(rhoa_obs) & (rhoa_obs > 0) & np.isfinite(err_obs) & (err_obs > 0)
    if good.sum() < 3:
        return th0, {'success': False, 'reason': f'only {good.sum()} valid obs',
                     'rmse_log': np.nan, 'nfev': 0}
    t_g, r_g, e_g = t_obs[good], rhoa_obs[good], err_obs[good]
    w_g = _weights(r_g, e_g, cfg['LOG_EPS'])

    if algo == 'trf':
        th, info = _invert_trf(t_g, r_g, w_g, z_centers, cfg)
    elif algo == 'lm':
        th, info = _invert_lm(t_g, r_g, w_g, z_centers, cfg)
    elif algo == 'occam':
        oc = dict(DEFAULT_OCCAM_CFG)
        if occam_cfg: oc.update(occam_cfg)
        th, info = _invert_occam(t_g, r_g, w_g, z_centers, cfg, oc)
    else:
        raise ValueError(f'未知算法 {algo!r}, 可选 {ALGO_CHOICES}')

    pred = forward_rhoa(th, t_g, z_centers, cfg['RHO_REF_DIFF'])
    info['rmse_log'] = float(np.sqrt(np.mean((np.log10(np.maximum(pred, 1e-3))
                                              - np.log10(r_g)) ** 2)))
    rho_lin = 10.0 ** th
    info.update(rho_min=float(rho_lin.min()), rho_med=float(np.median(rho_lin)),
                rho_max=float(rho_lin.max()))
    return th, info


def _cfg_hash(cfg: dict, algo: str) -> str:
    s = json.dumps({k: str(v) for k, v in sorted(cfg.items())}, sort_keys=True) + '|' + algo
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _cache_path(out_base: Path, st: str, algo: str, cfg_hash: str) -> Path:
    return out_base / '_inv_cache' / f'{st}_{algo}_{cfg_hash}.npz'


def _algo_station_worker(args: dict):
    """模块级 worker (Windows spawn 可 pickle)。"""
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
    th, info = invert_one_station_algo(
        args['t_obs'], args['rhoa_obs'], args['err_obs'],
        args['z_c'], args['cfg'], args['algo'], args.get('occam_cfg'))
    if cp:
        try:
            np.savez_compressed(cp, th=th, info=info)
        except Exception:
            pass
    return idx, st, th, info, False


def invert_line_algo(line_tag: str, out_root: str | Path, algo: str = 'trf',
                     user_cfg: Optional[dict] = None,
                     occam_cfg: Optional[dict] = None,
                     n_jobs: int = 1, use_cache: bool = False,
                     log_fn: Optional[Callable[[str], None]] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_INV_CFG)
    if user_cfg: cfg.update(user_cfg)
    out_base = Path(out_root) / line_tag
    dat = out_base / f'{line_tag}_occam.dat'
    if not dat.is_file():
        log_fn(f'  [SKIP] {line_tag}: 无 {dat.name} (先跑预处理)')
        return {}
    log_fn(f'\n=== 1D 反演[{algo}] {line_tag}  N_LAYERS={cfg["N_LAYERS"]} λ_smooth={cfg["LAMBDA_SMOOTH"]} ===')
    df = parse_occam_dat(dat)
    stations = sorted(df.Station.unique())
    z_c = depth_layer_centers(cfg['N_LAYERS'], cfg['Z_MIN_M'], cfg['Z_MAX_M'])
    cfg_hash = _cfg_hash(cfg, algo)
    if use_cache:
        (out_base / '_inv_cache').mkdir(parents=True, exist_ok=True)
    R_log = np.zeros((len(stations), cfg['N_LAYERS']))
    summary = []
    cache_hits = 0

    args_list = []
    for idx, st in enumerate(stations):
        sub = df[df.Station == st].sort_values('Time_s')
        cp = _cache_path(out_base, st, algo, cfg_hash) if use_cache else None
        args_list.append({
            'idx': idx, 'st': st,
            't_obs': sub.Time_s.values, 'rhoa_obs': sub.AppRes.values,
            'err_obs': sub.StdDev.values,
            'z_c': z_c, 'cfg': cfg, 'algo': algo, 'occam_cfg': occam_cfg,
            'cache_path': str(cp) if cp else None,
        })

    def _collect(idx, st, th, info, hit):
        nonlocal cache_hits
        R_log[idx] = th
        info['Station'] = st
        summary.append(info)
        if hit: cache_hits += 1

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as exe:
            futures = [exe.submit(_algo_station_worker, a) for a in args_list]
            for fut in as_completed(futures):
                _collect(*fut.result())
    else:
        for a in args_list:
            _collect(*_algo_station_worker(a))
    if cache_hits:
        log_fn(f'  [CACHE] {line_tag}[{algo}] 命中 {cache_hits}/{len(stations)} 站')
    R_lin = 10.0 ** R_log
    np.save(out_base / f'{line_tag}_inv_{algo}_R.npy', R_lin)
    np.save(out_base / f'{line_tag}_inv_{algo}_depth.npy', z_c)
    pd.DataFrame(summary).to_csv(out_base / f'{line_tag}_inv_{algo}_summary.csv',
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
    log_fn(f'  [SUMMARY] {line_tag}[{algo}]: {len(stations)}站 RMSE_log10_med={rmse_med:.3f}')
    return {'stations': len(stations), 'rmse_median_log': rmse_med, 'algo': algo}


def _algo_line_worker(line_tag, out_root, algo, user_cfg, occam_cfg, use_cache):
    """子进程入口: 单线单算法 1D 反演 (站内并行关闭, 防 CPU 超订)。"""
    logs = []
    try:
        st = invert_line_algo(line_tag, out_root, algo, user_cfg, occam_cfg,
                              n_jobs=1, use_cache=use_cache, log_fn=logs.append)
        return line_tag, st, logs
    except Exception as e:
        import traceback
        logs.append(f'  ✗ {line_tag}: 异常: {e}\n{traceback.format_exc()}')
        return line_tag, {'error': str(e)}, logs


def run_all_algo(out_root: str | Path, algo: str = 'trf',
                 user_cfg: Optional[dict] = None, occam_cfg: Optional[dict] = None,
                 n_jobs: int = 1, use_cache: bool = False,
                 n_line_jobs: int = 1,
                 log_fn: Optional[Callable[[str], None]] = None) -> dict:
    from .parallel_utils import map_lines
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    lines = sorted([p.name for p in out_root.iterdir() if p.is_dir()
                    and re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== 多算法 1D 反演  algo={algo}  测线={lines} '
          f'站内n_jobs={n_jobs} 线级n_line_jobs={n_line_jobs} ==========')
    if n_line_jobs in (None, 0):
        from .parallel_utils import auto_line_jobs
        n_line_jobs = auto_line_jobs(len(lines))
    if n_line_jobs > 1 and len(lines) > 1:
        return map_lines(lines, _algo_line_worker,
                         worker_args=(str(out_root), algo, user_cfg,
                                      occam_cfg, use_cache),
                         n_line_jobs=n_line_jobs, log_fn=log_fn,
                         label=f'1D反演[{algo}]')
    return {lt: invert_line_algo(lt, out_root, algo, user_cfg, occam_cfg,
                                 n_jobs, use_cache, log_fn)
            for lt in lines}


def run_all_algos_compare(out_root: str | Path, algos=('trf', 'lm', 'occam'),
                          user_cfg: Optional[dict] = None,
                          occam_cfg: Optional[dict] = None,
                          n_jobs: int = 1, use_cache: bool = False,
                          n_line_jobs: int = 1,
                          log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """跑全部算法 + 每线生成对比断面图 + 全线算法对比 CSV."""
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    results = {a: run_all_algo(out_root, a, user_cfg, occam_cfg,
                               n_jobs, use_cache, n_line_jobs, log_fn)
               for a in algos}
    comp_rows = []
    for lt in sorted({k for r in results.values() for k in r.keys()}):
        Rs, rmses = [], {}
        for a in algos:
            r = results[a].get(lt) or {}
            if not r: continue
            rmses[a] = r.get('rmse_median_log', np.nan)
            comp_rows.append({'line': lt, 'algo': a,
                              'stations': r.get('stations'),
                              'rmse_median_log': r.get('rmse_median_log')})
            p = out_root / lt / f'{lt}_inv_{a}_R.npy'
            if p.is_file():
                Rs.append((a, np.load(p)))
        if len(Rs) >= 2:
            z_c = np.load(out_root / lt / f'{lt}_inv_{Rs[0][0]}_depth.npy')
            fig, axes = plt.subplots(1, len(Rs), figsize=(6 * len(Rs), 6), sharey=True)
            if len(Rs) == 1: axes = [axes]
            for ax, (a, R) in zip(axes, Rs):
                X, Y = np.meshgrid(np.arange(R.shape[0]), np.log10(z_c), indexing='ij')
                im = ax.pcolormesh(X, Y, np.log10(R), shading='auto', cmap='jet_r')
                ax.invert_yaxis(); ax.set_title(f'{lt} [{a}] RMSE={rmses.get(a, np.nan):.3f}')
                fig.colorbar(im, ax=ax).set_label('log10(ρ)')
            fig.tight_layout()
            png = out_root / lt / f'{lt}_inv_compare.png'
            fig.savefig(png, dpi=170); plt.close(fig)
            log_fn(f'  [OUT] 算法对比断面 → {png}')
    if comp_rows:
        csv_p = out_root / 'inv_algos_compare.csv'
        pd.DataFrame(comp_rows).to_csv(csv_p, index=False, encoding='utf-8-sig')
        log_fn(f'  [OUT] 全线算法对比 → {csv_p}')
    return results
