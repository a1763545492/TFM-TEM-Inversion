# -*- coding: utf-8 -*-
"""
tfm_engine.lci_inversion (M12)
==============================
LCI — Laterally Constrained Inversion (Auken & Christiansen 2004)
对整条测线做**联合反演**，在常规 1D 纵向平滑基础上增加相邻测站间横向约束：

  Φ(θ) = Σᵢ ||Wᵢ (dᵢ − F(θᵢ))||²          ← 各站数据拟合
        + λₛ Σᵢ ||D θᵢ||²                  ← 每站纵向光滑
        + λₗₐₜ Σᵢ~ⱼ wᵢⱼ ||R_depth (θᵢ − θⱼ)||²  ← 相邻站横向约束

实现: scipy.optimize.least_squares (trf + lsmr 稀疏迭代求解),
      提供 jac_sparsity 显式稀疏结构，数值微分按深度列着色，
      将 1475 参数的微分代价压至 ~n_layers 次残差调用。

热启动: 先用独立 TRF 逐站反演结果拼接 θ_all0，显著加速收敛。
输出: 拟 2D 断面图 + 与独立 1D 拼接断面对比图。

完全离线，纯 numpy + scipy。
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix, block_diag, vstack as sp_vstack

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from .inversion import (parse_occam_dat, depth_layer_centers, forward_rhoa,
                        DEFAULT_INV_CFG, analytic_jacobian)
from .inv_algorithms import _weights, _data_resid, _numeric_jacobian

DEFAULT_LCI_CFG = dict(
    N_LAYERS=25, Z_MIN_M=5.0, Z_MAX_M=1500.0,
    RHO_INIT=100.0, RHO_BOUNDS=(0.1, 1e8),
    LAMBDA_SMOOTH=0.3,
    LAMBDA_LAT=1.0,          # ★ 首选调参 (0.1→1.0→10), 越大横向越匀
    LAT_DEPTH_WEIGHT='uniform',  # uniform / increase / decrease
    MAX_NFEV=400,
    ERROR_RHOA_FRAC=0.05,
    RHO_REF_DIFF=1000.0,
    LOG_EPS=1e-10,
    WARM_START=True,         # 独立 TRF 热启动
    JAC_FD_STEP=1e-4,
)


def _build_lat_weight_matrix(n_l: int, mode: str) -> np.ndarray:
    """R_depth: (n_l, n_l) 对角阵，控制横向约束随深度的强弱。"""
    R = np.eye(n_l)
    if mode == 'increase':
        diag = np.linspace(1.0, 3.0, n_l)  # 深部约束更强
    elif mode == 'decrease':
        diag = np.linspace(3.0, 1.0, n_l)  # 浅部约束更强
    elif mode == 'uniform':
        diag = np.ones(n_l)
    else:
        diag = np.ones(n_l)
    return np.diag(diag)


def _build_jac_sparsity(t_list, n_st: int, n_l: int) -> lil_matrix:
    """构建全局雅可比的稀疏结构 (M_total, N_total)。"""
    n_d_total = sum(len(t) for t in t_list)
    M_smooth = n_st * (n_l - 1)
    M_lat = (n_st - 1) * n_l
    M_total = n_d_total + M_smooth + M_lat
    N_total = n_st * n_l

    sp = lil_matrix((M_total, N_total), dtype=int)
    row = 0
    # 数据残差: 块对角
    for i in range(n_st):
        nd = len(t_list[i])
        c0 = i * n_l
        sp[row:row + nd, c0:c0 + n_l] = 1
        row += nd
    # 纵向平滑: 块对角 (diff 矩阵)
    for i in range(n_st):
        c0 = i * n_l
        sp[row:row + n_l - 1, c0:c0 + n_l] = 1
        row += n_l - 1
    # 横向约束: 相邻站重叠块
    for i in range(n_st - 1):
        c_i = i * n_l
        c_j = (i + 1) * n_l
        sp[row:row + n_l, c_i:c_i + n_l] = 1
        sp[row:row + n_l, c_j:c_j + n_l] = 1
        row += n_l
    return sp


def _build_global_jac_analytic(t_list, r_list, e_list, n_l, z_c, rho_ref,
                               lam_s, lam_l, w_lat, R_depth, cfg):
    """LCI 全局残差的**解析稀疏雅可比** (M_total, n_st*n_l)。

    数据残差块: 每站块对角 = 插值权重矩阵 W_i·w_i (模型对 θ 线性, 精确)
    纵向光滑块: sqrt(λs)·D1 块对角
    横向约束块: sqrt(λl·w_i)·R_depth 与 −同, 相邻站两列块

    与 θ 无关 → 整个迭代只需构造 1 次, 彻底消除 least_squares 的数值微分
    (旧方案即使有着色稀疏结构仍需 ~n_l 次残差调用/迭代)。
    """
    n_st = len(t_list)
    # --- 数据块 (CSR 块对角) ---
    data_blocks = []
    for i in range(n_st):
        w_g = _weights(r_list[i], e_list[i], cfg['LOG_EPS'])
        Jd_i = analytic_jacobian(None, t_list[i], z_c, rho_ref, w_g)
        data_blocks.append(csr_matrix(Jd_i))
    J_data = block_diag(data_blocks, format='csr')

    # --- 纵向光滑块 ---
    n_d = J_data.shape[1]
    sm_rows = []
    D1 = np.zeros((n_l - 1, n_l))
    D1[np.arange(n_l - 1), np.arange(n_l - 1)] = -np.sqrt(lam_s)
    D1[np.arange(n_l - 1), np.arange(1, n_l)] = np.sqrt(lam_s)
    for i in range(n_st):
        Ji = lil_matrix((n_l - 1, n_d))
        Ji[:, i * n_l:(i + 1) * n_l] = D1
        sm_rows.append(csr_matrix(Ji))
    J_smooth = sp_vstack(sm_rows, format='csr')

    # --- 横向约束块 ---
    lat_rows = []
    for i in range(n_st - 1):
        Ji = lil_matrix((n_l, n_d))
        B = np.sqrt(lam_l * w_lat[i]) * R_depth
        Ji[:, i * n_l:(i + 1) * n_l] = B
        Ji[:, (i + 1) * n_l:(i + 2) * n_l] = -B
        lat_rows.append(csr_matrix(Ji))
    J_lat = sp_vstack(lat_rows, format='csr') if lat_rows else csr_matrix((0, n_d))

    return sp_vstack([J_data, J_smooth, J_lat], format='csr')


def invert_line_lci(line_tag: str, out_root: str | Path,
                    user_cfg: Optional[dict] = None,
                    n_jobs: int = 1,
                    log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """LCI 整条测线联合反演主入口。"""
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_LCI_CFG)
    if user_cfg:
        cfg.update(user_cfg)
    out_base = Path(out_root) / line_tag
    dat = out_base / f'{line_tag}_occam.dat'
    if not dat.is_file():
        log_fn(f'  [SKIP] {line_tag}: 无 {dat.name} (先跑预处理)')
        return {}

    log_fn(f'\n=== LCI横向约束反演 {line_tag}  N_LAYERS={cfg["N_LAYERS"]} '
           f'λ_smooth={cfg["LAMBDA_SMOOTH"]} λ_lat={cfg["LAMBDA_LAT"]} ===')

    df = parse_occam_dat(dat)
    stations = sorted(df.Station.unique())
    n_st = len(stations)
    n_l = cfg['N_LAYERS']
    z_c = depth_layer_centers(n_l, cfg['Z_MIN_M'], cfg['Z_MAX_M'])
    lo, hi = np.log10(cfg['RHO_BOUNDS'][0]), np.log10(cfg['RHO_BOUNDS'][1])

    # 提取每站数据 + 坐标
    t_list, r_list, e_list, x_list = [], [], [], []
    for st in stations:
        sub = df[df.Station == st].sort_values('Time_s')
        t_list.append(sub.Time_s.values)
        r_list.append(sub.AppRes.values)
        e_list.append(sub.StdDev.values)
        # X 坐标取第一个有效值，否则用站序号
        xs = sub['X'].values if 'X' in sub.columns else sub['Northing'].values if 'Northing' in sub.columns else None
        if xs is not None and np.isfinite(xs).any():
            x_list.append(float(xs[np.isfinite(xs)][0]))
        else:
            x_list.append(float(len(x_list)))

    # 站间距归一化权重 (间距越大约束越弱)
    x_arr = np.array(x_list)
    if n_st >= 2:
        dist = np.diff(x_arr)
        med_dist = max(np.median(np.abs(dist)), 1e-3)
        w_lat = med_dist / (np.abs(dist) + 0.1 * med_dist)  # (n_st-1,)
    else:
        w_lat = np.ones(max(0, n_st - 1))

    # 深度加权矩阵
    R_depth = _build_lat_weight_matrix(n_l, cfg['LAT_DEPTH_WEIGHT'])

    # ---------- 热启动 ----------
    theta0 = np.full(n_st * n_l, np.log10(cfg['RHO_INIT']))
    warm_info = []
    if cfg['WARM_START']:
        log_fn(f'  [WARM] 热启动: 独立 TRF 逐站反演 {n_st} 站 ...')
        from .inv_algorithms import invert_one_station_algo
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            args_list = []
            for i, st in enumerate(stations):
                good = np.isfinite(r_list[i]) & (r_list[i] > 0) & np.isfinite(e_list[i]) & (e_list[i] > 0)
                if good.sum() < 3:
                    warm_info.append({'Station': st, 'success': False, 'reason': 'few_data'})
                    continue
                c = dict(cfg)
                c['LAMBDA_SMOOTH'] = float(c['LAMBDA_SMOOTH'])  # 防止进程传参问题
                args_list.append({
                    't_obs': t_list[i][good], 'rhoa_obs': r_list[i][good],
                    'err_obs': e_list[i][good], 'z_centers': z_c,
                    'cfg': c, 'algo': 'trf', 'occam_cfg': None,
                })
            with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                futures = {exe.submit(invert_one_station_algo, **a): i
                           for i, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    ii = futures[fut]
                    th, info = fut.result()
                    theta0[ii * n_l:(ii + 1) * n_l] = th
                    info['th'] = th  # 保存 theta 供对比图
                    info['Station'] = stations[ii]
                    warm_info.append(info)
        else:
            for i, st in enumerate(stations):
                th, info = invert_one_station_algo(
                    t_list[i], r_list[i], e_list[i], z_c, cfg, 'trf')
                theta0[i * n_l:(i + 1) * n_l] = th
                info['th'] = th  # 保存 theta 供对比图
                info['Station'] = st
                warm_info.append(info)
        succ = sum(1 for w in warm_info if w.get('success', False))
        log_fn(f'  [WARM] 热启动完成: {succ}/{n_st} 站成功')

    # ---------- 构建全局残差 ----------
    lam_s = float(cfg['LAMBDA_SMOOTH'])
    lam_l = float(cfg['LAMBDA_LAT'])
    fd = float(cfg['JAC_FD_STEP'])
    rho_ref = cfg['RHO_REF_DIFF']

    def global_resid(theta_all: np.ndarray) -> np.ndarray:
        theta_all = np.clip(theta_all, lo, hi)
        # 数据残差
        r_data = []
        for i in range(n_st):
            th_i = theta_all[i * n_l:(i + 1) * n_l]
            w_g = _weights(r_list[i], e_list[i], cfg['LOG_EPS'])
            r_data.append(_data_resid(th_i, t_list[i], r_list[i], w_g, z_c, rho_ref))
        # 纵向平滑
        r_smooth = []
        for i in range(n_st):
            th_i = theta_all[i * n_l:(i + 1) * n_l]
            r_smooth.append(np.sqrt(lam_s) * np.diff(th_i))
        # 横向约束
        r_lat = []
        for i in range(n_st - 1):
            th_i = theta_all[i * n_l:(i + 1) * n_l]
            th_j = theta_all[(i + 1) * n_l:(i + 2) * n_l]
            diff = R_depth @ (th_i - th_j)
            r_lat.append(np.sqrt(lam_l * w_lat[i]) * diff)
        return np.concatenate(r_data + r_smooth + r_lat)

    # ---------- 稀疏雅可比结构 ----------
    sp = _build_jac_sparsity(t_list, n_st, n_l)
    sp_csr = sp.tocsr()

    # ---------- 解析全局雅可比 (常量, 一次构造; numeric 模式回退到稀疏着色差分) ----------
    jac_mode = cfg.get('JAC_MODE', 'analytic')
    if jac_mode == 'numeric':
        jac_arg = '2-point'
        log_fn('  [LCI] JAC_MODE=numeric (稀疏结构着色差分, 仅用于校验)')
    else:
        J_global = _build_global_jac_analytic(
            t_list, r_list, e_list, n_l, z_c, rho_ref,
            lam_s, lam_l, w_lat, R_depth, cfg)
        # 形状自检, 与残差/稀疏结构严格一致
        assert J_global.shape == sp_csr.shape, (J_global.shape, sp_csr.shape)
        jac_arg = lambda thv: J_global
        log_fn(f'  [LCI] 解析稀疏雅可比: shape={J_global.shape} '
               f'nnz={J_global.nnz} (全迭代复用, 零数值微分)')

    # ---------- 联合反演 ----------
    nfev_data = {'nfev': 0}

    def resid_wrapper(thv):
        nfev_data['nfev'] += 1
        return global_resid(thv)

    log_fn(f'  [LCI] 联合反演启动: {n_st}站×{n_l}层={n_st*n_l}参数 '
           f'残差维度={sp.shape[0]}')
    bounds_lo = np.full(n_st * n_l, lo)
    bounds_hi = np.full(n_st * n_l, hi)

    ls_kwargs = dict(bounds=(bounds_lo, bounds_hi),
                     method='trf', max_nfev=cfg['MAX_NFEV'],
                     x_scale='jac', tr_solver='lsmr', verbose=0)
    if jac_mode == 'numeric':
        ls_kwargs['jac_sparsity'] = sp_csr
    else:
        ls_kwargs['jac'] = jac_arg
    res = least_squares(resid_wrapper, theta0, **ls_kwargs)

    theta_opt = np.clip(res.x, lo, hi).reshape(n_st, n_l)
    R_lin = 10.0 ** theta_opt
    log_fn(f'  [LCI] 收敛: nfev={res.nfev} cost={res.cost:.4f} '
           f'success={res.success}')

    # ---------- 逐站评估 ----------
    summary = []
    for i, st in enumerate(stations):
        pred = forward_rhoa(theta_opt[i], t_list[i], z_c, rho_ref)
        rmse_log = float(np.sqrt(np.mean(
            (np.log10(np.maximum(pred, 1e-3)) - np.log10(r_list[i])) ** 2)))
        summary.append({
            'Station': st, 'success': True,
            'rmse_log': rmse_log,
            'rho_min': float(R_lin[i].min()),
            'rho_med': float(np.median(R_lin[i])),
            'rho_max': float(R_lin[i].max()),
        })
    rmses = [s['rmse_log'] for s in summary if np.isfinite(s['rmse_log'])]
    rmse_med = float(np.median(rmses)) if rmses else np.nan

    # ---------- 保存 ----------
    np.save(out_base / f'{line_tag}_inv_lci_R.npy', R_lin)
    np.save(out_base / f'{line_tag}_inv_lci_depth.npy', z_c)
    pd.DataFrame(summary).to_csv(out_base / f'{line_tag}_inv_lci_summary.csv',
                                 index=False, encoding='utf-8-sig')

    # ---------- 拟 2D 断面图 ----------
    if n_st >= 2:
        # 独立 1D 断面 (热启动结果)
        R_warm = np.zeros((n_st, n_l))
        for i, info in enumerate(warm_info):
            if info.get('success') and 'th' in info:
                R_warm[i] = 10.0 ** info['th']
            else:
                R_warm[i] = cfg['RHO_INIT']

        fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                                 gridspec_kw={'width_ratios': [1, 1, 0.05]})
        ax1, ax2, cax = axes
        Y = np.log10(z_c)
        X = np.arange(n_st)
        Xmesh, Ymesh = np.meshgrid(X, Y, indexing='ij')

        # 色标用数据 2~98 分位数 (物理上下界跨9个数量级会导致对比度不足)
        all_vals = np.concatenate([np.log10(np.clip(R_lin, 1e-3, None)).ravel(),
                                   np.log10(np.clip(R_warm, 1e-3, None)).ravel()])
        vmin, vmax = np.percentile(all_vals[np.isfinite(all_vals)], [2, 98])
        # LCI
        Z_lci = np.log10(np.clip(R_lin, cfg['RHO_BOUNDS'][0], cfg['RHO_BOUNDS'][1]))
        im1 = ax1.pcolormesh(Xmesh, Ymesh, Z_lci, shading='auto',
                             cmap='jet_r', vmin=vmin, vmax=vmax)
        # 独立 1D
        Z_ind = np.log10(np.clip(R_warm, cfg['RHO_BOUNDS'][0], cfg['RHO_BOUNDS'][1]))
        im2 = ax2.pcolormesh(Xmesh, Ymesh, Z_ind, shading='auto',
                             cmap='jet_r', vmin=vmin, vmax=vmax)

        for ax, title in [(ax1, f'LCI (λ_lat={lam_l})'),
                          (ax2, '独立1D (TRF)')]:
            ax.invert_yaxis()
            ax.set_xlabel('测点序号')
            ax.set_ylabel('深度 (m, log)')
            yt = np.linspace(np.log10(cfg['Z_MIN_M']), np.log10(cfg['Z_MAX_M']), 6)
            ax.set_yticks(yt); ax.set_yticklabels([f'{10**v:.0f}' for v in yt])
            step = max(1, n_st // 10)
            ax.set_xticks(np.arange(0, n_st, step))
            ax.set_xticklabels([stations[i] for i in range(0, n_st, step)],
                               rotation=45, ha='right')
            ax.set_title(title)

        fig.colorbar(im1, cax=cax).set_label('log10(ρ/Ω·m)')
        fig.suptitle(f'{line_tag} 拟2D断面对比  RMSE_log10_med={rmse_med:.3f}', y=1.02)
        fig.tight_layout()
        png = out_base / f'{line_tag}_inv_lci_section.png'
        fig.savefig(png, dpi=180); plt.close(fig)
        log_fn(f'  [OUT] LCI 断面 → {png}')

    log_fn(f'  [SUMMARY] {line_tag}[lci]: {n_st}站 RMSE_log10_med={rmse_med:.3f}')
    return {
        'stations': n_st, 'rmse_median_log': rmse_med,
        'algo': 'lci', 'nfev': int(res.nfev), 'cost': float(res.cost),
        'success': bool(res.success),
    }


def _lci_line_worker(line_tag, out_root, user_cfg):
    """子进程入口: 单线 LCI (热启动站内并行关闭, 防 CPU 超订)。"""
    logs = []
    try:
        st = invert_line_lci(line_tag, out_root, user_cfg, n_jobs=1,
                             log_fn=logs.append)
        return line_tag, st, logs
    except Exception as e:
        import traceback
        logs.append(f'  ✗ {line_tag}: 异常: {e}\n{traceback.format_exc()}')
        return line_tag, {'error': str(e)}, logs


def run_all_lci(out_root: str | Path,
                user_cfg: Optional[dict] = None,
                n_jobs: int = 1,
                n_line_jobs: int = 1,
                log_fn: Optional[Callable[[str], None]] = None) -> dict:
    from .parallel_utils import map_lines, auto_line_jobs
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    lines = sorted([p.name for p in out_root.iterdir() if p.is_dir()
                    and re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== LCI 横向约束反演  测线={lines} '
          f'热启动n_jobs={n_jobs} 线级n_line_jobs={n_line_jobs} ==========')
    if n_line_jobs in (None, 0):
        n_line_jobs = auto_line_jobs(len(lines))
    if n_line_jobs > 1 and len(lines) > 1:
        return map_lines(lines, _lci_line_worker,
                         worker_args=(str(out_root), user_cfg),
                         n_line_jobs=n_line_jobs, log_fn=log_fn,
                         label='LCI联合反演')
    return {lt: invert_line_lci(lt, out_root, user_cfg, n_jobs, log_fn)
            for lt in lines}
