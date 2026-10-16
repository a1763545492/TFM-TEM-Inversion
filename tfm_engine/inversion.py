# -*- coding: utf-8 -*-
"""
inversion.py — 1D 层状 least_squares 反演 (经验 #1477534 规范)
===============================================
读取 `{out_dir}/{测线}/X线_occam.dat` → R[station, N_LAYERS] 张量 + 断面 PNG.

核心残差设计 (稳定):
  r_data[i] = (log10(ρ_pred(t_i)) - log10(ρ_obs(t_i))) / max(log10(err_i/ρ_obs_i), 0.05)   Ng 条
  r_smooth[j] = sqrt(LAMBDA_SMOOTH) * (θ_{j+1} - θ_j)                          (N-1) 条
  拼接 resid = [r_data, r_smooth]  → scipy.optimize.least_squares(TRF)

参数:
  θ = log10(rho_layers)  所以 bounds = [log10(0.1), log10(1e8)] = [-1, 8]
  N_LAYERS = 25 (行业默认, 可改)
  LAMBDA_SMOOTH = 0.3 (默认, 单变量首选调参)

正演: Boerner-style 扩散深度快速近似 (骨架版).  商业 Occam1D 用 Wait/Kaufman 严格正演.
"""
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import least_squares

rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','Arial']
rcParams['axes.unicode_minus'] = False

MU0 = 4.0 * np.pi * 1e-7
DEFAULT_INV_CFG = {
    'N_LAYERS': 25,
    'Z_MIN_M': 5.0,
    'Z_MAX_M': 1500.0,
    'RHO_BOUNDS': (0.1, 1e8),
    'RHO_INIT': 100.0,
    'RHO_REF_DIFF': 100.0,
    'LAMBDA_SMOOTH': 0.3,
    'LOG_EPS': 1e-3,
    'MAX_NFEV': 200,
    'METHOD': 'trf',
}


def parse_occam_dat(path: Path) -> pd.DataFrame:
    """Occam.dat → DataFrame (9 列, 双引号字符串字段)"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if not ln or ln.startswith('#'): continue
            toks = re.findall(r'"[^"]*"|[^\s]+', ln)
            if len(toks) < 9: continue
            try:
                rows.append({
                    'X': float(toks[0]), 'Y': float(toks[1]), 'Z': float(toks[2]),
                    'Time_s': float(toks[3]), 'AppRes': float(toks[4]), 'StdDev': float(toks[5]),
                    'Station': toks[6].strip('"'), 'Section': toks[7].strip('"'),
                    'Instruments': toks[8].strip('"'),
                })
            except ValueError: continue
    return pd.DataFrame(rows)


def depth_layer_centers(n_layers: int, z_min: float, z_max: float) -> np.ndarray:
    """对数等比层中心"""
    edges = np.geomspace(z_min, z_max, n_layers + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def forward_rhoa(theta_log10: np.ndarray, t_obs: np.ndarray,
                 z_centers: np.ndarray, rho_ref: float) -> np.ndarray:
    """扩散深度近似 → rhoa 预测. 与 preprocess 公式一致 (Boerner-style)."""
    rho_pred_log_layers = theta_log10
    log_z = np.log10(np.maximum(z_centers, 1e-3))
    # δ(t) = sqrt(2*ρ_ref*t/μ0)    经验缩放: z(t) = δ / 2
    z_of_t = np.sqrt(2.0 * rho_ref * t_obs / MU0) * 0.5
    log_zt = np.log10(np.maximum(z_of_t, 1e-3))
    log_rhoa = np.interp(log_zt, log_z, rho_pred_log_layers)
    return 10.0 ** log_rhoa


def interp_weight_matrix(t_obs: np.ndarray, z_centers: np.ndarray,
                         rho_ref: float) -> np.ndarray:
    """解析雅可比核: 因 log10(ρa_pred) = W @ θ (np.interp 对 θ 分段线性),
    返回插值权重矩阵 W, shape=(Ng, N_layers), 每行至多 2 个非零元。

    性质:
      - 构造只需 1 次 z(t) 计算, **与层数无关**, 无需任何额外正演调用
        (数值差分需 N_layers+1 次正演, 25 层时慢约 26 倍)
      - 区间外 (扩散深度浅于/深于层网格) np.interp 钳到端点 → 权重全给端点层
      - 与 forward_rhoa 严格一致 (机器精度)
    """
    log_z = np.log10(np.maximum(z_centers, 1e-3))
    z_of_t = np.sqrt(2.0 * rho_ref * np.maximum(t_obs, 1e-12) / MU0) * 0.5
    log_zt = np.log10(np.maximum(z_of_t, 1e-3))
    n_g = t_obs.size
    n_l = z_centers.size
    W = np.zeros((n_g, n_l), dtype=np.float64)
    below = log_zt <= log_z[0]
    above = log_zt >= log_z[-1]
    W[below, 0] = 1.0
    W[above, -1] = 1.0
    inner = ~(below | above)
    if inner.any():
        idx = np.searchsorted(log_z, log_zt[inner]) - 1
        idx = np.clip(idx, 0, n_l - 2)
        alpha = (log_zt[inner] - log_z[idx]) / (log_z[idx + 1] - log_z[idx])
        rows = np.where(inner)[0]
        W[rows, idx] = 1.0 - alpha
        W[rows, idx + 1] = alpha
    return W


def analytic_jacobian(theta_log10: np.ndarray, t_obs: np.ndarray,
                      z_centers: np.ndarray, rho_ref: float,
                      weights: np.ndarray) -> np.ndarray:
    """加权对数残差 r = (log10 ρa_pred − log10 ρa_obs)·w 对 θ 的解析雅可比。

    与数值前向差分 (_numeric_jacobian) 在模型定义域内完全等价
    (模型对 θ 为线性, 截断误差=0), 成本由 N_layers+1 次正演降为 1 次。
    `theta_log10` 仅为保持签名一致而保留, 不参与计算。
    """
    W = interp_weight_matrix(t_obs, z_centers, rho_ref)
    return W * np.asarray(weights, dtype=np.float64)[:, None]


def invert_one_station(t_obs: np.ndarray, rhoa_obs: np.ndarray, err_obs: np.ndarray,
                       z_centers: np.ndarray, cfg: dict,
                       log_path: Path) -> tuple[np.ndarray, dict]:
    n_l = len(z_centers)
    log_b_lo = np.log10(cfg['RHO_BOUNDS'][0])
    log_b_hi = np.log10(cfg['RHO_BOUNDS'][1])
    theta_init = np.full(n_l, np.log10(cfg['RHO_INIT']))

    good = np.isfinite(rhoa_obs) & (rhoa_obs > 0) & np.isfinite(err_obs) & (err_obs > 0)
    if good.sum() < 3:
        return theta_init, {'success': False, 'reason': f'only {good.sum()} valid obs',
                            'rmse_log': np.nan, 'nfev': 0, 'cost': np.nan,
                            'rho_min': np.nan, 'rho_med': np.nan, 'rho_max': np.nan}
    t_g, r_g, e_g = t_obs[good], rhoa_obs[good], err_obs[good]
    # 1/σ 对数权重 (下限 0.05 = 约 ±12%)
    log_err_frac = np.abs(np.log10(e_g / r_g + cfg['LOG_EPS']))
    inv_w = 1.0 / np.maximum(log_err_frac, 0.05)

    def resid(theta):
        pred = forward_rhoa(theta, t_g, z_centers, cfg['RHO_REF_DIFF'])
        pred = np.maximum(pred, cfg['LOG_EPS'])
        r_data = (np.log10(pred) - np.log10(np.maximum(r_g, cfg['LOG_EPS']))) * inv_w
        r_smooth = np.sqrt(cfg['LAMBDA_SMOOTH']) * np.diff(theta)
        return np.concatenate([r_data, r_smooth])

    def jac(theta):
        """解析雅可比: 数据块 = 插值权重矩阵 W·inv_w; 光滑块 = sqrt(λ)·D1。"""
        if cfg.get('JAC_MODE', 'analytic') == 'numeric':
            base = forward_rhoa(theta, t_g, z_centers, cfg['RHO_REF_DIFF'])
            lb = np.log10(np.maximum(base, cfg['LOG_EPS']))
            Jd = np.zeros((t_g.size, n_l))
            for j in range(n_l):
                tp = theta.copy(); tp[j] += 1e-4
                lp = np.log10(np.maximum(forward_rhoa(tp, t_g, z_centers, cfg['RHO_REF_DIFF']),
                                         cfg['LOG_EPS']))
                Jd[:, j] = (lp - lb) / 1e-4
            Jd *= inv_w[:, None]
        else:
            Jd = analytic_jacobian(theta, t_g, z_centers, cfg['RHO_REF_DIFF'], inv_w)
        Dm = np.zeros((n_l - 1, n_l))
        Dm[np.arange(n_l - 1), np.arange(n_l - 1)] = -np.sqrt(cfg['LAMBDA_SMOOTH'])
        Dm[np.arange(n_l - 1), np.arange(1, n_l)] = np.sqrt(cfg['LAMBDA_SMOOTH'])
        return np.vstack([Jd, Dm])

    res = least_squares(
        resid, theta_init, jac=jac,
        bounds=(np.full(n_l, log_b_lo), np.full(n_l, log_b_hi)),
        method=cfg['METHOD'], max_nfev=cfg['MAX_NFEV'], x_scale='jac',
    )
    pred_final = forward_rhoa(res.x, t_g, z_centers, cfg['RHO_REF_DIFF'])
    rmse_log = float(np.sqrt(np.mean((np.log10(pred_final) - np.log10(r_g)) ** 2)))
    rho_lin = 10.0 ** res.x
    info = {'success': bool(res.success), 'rmse_log': rmse_log,
            'nfev': int(res.nfev), 'cost': float(res.cost),
            'rho_min': float(rho_lin.min()), 'rho_med': float(np.median(rho_lin)),
            'rho_max': float(rho_lin.max())}
    return res.x, info


def invert_line(line_tag: str, out_root: Path,
                user_cfg: Optional[dict] = None,
                log_fn: Optional[Callable[[str], None]] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_INV_CFG)
    if user_cfg: cfg.update(user_cfg)
    out_base = out_root / line_tag
    dat = out_base / f'{line_tag}_occam.dat'
    if not dat.is_file():
        log_fn(f'  [SKIP] {line_tag}: 无 {dat.name}')
        return {}
    log_fn(f'\n=== 1D 反演 {line_tag} N_LAYERS={cfg["N_LAYERS"]} λ={cfg["LAMBDA_SMOOTH"]} ===')
    df = parse_occam_dat(dat)
    stations = sorted(df.Station.unique())
    z_c = depth_layer_centers(cfg['N_LAYERS'], cfg['Z_MIN_M'], cfg['Z_MAX_M'])
    R_log = np.zeros((len(stations), cfg['N_LAYERS']))
    summary = []
    log_path = out_base / f'{line_tag}_1d_inversion.log'
    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write(f'# 1D Inversion for {line_tag} cfg={json.dumps(cfg, ensure_ascii=False)}\n')
    for idx, st in enumerate(stations):
        sub = df[df.Station == st].sort_values('Time_s')
        theta, info = invert_one_station(sub.Time_s.values, sub.AppRes.values,
                                         sub.StdDev.values, z_c, cfg, log_path)
        R_log[idx, :] = theta
        info.update({'Station': st})
        summary.append(info)
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f'  {st}: RMSE_log10={info["rmse_log"]:.3f}  nfev={info["nfev"]}  '
                     f'ρ(min/med/max)={info.get("rho_min",0):.1f}/{info.get("rho_med",0):.1f}/{info.get("rho_max",0):.1f}'
                     f'  ok={info.get("success")}\n')
    R_lin = 10.0 ** R_log
    np.save(out_base / f'{line_tag}_1d_inversion_R.npy', R_lin)
    np.save(out_base / f'{line_tag}_1d_inversion_depth.npy', z_c)
    pd.DataFrame(summary).to_csv(out_base / f'{line_tag}_1d_inversion_summary.csv',
                                 index=False, encoding='utf-8-sig')

    # 断面图 (经验 #1477534 统一 R[pickets, 层] → pcolormesh)
    if len(stations) >= 2:
        X, Y = np.meshgrid(np.arange(len(stations)), np.log10(z_c), indexing='ij')
        Z = np.log10(np.clip(R_lin, cfg['RHO_BOUNDS'][0], cfg['RHO_BOUNDS'][1]))
        fig, ax = plt.subplots(figsize=(max(8, len(stations) * 0.4 + 2), 6))
        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet_r')
        ax.set_xlabel(f'测点 ({len(stations)} 个 station, 权威编码)')
        ax.set_ylabel('深度 Depth (m, log 轴)')
        yt_l = np.linspace(np.log10(cfg['Z_MIN_M']), np.log10(cfg['Z_MAX_M']), 6)
        ax.set_yticks(yt_l); ax.set_yticklabels([f'{10**v:.0f}m' for v in yt_l])
        step = max(1, len(stations)//10)
        ax.set_xticks(np.arange(0, len(stations), step))
        ax.set_xticklabels([stations[i] for i in range(0, len(stations), step)], rotation=45)
        ax.invert_yaxis()
        ax.set_title(f'{line_tag}  1D 反演真电阻率断面  (N_LAYERS={cfg["N_LAYERS"]}  λ_smooth={cfg["LAMBDA_SMOOTH"]})')
        cb = fig.colorbar(im, ax=ax); cb.set_label('log10(ρ / Ω·m)')
        fig.tight_layout()
        png = out_base / f'{line_tag}_1d_inversion_section.png'
        fig.savefig(png, dpi=180); plt.close(fig)
        log_fn(f'  [OUT] 断面图 → {png}')
    rmses = [s['rmse_log'] for s in summary if np.isfinite(s.get('rmse_log', np.nan))]
    rmse_med = float(np.median(rmses)) if rmses else np.nan
    log_fn(f'  [SUMMARY] {len(stations)}  stations  RMSE_log10_median={rmse_med:.3f}'
          f'  (0.3 以内良好, 0.3~1.0 骨架版合理)')
    return {'stations': len(stations), 'depth_max': float(z_c[-1]), 'rmse_median_log': rmse_med}


def _invert_line_worker(line_tag: str, out_root: str, user_cfg: Optional[dict]):
    """子进程入口: 单条测线 1D 反演, 日志随结果返回 (spawn 可 pickle)。"""
    logs: list[str] = []
    try:
        st = invert_line(line_tag, Path(out_root), user_cfg, logs.append)
        return line_tag, st, logs
    except Exception as e:
        import traceback
        logs.append(f'  ✗ {line_tag}: 异常: {e}\n{traceback.format_exc()}')
        return line_tag, {'error': str(e)}, logs


def run_all_inversions(out_root: str | Path, user_cfg: Optional[dict] = None,
                       log_fn: Optional[Callable[[str], None]] = None,
                       n_line_jobs: int = 1) -> dict:
    from .parallel_utils import map_lines
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    lines = sorted([p.name for p in out_root.iterdir() if p.is_dir() and
                    re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== 启动 1D 反演  目标测线={lines} '
          f'线级并行 n_line_jobs={n_line_jobs} ==========')
    return map_lines(lines, _invert_line_worker,
                     worker_args=(str(out_root), user_cfg),
                     n_line_jobs=n_line_jobs, log_fn=log_fn, label='1D反演')
