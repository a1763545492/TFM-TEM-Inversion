# -*- coding: utf-8 -*-
"""
export_formats.py — 反演断面标准化导出
================================================================
把 2D 反演电阻率断面 (R[station, layer] + 深度 + 测站坐标) 导出为
地学软件通用格式, 全部本地文件、完全离线:

  1. CSV  点数据 (tidy): station_code, X_m, Y_m, chainage_m, depth_m, rho_ohmm
  2. XYZ  Surfer/GMT 文本网格 (x=测线里程, y=深度负值向上, z=ρ 或 log10ρ)
  3. GRD  Surfer DSAA 二进制文本网格 (Surfer 8 及以前/Golden Software 全兼容)
  4. TIF  GeoTIFF (rasterio 为可选依赖, 缺失时明确提示并跳过)

坐标来源: {line}_pickup_xyz.csv (预处理 S8 输出, 含 station_code/X_m/Y_m);
缺失时回退为等间距 50 m。
深度轴: 断面按 Surfer 习惯 y = -depth (向上为正), 单位 m。
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .inversion import parse_occam_dat

# 结果文件候选: (R, depth)
def _result_paths(out_base: Path, line_tag: str, algo: str):
    if algo in (None, 'basic', '1d'):
        return (out_base / f'{line_tag}_1d_inversion_R.npy',
                out_base / f'{line_tag}_1d_inversion_depth.npy')
    return (out_base / f'{line_tag}_inv_{algo}_R.npy',
            out_base / f'{line_tag}_inv_{algo}_depth.npy')


def load_section(out_root: str | Path, line_tag: str,
                 algo: str = 'trf') -> dict:
    """加载断面网格 + 测站坐标/里程。

    返回:
        R          (n_st, n_l) 线性电阻率 Ω·m
        z_centers  (n_l,) 层中心深度 m
        chainage   (n_st,) 沿测线累计里程 m
        xy         (n_st, 2) 权威 X/Y 坐标
        stations   list[str] 与 R 行序一致 (反演序)
    """
    out_base = Path(out_root) / line_tag
    r_p, z_p = _result_paths(out_base, line_tag, algo)
    if not r_p.is_file():
        raise FileNotFoundError(f'反演结果不存在: {r_p}')
    R = np.load(r_p)
    z_centers = np.load(z_p) if z_p.is_file() else \
        np.geomspace(5.0, 1500.0, R.shape[1])

    # 测站序 (与 invert_line 一致: occam.dat Station 排序)
    dat_p = out_base / f'{line_tag}_occam.dat'
    if dat_p.is_file():
        stations = sorted(parse_occam_dat(dat_p).Station.unique())
    else:
        stations = [f'S{i+1:03d}' for i in range(R.shape[0])]
    n = min(len(stations), R.shape[0])
    stations = stations[:n]
    R = R[:n]

    # 权威坐标: pickup_xyz.csv
    xy = np.full((n, 2), np.nan)
    pickup = out_base / f'{line_tag}_pickup_xyz.csv'
    if pickup.is_file():
        dfp = pd.read_csv(pickup)
        cmap = dict(zip(dfp['station_code'].astype(str),
                        zip(pd.to_numeric(dfp['X_m'], errors='coerce'),
                            pd.to_numeric(dfp['Y_m'], errors='coerce'))))
        for i, st in enumerate(stations):
            if st in cmap:
                xx, yy = cmap[st]
                if np.isfinite(xx) and np.isfinite(yy):
                    xy[i] = (xx, yy)

    # 缺失坐标回退等间距 50 m
    bad = ~np.isfinite(xy[:, 0])
    if bad.any():
        # 优先用已有坐标确定方向, 缺的填序号*50
        if not bad.all():
            good_idx = np.where(~bad)[0]
            # 沿 X 方向为测线方向的常见情形
            xy[bad, 0] = np.arange(n)[bad] * 50.0
            xy[bad, 1] = np.nanmedian(xy[~bad, 1])
        else:
            xy[:, 0] = np.arange(n) * 50.0
            xy[:, 1] = 0.0

    # 累计里程 (相邻站 hypot)
    chainage = np.concatenate([[0.0], np.cumsum(
        np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))])
    return {'R': R, 'z_centers': np.asarray(z_centers, dtype=float),
            'chainage': chainage, 'xy': xy, 'stations': stations}


# ============================================================
# 1) CSV 点数据 (tidy)
# ============================================================
def export_csv(sec: dict, out_csv: str | Path,
               value_scale: str = 'linear') -> Path:
    """导出长表 CSV: 每站每层一行, Excel 可直接打开。"""
    R = sec['R']; z = sec['z_centers']
    n_st, n_l = R.shape
    rows = []
    for i in range(n_st):
        for j in range(n_l):
            rho = float(R[i, j])
            rows.append({
                'station_code': sec['stations'][i],
                'X_m': sec['xy'][i, 0], 'Y_m': sec['xy'][i, 1],
                'chainage_m': sec['chainage'][i],
                'depth_m': float(z[j]),
                'rho_ohmm': rho,
                'log10_rho': np.log10(max(rho, 1e-9)),
            })
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return out_csv


# ============================================================
# 2) XYZ 文本网格 (Surfer/GMT)
# ============================================================
def export_xyz(sec: dict, out_xyz: str | Path,
               value_scale: str = 'linear') -> Path:
    """导出 Surfer XYZ 文本: x=里程, y=-深度(向上), z=ρ/log10ρ。

    节点数 = n_station × n_layer (反演网格本身, 不做插值)。
    """
    R = sec['R']; z = sec['z_centers']
    x = sec['chainage']
    y = -z  # Surfer 纵轴向上
    Z = np.log10(np.maximum(R, 1e-9)) if value_scale == 'log10' else R
    lines = []
    for j in range(len(y)):          # 先浅后深 (y 由大到小)
        for i in range(len(x)):
            lines.append(f'{x[i]:.3f} {y[j]:.3f} {Z[i, j]:.5g}')
    out_xyz = Path(out_xyz)
    out_xyz.write_text('\n'.join(lines), encoding='ascii')
    return out_xyz


# ============================================================
# 3) Surfer GRD (DSAA)
# ============================================================
def export_surfer_grd(sec: dict, out_grd: str | Path,
                      value_scale: str = 'linear') -> Path:
    """导出 Golden Software DSAA 格式 GRD。

    DSAA 约定: 数据矩阵第 1 行对应 ymax(最浅, y=-z_min),
    末行对应 ymin(最深)。
    """
    R = sec['R']; z = sec['z_centers']
    x = sec['chainage']
    ymin, ymax = float(-z[-1]), float(-z[0])
    Z = np.log10(np.maximum(R, 1e-9)) if value_scale == 'log10' else \
        R.astype(float).copy()
    nx, ny = len(x), len(z)
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    # DSAA 行序: 第 1 行 = ymax(浅层 j=0), 即 R.T 正序
    mat = Z.T  # (ny, nx), 行 j 由浅→深
    out_grd = Path(out_grd)
    with open(out_grd, 'w', encoding='ascii') as f:
        f.write('DSAA\n')
        f.write(f'{nx} {ny}\n')
        f.write(f'{x[0]:.3f} {x[-1]:.3f}\n')
        f.write(f'{ymin:.3f} {ymax:.3f}\n')
        f.write(f'{zmin:.6g} {zmax:.6g}\n')
        for row in mat:
            f.write(' '.join(f'{v:.6g}' for v in row) + '\n')
    return out_grd


# ============================================================
# 4) GeoTIFF (rasterio 可选)
# ============================================================
def export_geotiff(sec: dict, out_tif: str | Path,
                   value_scale: str = 'linear',
                   epsg: Optional[int] = None) -> Optional[Path]:
    """断面栅格导出为 GeoTIFF。

    像元: 宽=平均站间距(m), 高=平均层厚(m); 坐标为局部工程坐标
    (x=里程, y=-深度向上)。epsg 给定时写入 CRS, 否则为无 CRS 局部米制栅格。
    rasterio 为可选依赖, 未安装时返回 None 并由调用方提示。
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return None

    R = sec['R']; z = sec['z_centers']
    x = sec['chainage']
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 50.0
    dz = float(np.median(np.diff(z))) if len(z) > 1 else 10.0
    Z = np.log10(np.maximum(R, 1e-9)) if value_scale == 'log10' else \
        R.astype(float)
    # 栅格第 1 行 = 顶部(最浅 y=-z0); 像元高 dz, y 向下递减 → -dz
    transform = from_origin(float(x[0]) - dx / 2,
                            float(-z[0]) + dz / 2, dx, dz)
    out_tif = Path(out_tif)
    profile = dict(driver='GTiff', height=Z.shape[1], width=Z.shape[0],
                   count=1, dtype='float32', transform=transform,
                   nodata=-9999.0)
    if epsg:
        profile['crs'] = rasterio.crs.CRS.from_epsg(int(epsg))
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(Z.T.astype('float32'), 1)
    return out_tif


# ============================================================
# 批量入口
# ============================================================
def export_line(out_root: str | Path, line_tag: str, algo: str = 'trf',
                formats: tuple = ('csv', 'xyz', 'grd'),
                value_scale: str = 'linear', epsg: Optional[int] = None,
                out_dir: Optional[str | Path] = None,
                log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """单测线导出。formitats 可选 csv/xyz/grd/geotiff。"""
    log_fn = log_fn or (lambda s: print(s))
    sec = load_section(out_root, line_tag, algo)
    out_dir = Path(out_dir) if out_dir else Path(out_root) / line_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{line_tag}_{algo}'
    done = {}
    if 'csv' in formats:
        p = export_csv(sec, out_dir / f'{tag}_section.csv', value_scale)
        done['csv'] = p; log_fn(f'  [导出] CSV 点数据 → {p}')
    if 'xyz' in formats:
        p = export_xyz(sec, out_dir / f'{tag}_section.xyz', value_scale)
        done['xyz'] = p; log_fn(f'  [导出] Surfer XYZ → {p}')
    if 'grd' in formats:
        p = export_surfer_grd(sec, out_dir / f'{tag}_section.grd', value_scale)
        done['grd'] = p; log_fn(f'  [导出] Surfer GRD(DSAA) → {p}')
    if 'geotiff' in formats:
        p = export_geotiff(sec, out_dir / f'{tag}_section.tif',
                           value_scale, epsg)
        if p is None:
            log_fn('  [导出] GeoTIFF 跳过: 未安装可选依赖 rasterio '
                   '(pip install rasterio)')
        else:
            done['geotiff'] = p; log_fn(f'  [导出] GeoTIFF → {p}')
    return done


def export_all_lines(out_root: str | Path, algo: str = 'trf',
                     formats: tuple = ('csv', 'xyz', 'grd'),
                     value_scale: str = 'linear', epsg: Optional[int] = None,
                     log_fn: Optional[Callable[[str], None]] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    lines = sorted([p.name for p in out_root.iterdir() if p.is_dir()
                    and re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== 标准化导出 algo={algo} fmt={formats} '
          f'测线={lines} ==========')
    result = {}
    for lt in lines:
        try:
            result[lt] = export_line(out_root, lt, algo, formats,
                                     value_scale, epsg, log_fn=log_fn)
        except Exception as e:
            log_fn(f'  ✗ {lt}: 导出失败: {e}')
            result[lt] = {'error': str(e)}
    return result
