# -*- coding: utf-8 -*-
"""
coord_loader.py — 加载权威坐标文件 `myfile/坐标文件.xls` (单一数据源契约)
========================================================================
两个 Sheet:
  1.「MN坐标」: station_code (S1-1..S5-16) → (northing, easting, elev)  共 64 行 (4 测线 × 16 点)
  2.「源坐标」: (A1,0), (B1,1), (A2,0), (B2,1) → 大定源回线对角点
输出:
  - StationMap  = dict[station_code] → (X_northing, Y_easting, Z_elev)    (64 条)
  - TxAreaMap   = dict['A1B1'/'A2B2'] → 面积 (m²)  = |Δn × Δe|
    - A1B1 ≈ 4,452,697 m²  (奇号线 1/5 用)
    - A2B2 ≈ 1,724,449 m²  (偶号线 2/4 用)
  - TxCornerMap = dict['A1'/'B1'/'A2'/'B2'] → (X_northing, Y_easting)
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import pandas as pd


def load_authoritative_coordinates(xls_path: Path,
                                    log_fn: Callable[[str], None] = None) -> dict:
    log_fn = log_fn or (lambda s: None)
    if not xls_path.is_file():
        raise FileNotFoundError(f'权威坐标文件不存在: {xls_path}')
    df_mn = pd.read_excel(xls_path, sheet_name='MN坐标', header=None,
                          names=['station_code', 'X', 'Y', 'Z'])
    df_src = pd.read_excel(xls_path, sheet_name='源坐标', header=None,
                           names=['tag', 'st_idx', 'X', 'Y'])
    StationMap: dict[str, tuple[float, float, float]] = {}
    for _, r in df_mn.iterrows():
        st_code = str(r.station_code).strip()
        if not st_code or pd.isna(r.X) or pd.isna(r.Y):
            continue
        StationMap[st_code] = (float(r.X), float(r.Y), 0.0 if pd.isna(r.Z) else float(r.Z))

    TxCornerMap: dict[str, tuple[float, float]] = {}
    for _, r in df_src.iterrows():
        tag = str(r.tag).strip()
        if not tag or pd.isna(r.X) or pd.isna(r.Y): continue
        TxCornerMap[tag] = (float(r.X), float(r.Y))

    TxAreaMap: dict[str, float] = {}
    if 'A1' in TxCornerMap and 'B1' in TxCornerMap:
        n0, e0 = TxCornerMap['A1']; n1, e1 = TxCornerMap['B1']
        TxAreaMap['A1B1'] = abs((n1 - n0) * (e1 - e0))
    if 'A2' in TxCornerMap and 'B2' in TxCornerMap:
        n0, e0 = TxCornerMap['A2']; n1, e1 = TxCornerMap['B2']
        TxAreaMap['A2B2'] = abs((n1 - n0) * (e1 - e0))

    # 每线 station 覆盖完整性校验
    from collections import Counter
    prefix_counter = Counter()
    for k in StationMap:
        if '-' in k: prefix_counter[k.split('-')[0]] += 1
    log_fn(f'  [coord] 权威坐标加载: {len(StationMap)} 个 stations = {dict(prefix_counter)}; '
          f'Tx 面积 A1B1={TxAreaMap.get("A1B1",0):,.0f}m²  A2B2={TxAreaMap.get("A2B2",0):,.0f}m²')
    return {
        'StationMap': StationMap,
        'TxCornerMap': TxCornerMap,
        'TxAreaMap': TxAreaMap,
    }
