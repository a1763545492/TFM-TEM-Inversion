# -*- coding: utf-8 -*-
"""
data_check.py — Step0.5 数据体检 (前置坏数据检测)
==================================================================
在整条流水线 (预处理/反演) 之前运行, 用最小代价发现:

  结构级 (必查, 秒级):
    · 测线目录 (1-5线) 缺失 / Excel 元数据缺失 / 权威坐标文件缺失
    · section 目录缺发射电流档 (CXXST01)
    · Excel 声明的 (section, instrument) 与磁盘接收档 dat 不对齐
    · section 目录为空 / 无接收档

  文件级 (默认抽检, deep=True 全量):
    · dat 解析失败 (header 魔法值/版本/最小大小校验不过)
    · 文件截断 (期望字节 vs 实际字节不符, dat_parser 自动截断/告警)
    · 波形 cube NaN 比例 / 死通道 (std 过低) / 幅值越界 (counts 级)

体检结果分级: ERROR (阻断流水线) / WARN (建议人工复核) / OK。
产物: {out_root}/data_check_report.json + 控制台摘要。
返回码: 0=OK, 1=有 WARN, 2=有 ERROR。

设计原则: 不重复 preprocess 的逐站 QC (死道/NaN比例/跳变, 那是
gate 提取后的物理量级检查), 本模块只做 "数据能不能进流水线" 的
前置可行性检查。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .dat_parser import load_wlf_dat
from .preprocess import run_inventory, PER_CH

# 原始波形 (counts 级) 合理幅值界; 超出视为采集异常
_RAW_AMP_ABS_MAX = 1.0e9
_DEAD_CH_STD_MIN = 100.0        # preprocess.QC_DEAD_CH_STD 同值
_NAN_RATIO_MAX = 0.50           # cube NaN 比例上限


def check_station_dat(fp: Path, log_fn: Callable[[str], None]) -> dict:
    """单个 dat 文件体检。返回 {'status': ok|warn|error, 'issues': [...]}"""
    issues: list[str] = []
    status = 'ok'
    try:
        d = load_wlf_dat(fp, log_fn=log_fn)
    except Exception as e:
        return {'status': 'error',
                'issues': [f'解析失败: {type(e).__name__}: {e}'],
                'n_stakes': 0, 'nan_ratio': None, 'dead_ch': None}

    cube = d.cube  # (4, PER_CH, Np)
    n_ch, n_samp, n_stakes = cube.shape
    if n_stakes == 0:
        return {'status': 'error', 'issues': ['数据桩数 Np=0'],
                'n_stakes': 0, 'nan_ratio': None, 'dead_ch': None}

    # NaN 比例 (Hx 磁分量 ch=2 为主, 但全通道统一体检)
    nan_ratio = float(np.mean(~np.isfinite(cube)))
    if nan_ratio > _NAN_RATIO_MAX:
        issues.append(f'NaN/Inf 比例 {nan_ratio:.1%} > { _NAN_RATIO_MAX:.0%}')
        status = 'error'

    # 死通道: 某通道 std 过低 (std 按通道-桩平均)
    ch_std = np.nanstd(cube, axis=(1, 2))
    dead = [int(i) for i, s in enumerate(ch_std) if s < _DEAD_CH_STD_MIN]
    if 2 in dead:  # Hx 主分量死道 = 致命
        issues.append(f'Hx(ch2) 死通道 std={ch_std[2]:.1f} < {_DEAD_CH_STD_MIN}')
        status = 'error'
    elif dead:
        issues.append(f'通道死道 {dead} (std<{_DEAD_CH_STD_MIN})')
        status = max(status, 'warn', key=lambda s: {'ok': 0, 'warn': 1, 'error': 2}[s])

    # 幅值越界
    absmax = float(np.nanmax(np.abs(cube)))
    if absmax > _RAW_AMP_ABS_MAX:
        issues.append(f'幅值越界 |max|={absmax:.3g} > {_RAW_AMP_ABS_MAX:.3g}')
        status = 'error'

    return {'status': status, 'issues': issues, 'n_stakes': int(n_stakes),
            'nan_ratio': nan_ratio,
            'dead_ch': dead or None,
            'absmax': absmax}


def check_line(line_tag: str, line_inv: dict, coordinates: dict,
               deep: bool, log_fn: Callable[[str], None]) -> dict:
    """单测线体检: 结构对齐 + (抽检/全量) dat 文件。"""
    issues_e: list[str] = []   # error
    issues_w: list[str] = []   # warn
    meta = line_inv.get('meta') or {}
    sections = line_inv.get('sections_disk') or {}

    if not meta:
        issues_e.append('Excel 元数据缺失 (parse_em_xlsx 未返回)')
    if not sections:
        issues_e.append('无任何数字 section 子目录')
        return {'line': line_tag, 'status': 'error',
                'errors': issues_e, 'warns': issues_w, 'dat_checks': {}}

    # Excel 声明的 (section, instrument) 集合
    excel_pairs: set[tuple[str, str]] = set()
    for sec_key, sec in (meta.get('sections') or {}).items():
        for inst in (sec.get('instruments') or []):
            excel_pairs.add((sec_key, str(inst).zfill(3)))

    # 磁盘 (section, instrument) 集合
    disk_pairs: set[tuple[str, str]] = set()
    for sec_key, sd in sections.items():
        recv = sd.get('recv_dats') or {}
        for inst in recv:
            disk_pairs.add((sec_key, str(inst).zfill(3)))
        if sd.get('tx_dat') is None:
            issues_e.append(f'section {sec_key}: 缺发射电流档 CXXST01.dat')
        if not recv:
            issues_w.append(f'section {sec_key}: 无接收档 dat')

    missing_disk = excel_pairs - disk_pairs
    if missing_disk:
        issues_e.append(f'Excel 声明但磁盘缺失 dat: {sorted(missing_disk)[:10]}'
                        + (f' 等{len(missing_disk)}项' if len(missing_disk) > 10 else ''))
    orphan = disk_pairs - excel_pairs
    if orphan:
        issues_w.append(f'磁盘有 dat 但 Excel 未声明: {sorted(orphan)[:10]}')

    # 权威坐标覆盖检查
    n_coord = 0
    try:
        n_coord = len(coordinates.get('by_station', coordinates))
    except Exception:
        n_coord = len(coordinates) if coordinates else 0
    if n_coord == 0:
        issues_e.append('权威坐标文件未解析到任何测站')

    # dat 文件级体检 (deep=True 全量; 否则每 section 抽 1 个)
    dat_checks: dict[str, dict] = {}
    pairs_to_check: list[tuple[str, Path]] = []
    for sec_key, sd in sections.items():
        recv = sd.get('recv_dats') or {}
        items = sorted(recv.items())
        take = items if deep else items[:1]
        pairs_to_check += [(sec_key, p) for _, p in take]

    for sec_key, fp in pairs_to_check:
        r = check_station_dat(fp, log_fn)
        dat_checks[f'{sec_key}/{fp.name}'] = r
        if r['status'] == 'error':
            issues_e.append(f'dat 体检失败 {sec_key}/{fp.name}: {r["issues"]}')
        elif r['status'] == 'warn':
            issues_w.append(f'dat 体检告警 {sec_key}/{fp.name}: {r["issues"]}')

    status = 'error' if issues_e else ('warn' if issues_w else 'ok')
    return {'line': line_tag, 'status': status,
            'errors': issues_e, 'warns': issues_w,
            'n_excel_pairs': len(excel_pairs), 'n_disk_pairs': len(disk_pairs),
            'n_coordinates': n_coord, 'dat_checks': dat_checks}


def check_project(project_root: str | Path, out_root: Optional[str | Path] = None,
                  deep: bool = False, log_fn: Optional[Callable[[str], None]] = None
                  ) -> dict:
    """全项目数据体检 (Step0.5)。

    返回 dict: rc(0/1/2) + lines 各线详情 + errors/warns 汇总。
    out_root 给定时落盘 data_check_report.json。
    """
    log_fn = log_fn or (lambda s: print(s))
    project_root = Path(project_root)
    t0 = time.time()
    log_fn('\n========== [Step0.5] 数据体检 ==========')
    log_fn(f'  数据根: {project_root}   deep={deep}')

    errors: list[str] = []
    warns: list[str] = []
    lines: dict[str, dict] = {}

    try:
        inv = run_inventory(project_root, log_fn)
    except Exception as e:
        log_fn(f'  [ERROR] inventory 失败: {e}')
        report = {'rc': 2, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'errors': [f'inventory 失败: {e}'], 'warns': [], 'lines': {}}
        _save_report(report, out_root)
        return report

    for line_tag, line_inv in inv.get('lines', {}).items():
        r = check_line(line_tag, line_inv, inv.get('coordinates') or {},
                       deep=deep, log_fn=log_fn)
        lines[line_tag] = r
        for e in r['errors']:
            errors.append(f'[{line_tag}] {e}')
        for w in r['warns']:
            warns.append(f'[{line_tag}] {w}')

    # 全局: 坐标文件
    if not (project_root / '坐标文件.xls').is_file() \
            and not list(project_root.glob('*.xls')):
        errors.append('权威坐标文件 (坐标文件.xls) 缺失')

    rc = 2 if errors else (1 if warns else 0)
    report = {'rc': rc, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
              'project_root': str(project_root), 'deep': deep,
              'lines': lines, 'errors': errors, 'warns': warns,
              'elapsed_s': round(time.time() - t0, 2)}
    _save_report(report, out_root)

    # 摘要
    tag = {0: 'OK', 1: 'WARN', 2: 'ERROR'}[rc]
    log_fn(f'  ---- 体检结论: [{tag}] rc={rc} '
           f'({len(lines)} 测线, {len(errors)} 错误, {len(warns)} 告警, '
           f'{report["elapsed_s"]:.1f}s) ----')
    for e in errors[:20]:
        log_fn(f'    [ERROR] {e}')
    for w in warns[:20]:
        log_fn(f'    [WARN ] {w}')
    if rc == 0:
        log_fn('  ✓ 数据结构完整, 可进入预处理流水线')
    return report


def _save_report(report: dict, out_root: Optional[str | Path]) -> None:
    if out_root is None:
        return
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'data_check_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


__all__ = ['check_project', 'check_line', 'check_station_dat']
