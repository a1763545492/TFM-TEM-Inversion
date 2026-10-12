# -*- coding: utf-8 -*-
"""
preprocess.py — 四阶段预处理流水线 (完整重新实现, 权威 Excel schema 驱动)
======================================================================
阶段 1 Inventory  : project_root → 每条测线 Excel 解析 + section 目录枚举 + DAT 文件绑定
阶段 2 Load       : 按 Excel 映射链逐 DAT 读取 (auto_truncate_if_mismatch)
阶段 3 Cube Align : 关键! 对 每 section(=大写档 CXXST01) 下 3 台仪器 × 每仪器 Np=5 →
                    映射到 Excel「接收对应点位」的 station 集合, 合并成 16 条 station_cube.
                    合并策略: 同一 station 多仪器/多通道 → 按通道加权平均 (Hx优先).
阶段 4 ADC + ρa + QC + Export:
                    S4 ADC (G=2^22)  S5 去DC  S6 25 gates  S7 ρa (中心回线公式, 奇偶号线 Tx 面积不同)
                    S8 QC + Occam.dat + ρa section PNG + pickup_xyz CSV + cfg JSON

输入: project_root (myfile/), out_dir, cfg 超参 dict
输出: out_dir/{1,2,4,5}线/*_occam.dat | *_rhoa_section.png | *_pickup_xyz.csv
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 离线无头
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','Arial']
rcParams['axes.unicode_minus'] = False

# 本地模块
from .dat_parser import load_wlf_dat, PER_CHANNEL, auto_truncate_if_mismatch, WlfDatData, WlfDatHeader
from .excel_meta import parse_em_xlsx, LINE_NAME_TO_CODE, LINE_CODE_TO_NAME
from .coord_loader import load_authoritative_coordinates

# 默认超参 (单变量少步长, 可由 GUI/CLI 覆盖)
DEFAULT_CFG = {
    'G_ADC_COUNTS_PER_VOLT': 4_194_304,   # 2^22 = 4,194,304  counts/V (标定档半量程实锤)
    'RHOA_SCALE_CAL'       : 1_000_000,   # ★ 第一调参入口 (ρa 全局校准)
    'TEM_N_GATES'          : 25,
    'T_MIN_US'             : 2.0,
    'T_MAX_US'             : 20_000.0,
    'DC_OFFSET_FRAC'       : 0.10,        # TEM 关断前 前 10% median 去DC
    'MU0'                  : 4.0 * np.pi * 1e-7,
    'ERR_MIN_FRAC'         : 0.02,        # 2% 下限
    'QC_DEAD_CH_STD'       : 100,
    'QC_RHOA_NAN_RATIO_MAX': 0.50,
    'QC_JUMP_RATIO_MAX'    : 20.0,
}

PER_CH = PER_CHANNEL     # 544,859  锁死
HX_CH = 2                # 磁分量 Hx 索引 (TEM 主分量)


@dataclass
class StationObservation:
    """最终对齐到权威 station_code (S1-1..S5-16) 的一个观测点"""
    line_tag: str
    line_code: str
    station_code: str
    section: str
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    tx_area_m2: float = 0.0
    i_tx_amps: float = 0.0
    mag_bar: str = ""
    instruments: list[str] = field(default_factory=list)
    Vt: Optional[np.ndarray] = None   # shape=(Ng,) 提取后的 Hx-gated
    rhoa: Optional[np.ndarray] = None # shape=(Ng,)
    err: Optional[np.ndarray] = None  # shape=(Ng,)
    qc_flags: dict = field(default_factory=dict)


# ============================================================
# S1 Inventory: project_root → 4 测线 sections+instruments+DAT 绑定
# ============================================================
def run_inventory(project_root: Path, log_fn: Callable[[str], None]) -> dict:
    """project_root = myfile/.  扫描 4 个 X线 子目录, 读 Excel + 列举 section 子目录.
    返回结构:
      {'line_tag' → {'meta': <parse_em_xlsx>, 'sections_disk': {section_key → {instrument → Path(dat_on_disk),
                                                                               tx_dat → Path(CXXST01)}}}}
    """
    lines_dirs = [p for p in project_root.iterdir() if p.is_dir()
                  and re.fullmatch(r'[1-5]线', p.name)]
    if not lines_dirs:
        raise FileNotFoundError(f'{project_root} 下未找到任何 [1-5]线 子目录')
    inventory = {}
    for line_dir in sorted(lines_dirs):
        line_tag = line_dir.name
        log_fn(f'\n==== Inventory: {line_tag} ====')
        meta = parse_em_xlsx(line_dir, log_fn)
        # 找 section 子目录  (1线/17, 1线/27;  数值目录)
        sec_dirs = [p for p in line_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        sections_disk: dict[str, dict] = {}
        for sd in sec_dirs:
            sec_key = sd.name.zfill(3)  # "17" → "017"
            recv_insts: dict[str, Path] = {}
            tx_dat = None
            for f in sd.iterdir():
                if not f.suffix.lower() == '.dat': continue
                name = f.name
                if name.upper() == name:          # 全大写 = 发射电流档 (CXXST01)
                    tx_dat = f
                else:                              # 全小写 = 接收档
                    m = re.match(r'^c(\d{3})st(\d{3})\.dat', name.lower())
                    if m and m.group(1) == sec_key:
                        recv_insts[m.group(2)] = f   # "001" → Path
            sections_disk[sec_key] = {'recv_dats': recv_insts, 'tx_dat': tx_dat,
                                       'dir': sd}
            log_fn(f'  section {sec_key}: tx={tx_dat.name if tx_dat else "MISSING"},'
                  f' recv instruments={sorted(recv_insts.keys())}')
        inventory[line_tag] = {'meta': meta, 'sections_disk': sections_disk}
    # 权威坐标全局加载 1 次
    coord_xls = project_root / '坐标文件.xls'
    if not coord_xls.is_file():
        # 回退: 同目录下的 xls
        for f in project_root.glob('*.xls'):
            coord_xls = f; break
    coordinates = load_authoritative_coordinates(coord_xls, log_fn)
    return {'project_root': project_root,
            'lines': inventory, 'coordinates': coordinates}


# ============================================================
# S2~S8 单测线完整处理
# ============================================================
def process_one_line(line_tag: str, inv: dict, cfg: dict, out_dir: Path,
                     log_fn: Callable[[str], None]) -> dict:
    meta = inv['lines'][line_tag]['meta']
    line_code = meta['line_code']
    coord = inv['coordinates']
    StationMap = coord['StationMap']; TxAreaMap = coord['TxAreaMap']
    line_num = int(line_tag.rstrip('线'))   # "1线" → 1
    tx_area = TxAreaMap['A1B1'] if (line_num % 2 == 1) else TxAreaMap['A2B2']  # 奇偶号线
    out_base = out_dir / line_tag
    out_base.mkdir(parents=True, exist_ok=True)

    # --- S2 逐 section + instrument + Np → 读 cube, 提取 CH2(Hx) 关断后 gates 25 道
    #     对应到权威 station_code
    station_obs: dict[str, StationObservation] = {}
    # 先初始化 权威 station (16 个)
    for st_code in sorted(StationMap.keys()):
        if not st_code.startswith(line_code + '-'): continue
        station_obs[st_code] = StationObservation(
            line_tag=line_tag, line_code=line_code, station_code=st_code,
            section="", x=StationMap[st_code][0], y=StationMap[st_code][1],
            z=StationMap[st_code][2], tx_area_m2=tx_area,
        )
    log_fn(f'  [S2] 权威 station 初始化: {len(station_obs)} 个 ({sorted(list(station_obs.keys()))[:3]}...)')

    # channel_station_rows 里的 (date, order, instrument, ch, section, station)
    # ★ 关键映射: 每 (section_key, instrument_code) 独立 station 序列 = 该仪器在该 section 滚动排列的 Np 个 station.
    #   例如 1线017/inst001 的 Excel 顺序(S1-10,S1-11,S1-12) 就是该仪器 picket=0→10, 1→11, 2→12...
    #   (同一 channel 1/2/3 多行出现的同一 station 在顺序中只保留首次).
    sec_inst_station_lists: dict[tuple[str, str], list[str]] = {}  # (sec, inst) → ordered stations unique
    for sec, sec_meta in meta['sections'].items():
        order = sec_meta['order']
        rows_same = [r for r in meta['channel_station_rows'] if r['order'] == order]
        # 先按 instrument 分组
        by_inst: dict[str, list[str]] = {}
        for r in rows_same:
            inst_r = r['instrument']
            st = r['station']
            if inst_r not in by_inst: by_inst[inst_r] = []
            if st not in by_inst[inst_r]:  # 保留首次出现顺序
                by_inst[inst_r].append(st)
        # 再绑定到 sec key
        for inst_r, stations_in_order in by_inst.items():
            sec_inst_station_lists[(sec, inst_r)] = stations_in_order
        sec_summary = {inst_r: lst for (sec_r, inst_r), lst in sec_inst_station_lists.items() if sec_r == sec}
        short = {k: (v if len(v) <= 6 else v[:6] + ['...+%d' % (len(v)-6)]) for k, v in sec_summary.items()}
        log_fn(f'    section {sec} (order {order}): per-inst stations seq = {short}')

    section_keys = sorted(meta['sections'].keys())
    # 每个 station 的多个 (cube+channel+picket) Vt 样本, 收集后最后 median
    station_samples: dict[str, list[np.ndarray]] = {k: [] for k in station_obs}
    station_i_tx_map: dict[str, float] = {}

    # 关断索引 (按整线统一): 先扫描 所有 section+instrument+picket 的 Hx 找 max 位置众数
    cutoff_candidates: list[int] = []
    loaded_cubes: dict[tuple[str, str], WlfDatData] = {}  # (section, inst) → WlfDatData

    for sec in section_keys:
        disk = inv['lines'][line_tag]['sections_disk'].get(sec, {})
        recv_dats: dict[str, Path] = disk.get('recv_dats', {})
        sec_meta = meta['sections'][sec]
        i_tx = float(sec_meta['i_tx_amps'] or 30.0)
        for inst in ('001', '002', '003'):
            if inst not in recv_dats: continue
            try:
                wd = load_wlf_dat(recv_dats[inst], log_fn=log_fn, validate=True, strict_per_ch=False)
            except Exception as e:
                log_fn(f'  [WARN] 读取 {sec}/{inst} 失败: {e}')
                continue
            loaded_cubes[(sec, inst)] = wd
            cube32 = wd.cube.astype(np.int32)
            # 对每个 picket 收集 Hx(|·|top10 index 众数) → cutoff candidate
            for pk in range(wd.n_pickets):
                hx = np.abs(cube32[HX_CH, :, pk].astype(np.float64))
                top_k = hx.argsort()[-20:][::-1]
                # 取连续 10 窗口内的最小 idx (关断脉冲起点)
                if len(top_k):
                    cutoff_candidates.append(int(np.min(top_k)))
    # 取 cutoff 众数 (夹 200,000 ~ 400,000 安全范围)
    cutoff_idx = 270_000
    if cutoff_candidates:
        from collections import Counter
        c = Counter(cutoff_candidates); cutoff_idx = c.most_common(1)[0][0]
    cutoff_idx = int(np.clip(cutoff_idx, 200_000, 400_000))
    # 25 gates: cutoff_idx → PER_CH-1, geomspace (对数等比)
    idx_gates = np.geomspace(cutoff_idx, PER_CH - 1, num=cfg['TEM_N_GATES'], endpoint=True).astype(np.int64)
    idx_gates = np.unique(idx_gates)
    # 时间轴: 映射到 t∈[2μs, 20ms] 对数 (实际 WLF 时钟未标定, 用相对 gates)
    t_us = np.geomspace(cfg['T_MIN_US'], cfg['T_MAX_US'], num=len(idx_gates))
    t_s = t_us * 1e-6

    # --- S3 Cube → Vt (1 个 station = 多个 instrument/picket/channel 的 median) ---
    for sec in section_keys:
        sec_meta = meta['sections'][sec]
        i_tx = float(sec_meta['i_tx_amps'] or 30.0)
        for inst in ('001', '002', '003'):
            wd = loaded_cubes.get((sec, inst))
            if wd is None: continue
            stations_ordered = sec_inst_station_lists.get((sec, inst), [])
            if not stations_ordered:
                log_fn(f'  [WARN] {sec}/{inst}: Excel 无对应接收点位映射, 跳过')
                continue
            # 匹配 station (按 picket_idx ∈ 0..Np-1 → stations_ordered[pk])
            for pk in range(wd.n_pickets):
                if pk >= len(stations_ordered):
                    break
                st_code = stations_ordered[pk]
                if st_code not in station_obs:
                    continue
                so = station_obs[st_code]
                so.section = so.section or sec
                so.i_tx_amps = so.i_tx_amps or i_tx
                if inst not in so.instruments:
                    so.instruments.append(inst)
                station_i_tx_map[st_code] = i_tx
                # 提取 gates: Hx channel
                cube64 = wd.cube.astype(np.float64)
                # S4 ADC + S5 去 DC (关断前前 DC_OFFSET_FRAC median)
                ch_pk = cube64[:, :, pk] / cfg['G_ADC_COUNTS_PER_VOLT']
                dc_roi = slice(0, max(5000, int(cutoff_idx * cfg['DC_OFFSET_FRAC'])))
                dc = np.median(ch_pk[:, dc_roi], axis=1, keepdims=True)
                ch_pk -= dc
                hx_gated = np.abs(ch_pk[HX_CH, :])[idx_gates]   # shape=(Ng,)
                station_samples.setdefault(st_code, []).append(hx_gated)

    # 汇总 station_samples → median Vt
    n_samples_lst = [len(v) for v in station_samples.values() if v]
    ns_med = int(np.median(n_samples_lst)) if n_samples_lst else 0
    log_fn(f'  [S3-S5] 对齐到权威 station: {len(station_samples)} 个, 每 station 合并样本数 median={ns_med}  (instrument×picket 合并)')

    Ng = len(t_s)
    obs_list: list[StationObservation] = []
    mu0 = cfg['MU0']
    coef_front = (mu0 / (4 * np.pi)) ** 1.5
    for st_code, so in station_obs.items():
        so.i_tx_amps = station_i_tx_map.get(st_code, so.i_tx_amps or 30.0)
        samples = station_samples.get(st_code, [])
        if not samples:
            so.Vt = np.full(Ng, np.nan)
            so.rhoa = np.full(Ng, np.nan)
            so.err = np.full(Ng, np.nan)
            obs_list.append(so)   # 16 权威 station 全进 obs_list, QC 判为 dead
            continue
        arr = np.vstack(samples)  # (Nsamples, Ng)
        Vt = np.median(arr, axis=0)                 # 跨样本 median 合并
        so.Vt = Vt
        # S7 ρa
        I = so.i_tx_amps
        Ae = so.tx_area_m2 or 0
        if Ae <= 0: Ae = tx_area
        M = I * Ae
        coef = coef_front * (2.0 * M) ** (2.0 / 3.0) * cfg['RHOA_SCALE_CAL']
        rhoa = np.zeros(Ng, dtype=np.float64)
        for gi in range(Ng):
            t = max(t_s[gi], 1e-10)
            v = max(Vt[gi], 1e-30)
            rhoa[gi] = coef * (1.0 / t) ** 1.5 / (v ** (2.0 / 3.0))
        # 误差估计: 2% 下限 + 相邻道 log 差 / 2
        log_r = np.log10(np.clip(rhoa, 1e-2, 1e12))
        spread = np.diff(log_r)
        spread = np.concatenate([[spread[0] if len(spread) else 0.0], spread,
                                 [spread[-1] if len(spread) else 0.0]])
        spread = np.convolve(spread, [0.5, 0.5], mode='valid') if len(spread) >= 2 else spread
        if len(spread) != Ng: spread = np.pad(spread, (0, Ng-len(spread)), mode='edge')[:Ng]
        err = rhoa * np.maximum(cfg['ERR_MIN_FRAC'], np.abs(spread) / 2.0)
        rhoa = np.where((np.isfinite(rhoa)) & (rhoa > 0), rhoa, np.nan)
        so.rhoa = rhoa; so.err = err
        obs_list.append(so)

    # 排序: 按 station 号升序
    def _sort_key(s: StationObservation):
        m = re.search(r'(\d+)$', s.station_code)
        return int(m.group(1)) if m else 999
    obs_list.sort(key=_sort_key)

    # --- S8 QC ---
    n_total = len(obs_list)
    qc_keep: list[int] = []
    for i, so in enumerate(obs_list):
        dead = so.Vt is None or not np.isfinite(so.Vt).any()
        if so.rhoa is None:
            nan_ratio = 1.0
        else:
            nan_ratio = float(np.isnan(so.rhoa).mean())
        # 跳变计数
        jc = 0
        if so.rhoa is not None:
            good = np.isfinite(so.rhoa) & (so.rhoa > 0)
            if good.sum() >= 2:
                rr = so.rhoa[good]
                jc = int((rr[1:] / (rr[:-1] + 1e-30) > cfg['QC_JUMP_RATIO_MAX']).sum())
        drop = dead or nan_ratio > cfg['QC_RHOA_NAN_RATIO_MAX']
        so.qc_flags = {'dead': dead, 'nan_ratio': nan_ratio,
                       'jump_count': jc, 'drop': drop}
        if not drop: qc_keep.append(i)
    log_fn(f'  [QC] {len(qc_keep)}/{n_total} 通过 (死道={sum(so.qc_flags["dead"] for so in obs_list)},'
          f' NaN>{cfg["QC_RHOA_NAN_RATIO_MAX"]*100:.0f}%={sum(so.qc_flags["nan_ratio"]>cfg["QC_RHOA_NAN_RATIO_MAX"] for so in obs_list)})')

    # --- 导出 Occam.dat (EMData_1.1 / OCCAMITER_FLEX 空格分隔 9 列) ---
    occam_path = out_base / f'{line_tag}_occam.dat'
    rows_out = []
    for i in qc_keep:
        so = obs_list[i]
        if so.rhoa is None: continue
        for gi in range(Ng):
            if not (np.isfinite(so.rhoa[gi]) and np.isfinite(so.err[gi]) and so.rhoa[gi] > 0): continue
            rows_out.append([so.x, so.y, so.z, t_s[gi], so.rhoa[gi],
                             so.err[gi], so.station_code, so.section, so.instruments])
    with open(occam_path, 'w', encoding='utf-8') as f:
        f.write(f'# OCCAMITER_FLEX / EMData_1.1 格式 | 测线={line_tag} |'
                f' 权威 stations 保持={len(station_obs)} | QC pass={len(qc_keep)} | Ng={Ng} | '
                f'ADC G={cfg["G_ADC_COUNTS_PER_VOLT"]:.0e}\n')
        f.write(f'# Units: X/Y/Z=m  Time=s  AppRes=Ohm*m  StdDev=Ohm*m\n')
        f.write(f'# Columns: X  Y  Z  Time  AppRes  StdDev  StationCode  SectionKey  Instruments\n')
        for r in rows_out:
            instrs = ','.join(r[8]) if isinstance(r[8], list) else str(r[8])
            f.write(f'{r[0]:14.4f} {r[1]:14.4f} {r[2]:10.3f} {r[3]:12.4e}'
                    f' {r[4]:12.4e} {r[5]:12.4e}  "{r[6]}"  "{r[7]}"  "{instrs}"\n')
    log_fn(f'  [S8] Occam.dat → {occam_path}  ({len(rows_out)} 行)')

    # --- 断面色带图 (经验 #1477534 统一 R[station, Ng] → pcolormesh) ---
    if len(qc_keep) >= 2:
        rhoa_mat = np.vstack([obs_list[i].rhoa for i in qc_keep])  # (N_keep, Ng)
        X = np.arange(rhoa_mat.shape[0])
        Y = np.log10(np.maximum(t_s, 1e-12))
        Xg, Yg = np.meshgrid(X, Y, indexing='ij')
        Z = np.log10(np.clip(rhoa_mat, 1e-1, 1e9))
        fig, ax = plt.subplots(figsize=(max(8, len(qc_keep) * 0.35 + 2), 5))
        im = ax.pcolormesh(Xg, Yg, Z, shading='auto', cmap='jet')
        yt = np.linspace(Y[0], Y[-1], 6)
        ax.set_yticks(yt)
        ax.set_yticklabels([f'$10^{{{v:.1f}}}$s' for v in yt])
        step = max(1, len(qc_keep)//12)
        ax.set_xticks(X[::step])
        ax.set_xticklabels([obs_list[q].station_code for q in qc_keep[::step]],
                           rotation=45)
        ax.set_xlabel('测点 (Sx-N 权威编码)')
        ax.set_ylabel('时间 Time (s, log 轴)')
        ax.set_title(f'{line_tag}  视电阻率 ρa(t) 剖面  (权威 stations {len(qc_keep)} × {Ng} 道)')
        cbar = fig.colorbar(im, ax=ax); cbar.set_label('log10(ρa / Ω·m)')
        fig.tight_layout()
        png_p = out_base / f'{line_tag}_rhoa_section.png'
        fig.savefig(png_p, dpi=180); plt.close(fig)
        log_fn(f'  [S8] ρa 剖面色带图 → {png_p}')

    # --- pickup_xyz.csv (Excel 直开 utf-8-sig) ---
    csv_p = out_base / f'{line_tag}_pickup_xyz.csv'
    rows = []
    for so in obs_list:
        if so.rhoa is not None and np.isfinite(so.rhoa).any():
            rm = float(np.nanmedian(so.rhoa))
        else:
            rm = float('nan')
        rows.append({
            'line_tag': so.line_tag, 'line_code': so.line_code,
            'station_code': so.station_code, 'section_key': so.section,
            'X_m': so.x, 'Y_m': so.y, 'Z_m': so.z,
            'tx_area_m2': so.tx_area_m2, 'i_tx_amps': so.i_tx_amps,
            'mag_bar': so.mag_bar,
            'instruments': ','.join(so.instruments),
            'qc_drop': so.qc_flags.get('drop', False),
            'qc_dead': so.qc_flags.get('dead', False),
            'qc_nan_ratio': so.qc_flags.get('nan_ratio', 0.0),
            'qc_jump_count': so.qc_flags.get('jump_count', 0),
            'rhoa_median_Ohm': rm,
            'Vt_samples_merged': len(station_samples.get(so.station_code, [])),
        })
    pd.DataFrame(rows).to_csv(csv_p, index=False, encoding='utf-8-sig')
    log_fn(f'  [S8] 测点元数据表 → {csv_p}')

    return {
        'stations_total': n_total,
        'stations_qc_pass': len(qc_keep),
        'rhoa_median_all': float(np.nanmedian(np.concatenate([o.rhoa for o in obs_list if o.rhoa is not None]))),
        'tx_area_m2': tx_area,
        'occam_rows': len(rows_out),
        'gates': Ng,
    }


# ============================================================
# 主入口
# ============================================================
def run_full_preprocess(project_root: str | Path, out_root: str | Path,
                        user_cfg: Optional[dict] = None,
                        log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """完整 4 阶段. project_root = D:/MyProject/myfile/
    返回 summary dict. log_fn 用于 GUI 文本框回显."""
    project_root = Path(project_root); out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_CFG)
    if user_cfg: cfg.update(user_cfg)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg_path = out_root / f'cfg_snapshot_{ts}.json'
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')

    log_fn(f'{"="*80}\n  TEM 预处理启动 @ {ts}\n'
          f'  项目根: {project_root}\n  输出根: {out_root}\n'
          f'  参数快照: {cfg_path}\n{"="*80}')

    inv = run_inventory(project_root, log_fn)
    line_tags = sorted(inv['lines'].keys())
    summary = {}
    for lt in line_tags:
        log_fn(f'\n{"█"*60}\n  处理测线: {lt}\n{"█"*60}')
        try:
            st = process_one_line(lt, inv, cfg, out_root, log_fn)
            summary[lt] = st
            log_fn(f'  ✓ {lt}: stations(qc_pass/total)={st["stations_qc_pass"]}/{st["stations_total"]}'
                  f'  ρa_median={st["rhoa_median_all"]:.2e} Ω·m  Occam_rows={st["occam_rows"]}')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_fn(f'  ✗ {lt}: 异常: {e}\n{tb}')
            summary[lt] = {'error': str(e)}
    log_fn(f'\n{"="*80}\n  SUMMARY\n{"="*80}')
    for lt, st in summary.items():
        log_fn(f'  {lt}: {st}')
    sum_path = out_root / f'preprocess_report_{ts}.json'
    sum_path.write_text(json.dumps({
        'timestamp': ts, 'project_root': str(project_root),
        'cfg': cfg, 'summary': summary,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    log_fn(f'📄 最终报告 → {sum_path}')
    return summary
