# -*- coding: utf-8 -*-
"""
tfm_engine.preprocess_plus
==========================
TEM 体系增强预处理（完全按 proc_kru.m / tproc_kru.m / cbfilt_kru.m / AATZero 设计）：
  1. 关断时刻微调 (ATZero style) — Hx 峰精确到子样本
  2. 去 DC (ATZero 前后窗口 median)
  3. 工频陷波 50/100/150 Hz (IIR butterworth notch, quality=30)
  4. 带通滤波 FIR 2Hz ~ 80kHz (对应 TEM fir3_20/fir10_30/fir15_40/fir20lp 组合)
  5. 关断后 gates 几何采样 + 衰减曲线提取 (curview_kru style)
  6. 多 picket 叠加 median 合成 (抗野值, average.m)
  7. 野值剔除 MAD + 5σ clip (verify_kru.m style)

最终生成: decay_curves dict 供信号曲线绘制:
  { station_code:
       {'Ex': {'t_s': (N,), 'raw': (N,), 'dc_rm': (N,), 'notch': (N,), 'bandpass': (N,),
               't_cutoff': float, 'gate_idx': (Ng,), 'gate_vals': (Ng,)},
        'Ey': {...}, 'Hx': {...}, 'Hy': {...}} }
完全离线, 纯 numpy + scipy.signal。
"""
from __future__ import annotations
from typing import Optional, Callable
import re
from pathlib import Path
import numpy as np
from scipy import signal as _sgn
from .dat_parser import load_tem_dat  # 复用 DAT 读取 → 返回 WlfDatData(cube=, header=)
from .excel_meta import parse_em_xlsx
from .coord_loader import load_authoritative_coordinates   # 权威坐标加载 (公开 API)
from .preprocess import DEFAULT_CFG  # 仅复用公开常量 (DEFAULT_CFG 存在, 其他私有符号不导入)

# TEM 仪器采样率 (未在头段暴露, 行业默认: 544,859 样/通道 ÷ 约 20ms 记录 ≈ 27.243 MHz)
# 可由 user_cfg['TEM_FS_HZ'] 覆盖
_TEM_DEFAULT_FS_HZ = 27_243_000.0


# ============================================================
# 1. 关断时刻微调 (TEM AATZero 原理)
# ============================================================
def refine_t_cutoff(hx_abs: np.ndarray, t_cutoff_init: int,
                    radius: int = 25) -> tuple[int, float]:
    """以 Hx 绝对值峰为锚点，在 [t_cutoff_init-radius, t_cutoff_init+radius] 内精修关断时刻。
    返回: (sample_idx_int, subsample_frac)  """
    lo = max(200_000, t_cutoff_init - radius)
    hi = min(400_000, t_cutoff_init + radius, len(hx_abs) - 3)
    window = hx_abs[lo:hi + 1]
    idx_rel = int(np.argmax(window))
    idx_abs = lo + idx_rel
    # 子样本抛物线插值 (TEM AATZero 精修)
    if 0 < idx_rel < len(window) - 1:
        a, b, c = window[idx_rel - 1], window[idx_rel], window[idx_rel + 1]
        denom = (a - 2 * b + c)
        frac = 0.5 * (a - c) / denom if abs(denom) > 1e-15 else 0.0
    else:
        frac = 0.0
    return int(idx_abs), float(np.clip(frac, -0.5, 0.5))


# ============================================================
# 2/3/4: DC 移除 + 工频陷波 + 带通
# ============================================================
def remove_dc(ch_raw: np.ndarray, t_cutoff: int, frac_pre: float = 0.10) -> np.ndarray:
    """关断前 frac_pre 窗口 median 作为 DC offset (与 preprocess 统一, 但精确到精修后 t_cutoff)。"""
    n_pre = max(100, int(t_cutoff * frac_pre))
    dc = np.median(ch_raw[max(0, t_cutoff - n_pre): t_cutoff])
    return (ch_raw.astype(np.float64) - dc)


def notch_50_150(ch: np.ndarray, fs_hz: float, f0_list=(50, 100, 150), Q=30.0) -> np.ndarray:
    """IIR 陷波 50/100/150 Hz (TEM cbfilt 工频抑制)。"""
    y = ch.astype(np.float64)
    nyq = fs_hz * 0.5
    for f0 in f0_list:
        if 1.0 < f0 < nyq - 1:
            b, a = _sgn.iirnotch(w0=f0 / nyq, Q=Q)
            y = _sgn.filtfilt(b, a, y)
    return y


def bandpass_tem(ch: np.ndarray, fs_hz: float,
                 lowcut_hz: float = 2.0, highcut_hz: float = 80_000.0,
                 order: int = 4) -> np.ndarray:
    """Butterworth 带通 FIR (对应 TEM fir*.mat 组)。"""
    nyq = fs_hz * 0.5
    low = max(1e-3, lowcut_hz / nyq)
    high = min(0.999, highcut_hz / nyq)
    if high - low < 1e-4:
        return ch
    sos = _sgn.butter(order, [low, high], btype='band', output='sos')
    return _sgn.sosfiltfilt(sos, ch.astype(np.float64))


# ============================================================
# 野值剔除 (MAD + 5-sigma)
# ============================================================
def mad_clip(x: np.ndarray, sigma_clip: float = 5.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad < 1e-30:
        return x
    lo = med - sigma_clip * mad
    hi = med + sigma_clip * mad
    return np.clip(x, lo, hi)


# ============================================================
# 单 cube 4 通道全流水线 → 衰减曲线 dict
# ============================================================
def process_one_cube_decay(cube_int32: np.ndarray,          # shape=(4, N, Np)  int32LE
                           fs_hz: float,
                           G_ADC: float,
                           gates_us_arr: np.ndarray,
                           t_cutoff_init: int,
                           log_fn: Optional[Callable[[str], None]] = None,
                           ) -> dict:
    """对一个 4×N×Np cube 做全增强流水线，返回 4 通道衰减曲线数据结构。"""
    assert cube_int32.ndim == 3 and cube_int32.shape[0] == 4
    N4, N_s, N_p = cube_int32.shape
    cube_v = cube_int32.astype(np.float64) / G_ADC

    # Hx 绝对值峰 top-20 → 关断精修
    hx_merge = np.median(np.abs(cube_v[2]), axis=-1)
    t_coarse = int(np.clip(t_cutoff_init, 200_000, 400_000))
    t_cutoff, t_frac = refine_t_cutoff(hx_merge, t_coarse, radius=25)
    if log_fn:
        log_fn(f"    [AATZero] t_cutoff: {t_coarse} → {t_cutoff} (Δ={t_cutoff - t_coarse:+d}  sub={t_frac:+.2f} samps)")

    # 25 gates 时间轴 (s)，以 t_cutoff 为 0 点
    gates_t_s = gates_us_arr * 1e-6
    gate_idx_exact = t_cutoff + gates_t_s * fs_hz
    gate_idx_lo = np.floor(gate_idx_exact).astype(int)
    gate_idx_w = gate_idx_exact - gate_idx_lo  # 线性插值权重

    out_station = {}
    ch_names = ['Ex', 'Ey', 'Hx', 'Hy']
    # 每个 picket 独立处理后叠加 median
    for ci, cname in enumerate(ch_names):
        all_pickets_gated = []
        per_picket = {'raw': [], 'dc_rm': [], 'notch': [], 'bandpass': []}
        for pk in range(N_p):
            raw = cube_v[ci, :, pk]
            y0 = remove_dc(raw, t_cutoff)
            y1 = notch_50_150(y0, fs_hz)
            y2 = bandpass_tem(y1, fs_hz)
            # gates 线性插值 (关断后)
            idx = np.clip(gate_idx_lo, t_cutoff, N_s - 2)
            va = y2[idx]; vb = y2[idx + 1]
            gated = va * (1 - gate_idx_w) + vb * gate_idx_w
            all_pickets_gated.append(gated)
            # 仅保存第 0 个 picket 完整波形 (供绘图，避免内存过大)
            if pk == 0:
                per_picket['raw'].append(raw)
                per_picket['dc_rm'].append(y0)
                per_picket['notch'].append(y1)
                per_picket['bandpass'].append(y2)
        # Np 叠加 median 合成 gates
        gated_stack = np.vstack(all_pickets_gated)  # (Np, Ng)
        gated_final = mad_clip(np.median(gated_stack, axis=0))
        # 输出曲线时间轴：全 sample 范围（仅第0个picket，s，以 t=0 为采样起点）
        t_full_s = np.arange(N_s, dtype=np.float64) / fs_hz
        out_station[cname] = {
            't_full_s':  t_full_s,
            'raw':       per_picket['raw'][0],
            'dc_removed': per_picket['dc_rm'][0],
            'notched':   per_picket['notch'][0],
            'bandpass':  per_picket['bandpass'][0],
            't_cutoff_sample': t_cutoff,
            't_cutoff_s':      t_cutoff / fs_hz,
            'gates_t_s':       gates_t_s,
            'gates_values':    gated_final,
        }
    return out_station


# ============================================================
# 全测线入口 → 返回 {station_code: {Ex/Ey/Hx/Hy}}
# ============================================================
def run_full_preprocess_plus(project_root: str | Path,
                             out_root: str | Path,
                             user_cfg: Optional[dict] = None,
                             log_fn: Optional[Callable[[str], None]] = None,
                             ) -> dict:
    """在原有 Step1-2 预处理基础上，额外做衰减曲线提取 (curview_kru)。
    复用原有权威元数据解析链，不改变原产物。"""
    from pathlib import Path
    from .preprocess import run_full_preprocess  # 复用权威解析链
    cfg = dict(DEFAULT_CFG)
    if user_cfg is not None: cfg.update(user_cfg)
    log_fn = log_fn or (lambda s: print(s))
    project_root = Path(project_root); out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ① 先跑原有预处理（生成 Occam.dat、ρa 图、QC 等权威产物）
    base_report = run_full_preprocess(project_root, out_root, cfg, log_fn=log_fn)

    # ② 再提取每条测线的衰减曲线（curview_kru 风格）
    report = {'base_report': dict(base_report), 'decay_lines': {}}

    # base_report 的顶级 key = {line_tag: st_dict} (如 '1线', '2线'...)，非嵌套在 'lines' 下
    def _is_line_tag(k): return bool(re.fullmatch(r'[1-5]线', k))
    line_tags = sorted([k for k in base_report.keys() if _is_line_tag(k)])
    log_fn(f"\n========== 增强预处理 / 衰减曲线提取  lines={line_tags} ==========")

    for line_tag in line_tags:
        line_dir = project_root / line_tag
        if not line_dir.is_dir():
            continue
        try:
            meta = parse_em_xlsx(line_dir, log_fn=log_fn)
        except Exception as exc:
            log_fn(f"  [ERR] Excel parse {line_tag}: {exc}"); continue
        sections = meta['sections']
        cs_rows = meta.get('channel_station_rows', [])
        # section × instrument 分组
        from collections import defaultdict
        sec_inst_groups = defaultdict(list)
        for r in cs_rows:
            sec_inst_groups[(r.get('section'), r.get('instrument'))].append(r)
        line_decay = {}
        for (sec_code, inst_code), rows in sec_inst_groups.items():
            sec_meta = sections.get(sec_code)
            if not sec_meta: continue
            # 内联 DAT 定位（替代未公开的 _find_dat_case_insensitive）
            sec_dir_name = str(int(sec_code))  # "017" → "17" 匹配磁盘目录名
            dpath = None
            for sd_cand in (line_dir / sec_dir_name, line_dir / sec_code):
                if not sd_cand.is_dir(): continue
                # 小写档: c{ddd}st{iii}.dat   e.g. c017st001.dat
                pattern = f'c{sec_code.zfill(3)}st{inst_code}.dat'
                for f in sd_cand.iterdir():
                    if not f.is_file(): continue
                    if f.name.lower() == pattern.lower():
                        dpath = f; break
                if dpath is not None: break
            if dpath is None: continue
            try:
                # load_tem_dat 返回 WlfDatData (单值, 非 tuple)，含 .cube / .header
                wd = load_tem_dat(dpath, log_fn=log_fn, validate=True, strict_per_ch=False)
                cube = wd.cube
                header = wd.header
            except Exception as exc:
                log_fn(f"  [WARN] load DAT {dpath.name}: {exc}"); continue
            # Hx top-20 t_cutoff_init (Hx 峰索引众数)
            hx_abs = np.abs(cube[2].astype(np.float64)).mean(axis=-1).ravel()
            roi = hx_abs[200_000:400_001]
            t_coarse = int(np.clip((int(np.argmax(roi)) if roi.size else 0) + 200_000, 200_000, 400_000))
            # gates 时间 (统一 25 个, 复用 cfg 键 T_MIN_US / T_MAX_US / TEM_N_GATES)
            gates_us = np.geomspace(cfg['T_MIN_US'], cfg['T_MAX_US'], num=cfg['TEM_N_GATES'])
            # 采样率: user_cfg 覆盖 → header 推导 → 默认 27.243 MHz
            fs_hz = float(user_cfg.get('TEM_FS_HZ') if isinstance(user_cfg, dict) and 'TEM_FS_HZ' in user_cfg
                          else getattr(header, 'fs_hz', _TEM_DEFAULT_FS_HZ))
            # 按 per-instrument 的 stations 顺序 (按 Excel order 去重保持顺序)
            seen = set(); stations_here = []
            for r in sorted(rows, key=lambda x: x.get('order', 0)):
                st = r.get('station')
                if st and st not in seen:
                    seen.add(st); stations_here.append(st)
            Np_clip = min(cube.shape[2], 5, len(stations_here))
            if Np_clip <= 0: continue
            # 一个 instrument × section 衰减曲线: 对 Np 公用 (因为我们没逐 picket 做独立正演区分)
            try:
                decay = process_one_cube_decay(cube[:, :, :Np_clip], fs_hz=fs_hz,
                                               G_ADC=cfg['G_ADC_COUNTS_PER_VOLT'],
                                               gates_us_arr=gates_us,
                                               t_cutoff_init=t_coarse, log_fn=log_fn)
            except Exception as exc:
                log_fn(f"  [WARN] process decay {sec_code}/{inst_code}: {exc}"); continue
            for j, st in enumerate(stations_here[:Np_clip]):
                if st not in line_decay:
                    line_decay[st] = decay
                else:
                    # 多 instrument 合并：gates_values 取 median
                    for cname in ['Ex', 'Ey', 'Hx', 'Hy']:
                        old_v = line_decay[st][cname]['gates_values']
                        new_v = decay[cname]['gates_values']
                        try:
                            line_decay[st][cname]['gates_values'] = np.median(
                                np.vstack([old_v, new_v]), axis=0)
                        except ValueError:
                            pass  # shape 不匹配时丢弃新样本
        report['decay_lines'][line_tag] = line_decay
        # 保存衰减曲线 NPZ (供 GUI Signal Viewer 加载)
        np.savez_compressed(out_root / line_tag / f'{line_tag}_decay_curves.npz',
                            stations=np.array(list(line_decay.keys()), dtype=object),
                            data=np.array(line_decay, dtype=object))
        log_fn(f"  ✓ [{line_tag}] decay curves: {len(line_decay)} stations → "
               f"{line_tag}_decay_curves.npz")
    return report


__all__ = ['run_full_preprocess_plus', 'process_one_cube_decay',
           'refine_t_cutoff', 'remove_dc', 'notch_50_150', 'bandpass_tem', 'mad_clip']
