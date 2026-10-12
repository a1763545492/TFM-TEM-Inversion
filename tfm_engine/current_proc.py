# -*- coding: utf-8 -*-
"""
tfm_engine.current_proc (M8)
============================
WLF 体系 发射电流数据处理模块（对应 WLF 大写校准档 CXXST01.DAT 的电流支路）：
  - 逐 section 读取发射电流档 (大写 DAT, Np 桩 × 4 通道 × 544859 样点)
  - 自动识别电流监测通道 (平台幅值最大的通道)
  - 提取: 平台电流幅值 (counts/V)、关断时刻 t_off、平台纹波 %、通断占空
  - 与 Excel 权威电流 I_tx (发射时间 sheet, A) 对照 QC
  - 输出: 每测线 {X线}_current_summary.csv + {X线}_current_waveforms.png

物理说明:
  电流档由发射机同步记录, 幅值以 ADC counts 为单位; 换算 V 用 G_ADC=2^22 (与 preprocess 统一).
  真正的电流安培值以 Excel「发射时间」sheet 的 I_tx 为权威 (单一数据源契约),
  本模块提供波形级 QC: 平台是否稳定、关断是否干净、各桩是否一致。

完全离线, 纯 numpy + matplotlib(Agg)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from .dat_parser import load_wlf_dat, PER_CHANNEL
from .preprocess import run_inventory, DEFAULT_CFG

_WLF_DEFAULT_FS_HZ = 27_243_000.0   # 与 preprocess_plus 统一 (544859 样 / ~20ms)
_CH_NAMES = ['Ex', 'Ey', 'Hx', 'Hy']


@dataclass
class CurrentQC:
    """单 section 电流档 QC 结果"""
    line_tag: str
    section: str
    file_name: str
    n_pickets: int
    channel_used: str
    amplitude_counts: float      # 平台幅值 (DC去除后, counts)
    amplitude_v: float           # 换算 V (G_ADC)
    ripple_pct: float            # 平台纹波 std/幅值 %
    t_off_ms: Optional[float]    # 关断时刻 (相对记录起点, ms); None=未检测到
    duty_pct: float              # 导通占空 %
    i_tx_excel_a: float          # Excel 权威电流 (A)
    qc_ok: bool
    qc_notes: str = ""


# ============================================================
# 1. 单通道波形分析: 平台 / 关断 / 纹波
# ============================================================
def analyze_current_waveform(wave: np.ndarray, fs_hz: float) -> dict:
    """wave: 1D float counts (已多桩 median 合成). 返回平台/关断/纹波指标."""
    n = wave.size
    w0 = wave - np.median(wave)                       # 去 DC
    env = np.abs(w0)
    # 平滑包络 (窗口≈0.1ms, 至少 31 点, 奇数)
    win = max(31, int(fs_hz * 1e-4) | 1)
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    kern = np.ones(win) / win
    env_s = np.convolve(env, kern, mode='same')
    peak = float(env_s.max())
    if peak <= 0:
        return {'amplitude': 0.0, 'ripple_pct': np.nan, 't_off_s': None, 'duty_pct': 0.0}
    on_mask = env_s > 0.5 * peak
    duty = float(on_mask.mean() * 100.0)
    # 纹波: 双极性方波分正/负半周分别统计 (否则 std≈幅值, 指标失真)
    w_pos = w0[on_mask & (w0 > 0)]
    w_neg = w0[on_mask & (w0 < 0)]
    med_pos = float(np.median(w_pos)) if w_pos.size > 10 else 0.0
    med_neg = float(np.median(w_neg)) if w_neg.size > 10 else 0.0
    if med_pos > 0.3 * peak and abs(med_neg) > 0.3 * peak:
        # 双极性: 幅值 = 两半周 |中值| 平均, 纹波 = 两半周相对std 平均
        plateau = (med_pos + abs(med_neg)) / 2.0
        ripple = 0.5 * (float(np.std(w_pos)) / med_pos
                        + float(np.std(w_neg)) / abs(med_neg)) * 100.0
    else:
        plateau = float(np.median(env[on_mask])) if on_mask.any() else 0.0
        ripple = float(np.std(w0[on_mask]) / plateau * 100.0) if plateau > 0 else np.nan
    # 关断: 最后一个 on→off 跳变沿
    idx_on = np.flatnonzero(on_mask)
    t_off_s = None
    if idx_on.size and idx_on[-1] < n - 2:
        t_off_s = float(idx_on[-1]) / fs_hz
    return {'amplitude': plateau, 'ripple_pct': ripple, 't_off_s': t_off_s, 'duty_pct': duty}


def process_one_tx_dat(dat_path: Path, line_tag: str, section: str,
                       i_tx_excel: float, cfg: dict,
                       log_fn: Callable[[str], None],
                       force_channel: Optional[str] = None) -> tuple[CurrentQC, dict]:
    """读 1 个大写电流档 → CurrentQC + 波形数据 (供绘图).
    force_channel: 强制指定电流监测通道 ('Ex'/'Ey'/'Hx'/'Hy'), None=自动选幅值最大者."""
    fs = float(cfg.get('TEM_FS_HZ', _WLF_DEFAULT_FS_HZ))
    g_adc = float(cfg.get('G_ADC_COUNTS_PER_VOLT', DEFAULT_CFG['G_ADC_COUNTS_PER_VOLT']))
    d = load_wlf_dat(dat_path, log_fn=log_fn)
    cube = d.cube.astype(np.float64)                 # (4, 544859, Np)
    med_wave = np.median(cube, axis=2)               # (4, 544859) 多桩 median 抗野值

    # 选通道: 平台幅值最大者 = 电流监测通道 (或强制指定)
    best = None
    per_ch = {}
    for ci, chn in enumerate(_CH_NAMES):
        a = analyze_current_waveform(med_wave[ci], fs)
        per_ch[chn] = {'wave': med_wave[ci], **a}
        if best is None or a['amplitude'] > best[1]['amplitude']:
            best = (chn, a)
    if force_channel and force_channel in _CH_NAMES:
        ch_used = force_channel
        a = per_ch[ch_used]
    else:
        ch_used, a = best

    # QC 判定
    notes = []
    ok = True
    if a['amplitude'] <= 0:
        ok = False; notes.append('无平台幅值')
    if a['t_off_s'] is None:
        ok = False; notes.append('未检测到关断沿')
    if np.isfinite(a['ripple_pct']) and a['ripple_pct'] > 15.0:
        notes.append(f'纹波偏大 {a["ripple_pct"]:.1f}%')
    # 多桩一致性: 每桩幅值离散度
    pk_amp = [analyze_current_waveform(cube[list(_CH_NAMES).index(ch_used), :, p], fs)['amplitude']
              for p in range(d.n_pickets)]
    pk_amp = np.array([x for x in pk_amp if x > 0])
    if pk_amp.size >= 2:
        spread = float(np.std(pk_amp) / np.mean(pk_amp) * 100.0)
        if spread > 10.0:
            notes.append(f'桩间幅值离散 {spread:.1f}%')
    qc = CurrentQC(
        line_tag=line_tag, section=section, file_name=dat_path.name,
        n_pickets=d.n_pickets, channel_used=ch_used,
        amplitude_counts=a['amplitude'], amplitude_v=a['amplitude'] / g_adc,
        ripple_pct=a['ripple_pct'],
        t_off_ms=None if a['t_off_s'] is None else a['t_off_s'] * 1e3,
        duty_pct=a['duty_pct'], i_tx_excel_a=i_tx_excel,
        qc_ok=ok, qc_notes='; '.join(notes),
    )
    return qc, {'fs': fs, 'per_ch': per_ch, 'ch_used': ch_used}


# ============================================================
# 2. 测线级: 全 section 处理 + CSV + 波形 PNG
# ============================================================
def process_line_current(line_tag: str, inv: dict, out_dir: Path,
                         cfg: dict, log_fn: Callable[[str], None],
                         force_channel_map: Optional[dict] = None) -> pd.DataFrame:
    """force_channel_map: {section: 'Ex'} 逐 section 强制电流通道."""
    meta = inv['lines'][line_tag]['meta']
    secs_disk = inv['lines'][line_tag]['sections_disk']
    out_base = out_dir / line_tag
    out_base.mkdir(parents=True, exist_ok=True)

    qcs, waves = [], {}
    for sec in sorted(secs_disk.keys()):
        tx_dat = secs_disk[sec].get('tx_dat')
        if tx_dat is None:
            log_fn(f'  [SKIP] {line_tag} sec{sec}: 无大写电流档')
            continue
        i_tx = float(meta['sections'].get(sec, {}).get('i_tx_amps', 0.0))
        fch = (force_channel_map or {}).get(sec)
        try:
            qc, wv = process_one_tx_dat(tx_dat, line_tag, sec, i_tx, cfg, log_fn,
                                        force_channel=fch)
        except Exception as e:
            log_fn(f'  [ERR] {line_tag} sec{sec} {tx_dat.name}: {e}')
            continue
        qcs.append(qc)
        waves[sec] = wv
        log_fn(f'  sec{sec}: ch={qc.channel_used} 幅值={qc.amplitude_counts:.3e}counts '
               f'({qc.amplitude_v:.3f}V) t_off={qc.t_off_ms and f"{qc.t_off_ms:.2f}ms"} '
               f'纹波={qc.ripple_pct:.2f}% I_tx_excel={qc.i_tx_excel_a:.1f}A '
               f'{"OK" if qc.qc_ok else "QC!"}{(" " + qc.qc_notes) if qc.qc_notes else ""}')
    if not qcs:
        return pd.DataFrame()

    df = pd.DataFrame([vars(q) for q in qcs])
    csv_p = out_base / f'{line_tag}_current_summary.csv'
    df.to_csv(csv_p, index=False, encoding='utf-8-sig')
    log_fn(f'  [OUT] 电流汇总 → {csv_p}')

    # 波形 PNG: 每 section 1 行 (全记录 + 关断邻域放大)
    nsec = len(waves)
    fig, axes = plt.subplots(nsec, 2, figsize=(13, 2.4 * nsec), squeeze=False)
    for r, (sec, wv) in enumerate(sorted(waves.items())):
        ch = wv['ch_used']
        w = wv['per_ch'][ch]['wave']
        fs = wv['fs']
        t_ms = np.arange(w.size) / fs * 1e3
        a = wv['per_ch'][ch]
        ax = axes[r][0]
        ax.plot(t_ms, w - np.median(w), lw=0.5, color='tab:blue')
        if a['t_off_s'] is not None:
            ax.axvline(a['t_off_s'] * 1e3, color='r', ls='--', lw=1, label='t_off')
            ax.legend(fontsize=8)
        ax.set_ylabel(f'sec{sec}\n{ch} counts', fontsize=9)
        ax.set_title(f'{line_tag} 发射电流波形 (全记录)', fontsize=10, loc='left')
        ax.grid(alpha=0.3)
        # 关断邻域放大
        ax2 = axes[r][1]
        if a['t_off_s'] is not None:
            c = int(a['t_off_s'] * fs)
            hw = int(0.002 * fs)                      # ±2ms
            sl = slice(max(0, c - hw), min(w.size, c + hw))
            ax2.plot(t_ms[sl] - a['t_off_s'] * 1e3, (w - np.median(w))[sl], lw=0.7, color='tab:red')
            ax2.axvline(0, color='k', ls='--', lw=1)
            ax2.set_title('关断沿 ±2ms', fontsize=10)
            ax2.set_xlabel('t - t_off (ms)', fontsize=9)
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '无关断沿', ha='center', va='center', transform=ax2.transAxes)
    fig.tight_layout()
    png = out_base / f'{line_tag}_current_waveforms.png'
    fig.savefig(png, dpi=160)
    plt.close(fig)
    log_fn(f'  [OUT] 电流波形图 → {png}')
    return df


def run_all_current(project_root: str | Path, out_dir: str | Path,
                    user_cfg: Optional[dict] = None,
                    log_fn: Optional[Callable[[str], None]] = None,
                    force_channel_map: Optional[dict] = None) -> dict:
    """全流程入口: 4 测线电流档处理.
    force_channel_map: {'1线': {'027': 'Ex'}, ...} 按 测线→section 强制电流通道."""
    log_fn = log_fn or (lambda s: print(s))
    cfg = dict(DEFAULT_CFG)
    if user_cfg:
        cfg.update(user_cfg)
    project_root = Path(project_root)
    out_dir = Path(out_dir)
    # 持久化通道覆盖配置: <out_dir>/current_channel_override.json
    if force_channel_map is None:
        import json as _json
        ov = out_dir / 'current_channel_override.json'
        if ov.is_file():
            try:
                force_channel_map = _json.loads(ov.read_text(encoding='utf-8'))
                log_fn(f'  [CFG] 加载通道覆盖: {ov}')
            except Exception:
                pass
    log_fn('\n========== 发射电流数据处理 (current_proc, 双极性纹波修正) ==========')
    inv = run_inventory(project_root, log_fn)
    results = {}
    for line_tag in sorted(inv['lines'].keys()):
        log_fn(f'\n--- {line_tag} ---')
        df = process_line_current(line_tag, inv, out_dir, cfg, log_fn,
                                  force_channel_map=(force_channel_map or {}).get(line_tag))
        results[line_tag] = {'sections': len(df),
                             'qc_ok': int(df.qc_ok.sum()) if len(df) else 0}
    return results
