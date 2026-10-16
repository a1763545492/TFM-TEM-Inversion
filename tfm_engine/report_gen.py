# -*- coding: utf-8 -*-
"""
report_gen.py — 自动化 HTML 报告生成器
================================================================
一键汇总输出根目录下的全流程产物, 生成**单文件自包含** HTML 报告
(图片 base64 内嵌, 可离线打开/邮件发送, 不依赖任何外部文件/网络):

  ① 概览: 时间戳、项目/输出路径、关键参数、各线 QC 指标卡
  ② 预处理 QC: ρa 剖面 PNG、测点统计表、ρa 全项目分位数
  ③ 反演断面: 各算法断面 PNG (按算法分组)
  ④ 多算法对比: compare_sections / fits / rmse + all_lines_compare_stats
  ⑤ LCI 拟2D、MCMC 后验等其他成果图
  ⑥ 参数快照 cfg_snapshot_*.json

各章节按产物实际存在情况动态生成, 无该类产物则自动跳过。
"""
from __future__ import annotations
import base64
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ALGO_LABELS = {
    'trf': 'TRF', 'lm': 'LM', 'occam': 'Occam',
    'occam_adaptive': 'Occam自适应', 'lm_adaptive': 'LM自适应',
    'cgls': 'CGLS', 'mcmc': 'MCMC', 'lci': 'LCI横向约束',
}

CSS = """
body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:0;background:#f4f6fa;color:#222}
h1{color:#1a3a6b;margin:0 0 4px}
h2{color:#1a3a6b;border-left:5px solid #2f6db3;padding-left:10px;margin-top:36px}
h3{color:#33507a;margin:18px 0 6px}
.wrap{max-width:1180px;margin:0 auto;padding:24px 30px 60px;background:#fff;
     box-shadow:0 0 14px rgba(0,0,0,.08)}
.meta{color:#666;font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.card{background:#eef4fc;border:1px solid #cdddf2;border-radius:8px;padding:10px 16px;min-width:150px}
.card .k{font-size:12px;color:#5a7398}
.card .v{font-size:20px;font-weight:bold;color:#1a3a6b}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 18px}
th,td{border:1px solid #d4dbe6;padding:5px 9px;text-align:center}
th{background:#2f6db3;color:#fff}
tr:nth-child(even){background:#f5f8fd}
img{max-width:100%;border:1px solid #d4dbe6;border-radius:6px;margin:6px 0 18px;
    cursor:zoom-in;background:#fff}
pre{background:#1e2530;color:#dfe7f3;padding:12px;border-radius:6px;
   overflow:auto;font-size:12px;max-height:360px}
.toc a{color:#2f6db3;text-decoration:none;margin-right:16px;white-space:nowrap}
.badge{display:inline-block;background:#e8f5e9;color:#1b6b2a;border-radius:10px;
       padding:1px 10px;font-size:12px;margin-left:8px}
.warn{background:#fff7e6;border-left:4px solid #f0a020;padding:8px 12px;margin:10px 0}
.modal{display:none;position:fixed;z-index:99;left:0;top:0;width:100%;height:100%;
       background:rgba(0,0,0,.82)}
.modal img{max-width:94%;max-height:94%;margin:2% 3%;border:none;cursor:zoom-out}
"""

JS = """
document.addEventListener('click',function(e){
  if(e.target.tagName==='IMG'){
    let m=document.getElementById('im');
    m.style.display='block';m.firstChild.src=e.target.src;
  }
});
function closeM(){document.getElementById('im').style.display='none'}
"""


def _img64(p: Path) -> str:
    mime = 'png' if p.suffix.lower() == '.png' else 'jpeg'
    b = base64.b64encode(p.read_bytes()).decode('ascii')
    return f'data:image/{mime};base64,{b}'


def _df_to_html(df: pd.DataFrame, float_fmt: str = '{:.4f}') -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else float_fmt.format(v))
    return df.to_html(index=False, escape=True, border=0)


def _latest_glob(root: Path, pattern: str) -> Optional[Path]:
    files = sorted(root.glob(pattern))
    return files[-1] if files else None


def _line_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()
                   and re.fullmatch(r'[1-5]线', p.name)],
                  key=lambda p: p.name)


def _section_pngs(line_dir: Path) -> dict[str, Path]:
    """该测线下所有反演断面图 {algo/name: path}。"""
    out = {}
    basic = line_dir / f'{line_dir.name}_1d_inversion_section.png'
    if basic.is_file():
        out['basic'] = basic
    for algo, label in ALGO_LABELS.items():
        p = line_dir / f'{line_dir.name}_inv_{algo}_section.png'
        if p.is_file():
            out[algo] = p
    return out


def _gather_preprocess(root: Path) -> tuple[Optional[dict], pd.DataFrame]:
    """读最新 preprocess_report_*.json + 汇总 pickup CSV。"""
    rep = _latest_glob(root, 'preprocess_report_*.json')
    pre_meta = None
    if rep:
        try:
            pre_meta = json.loads(rep.read_text(encoding='utf-8'))
        except Exception:
            pre_meta = None
    rows = []
    for ld in _line_dirs(root):
        f = ld / f'{ld.name}_pickup_xyz.csv'
        if f.is_file():
            d = pd.read_csv(f)
            rows.append(pd.DataFrame({
                '测线': ld.name,
                '测站数': [len(d)],
                '剔除站': [int(d['qc_drop'].astype(bool).sum())
                         if 'qc_drop' in d else 0],
                '死道站': [int(d['qc_dead'].astype(bool).sum())
                         if 'qc_dead' in d else 0],
                'ρa中位(Ω·m)': [pd.to_numeric(d.get('rhoa_median_Ohm'),
                                        errors='coerce').median()],
            }))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pre_meta, df


def _gather_inversion_stats(root: Path) -> pd.DataFrame:
    """聚合所有算法 summary CSV → 全线×算法 RMSE 表。"""
    rows = []
    for ld in _line_dirs(root):
        for algo in ALGO_LABELS:
            f = ld / f'{ld.name}_inv_{algo}_summary.csv'
            if not f.is_file():
                continue
            d = pd.read_csv(f)
            v = pd.to_numeric(d.get('rmse_log'), errors='coerce').dropna()
            if not len(v):
                continue
            succ = pd.to_numeric(d.get('success'), errors='coerce')
            rows.append({
                '测线': ld.name, '算法': ALGO_LABELS[algo],
                '站数': len(d),
                'RMSE中位': float(v.median()), 'RMSE均值': float(v.mean()),
                'RMSE_P10': float(v.quantile(0.1)),
                'RMSE_P90': float(v.quantile(0.9)),
                '成功率': float(succ.fillna(1).mean()) if len(succ) else np.nan,
            })
    return pd.DataFrame(rows)


def generate_report(out_root: str | Path,
                    html_path: Optional[str | Path] = None,
                    title: str = 'TFM 时频电磁法反演报告',
                    log_fn: Optional[Callable[[str], None]] = None) -> Path:
    """生成单文件 HTML 报告, 返回路径。"""
    log_fn = log_fn or (lambda s: print(s))
    root = Path(out_root)
    line_dirs = _line_dirs(root)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_path = Path(html_path) if html_path else \
        root / f'tfm_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'

    parts = []
    parts.append(f'<h1>{html.escape(title)}</h1>')
    parts.append(f'<div class="meta">生成时间: {ts} ｜ 输出目录: '
                 f'{html.escape(str(root))} ｜ 测线数: {len(line_dirs)} ｜ '
                 '完全离线生成</div>')

    # ---- 目录 ----
    toc = ['<div class="toc"><b>目录:</b>']
    sections = [('sec_pre', '① 预处理 QC'), ('sec_inv', '② 反演断面'),
                ('sec_cmp', '③ 多算法对比'), ('sec_other', '④ 其他成果'),
                ('sec_param', '⑤ 参数快照')]
    toc += [f'<a href="#{i}">{n}</a>' for i, n in sections]
    toc.append('</div>')
    parts.append(''.join(toc))

    # ---- ① 预处理 ----
    pre_meta, pre_df = _gather_preprocess(root)
    cards = []
    if pre_meta and pre_df is not None and len(pre_df):
        total_st = int(pre_df['测站数'].sum())
        drop_st = int(pre_df['剔除站'].sum())
        med = pd.to_numeric(pre_df['ρa中位(Ω·m)'], errors='coerce').median()
        cards += [
            ('测线数', len(line_dirs)), ('测站总数', total_st),
            ('QC剔除', drop_st),
            ('ρa中位', f'{med:.1f}' if np.isfinite(med) else '-'),
        ]
    if not cards:
        # 无预处理报告时至少给测线卡片
        cards = [('测线数', len(line_dirs))]
    parts.append('<h2 id="sec_pre">① 预处理 QC</h2>')
    parts.append('<div class="cards">' + ''.join(
        f'<div class="card"><div class="k">{html.escape(str(k))}</div>'
        f'<div class="v">{html.escape(str(v))}</div></div>'
        for k, v in cards) + '</div>')
    if len(pre_df):
        parts.append(_df_to_html(pre_df, '{:.2f}'))
    for ld in line_dirs:
        png = ld / f'{ld.name}_rhoa_section.png'
        if png.is_file():
            parts.append(f'<h3>{ld.name} 视电阻率剖面</h3>')
            parts.append(f'<img src="{_img64(png)}" alt="{ld.name} rhoa">')
    if pre_meta is None and not len(pre_df):
        parts.append('<div class="warn">未发现预处理产物 (preprocess_report / '
                     'pickup_xyz / rhoa_section)。</div>')

    # ---- ② 反演断面 ----
    parts.append('<h2 id="sec_inv">② 反演断面</h2>')
    inv_stats = _gather_inversion_stats(root)
    if len(inv_stats):
        parts.append(_df_to_html(inv_stats, '{:.4f}'))
    any_sec = False
    for ld in line_dirs:
        pngs = _section_pngs(ld)
        for algo, png in pngs.items():
            any_sec = True
            label = ALGO_LABELS.get(algo, '基础1D反演')
            parts.append(f'<h3>{ld.name} · {html.escape(label)}</h3>')
            parts.append(f'<img src="{_img64(png)}">')
    if not any_sec:
        parts.append('<div class="warn">未发现反演断面图。</div>')

    # ---- ③ 多算法对比 ----
    parts.append('<h2 id="sec_cmp">③ 多算法对比 QC</h2>')
    cmp_csv = root / 'all_lines_compare_stats.csv'
    if cmp_csv.is_file():
        parts.append('<h3>全线算法 RMSE 总表</h3>')
        parts.append(_df_to_html(pd.read_csv(cmp_csv), '{:.4f}'))
    for ld in line_dirs:
        block = []
        for key, name in [('sections', '并排断面'), ('rmse', '逐站RMSE'),
                          ('fits', '逐站拟合曲线')]:
            png = ld / f'{ld.name}_compare_{key}.png'
            if png.is_file():
                block.append(f'<h3>{ld.name} · {name}</h3>'
                             f'<img src="{_img64(png)}">')
        if block:
            parts.append(''.join(block))
    if not cmp_csv.is_file() and not any(
            (ld / f'{ld.name}_compare_sections.png').is_file()
            for ld in line_dirs):
        parts.append('<div class="warn">未发现对比成果 (运行 model_compare 后生成)。</div>')

    # ---- ④ 其他成果 (MCMC 后验等) ----
    parts.append('<h2 id="sec_other">④ 其他成果</h2>')
    other = []
    for ld in line_dirs:
        for png in sorted(ld.glob('*mcmc*posterior*.png')):
            other.append(f'<h3>{ld.name} · MCMC 后验</h3>'
                         f'<img src="{_img64(png)}">')
    parts.append(''.join(other) if other else
                 '<div class="meta">无其他附加成果图。</div>')

    # ---- ⑤ 参数快照 ----
    parts.append('<h2 id="sec_param">⑤ 参数快照</h2>')
    snap = _latest_glob(root, 'cfg_snapshot_*.json')
    if snap:
        try:
            cfg = json.loads(snap.read_text(encoding='utf-8'))
            parts.append(f'<div class="meta">{html.escape(snap.name)}</div>')
            parts.append('<pre>' + html.escape(
                json.dumps(cfg, ensure_ascii=False, indent=2)) + '</pre>')
        except Exception as e:
            parts.append(f'<div class="warn">参数快照读取失败: {e}</div>')
    else:
        parts.append('<div class="meta">无 cfg_snapshot。</div>')

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<div id="im" class="modal" onclick="closeM()"><img src=""></div>
<div class="wrap">{''.join(parts)}</div>
<script>{JS}</script>
</body></html>"""
    html_path.write_text(doc, encoding='utf-8')
    size_kb = html_path.stat().st_size / 1024
    log_fn(f'[报告] HTML → {html_path} ({size_kb:.0f} KB, 单文件自包含)')
    return html_path
