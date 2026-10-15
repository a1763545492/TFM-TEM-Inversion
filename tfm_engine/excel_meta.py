# -*- coding: utf-8 -*-
"""
excel_meta.py — 权威 Excel 元数据解析 (单一数据源契约 #100025880)
===============================================================
输入: `<project_root>/<测线>/发射接收备注信息.xlsx`  (每测线 1 份, 4 个 sheet)

Sheet 契约 (已实锤 1/2/5线):
  1.「接收对应点位」: 列=[日期, 排列, 点号=Sx-N, 通道=1/2/3, 仪器号=001/002/003, 备注]
     ffill 语义: 日期/排列/点号仅在首行出现, 后续 NaN 行从上一行 carry. 通道=非空行 为有效行.
     3 台仪器 × 每个排列各占 6 行 (3 通道 × 5 滚动采样 = 对应 station_code 排列覆盖).
  2.「发射对应关系」: 3 行 (表头, 排列1, 排列2).  列 2=大写档文件名(TXCAL01), 列 4..6=3 台仪器接收文件名
     接收文件名原始是 `001_YYMMDDHHMMSS` 时间戳, 但实际目录里已统一重命名为
     `c[section]st001.dat` / `c[section]st002.dat` / `c[section]st003.dat` —
     用 "仪器号末尾 1 位" 作为索引匹配.
  3.「发射时间」: 列 = [文件名=TXCAL01, 发射日期, 开始时间, 结束时间, 电流大小="30A"]
     提取 电流大小 I_tx = float.  ★ 权威, 不用 fallback.
  4.「坐标」: 列 = [仪器号, 线号=S1/S2/S4/S5, 点号=N, 磁棒号(如 4238/1619/1618), A2(发射源对角), B2(发射源对角)]
     用于 仪器+点号 → 磁棒号 映射; 也给出 A2/B2 坐标 (5线 这里给出 A1/B1 坐标)

输出 5 个数据对象 (Station / Section / Instrument / Tx / ChannelStation):
  - line_tag: 测线标签 ("1线"/"2线"/"4线"/"5线")
  - line_code: "S1"/"S2"/"S4"/"S5"
  - sections: dict[section_str] = {
        'date': '12.26',
        'order': 1,
        'tx_file': 'C017ST01',
        'instruments': {
            '001': {'recv_filename_on_disk_orig': '001_251225124239',
                    'recv_dat_expected': 'c017st001.dat',   # 规范名
                   },
            '002': { ... }, '003': { ... }
        },
        'i_tx_amps': 30.0,
        'start_time': '13:46:00',
        'end_time': '14:23:00',
    }
  - channel_station_rows: list[{date, order, station='S1-10', channel=1, instrument='001',
                                 section, line_code}]
  - instrument_mag_mapping: dict[(line_code, station_int, instrument)] = mag_bar_code 或 None
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Callable, Optional
import pandas as pd

LINE_NAME_TO_CODE = {'1线':'S1', '2线':'S2', '3线':'S3', '4线':'S4', '5线':'S5'}
LINE_CODE_TO_NAME = {v:k for k,v in LINE_NAME_TO_CODE.items()}


def _extract_inst_from_filename(fn: str) -> str:
    """从 Excel 「发射对应关系」第 4~6 列 提取仪器号 (001/002/003):
    支持 2 种格式:
      '001_251225124239'  → prefix=001
      'c017st001.dat'     → st001 → 001
    """
    fn = str(fn)
    m1 = re.match(r'^(\d{3})[_\-]', fn)
    if m1: return m1.group(1)
    m2 = re.search(r'[sS][tT](?:00)?(\d)', fn)
    if m2: return f"{int(m2.group(1)):03d}"
    return ""


def _parse_current_amps(text: str) -> float:
    """电流大小列 '30A' / '37 A' / 37 / 30.0 → float"""
    s = str(text).strip().upper().replace('A','').strip()
    try: return float(s)
    except ValueError:
        m = re.search(r'([\d.]+)', s)
        return float(m.group(1)) if m else float('nan')


def _ffill_cols(df_no_header: pd.DataFrame, col_indices: list[int]) -> pd.DataFrame:
    """对指定列索引做 forward fill (原地 copy)"""
    out = df_no_header.copy()
    for c in col_indices:
        last = None
        for i in range(len(out)):
            v = out.iat[i, c]
            if pd.notna(v) and str(v).strip() != "":
                last = v
            elif last is not None:
                out.iat[i, c] = last
    return out


def parse_em_xlsx(line_dir: Path, log_fn: Callable[[str], None] = None) -> dict:
    """
    解析 `<line_dir>/发射接收备注信息.xlsx`.
    返回 dict:
      {
        'line_tag': '1线', 'line_code': 'S1',
        'sections': dict, 'channel_station_rows': list,
        'instrument_mag': dict, 'source_corners': dict
      }
    """
    log_fn = log_fn or (lambda s: None)
    line_tag = line_dir.name  # "1线"
    if line_tag not in LINE_NAME_TO_CODE:
        raise ValueError(f'line_dir.name "{line_tag}" 不匹配 X线 模式')
    line_code = LINE_NAME_TO_CODE[line_tag]
    xlsx = line_dir / '发射接收备注信息.xlsx'
    if not xlsx.is_file():
        raise FileNotFoundError(f'Excel 不存在: {xlsx}')
    xl = pd.ExcelFile(xlsx)
    log_fn(f'  [Excel] {xlsx.name} sheets={xl.sheet_names}')

    # ============================================================
    # Sheet 2: 发射对应关系 →  section → 3 仪器 × {tx_file,recv_file} 映射
    # ---- 关键修正 ----
    #  1线 「排列」列全写为1, 造成 order 冲突.  解决方案:
    #   ★ 发射表的「行序」= 权威 section 顺序号 (1,2,3...), 覆盖 Excel 「排列」列值.
    # ============================================================
    s_t = pd.read_excel(xlsx, sheet_name='发射对应关系', header=None)
    sections: dict[str, dict] = {}
    section_order_to_key: dict[int, str] = {}   # 行序 1..N → section_key
    seq_idx = 0
    for i in range(2, len(s_t)):  # 跳过 2 行表头
        r = s_t.iloc[i]
        if pd.isna(r.iloc[2]) or str(r.iloc[2]).strip() == "": continue
        seq_idx += 1          # 行序 (1-based), 权威顺序号
        date_val = r.iloc[0] if pd.notna(r.iloc[0]) else ""
        # order_val 不再依赖 Excel「排列」列; 直接用行序.
        order_val = seq_idx
        tx_file = str(r.iloc[2]).strip()
        # 解析 section = TXCAL01 → XX.  大写规则: C17 → 017
        m = re.match(r'^[cC](\d{2,4})[sS][tT]', tx_file)
        if not m:
            log_fn(f'  [WARN] 发射文件名 {tx_file} 不匹配 TXCAL01, 跳过')
            seq_idx -= 1
            continue
        section_key = m.group(1).zfill(3)  # 17 → 017
        instruments: dict[str, dict] = {}
        for col in [4, 5, 6]:
            if col >= len(r): break
            if pd.isna(r.iloc[col]) or str(r.iloc[col]).strip() == "": continue
            recv_orig = str(r.iloc[col]).strip()
            inst = _extract_inst_from_filename(recv_orig)
            if not inst: continue
            instruments[inst] = {
                'recv_orig_in_excel': recv_orig,
                'recv_expected_on_disk': f'c{section_key}st{inst}.dat'.lower(),
            }
        sections[section_key] = {
            'date': str(date_val), 'order': order_val,
            'tx_file': tx_file, 'instruments': instruments,
            'i_tx_amps': None, 'start_time': None, 'end_time': None,
        }
        section_order_to_key[order_val] = section_key
    log_fn(f'  [发射对应关系] section 映射: {list(sections.keys())}  (行序→key={section_order_to_key})')

    # ============================================================
    # Sheet 3: 发射时间  → I_tx (权威)
    # ============================================================
    s_i = pd.read_excel(xlsx, sheet_name='发射时间', header=None)
    for i in range(1, len(s_i)):
        r = s_i.iloc[i]
        if pd.isna(r.iloc[0]): continue
        tx_fn = str(r.iloc[0]).strip()
        m = re.match(r'^[cC](\d{2,4})', tx_fn)
        if not m: continue
        sk = m.group(1).zfill(3)
        if sk not in sections: continue
        sections[sk]['start_time'] = str(r.iloc[2]) if pd.notna(r.iloc[2]) else None
        sections[sk]['end_time']   = str(r.iloc[3]) if pd.notna(r.iloc[3]) else None
        sections[sk]['i_tx_amps']  = _parse_current_amps(r.iloc[4]) if pd.notna(r.iloc[4]) else None
    for sk, sec in sections.items():
        if not sec['i_tx_amps']:
            log_fn(f'  [WARN] section {sk} 未解析到电流大小, fallback=30A')
            sec['i_tx_amps'] = 30.0
    log_fn(f'  [发射时间] I_tx (A): { {sk: sec["i_tx_amps"] for sk,sec in sections.items()} }')

    # ============================================================
    # Sheet 1: 接收对应点位 → (instrument, channel, date, order) → station_code
    # ---- 关键修正 ----
    #  (A) section 区分键: 5线 Excel 「排列」列(索引1)全写为1, 但第 6 列(索引5, 备注)写有
    #      "一排列"/"二排列" 作为段落分隔符.  所以分组键扩展为:
    #      (date_str, 排列列值, 备注列段落标签) —— 首次出现顺序 = 权威 section 顺序号.
    #      (1/2/4线 备注列全空 → ffill 后不影响之前逻辑)
    #  (B) 点号前缀误写 → 丢弃前缀提取末尾数字 + "{line_code}-{N}" 重建权威编码.
    # ============================================================
    s_r = pd.read_excel(xlsx, sheet_name='接收对应点位', header=None)
    # 列 0=日期, 1=排列(order), 2=点号(station), 3=通道, 4=仪器号, 5=备注排列段落.  ffill 全部
    FILL_COLS = [0, 1, 2, 4]
    if 5 < s_r.shape[1]: FILL_COLS.append(5)
    s_r = _ffill_cols(s_r, FILL_COLS)
    # (A) 构建 (date, 排列值, 备注排列段落) → 顺序号 1..N 映射 (首次出现顺序)
    group_order_map: dict[tuple, int] = {}
    next_ord = 1
    for i in range(1, len(s_r)):
        r = s_r.iloc[i]
        if pd.isna(r.iloc[3]): continue  # 以通道非空行为锚点
        d_v = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else ""
        o_v = int(r.iloc[1]) if pd.notna(r.iloc[1]) else -1
        p5_v = str(r.iloc[5]).strip() if (5 < s_r.shape[1] and pd.notna(r.iloc[5])) else ""
        # 去掉 "排列" 二字前面的 "一"/"二" 保留语义 (但实际完整字符串当key直接用即可)
        key = (d_v, o_v, p5_v)
        if key not in group_order_map:
            group_order_map[key] = next_ord; next_ord += 1

    cs_rows: list[dict] = []
    for i in range(1, len(s_r)):
        r = s_r.iloc[i]
        ch_raw = r.iloc[3]
        if pd.isna(ch_raw): continue  # 只取通道非空
        try: ch = int(ch_raw)
        except: continue
        date_v = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else ""
        order_v_raw = int(r.iloc[1]) if pd.notna(r.iloc[1]) else -1
        p5_v = str(r.iloc[5]).strip() if (5 < s_r.shape[1] and pd.notna(r.iloc[5])) else ""
        order_v = group_order_map.get((date_v, order_v_raw, p5_v), 0)  # 顺序号 = 权威
        station_raw = str(r.iloc[2]).strip() if pd.notna(r.iloc[2]) else ""
        inst_v_raw = int(r.iloc[4]) if pd.notna(r.iloc[4]) else -1
        inst = f"{inst_v_raw:03d}" if inst_v_raw >= 0 else ""
        if not station_raw or not inst or order_v < 1: continue
        # (B) 丢弃 station_raw 前缀, 提取末尾数字 → 重建 "{line_code}-{N}" 权威编码
        m_st = re.search(r'(\d+)$', station_raw)
        if not m_st: continue
        st_num = int(m_st.group(1))
        station_code_authoritative = f"{line_code}-{st_num}"
        # 匹配 section (按顺序号 order_v → section_key)
        section_key = section_order_to_key.get(order_v, "")
        cs_rows.append({
            'date': date_v, 'order': order_v, 'section': section_key,
            'station': station_code_authoritative, 'station_raw_in_excel': station_raw,
            'channel': ch, 'instrument': inst,
            'line_code': line_code,
        })
    log_fn(f'  [接收对应点位] (date,排列,备注段落)→seq#: {group_order_map}')
    log_fn(f'  [接收对应点位] 解析 {len(cs_rows)} 有效行 (前缀已修正为权威 {line_code})')
    # 统计每 section 覆盖的 station 集合
    from collections import defaultdict
    stations_by_sec = defaultdict(set)
    for r in cs_rows:
        if r['section']: stations_by_sec[r['section']].add(r['station'])
    for sk in sorted(sections.keys()):
        st_set = stations_by_sec.get(sk, set())
        sample = sorted(list(st_set))[:5]
        log_fn(f'    section {sk} (seq#{sections[sk]["order"]}): 覆盖 stations {sample}... 共 {len(st_set)} 个')

    # ============================================================
    # Sheet 4: 坐标 (仪器号, 线号, 点号, 磁棒, 发射源 A1/B1 或 A2/B2 坐标)
    # ============================================================
    s_c = pd.read_excel(xlsx, sheet_name='坐标', header=None)
    instrument_mag: dict[tuple[str, int, str], str] = {}
    source_corners: dict[str, tuple[float, float]] = {}
    # 前 2 行是多级表头
    for i in range(2, len(s_c)):
        r = s_c.iloc[i]
        inst_raw = r.iloc[0]
        if pd.isna(inst_raw): continue
        try: inst = f"{int(inst_raw):03d}"
        except: inst = str(inst_raw).strip()
        lc = str(r.iloc[1]).strip() if pd.notna(r.iloc[1]) else line_code
        st_raw = r.iloc[2]
        st = int(st_raw) if pd.notna(st_raw) and str(st_raw).strip() != "" else -1
        mag_raw = r.iloc[3]
        mag_bar = str(mag_raw).strip() if pd.notna(mag_raw) else ""
        # 5 线 sheet: 列 4=X_receive, 列 5=Y_receive, 列 6=磁棒? 或 其它线: 列 3=磁棒, 列 4=A2_X
        # 容错: 只取 (inst, station) → 磁棒号 (如果是数值且长度>=4)
        if mag_bar != "" and re.fullmatch(r'\d{3,5}', mag_bar):
            instrument_mag[(lc, st, inst)] = mag_bar
        # 发射源角点: 5 线 这里给的是 A1/B1 (4829133.986 / 4831169.919 匹配 坐标文件.xls 源坐标 A1/B1)
        # 1/2/4 线 这里给的是 A2/B2 (4834028.26 / 4836946.11)
        # 解析思路: 若存在 大值 (4,829,xxx 级别) = 坐标, 取一对 (A_X,A_Y) 和 (B_X,B_Y)
        coords = []
        for ci in range(4, min(len(r), 10)):
            v = r.iloc[ci]
            if pd.notna(v) and isinstance(v, (int, float)) and abs(v) > 100000:
                coords.append((ci, float(v)))
        if len(coords) >= 4:
            ax_i, ax_v = coords[0]
            ay_i, ay_v = coords[1]
            bx_i, bx_v = coords[2]
            by_i, by_v = coords[3]
            # 判断是 A1/B1 (Y≈610xxx) 还是 A2/B2 (Y≈614xxx)
            label_prefix = "A1" if ay_v < 612500 else "A2"
            source_corners[label_prefix]   = (ax_v, ay_v)
            source_corners["B" + label_prefix[1:]] = (bx_v, by_v)
    log_fn(f'  [坐标Sheet] 仪器-点号→磁棒 共 {len(instrument_mag)} 条;  发射源角点: {source_corners}')

    return {
        'line_tag': line_tag, 'line_code': line_code,
        'sections': sections, 'channel_station_rows': cs_rows,
        'instrument_mag': instrument_mag, 'source_corners': source_corners,
        'section_order_to_key': section_order_to_key,
    }
