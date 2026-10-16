# -*- coding: utf-8 -*-
"""
make_demo_data.py — TFM-TEM-Inversion 合成示例数据生成器
==========================================================
在仓库根目录下生成一套完整的 WLF 体系演示工程::

    demo_data/myfile_demo/
      坐标文件.xls                      (xlsx 内容, 2 sheet: MN坐标 / 源坐标)
      1线/
        发射接收备注信息.xlsx            (4 sheet: 接收对应点位/发射对应关系/发射时间/坐标)
        17/  C017ST01.DAT  c017st001.dat  c017st002.dat  c017st003.dat
        27/  C027ST01.DAT  c027st001.dat  c027st002.dat  c027st003.dat

DAT 二进制契约 (与 tfm_engine/dat_parser.py 100% 一致):
  文件大小 = 2048 + 4 × 544859 × Np × 4 字节  (Header + int32 LE 数据区)
  Header 前 64 字节 = 32 个 int16 LE:
    [11..14]=8224  [15]=1024  [26]=0x0102(258)  [29]=1  [27]=256+Np
    [4:15] = ASCII 签名 (大写电流档含 "Xi'An,China"; 小写接收档含 "area")
  数据区 int32 LE, reshape(4, 544859, Np), order='C', 通道 0=Ex 1=Ey 2=Hx 3=Hy

说明: Np=3 (契约允许 1..10), 使 2 section × 3 仪器 × 3 桩 = 18 槽位
      覆盖全部 16 个测站 (S1-1..S1-16).

运行:  python make_demo_data.py        (幂等, 重复运行覆盖重建)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量 (锁死)
FS = 27_243_000.0          # 采样率 Hz (544859 样点 ≈ 20ms)
N = 544859                 # 每通道固定样点数
NP = 3                     # 每文件桩数 (Np, 契约范围 1..10)
HEADER_BYTES = 2048
G_ADC = 4_194_304          # 2^22 counts/V (仅注释用, 数据本身是 counts)

REPO_ROOT = Path(__file__).resolve().parent
DEMO_ROOT = REPO_ROOT / "demo_data" / "myfile_demo"
LINE_DIR = DEMO_ROOT / "1线"

# 测站规划: section -> instrument -> [stations]  (picket 顺序 = 列表顺序)
STATION_PLAN = {
    "017": {"001": ["S1-1", "S1-2", "S1-3"],
            "002": ["S1-4", "S1-5", "S1-6"],
            "003": ["S1-7", "S1-8", "S1-9"]},
    "027": {"001": ["S1-10", "S1-11", "S1-12"],
            "002": ["S1-13", "S1-14", "S1-15"],
            "003": ["S1-16"]},
}
SECTION_DIRS = {"017": "17", "027": "27"}
I_TX_AMPS = 30.0
TX_AREA = {"A1": (4832000.0, 613400.0), "B1": (4834000.0, 615600.0),   # 2000×2200 m
           "A2": (4832600.0, 613700.0), "B2": (4834600.0, 615900.0)}


# ---------------------------------------------------------------- DAT 写入
def _encode_sig(sig: str) -> np.ndarray:
    """i16_slice_to_ascii 的逆过程: 每 2 个 ASCII 字节装入 1 个 int16 (低字节在前)."""
    raw = sig.encode("ascii")[:22].ljust(22, b"\x00")
    out = np.zeros(11, dtype=np.int16)
    for k in range(11):
        out[k] = raw[2 * k] | (raw[2 * k + 1] << 8)
    return out


def _build_header(n_pickets: int, sig: str) -> bytes:
    i16 = np.zeros(32, dtype="<i2")
    i16[4:15] = _encode_sig(sig)
    i16[11] = i16[12] = i16[13] = i16[14] = 8224
    i16[15] = 1024
    i16[26] = 0x0102
    i16[29] = 1
    i16[27] = 256 + n_pickets
    return i16.tobytes() + b"\x00" * (HEADER_BYTES - 64)


def _write_dat(path: Path, cube: np.ndarray, sig: str) -> None:
    """cube: float/int array, 内部转 int32 LE, shape=(4, N, NP), order='C'."""
    assert cube.shape == (4, N, NP), cube.shape
    data = np.rint(cube).clip(-2**31, 2**31 - 1).astype("<i4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_build_header(NP, sig))
        f.write(data.tobytes(order="C"))
    expect = HEADER_BYTES + 4 * N * NP * 4
    actual = path.stat().st_size
    assert actual == expect, f"{path.name}: {actual} != {expect}"
    print(f"    [DAT] {path.relative_to(DEMO_ROOT)}  {actual / 1e6:.1f} MB")


# ---------------------------------------------------------------- 波形合成
def _tx_cube(rng: np.random.Generator) -> np.ndarray:
    """发射电流档 (大写 CXXST01.DAT):
    Ex=双极性方波 ±1.9e6 counts (T=0.25ms, 占空~95%), t≈19.99ms 关断归零;
    Hx=常值平台 3.9e6; Ey/Hy=小幅噪声. 5% 高斯噪声."""
    t = np.arange(N) / FS
    T = 0.25e-3
    ph = (t % T) / T
    sq = np.where(ph < 0.475, 1.0,
                  np.where(ph < 0.5, 0.0,
                           np.where(ph < 0.975, -1.0, 0.0)))
    i_cut = int(round(19.99e-3 * FS))
    sq[i_cut:] = 0.0
    cube = np.zeros((4, N, NP))
    for p in range(NP):
        amp = 1.0 + 0.02 * (p - 1)                       # 桩间幅值 ±2%
        cube[0, :, p] = 1.9e6 * amp * sq + 0.05 * 1.9e6 * rng.standard_normal(N)
        cube[1, :, p] = 1e4 * rng.standard_normal(N)
        cube[2, :, p] = 3.9e6 * amp + 0.02 * 3.9e6 * rng.standard_normal(N)
        cube[3, :, p] = 1e4 * rng.standard_normal(N)
    return cube


def _rx_cube(rng: np.random.Generator, seed_off: float) -> np.ndarray:
    """接收档 (小写 cXXXst00Y.dat):
    Hx: t<t_off(10ms) 发射期震荡(5e5); t≥t_off 指数衰减 A·exp(-t/τ)·cos 残余 + 噪声.
    A≈2e6 counts, τ≈2ms, 不同桩 τ/A 略有差异. Ex/Ey/Hy=小幅噪声."""
    t = np.arange(N) / FS
    t_off = 10.0e-3
    i_off = int(round(t_off * FS))                        # 272430
    n_post = N - i_off
    dt = t[i_off:] - t_off
    cube = np.zeros((4, N, NP))
    for p in range(NP):
        a_p = 2.0e6 * (1.0 + 0.10 * (p - 1) + 0.05 * seed_off)
        tau_p = 2.0e-3 * (1.0 + 0.15 * (p - 1) + 0.08 * seed_off)
        hx = np.empty(N)
        hx[:i_off] = (5.0e5 * np.sin(2 * np.pi * 4000.0 * t[:i_off])
                      + 1e4 * rng.standard_normal(i_off))
        hx[i_off:] = (a_p * np.exp(-dt / tau_p) * np.cos(2 * np.pi * 300.0 * dt)
                      + 1e4 * rng.standard_normal(n_post))
        cube[0, :, p] = 1e4 * rng.standard_normal(N)
        cube[1, :, p] = 1e4 * rng.standard_normal(N)
        cube[2, :, p] = hx
        cube[3, :, p] = 1e4 * rng.standard_normal(N)
    return cube


def write_dats() -> None:
    rng = np.random.default_rng(42)
    for sec, sdir in SECTION_DIRS.items():
        sec_dir = LINE_DIR / sdir
        print(f"  [section {sec}] -> 1线/{sdir}/")
        _write_dat(sec_dir / f"C{sec}ST01.DAT", _tx_cube(rng), "Xi'An,China WLF TX")
        for k, inst in enumerate(("001", "002", "003")):
            _write_dat(sec_dir / f"c{sec}st{inst}.dat",
                       _rx_cube(rng, seed_off=float(k)), "area recv WLF")


# ---------------------------------------------------------------- Excel 元数据
def write_meta_xlsx() -> None:
    """1线/发射接收备注信息.xlsx — 4 sheet, 契约见 tfm_engine/excel_meta.py."""
    xlsx = LINE_DIR / "发射接收备注信息.xlsx"

    # Sheet1 接收对应点位: 列=[日期,排列,点号,通道,仪器号,备注]  ffill 语义
    # 行 0 为表头(解析跳过); 日期/排列/点号/仪器号 仅在组首行出现, 后续 None.
    rows = []
    seq = 0
    for sec, inst_map in STATION_PLAN.items():
        seq += 1
        for inst, stations in inst_map.items():
            for st in stations:
                for ch in (1, 2, 3):
                    first_of_section = (st == inst_map[inst][0] and ch == 1
                                        and inst == list(inst_map)[0])
                    first_of_inst = (st == stations[0] and ch == 1)
                    first_of_station = (ch == 1)
                    rows.append([
                        "1.1" if first_of_section else None,          # 日期
                        seq if first_of_section else None,            # 排列
                        st if first_of_station else None,             # 点号
                        ch,                                           # 通道
                        int(inst) if (first_of_inst or first_of_section) else None,  # 仪器号
                        None,                                         # 备注
                    ])
    df_recv = pd.DataFrame(rows, columns=["日期", "排列", "点号", "通道", "仪器号", "备注"])

    # Sheet2 发射对应关系: 前 2 行表头(解析跳过), 数据行 列2=发射档 列4..6=3台接收档
    rows_tx = [["日期", "排列", "发射文件名", None, "接收仪器001", "接收仪器002", "接收仪器003"],
               [None] * 7]
    for j, sec in enumerate(("017", "027")):
        rows_tx.append(["1.1", j + 1, f"C{sec}ST01", None,
                        f"001_2601011{j}0000", f"002_2601011{j}0100", f"003_2601011{j}0200"])
    df_tx = pd.DataFrame(rows_tx)

    # Sheet3 发射时间: 行 0 表头; 列=[文件名,发射日期,开始时间,结束时间,电流大小]
    df_time = pd.DataFrame(
        [[f"C{sec}ST01", "2026-01-01", f"1{j}:00:00", f"1{j}:30:00", "30A"]
         for j, sec in enumerate(("017", "027"))],
        columns=["文件名", "发射日期", "开始时间", "结束时间", "电流大小"])

    # Sheet4 坐标: 前 2 行多级表头(解析跳过); 列0=仪器号 列1=线号 列2=点号 列3=磁棒号
    # 首个数据行 列4..7 = A2_X/A2_Y/B2_X/B2_Y (大坐标值)
    mag_pool = [4238, 1619, 1618, 4239, 1620, 4237, 1617, 1621,
                4236, 1616, 1622, 4235, 1615, 1623, 4234, 1614]
    rows_c = [["仪器号", "线号", "点号", "磁棒号", "A2_X", "A2_Y", "B2_X", "B2_Y"],
              [None] * 8]
    for i in range(16):
        row = [i % 3 + 1, "S1", i + 1, mag_pool[i], None, None, None, None]
        if i == 0:
            row[4:] = [4834050.0, 614450.0, 4836050.0, 616650.0]
        rows_c.append(row)
    df_coord = pd.DataFrame(rows_c)

    LINE_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df_recv.to_excel(w, sheet_name="接收对应点位", index=False)
        df_tx.to_excel(w, sheet_name="发射对应关系", index=False, header=False)
        df_time.to_excel(w, sheet_name="发射时间", index=False)
        df_coord.to_excel(w, sheet_name="坐标", index=False, header=False)
    print(f"  [Excel] {xlsx.relative_to(DEMO_ROOT)}  (4 sheets, {len(df_recv)} 接收行)")


def write_coord_xls() -> None:
    """demo_data/myfile_demo/坐标文件.xls — 内容为 xlsx (pandas 按魔数识别引擎).
    两 sheet 均无表头 (coord_loader 以 header=None 全行读取)."""
    xls = DEMO_ROOT / "坐标文件.xls"
    mn = [["S1-%d" % (i + 1), 4834000.0 + i * 25.0, 614000.0, 100.0 + 0.5 * i]
          for i in range(16)]
    df_mn = pd.DataFrame(mn)
    df_src = pd.DataFrame([["A1", 0, TX_AREA["A1"][0], TX_AREA["A1"][1]],
                           ["B1", 1, TX_AREA["B1"][0], TX_AREA["B1"][1]],
                           ["A2", 0, TX_AREA["A2"][0], TX_AREA["A2"][1]],
                           ["B2", 1, TX_AREA["B2"][0], TX_AREA["B2"][1]]])
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xls, engine="openpyxl") as w:
        df_mn.to_excel(w, sheet_name="MN坐标", index=False, header=False)
        df_src.to_excel(w, sheet_name="源坐标", index=False, header=False)
    area = abs((TX_AREA["B1"][0] - TX_AREA["A1"][0]) * (TX_AREA["B1"][1] - TX_AREA["A1"][1]))
    print(f"  [Excel] {xls.relative_to(DEMO_ROOT)}  (MN坐标 16站, A1B1={area:,.0f} m²)")


# ---------------------------------------------------------------- 自检
def self_check() -> None:
    """用引擎自身的 dat_parser 抽验一个发射档 + 一个接收档."""
    try:
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tfm_engine.dat_parser import load_wlf_dat
    except Exception as e:
        print(f"  [self-check] 跳过 (无法导入 tfm_engine): {e}")
        return
    for fp in (LINE_DIR / "17" / "C017ST01.DAT", LINE_DIR / "17" / "c017st001.dat"):
        d = load_wlf_dat(fp)
        print(f"  [self-check] {fp.name}: Np={d.n_pickets} sig='{d.header.ascii_sig}' "
              f"cube={d.cube.shape} Hx_peak={abs(d.cube[2]).max():.3e}")


def main() -> None:
    print("=" * 64)
    print("  TFM 示例数据生成器")
    print(f"  输出目录: {DEMO_ROOT}")
    print(f"  DAT: fs={FS / 1e6:.3f}MHz N={N} Np={NP}  每文件 "
          f"{(HEADER_BYTES + 4 * N * NP * 4) / 1e6:.1f} MB")
    print("=" * 64)
    write_dats()
    write_meta_xlsx()
    write_coord_xls()
    self_check()
    print("\n✓ 示例数据生成完成. 验证管线:")
    print(r"  python tfm_cli_run.py --project demo_data\myfile_demo --out demo_out "
          r"--current --inv-algo trf --no-occam")


if __name__ == "__main__":
    main()
