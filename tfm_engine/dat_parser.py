# -*- coding: utf-8 -*-
"""
dat_parser.py — TEM DAT 二进制格式统一读取器
格式契约（100% 锁死，之前 V1~V3 实锤校验通过）：
  File size = 2048 + 4 × 544,859 × Np × 4  bytes (Header + int32_LE data)
  Header = 2048 bytes, first 64 bytes = Int16_LE[0:32]
    Int16[27] - 256 = Np  (Np = number of pickets / rolling captures)
    Int16[5:6]  → ASCII "ra"/"ea" / "SurveyRegion" / "area"
    Int16[11:14] = 8,224 (magic, 恒固定)
    Int16[15] = 1,024 (magic)
    Int16[26] = 0x0102 = 258 (version word)
    Int16[29] = 1 (magic)
  Data = int32_LE (Little Endian), reshape order='C' → (4 channels, 544859 samples, Np pickets)
    channel 0=Ex 1=Ey 2=Hx 3=Hy (相关系数 r=0.996 实锤)
"""
from __future__ import annotations
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np

PER_CHANNEL = 544859     # 每通道固定样点数 (锁死)
HEADER_BYTES = 2048      # 头段字节数 (锁死)
PER_PICKET_BYTES = 4 * PER_CHANNEL * 4   # 每桩数据区 = 4ch × 544,859 × 4B (锁死)


def i16_slice_to_ascii(raw_i16: np.ndarray) -> str:
    """把 Int16 切片转 ASCII 字符串（每个 Int16 拆 LE 高低字节）。
    避免 ValueError: bytes must be in range(0, 256)"""
    buf = bytearray()
    for v in raw_i16:
        buf.append(int(v) & 0xFF)
        buf.append((int(v) >> 8) & 0xFF)
    try:
        s = buf.decode('ascii', errors='replace').replace('\x00', '.').strip('\x00 ')
    except Exception:
        s = ""
    return s


@dataclass
class WlfDatHeader:
    file_path: str
    file_size: int
    raw_int16: np.ndarray       # shape=(32,) Int16[0:32] from header
    n_pickets: int              # Np (Int16[27]-256, 锁死)
    ascii_sig: str              # Int16[4:15] 转 ASCII (SurveyRegion / area)
    version_word: int           # Int16[26] = 0x0102
    section_code: str = ""      # 从文件名解析: cXXst00Y / TXCAL01 → XX
    instrument_code: str = ""   # 从文件名解析: st00Y → Y (001/002/003; 大写档为空)


@dataclass
class WlfDatData:
    header: WlfDatHeader
    cube: np.ndarray            # dtype=int32 shape=(4, PER_CHANNEL, Np)

    @property
    def n_pickets(self) -> int: return self.header.n_pickets

    def channel_std(self) -> np.ndarray:
        """返回 shape=(4, Np) 每通道每桩标准差 (counts)"""
        return np.std(self.cube.astype(np.float64), axis=1)


def _expect_bytes_from_np(np_val: int) -> int:
    """由 Int16[27] → 算 Np → 计算期望字节数"""
    np_val = int(np_val)
    # 已知 Np=Int16[27]-256. 但 Int16[27] 本身是 unsigned 解释为 short, 可能负数
    np_pickets = (np_val & 0xFFFF) - 256
    if np_pickets <= 0 or np_pickets > 32:  # 经验范围: 1~5
        raise ValueError(f"Int16[27]={np_val} (u={np_val&0xFFFF}) → Np-256={np_pickets} out of range")
    return HEADER_BYTES + PER_PICKET_BYTES * np_pickets


def auto_truncate_if_mismatch(fp: Path, log_fn: Callable[[str], None]) -> int:
    """加载前检查: 按头段 Np 算出期望字节数. 若实际 size > 期望 且 (size-expect) % 4096 == 0 (扇区冗余),
    自动 truncate + .bak 备份. 返回实际截断的字节差额 (0=无需)."""
    with open(fp, 'rb') as f:
        f.seek(0, 2)
        actual = f.tell()
        f.seek(0)
        head64 = f.read(64)
    if len(head64) < 64:
        return 0
    i16s = struct.unpack('<32h', head64)
    try:
        expected = _expect_bytes_from_np(i16s[27])
    except ValueError:
        return 0
    diff = actual - expected
    if diff > 0 and diff % 4096 == 0:
        bak = fp.with_suffix(fp.suffix + '.before_truncate.bak')
        if not bak.exists():
            import shutil
            shutil.copy2(fp, bak)
        with open(fp, 'rb+') as f:
            f.truncate(expected)
        log_fn(f'  ✓ auto truncate {fp.name}: {actual}B→{expected}B (-{diff}B, 4KB扇区冗余), bak={bak.name}')
        return diff
    elif diff < 0:
        log_fn(f'  ⚠ size mismatch {fp.name}: actual={actual}B < expect={expected}B (损坏?)')
    return 0


def load_tem_dat(fp: Path, *, log_fn: Callable[[str], None] = None,
                 validate: bool = True, strict_per_ch: bool = True) -> WlfDatData:
    """唯一 DAT 读取入口 (单一数据源契约)."""
    log_fn = log_fn or (lambda s: None)
    if not fp.is_file():
        raise FileNotFoundError(f'DAT 不存在: {fp}')
    auto_truncate_if_mismatch(fp, log_fn)
    size = fp.stat().st_size
    if size < HEADER_BYTES + PER_PICKET_BYTES:
        raise ValueError(f'DAT 太小 {size}B < 最小 {HEADER_BYTES + PER_PICKET_BYTES}B: {fp}')
    with open(fp, 'rb') as f:
        head64 = f.read(64)
        header_rest = f.read(HEADER_BYTES - 64)
        i16s = struct.unpack('<32h', head64)
        raw_i16 = np.array(i16s, dtype=np.int16)
        # Np 锁死
        Np = (int(i16s[27]) & 0xFFFF) - 256
        if Np < 1 or Np > 10:
            raise ValueError(f'DAT Np={Np} out of [1,10] in {fp}')
        # 魔法值校验
        if validate:
            magics = {11:8224,12:8224,13:8224,14:8224, 15:1024, 26:0x0102, 29:1}
            for idx, val in magics.items():
                if (i16s[idx] & 0xFFFF) != val:
                    raise ValueError(f'DAT 魔法值 Int16[{idx}]={i16s[idx]&0xFFFF}≠{val} in {fp}')
        ascii_sig = i16_slice_to_ascii(raw_i16[4:15])
        # 数据区
        expected = HEADER_BYTES + PER_PICKET_BYTES * Np
        if size != expected:
            if strict_per_ch:
                raise ValueError(f'DAT size={size}B ≠ header+Np data={expected}B in {fp}. '
                                 f'请调用 auto_truncate_if_mismatch() 前先处理.')
            else:
                data_bytes = f.read(expected - HEADER_BYTES)
        else:
            data_bytes = f.read()
    cube = np.frombuffer(data_bytes, dtype=np.int32).reshape((4, PER_CHANNEL, Np), order='C')
    # 从文件名解析 section / instrument (c017st001.dat → sect=017, inst=001)
    m = re.match(r'^[cC](\d{2,4})[sS][tT](?:00)?(\d*)', fp.stem)
    section, inst = (m.group(1), (m.group(2) or '').zfill(3)) if m else ("", "")
    header = WlfDatHeader(
        file_path=str(fp), file_size=size, raw_int16=raw_i16,
        n_pickets=Np, ascii_sig=ascii_sig, version_word=int(i16s[26]) & 0xFFFF,
        section_code=section, instrument_code=inst if inst else ""
    )
    return WlfDatData(header=header, cube=cube)
