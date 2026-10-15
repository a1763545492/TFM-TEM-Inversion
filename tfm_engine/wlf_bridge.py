# -*- coding: utf-8 -*-
"""
tfm_engine.wlf_bridge (M7)
==========================
TEM 商业算法 EXE 全桥接（完全遵循用户授权："要是麻烦就调用这些算法处理数据"）：
  ┌──────────────┬──────────────────────────────────────────────────────────┐
  │ EXE 名称     │ 作用与输入输出接口                                      │
  ├──────────────┼──────────────────────────────────────────────────────────┤
  │ HED.EXE      │ 头段编辑：DAT Header 2048B 校验修复，对应 AATZero/关断  │
  │ VMD.EXE      │ 1D VMD 正演：输入 modelfile → 输出 VMD.DAT 标准曲线     │
  │ FMT24.exe  │ 格式转换：Occam.dat ↔ TEM .mod / .dat / .cur 互相转换   │
  │ SWP24.EXE /  │ 2D TEM 剖面 Occam 反演：modelfile + dat_in → 2D .inv     │
  │   2DINV.exe │   (支持 IP 激电参数 + 正则化 = TEM 2D反演)         │
  │ SWP_EXP.EXE /│ 2D/3D 结果导出：.inv → Surfer .GRD / .BLN                │
  │ swpc_exp.EXE │                                                          │
  │ sWPc_cur.EXE │ 曲线导出: 逐站 .cur (ASCII 衰减曲线, 与 curview_kru 对齐)│
  │ NE.EXE       │ 网络(?) / 仪器格式 NE 转换辅助                          │
  │ velcur.exe / │ VEL (速度/视深度) 曲线 + 报告                            │
  │ VEL_c_f.exe  │                                                          │
  │ cmdow.exe    │ GUI 窗口控制（可选自动化 TEM 主程序 tem.exe）            │
  └──────────────┴──────────────────────────────────────────────────────────┘
所有桥接：① 找 EXE → ② 写输入文件 → ③ 调 EXE（subprocess）→ ④ 解析输出；
         找不到 EXE → DRY-RUN（生成所有输入文件+中文提示 set TEM=...），
         100% 离线，失败不兜底。
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import subprocess
import shutil
import os

# EXE 目录根
TEM_EXE_ROOT = Path(r"D:\工作\物探软件\TEM\TEM\TEM\tem_mcr\TEM\EXE_PROCESS")
TEM_MAIN_EXE = Path(r"D:\工作\物探软件\TEM\TEM\TEM\tem.exe")  # 主 GUI 程序（可选）


def _find_exe(name: str, user_override: Optional[str] = None) -> Optional[Path]:
    if user_override:
        p = Path(user_override)
        if p.exists(): return p
    p = TEM_EXE_ROOT / name
    if p.exists(): return p
    # PATH 回退
    in_path = shutil.which(name)
    return Path(in_path) if in_path else None


# ===================================================================
# 1. FMT24.exe: 格式转换
# ===================================================================
def run_trans24_convert(in_dat: str | Path, out_format: str = 'tem_cur',
                        out_dir: Optional[str | Path] = None,
                        trans24_exe: Optional[str] = None,
                        log_fn: Optional[Callable[[str], None]] = None
                        ) -> dict:
    """调用 FMT24.exe 做格式互转。
    in_dat:  Occam 风格 .dat 输入
    out_format: 'tem_cur'=.cur 衰减曲线 | 'tem_mod'=.mod 模型 | 'occam' 反向
    """
    exe = _find_exe('FMT24.exe', trans24_exe)
    in_dat = Path(in_dat)
    out_dir = Path(out_dir or in_dat.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt_to_ext = {'tem_cur': '.cur', 'tem_mod': '.mod', 'occam': '.dat'}
    out_file = out_dir / (in_dat.stem + f'_FMT24_{out_format}' + fmt_to_ext.get(out_format, '.out'))
    if exe is None:
        return {'status': 'DRY-RUN',
                'reason': 'FMT24.exe not found (install MCR7.3 or set FMT24=path)',
                'out_expected': str(out_file),
                'cmd_hint': r'set FMT24=D:\工作\物探软件\TEM\...\FMT24.exe 然后重新运行'}
    cmd = [str(exe), str(in_dat), str(out_file), f'/fmt={out_format}']
    try:
        res = subprocess.run(cmd, cwd=str(out_dir), capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=300)
        ok = (res.returncode == 0) and out_file.exists()
        return {'status': 'OK' if ok else 'FAIL',
                'returncode': res.returncode,
                'stdout_tail': res.stdout[-800:], 'stderr_tail': res.stderr[-800:],
                'output_file': str(out_file) if ok else None}
    except Exception as exc:
        return {'status': 'ERROR', 'error': str(exc)}


# ===================================================================
# 2. 2DINV.exe: 2D 剖面反演 (对应 TEM 2D反演 + NewLimModR_reg)
# ===================================================================
def run_2d_inv_tem(line_occam_dat: str | Path,
                   out_2d_dir: Optional[str | Path] = None,
                   include_ip: bool = False,
                   swp24_exe: Optional[str] = None,
                   lambda_init: float = 1.0,
                   n_layers: int = 25,
                   rho_bounds: tuple[float, float] = (0.1, 1e8),
                   log_fn: Optional[Callable[[str], None]] = None,
                   ) -> dict:
    """调用 TEM 2DINV.EXE 做 2D 剖面 TEM 反演。
    完全按 TEM Occam 风格：LAMBDA_INIT 单变量调参 (0.1→1→10)。
    include_ip=True → 2D 极化率反演（ρ + m + τ + c 四参数层）。
    """
    log_fn = log_fn or (lambda s: print(s))
    dat = Path(line_occam_dat)
    line_tag = dat.parent.name if dat.parent.name.endswith('线') else dat.stem.split('_')[0]
    out_2d_dir = Path(out_2d_dir or (dat.parent / f'{line_tag}_2d_tem'))
    out_2d_dir.mkdir(parents=True, exist_ok=True)
    exe = _find_exe('2DINV.exe', swp24_exe) or _find_exe('SWP24.EXE', swp24_exe)

    # ① 写 2D 反演 CFG（与 TEM 2D反演 接口兼容）
    cfg_path = out_2d_dir / f'{line_tag}_2d_swp24.cfg'
    cfg = f"""[2D_INV]
LINE_TAG={line_tag}
DATA_FILE={dat.resolve()}
OUTPUT_DIR={out_2d_dir.resolve()}
INCLUDE_IP={'YES' if include_ip else 'NO'}
N_LAYERS={n_layers}
RHO_BOUNDS={rho_bounds[0]:.4g} {rho_bounds[1]:.4g}
LAMBDA_INIT={lambda_init:.3g}
MAX_ITER=50
TARGET_MISFIT=1.0
ROUGHNESS_TYPE=1
MESH_H_SPACING=50.0
[END]
"""
    cfg_path.write_text(cfg, encoding='utf-8')
    # ② 写 BAT 启动器（UTF-8 BOM + 安全 DRY-RUN）
    bat = out_2d_dir / f'run_{line_tag}_2d_tem.bat'
    cfg_rel = f'%~dp0{cfg_path.name}'
    exe_name = exe.name if exe else '2DINV.EXE'
    bat_content = f"""@echo off
@chcp 65001 >nul
setlocal enabledelayedexpansion
set "EXE={exe_name}"
set "CFG={cfg_rel}"
if not exist "%CFG%" (echo [FATAL] missing %CFG% & pause & exit /b 2)
where %EXE% >nul 2>nul
if errorlevel 1 (
    echo.
    echo [WARN] TEM 2DINV.EXE 未在 PATH 中
    echo   设置: set SWP24={TEM_EXE_ROOT}\\2DINV.exe
    echo   DRY-RUN 就绪 → CFG={cfg_rel}
    pause
    exit /b 0
)
echo ================================================================
echo  TEM 2D TEM Occam反演  线={line_tag}  LAMBDA_INIT={lambda_init}  IP={include_ip}
echo ================================================================
pushd "%~dp0"
%EXE% "%CFG%"
set RC=%ERRORLEVEL%
popd
echo 完成 ExitCode=%RC%
pause
endlocal
"""
    with open(bat, 'wb') as fh:
        fh.write(b'\xef\xbb\xbf' + bat_content.encode('utf-8'))

    if exe is None:
        log_fn(f"  [2D-TEM] DRY-RUN: 找不到 2DINV.EXE → 已生成 {bat}")
        return {'status': 'DRY-RUN',
                'reason': '2DINV/SWP24.EXE missing (set SWP24=... 或确认 TEM 安装)',
                'cfg': str(cfg_path), 'bat': str(bat)}
    # ③ 实际调用 2DINV.EXE
    cmd = [str(exe), str(cfg_path.resolve())]
    try:
        log_fn(f"  [2D-TEM] 启动: {exe.name}  LAMBDA_INIT={lambda_init}  IP={include_ip}")
        res = subprocess.run(cmd, cwd=str(out_2d_dir), capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=3600)
        inv_files = list(out_2d_dir.glob('*.inv')) + list(out_2d_dir.glob('*.bln')) + list(out_2d_dir.glob('*.grd'))
        return {'status': 'OK' if res.returncode == 0 else 'FAIL',
                'returncode': res.returncode,
                'output_files': [str(p) for p in inv_files],
                'stdout_tail': res.stdout[-1000:], 'stderr_tail': res.stderr[-1000:]}
    except Exception as exc:
        return {'status': 'ERROR', 'error': str(exc)}


# ===================================================================
# 3. sWPc_cur.EXE: 导出衰减曲线 ASCII (与 curview_kru 对齐)
# ===================================================================
def run_export_curves(station_dat: str | Path,
                      out_dir: Optional[str | Path] = None,
                      exe_path: Optional[str] = None,
                      log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """调用 sWPc_cur.EXE 导出 .cur 衰减曲线 ASCII。"""
    log_fn = log_fn or (lambda s: print(s))
    exe = _find_exe('sWPc_cur.EXE', exe_path)
    dat = Path(station_dat)
    out_dir = Path(out_dir or dat.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (dat.stem + '.cur')
    if exe is None:
        return {'status': 'DRY-RUN', 'out_expected': str(out),
                'hint': 'set SWPC={}\\sWPc_cur.EXE'.format(TEM_EXE_ROOT)}
    cmd = [str(exe), str(dat), str(out)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', timeout=120)
        return {'status': 'OK' if (res.returncode == 0 and out.exists()) else 'FAIL',
                'rc': res.returncode, 'out_file': str(out)}
    except Exception as exc:
        return {'status': 'ERROR', 'error': str(exc)}


# ===================================================================
# 4. 3D：调用 tem.exe（主GUI程序 cmdow 自动化 + FMT24/SWP24 组合）
# ===================================================================
def run_3d_inv_tem(line_occam_dats: list[Path],
                   out_3d_dir: Path,
                   lambda_init: float = 1.0,
                   include_ip: bool = False,
                   log_fn: Optional[Callable[[str], None]] = None,
                   ) -> dict:
    """TEM 3D 层析（按 3D层析 + tomo_kru）：
    策略 = 每条线跑 2D SWP24 → FMT24 统一坐标 → 3D 切片插值。
    找不到 TEM EXE → DRY-RUN：生成批量脚本 + 说明。
    """
    log_fn = log_fn or (lambda s: print(s))
    out_3d_dir = Path(out_3d_dir); out_3d_dir.mkdir(parents=True, exist_ok=True)
    # Step A: 逐线 2D
    statuses = []
    for dat in line_occam_dats:
        s = run_2d_inv_tem(dat,
                           out_2d_dir=out_3d_dir / dat.parent.name,
                           include_ip=include_ip, lambda_init=lambda_init, log_fn=log_fn)
        statuses.append((str(dat), s['status']))
    # Step B: FMT24 合并 → 3D grid
    trans_hint = run_trans24_convert(line_occam_dats[0], out_format='tem_mod',
                                     out_dir=out_3d_dir, log_fn=log_fn)
    # 写 3D 启动 BAT
    bat = out_3d_dir / 'run_TEM_3D_tomo.bat'
    with open(bat, 'wb') as fh:
        fh.write(b'\xef\xbb\xbf' + f"""@echo off
@chcp 65001 >nul
REM TEM 3D TEM 层析 / 切片反演启动脚本 (TEM tomo_kru 接口)
REM 前置: ① 各线 2D .inv 生成完毕  ② set TEM_EXE={TEM_MAIN_EXE} (可选主GUI自动化)
echo ================================================================
echo  3D 层析: 合并 2D → 3D 切片 → 体积反演 (TEM 3D层析)
echo  如未装 MCR7.3, 请先安装: {Path(TEM_MAIN_EXE).parent.parent}\\MCRInstaller.exe
echo ================================================================
echo  [1] FMT24 坐标合并 ...
where FMT24.exe >nul 2>nul && FMT24.exe *.dat merge_3d_grid.dat /fmt=grid3d
echo  [2] 3D 切片插值 (slice_kru + slicecont_kru)
set TEM=%TEM_MAIN_EXE%
if exist "%TEM%" echo  主 TEM GUI 就绪，手动操作: File → Load 2D Inversions → 3D Tomography
pause
""".encode('utf-8'))
    any_dry = any(s[1] == 'DRY-RUN' for s in statuses)
    return {'status': 'DRY-RUN' if any_dry else 'PIPELINE_READY',
            'per_line': statuses,
            'trans24_status': trans_hint['status'],
            'automation_bat': str(bat),
            'hint': '3D 计算量极大(≥1h)，默认 DRY-RUN。建议先逐线 2D QC 合格，再双击 run_TEM_3D_tomo.bat 启动 TEM。'}


__all__ = ['TEM_EXE_ROOT', 'TEM_MAIN_EXE',
           'run_trans24_convert', 'run_2d_inv_tem',
           'run_export_curves', 'run_3d_inv_tem']
