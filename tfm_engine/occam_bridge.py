# -*- coding: utf-8 -*-
"""
occam_bridge.py — Occam1D 商业工具链桥接 (启动文件 + model-p + 运行 BAT)
======================================================================
★ 调参纪律 (经验 #783912 实锤):
  - 一次只改一个参数: LAMBDA_INIT 首选 0.1 → 1.0 → 10
  - 不动 TARGET_MISFIT (1.0) / ROUGHNESS_TYPE (First) / MAX_ITER (50)
  - 只读不改 *.iter / *.resp 输出

对每条测线生成 3 个文件:
  - {LINE}_occam_out/{LINE}_occam1d.cfg     启动配置 (INI)
  - {LINE}_occam_out/{LINE}_model-p.dat     初始 25 层模型 (均匀半空间)
  - run_occam1d.bat  (单测线启动器, 纯 ASCII 防 CMD 乱码)
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Callable, Optional

CFG_TEMPLATE = """# ============================================================
# Occam1D 配置文件 | 测线={LINE} | 自动生成 by tfm_engine.occam_bridge
# ★ 调参纪律: 仅调整 LAMBDA_INIT (0.1→1.0→10)
# ============================================================

[DATA]
   FILE             = {DAT_IN}
   FORMAT           = EMData_1.1
   IGNORE_BAD       = 0
   ERR_FLOOR        = 0.02

[MODEL]
   TYPE             = 1D
   N_LAYERS         = {N_LAYERS}
   Z_BOTTOM_M       = {Z_MAX_M}
   GRID             = logarithmic
   RHO_INIT         = {RHO_INIT}
   RHO_BOUNDS       = {RHO_LO} {RHO_HI}
   MODEL_P_DAT      = {MODEL_P_PATH}

[INVERSE]
   MISFIT_CALC      = standard
   TARGET_MISFIT    = 1.0
   ROUGHNESS_TYPE   = First
   MAX_ITER         = 50
   LAMBDA_INIT      = {LAMBDA_INIT}
   LAMBDA_MIN       = 0.001
   LAMBDA_MAX       = 1000.0
   LAMBDA_ADAPT     = 1

[OUTPUT]
   DIR              = {OUT_DIR}
   PREFIX           = {LINE}
   SAVE_ITER        = 1
   SAVE_MODEL       = 1
   SAVE_PREDICTED   = 1
"""

MODEL_P_HEADER = """# Occam1D 1D 初始模型 | 测线={LINE}
# 格式: n_layers  n_params_per_layer
# 每行: thickness(m)  rho(Ohm*m)
# 末层 thickness >= 9999 = 半空间无限延伸
 {N_LAYERS}  1
"""


def _generate_layer_thicknesses(n: int, z_min: float, z_max: float) -> list[float]:
    """对数等比, 末层半空间用 9999"""
    import numpy as np
    edges = np.geomspace(z_min, z_max, n)
    thicks = (edges[1:] - edges[:-1]).tolist()
    # 首段 = edges[0]
    result = [float(edges[0])] + thicks
    # 填充到 n
    while len(result) < n: result.append(0.0)
    # 末层 = 9999 (半空间标识)
    if result: result[-1] = 9999.0
    return result[:n]


def materialize_line_config(line_tag: str, out_root: Path,
                            occam_exe: str = 'Occam1D.exe',
                            lambda_init: float = 1.0,
                            n_layers: int = 25,
                            z_min: float = 5.0,
                            z_max: float = 1500.0,
                            rho_init: float = 100.0,
                            rho_bounds: tuple[float, float] = (0.1, 1e8),
                            log_fn: Optional[Callable[[str], None]] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    out_base = out_root / line_tag
    dat_in = out_base / f'{line_tag}_occam.dat'
    if not dat_in.is_file():
        log_fn(f'  [SKIP] {line_tag}: Occam.dat 不存在')
        return {}
    out_dir = out_base / f'{line_tag}_occam_out'
    out_dir.mkdir(parents=True, exist_ok=True)
    model_p_name = f'{line_tag}_model-p.dat'
    cfg_name = f'{line_tag}_occam1d.cfg'
    model_p_path = out_dir / model_p_name
    cfg_path = out_dir / cfg_name

    # (1) 写 model-p
    thicks = _generate_layer_thicknesses(n_layers, z_min, z_max)
    with open(model_p_path, 'w', encoding='utf-8') as f:
        f.write(MODEL_P_HEADER.format(LINE=line_tag, N_LAYERS=n_layers))
        for t in thicks:
            f.write(f'{t:10.3f}  {rho_init:10.3f}\n')

    # (2) 写 cfg (INI)
    cfg_text = CFG_TEMPLATE.format(
        LINE=line_tag,
        DAT_IN=dat_in.resolve().as_posix().replace('/', '\\'),   # Windows 反斜杠
        N_LAYERS=n_layers, Z_MAX_M=z_max,
        RHO_INIT=rho_init, RHO_LO=rho_bounds[0], RHO_HI=rho_bounds[1],
        MODEL_P_PATH=model_p_path.resolve().as_posix().replace('/', '\\'),
        OUT_DIR=out_dir.resolve().as_posix().replace('/', '\\'),
        LAMBDA_INIT=lambda_init,
    )
    cfg_path.write_text(cfg_text, encoding='utf-8')

    # (3) 生成 per-line bat 启动器
    #     ★ UTF-8 + chcp 65001 避免中文"线"字乱码 (CMD 代码页 936 半汉字问题)
    #     ★ 用 %~dp0 取 BAT 自身所在目录, 相对路径引用 cfg/model-p, 不依赖绝对路径字符集
    bat = out_dir / f'run_{line_tag}_occam1d.bat'
    # cfg 和 model-p 就在与 BAT 同一目录 (%~dp0 = BAT 所在目录, 末尾带 \)
    cfg_rel = f'%~dp0{cfg_name}'
    model_rel = f'%~dp0{model_p_name}'
    out_dir_rel = '%~dp0'
    # 推上级目录的 X线_occam.dat
    dat_rel = f'%~dp0..\\{line_tag}_occam.dat'

    bat_content = f"""@echo off
REM ================================================================
REM  Occam1D 商业反演启动器 - 测线 {line_tag}
REM  生成工具: tfm_engine  (完全离线, 数据绝不上传云端)
REM ================================================================
@chcp 65001 >nul
setlocal enabledelayedexpansion

REM --------- ★ 首选调参: 仅修改上方 CFG 文件中的 LAMBDA_INIT (0.1→1.0→10) ---------

set "OCCAM={occam_exe}"
set "CFG={cfg_rel}"
set "MODEL_P={model_rel}"
set "DAT_IN={dat_rel}"
set "OUT_DIR={out_dir_rel}"

REM ----------- 路径自检 -----------
if not exist "%CFG%" (
    echo [FATAL] CFG 不存在: %CFG%
    pause
    exit /b 2
)
if not exist "%MODEL_P%" (
    echo [FATAL] MODEL-P 不存在: %MODEL_P%
    pause
    exit /b 2
)
if not exist "%DAT_IN%" (
    echo [FATAL] Occam.dat 输入不存在: %DAT_IN%
    pause
    exit /b 2
)

REM ----------- EXE 检测 (DRY-RUN 安全降级) -----------
where %OCCAM% >nul 2>nul
if errorlevel 1 (
    echo.
    echo [WARN] 未在 PATH 中找到 Occam 可执行: %OCCAM%
    echo        [解决方式] 设置环境变量后重试:
    echo            set OCCAM=D:\\path\\to\\OCCAMITER_FLEX.exe
    echo            或 直接修改本 BAT 第 17 行 set OCCAM=...
    echo.
    echo [DRY-RUN] 本批已生成的启动文件:
    echo   CFG   : %CFG%
    echo   MODEL : %MODEL_P%
    echo   INPUT : %DAT_IN%
    echo   OUTPUT: %OUT_DIR%
    echo [提示] 当 OCCAM 可执行准备就绪后, 重新双击本批即可开始商业反演.
    echo.
    pause
    exit /b 0
)

REM ----------- 启动商业反演 -----------
echo ================================================================
echo  启动 Occam1D 反演  测线={line_tag}  时间: %DATE% %TIME%
echo  LAMBDA_INIT = {lambda_init}   (★ 首选调参, 单变量 0.1→1.0→10)
echo ================================================================
pushd "%OUT_DIR%"
%OCCAM% "%CFG%"
set "RC=%ERRORLEVEL%"
popd
echo.
echo 完成  ExitCode=%RC%   (反演产物: %OUT_DIR%*.iter / *.resp -- 只读, 请不要修改)
pause
endlocal
"""
    # UTF-8 BOM 让 CMD 正确识别中文 (Win10 1903+ 原生支持; 旧版+chcp65001也兼容)
    with open(bat, 'wb') as fh:
        fh.write(b'\xef\xbb\xbf' + bat_content.encode('utf-8'))

    log_fn(f'  [Occam 桥接] {line_tag} →  cfg={cfg_path.name}  model-p={model_p_path.name}  bat={bat.name}')
    log_fn(f'    启动命令 (CMD): "{bat}"  或  set OCCAM=OCCAMITER_FLEX.exe 然后双击 bat')
    return {'cfg': str(cfg_path), 'model_p': str(model_p_path),
            'bat': str(bat), 'occam_exe': occam_exe}


def materialize_all(out_root: str | Path,
                    occam_exe: str = 'Occam1D.exe',
                    lambda_init: float = 1.0,
                    log_fn: Optional[Callable[[str], None]] = None) -> dict:
    import re
    out_root = Path(out_root)
    log_fn = log_fn or (lambda s: print(s))
    lines = sorted([p.name for p in out_root.iterdir()
                    if p.is_dir() and re.fullmatch(r'[1-5]线', p.name)])
    log_fn(f'\n========== Occam1D 桥接生成  lines={lines}  λ_init={lambda_init} ==========')
    results = {}
    for lt in lines:
        results[lt] = materialize_line_config(lt, out_root, occam_exe, lambda_init, log_fn=log_fn)
    return results
