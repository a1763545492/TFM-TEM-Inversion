# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# TFM-TEM-Inversion PyInstaller 打包配置 (v3.0.0)
# ============================================================
# 用途: 把 GUI + CLI + tfm_engine 打成单 exe, 现场人员免 Python 环境
# 安装: pip install pyinstaller
# 打包: pyinstaller tfm_app.spec --noconfirm
# 产物: dist/TFM.exe
#
# 注意:
#   · torch 默认不打进去 (体积 2GB+); DL 权重训练在装 torch 的开发机
#     上完成, 现场只做 numpy 推理 → 无需 torch
#   · docs/ 不进 exe, 随 dist 目录分发
# ============================================================

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / 'tfm_gui.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # 文档随包分发 (Help 页"打开文档目录"指向 docs/)
        (str(ROOT / 'docs'), 'docs'),
    ],
    hiddenimports=[
        # 子进程 / 动态导入需要显式声明
        'tfm_engine', 'tfm_engine.dat_parser', 'tfm_engine.excel_meta',
        'tfm_engine.coord_loader', 'tfm_engine.preprocess', 'tfm_engine.inversion',
        'tfm_engine.inv_algorithms', 'tfm_engine.adv_inversion',
        'tfm_engine.lci_inversion', 'tfm_engine.dl_inversion',
        'tfm_engine.dl_advanced', 'tfm_engine.data_check',
        'tfm_engine.ip_inversion', 'tfm_engine.debye_inversion',
        'tfm_engine.occam_bridge', 'tfm_engine.current_proc',
        'tfm_engine.fem_forward', 'tfm_engine.static_correction',
        'tfm_engine.resolution', 'tfm_engine.nn_initial',
        'tfm_engine.denoise_ae', 'tfm_engine.viz_3d', 'tfm_engine.project_db',
        'tfm_engine.parallel_utils', 'tfm_engine.model_compare',
        'tfm_engine.export_formats', 'tfm_engine.report_gen',
        'tfm_engine.timelapse', 'tfm_engine.fem2d_forward',
        'tfm_engine.anisotropy', 'tfm_engine.wlf_bridge',
        'tfm_engine.preprocess_plus', 'tfm_engine.signal_viz',
        'multiprocessing.spawn',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 明确排除重型可选依赖 (现场不需要)
        'torch', 'rasterio', 'IPython', 'jedi', 'pytest',
        'pandas.tests', 'scipy.tests', 'numpy.tests',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name='TFM',
    debug=False,
    strip=False,
    upx=True,
    console=False,           # GUI 程序无控制台窗口
    icon=None,               # 可放自定义 .ico
)

# ============================================================
# 生成后 dist/ 布局建议:
#   dist/TFM.exe              ← 主程序
#   dist/_internal/docs/...   ← 打包内文档 (onedir 模式自动)
# 建议用 onedir (默认) 而非 onefile: 启动快、杀毒误报少
# ============================================================
