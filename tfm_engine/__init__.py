# -*- coding: utf-8 -*-
"""
================================================================================
tfm_engine — 时频电磁法 (TEM/FEM) 线下完整数据处理引擎库 (v2.0)
================================================================================
完全离线，不联网。仅使用本地文件操作 + 标准科学库。

**设计核心（单一数据源契约）**：
  - 100% 权威元数据来自 2 个 Excel 文件
  - 大写档 (CXXST01.DAT) = 发射电流校准档
  - 小写档 (cxxst00Y.dat) = 接收原始采集档 (Y=001/002/003)
  - Np=5 = 一台仪器 1 排列下的 5 次滚动/时间片采集

**模块划分 (v2 共 11 个)**：
  [v1 基础 6]
  - dat_parser.py    : WLF DAT 二进制格式解析 (Header=2048, int32, shape=(4, 544859, Np))
  - excel_meta.py    : 4 sheets Excel 解析 → StationMap / SectionMap / TxMap
  - coord_loader.py  : 坐标文件.xls 解析 → station_code → (X, Y, Z) / source_code → (X,Y)
  - preprocess.py    : 4 阶段流水线 → ρa + Occam.dat + ρa 断面 PNG
  - inversion.py     : 1D least_squares 电阻率反演 → R[station, layer] + 断面 PNG
  - occam_bridge.py  : Occam1D cfg/model-p 生成 + 命令行桥接 (含 dry-run)
  [v2 新增 5]
  - forward.py           : 1D TEM 正演 (Guptasarma 61点汉克尔 + Anderson 频→时 + Cole-Cole IP)
  - preprocess_plus.py   : WLF 体系增强预处理 (AATZero 关断精修 + 陷波 + 带通 + MAD 野值 + 衰减曲线提取)
  - ip_inversion.py      : 1D 极化率(IP)反演 (Pelton Cole-Cole 4参数, share_τ/c 稳健模式)
  - signal_viz.py        : WLF curview/verify 风格信号可视化 (4×4 衰减曲线子图 + 多站 Hx 叠加)
  - wlf_bridge.py        : WLF EXE 桥接 (TRANS24 格式转换 + SWP24D 2D反演 + 3D tomo 流水线)
  [v3 新增 2]
  - current_proc.py      : 发射电流档处理 (平台幅值/关断时刻/纹波 QC + 波形 PNG)
  - inv_algorithms.py    : 1D 电阻率多算法反演 (trf / lm / 纯Python occam) + 算法对比
  [v4 新增 2]
  - adv_inversion.py     : 高级反演 (自适应Occam λ寻优+L曲线 / Nielsen LM / CGLS / MCMC不确定性)
                           + 多进程并行 + 逐站缓存断点续跑
  - debye_inversion.py   : IP 谱反演 (双Cole-Cole / Debye分解 NNLS 弛豫谱)

**统一日志接口**：所有模块接受 log_fn(str) 回调。
================================================================================
"""
__version__ = "2.3.0"

# v1 核心模块
from . import dat_parser
from . import excel_meta
from . import coord_loader
from . import preprocess
from . import inversion
from . import occam_bridge

# v2 扩展模块 (M1-M5, M7)
from . import forward           # 1D 正演 (LayeredModel / tem_forward_1d)
from . import preprocess_plus   # 增强预处理 + 衰减曲线 (AATZero/notch/bandpass)
from . import ip_inversion      # 1D IP 极化率反演 (Cole-Cole 4 参数)
from . import signal_viz        # 4×4 信号子图 + verify_kru 叠加
from . import wlf_bridge        # WLF EXE 桥接 (2D/3D/格式转换)

# v3 扩展模块
from . import current_proc      # 发射电流档处理
from . import inv_algorithms    # 多算法 1D 反演

# v4 扩展模块
from . import adv_inversion     # 高级反演 (自适应Occam/LM-自适应/CGLS/MCMC) + 并行缓存
from . import debye_inversion   # IP 谱反演 (双Cole-Cole / Debye NNLS)
# v5 扩展模块
from . import lci_inversion     # LCI 横向约束联合反演 (Auken & Christiansen) + 拟2D断面

__all__ = (
    # v1
    "dat_parser", "excel_meta", "coord_loader",
    "preprocess", "inversion", "occam_bridge",
    # v2
    "forward", "preprocess_plus", "ip_inversion",
    "signal_viz", "wlf_bridge",
    # v3
    "current_proc", "inv_algorithms",
    # v4
    "adv_inversion", "debye_inversion",
    # v5
    "lci_inversion",
)

