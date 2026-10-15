# TFM-TEM-Inversion

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Offline](https://img.shields.io/badge/100%25-Offline-brightgreen)

时频电磁法（TFM / TEM & IP）一体化处理系统 —— 对标商业 WLF 软件的**纯 Python** 实现。
覆盖从 WLF 原始二进制采集档（DAT）→ 预处理 → 视电阻率 → 1D 电阻率反演（多算法）→
IP 极化率反演 → Occam1D / WLF EXE 桥接的完整流程。

> **离线声明**：本软件 100% 离线运行。所有数据仅在本地磁盘读写，绝不上传任何云端或远程服务器。

---

## 功能特性

| 模块 | 功能 |
|---|---|
| 发射电流处理 | 大写校准档 CXXST01 波形分析：平台幅值、关断时刻 t_off、双极性纹波、桩间一致性 QC |
| 预处理 | Inventory → DAT 加载 → Cube 对齐（权威测站编码）→ ADC 换算 → 去 DC → 25 时间门 → ρa（中心回线公式）→ QC → Occam.dat |
| 增强预处理 | AATZero 关断精修、陷波、带通、MAD 野值剔除、衰减曲线提取（preprocess_plus） |
| 层状模型正演 | 1D TEM 正演双引擎：Boerner 扩散深度近似（默认，与 WLF Vt 量纲自洽）+ 真频域严格正演（复电阻率层状递推 + Guptasarma J₁ 汉克尔 + Key2012 正弦 DLF），支持 Cole-Cole IP |
| 1D 反演（多算法） | TRF（信赖域，默认）/ LM / 纯 Python Occam（光滑反演），支持三算法对比 |
| 1D 反演（高级算法） | 自适应 Occam（λ 自动寻优 + L 曲线诊断图）/ Nielsen 自适应阻尼 LM / CGLS 共轭梯度 / MCMC 后验不确定性量化 |
| 拟 2D 反演 | LCI 横向约束联合反演（Auken & Christiansen）：相邻测站模型加横向平滑，产出连续拟 2D 断面 |
| 工程性能 | 多进程并行逐站反演（`--n-jobs`）、逐站缓存断点续跑（`--use-cache`，cfg 变更自动失效） |
| IP 极化率反演 | Pelton Cole-Cole 4 参数（ρ, m, τ, c），share-τ/c 稳健模式 |
| IP 谱反演 | 双 Cole-Cole 项拟合、Debye 分解弛豫谱（NNLS 非负约束，debye_inversion） |
| 信号可视化 | WLF curview/verify 风格：4×4 衰减曲线子图、多站 Hx 叠加 |
| 桥接 | Occam1D cfg/model-p/BAT 生成（含 dry-run）；WLF TRANS24 格式转换、SWP24D 2D 反演、3D tomo 流水线 |

## 目录结构

```
TFM-TEM-Inversion/
  README.md            # 本文档
  requirements.txt     # 依赖清单
  run_tfm.bat          # Windows 一键启动（GUI 默认 / --cli 无头）
  tfm_gui.py           # GUI 入口（7 页签）
  tfm_cli_run.py       # CLI 无头入口（完整流水线）
  make_demo_data.py    # 合成示例数据生成器
  tfm_engine/          # 处理引擎包（16 个模块）
    dat_parser.py        excel_meta.py    coord_loader.py
    preprocess.py        preprocess_plus.py
    inversion.py         inv_algorithms.py  ip_inversion.py
    forward.py           occam_bridge.py    signal_viz.py
    wlf_bridge.py        current_proc.py
    adv_inversion.py     debye_inversion.py lci_inversion.py
```

## 安装

- Python ≥ 3.10
- `pip install -r requirements.txt`（numpy / scipy / pandas / matplotlib / openpyxl）

## 快速开始

### 1. 生成示例数据

```bat
python make_demo_data.py
```

生成 `demo_data/myfile_demo/`（1线 × 2 个 section，16 测站 S1-1..S1-16，
8 个 DAT 文件每个约 26 MB + 两份 Excel 元数据）。幂等，可重复运行。

### 2. 一键跑通全流程（CLI 无头）

```bat
python tfm_cli_run.py --project demo_data\myfile_demo --out demo_out --current --inv-algo trf --no-occam
```

或 `run_tfm.bat --cli --project demo_data\myfile_demo --out demo_out --current --no-occam`

### 3. GUI

双击 `run_tfm.bat` 或 `python tfm_gui.py`。7 个页签：

1. **Basic Pipeline** — 预处理 + 1D 反演主流水线
2. **Layer Model** — 层状模型设计器（N 层 × 5 参数）
3. **Forward Compare** — 模型正演 vs 实测曲线叠加
4. **Signal Viewer** — 4×4 衰减曲线 / 多站 Hx 信号检视
5. **IP Inversion** — 全线 / 单站 极化率反演
6. **WLF 2D/3D** — TRANS24 / SWP24D / 3D tomo 桥接
7. **Tx Current** — 发射电流档波形与 QC

## 示例运行结果

以下结果由 `make_demo_data.py` 生成的合成数据（1线 × 2 section，16 测站 S1-1..S1-16）经
`tfm_cli_run.py --current --inv-algo all` 全流程产出。预处理 16/16 站通过 QC（400 道 ρa 数据），
发射电流 QC 正常（Ex 双极性方波平台 1.90e6 counts ≈ 0.45 V，t_off = 19.99 ms）。

| 1D 反演算法 | 站数 | RMSE_log10 中位数 |
|---|---|---|
| TRF（信赖域，默认） | 16 | 0.827 |
| LM（Levenberg-Marquardt） | 16 | 0.827 |
| Occam（纯 Python 光滑反演） | 16 | 0.914 |

解读：三种算法在同一数据集上结果一致（RMSE_log10 < 1.0 属骨架版扩散深度正演的合理区间），
TRF/LM 拟合略优且带边界约束更稳健；Occam 光滑模型作交叉验证。若需压低整体 RMSE，
按调参纪律先调 `RHOA_SCALE_CAL`（ρa 全局校准），再调 `LAMBDA_SMOOTH` / `LAMBDA_INIT`。

| 发射电流波形 QC | ρa 视电阻率断面 |
|---|---|
| ![current](docs/images/current_waveforms.png) | ![rhoa](docs/images/rhoa_section.png) |

| 三算法反演断面对比 | TRF 反演真电阻率断面 |
|---|---|
| ![compare](docs/images/inv_compare.png) | ![trf](docs/images/inv_trf_section.png) |

## 数据格式说明

### DAT 二进制契约（tfm_engine/dat_parser.py 严格校验）

- 文件大小 = `2048 + 4 × 544859 × Np × 4` 字节（Header + int32 LE 数据区，Np 范围 1..10）
- Header 2048 字节，前 64 字节 = 32 个 int16 LE：
  `Int16[11..14]=8224`、`Int16[15]=1024`、`Int16[26]=0x0102`、`Int16[29]=1`、`Int16[27]=256+Np`；
  `Int16[4:15]` 为 ASCII 签名（大写电流档含 "Xi'An,China"，小写接收档含 "area"，每字符占一个 int16 低字节）
- 数据区 int32 LE，`reshape(4, 544859, Np, order='C')`，通道顺序 0=Ex 1=Ey 2=Hx 3=Hy
- 命名：大写档 `CXXST01.DAT`（发射电流校准档）；小写档 `cXXXst00Y.dat`（接收档，Y=001/002/003）

### Excel 四表契约（每测线 `X线/发射接收备注信息.xlsx`）

1. **接收对应点位**：列=[日期, 排列, 点号, 通道, 仪器号, 备注]；ffill 语义（日期/排列/点号/仪器号仅首行出现）
2. **发射对应关系**：前 2 行表头；数据行列 2=发射档名，列 4..6=三台仪器接收文件名（行序=权威 section 顺序）
3. **发射时间**：列=[文件名, 发射日期, 开始时间, 结束时间, 电流大小]；电流（如 "30A"）为权威 I_tx
4. **坐标**：前 2 行多级表头；列 0..3=[仪器号, 线号, 点号, 磁棒号]，可含发射源角点大坐标

### 坐标文件契约（项目根 `坐标文件.xls`）

- **MN坐标**（无表头）：station_code, X, Y, Z（如 S1-1..S1-16）
- **源坐标**（无表头）：A1/B1/A2/B2 四个回线角点；奇号线用 A1B1 面积、偶号线用 A2B2 面积

## 算法说明

- **正演（扩散深度近似，默认）**：Boerner 风格 δ(t)=√(2ρ_ref·t/μ₀)/2，将层状模型 log₁₀ρ 插值映射为 ρa(t)；
  与 WLF Vt 量纲体系三向自洽（预处理 Nabighian ρa ↔ 正演 ↔ 反演）
- **正演（真频域严格解）**：`tem_forward_1d_freq` —— 频率域复电阻率层状递推（空气层置顶，IP 走
  Pelton Cole-Cole 复电阻率路径）+ Guptasarma 47 点 J₁ 汉克尔 DLF（中心回线核）+ Key 2012 201 点
  正弦 DLF 频→时转换，输出真 SI dBz/dt [T/s]，晚期与 Ward & Hohmann t^-2.5 渐近精确一致
  （100 Ω·m 半空间晚期 ρa 恢复 ±5%）；再现扩散深度近似缺失的"良导中间层 S 控制"物理
  （三层 100/20/500 模型 5~20ms 低谷与超晚期向基底爬升）。配套 `apparent_resistivity_dbzdt_freq`
  （W&H 晚期渐近严格反解）与 `tem_forward_1d_freq_vt`（WLF Vt 量纲封装，反演/GUI 可直接替换）。
  IP 限制：晚期 dBz/dt 为 ~11 个量级的宽频相消残差，Cole-Cole 色散扰动会破坏 DLF 相消平衡产生
  伪影，故 t>2τ_eff 平滑回归无 IP 基线（blend 过渡），IP 反演灵敏区 t~τ 附近完整保留
- **TRF**：scipy least_squares（信赖域 + 边界 [0.1, 1e8] Ω·m），残差 = 加权对数数据残差 + √λ·层间平滑
- **LM**：Levenberg-Marquardt 经典阻尼最小二乘（软裁剪无界）
- **Occam（纯 Python）**：Constable 1987 光滑反演，外环迭代线性化 + λ 搜索使 χ²/N→1
- **自适应 Occam**：λ 不再预设候选集，改为 12 点对数网格自动寻优（χ²≤target 取最大 λ），输出 L 曲线诊断图
- **自适应 LM**：Nielsen 阻尼策略（cost 下降 λ×0.5 / 上升 λ×4），按 JᵀJ 对角线缩放，全程有界
- **CGLS**：模型空间共轭梯度（Rodi-Mackie 风格），免矩阵求逆，解 (JᵀJ+λDᵀD)δ=−Jᵀr
- **MCMC**：Metropolis-Hastings 后验采样（先 TRF 取 MAP 初值），输出后验均值/标准差/95% HPD 及后验剖面诊断图
- **LCI 横向约束反演**：Auken & Christiansen 2004 联合反演，相邻测站模型加横向平滑约束（按站间距加权），
  采用 scipy.sparse 块对角雅可比结构 + lsmr 迭代求解，支持 TRF 热启动加速，产出连续拟 2D 断面
- **IP 反演**：Pelton Cole-Cole 4 参数模型（ρ, m, τ, c），支持 τ/c 全线共享的稳健模式
- **IP 谱反演**：双 Cole-Cole 项（两种弛豫机制分离）与 Debye 分解（15 个对数等距弛豫 + NNLS 非负谱）

## 调参纪律（单变量原则，少步长）

1. **RHOA_SCALE_CAL**（预处理，默认 1e6）：ρa 全局校准倍率，第一调参入口（`--rhoa-scale`）
2. **LAMBDA_SMOOTH**（1D 反演，默认 0.3）：层间平滑权重（`--lambda-smooth`）
3. **LAMBDA_LAT**（LCI，默认 1.0）：横向约束权重，越大断面越匀，按 0.1→1→10 扫描（`--lambda-lat`）
4. **LAMBDA_INIT**（Occam，默认 1.0）：初始正则化因子，按 0.1→1→10 扫描（`--lambda-init`）

每次只动一个参数，观察 RMSE 与断面形态后再进下一步。

## 输出文件说明（out_dir/X线/）

| 文件 | 内容 |
|---|---|
| `X线_occam.dat` | OCCAMITER_FLEX / EMData_1.1 9 列数据（X Y Z Time AppRes StdDev Station Section Instruments） |
| `X线_rhoa_section.png` | 视电阻率 ρa(t) 剖面色带图 |
| `X线_pickup_xyz.csv` | 测点元数据表（坐标、QC 标志、ρa 中值等，utf-8-sig） |
| `X线_inv_{algo}_R.npy` / `_depth.npy` | 反演真电阻率张量 [station, 层] 与层中心深度 |
| `X线_inv_{algo}_summary.csv` | 逐站 RMSE / nfev / ρ 范围 |
| `X线_inv_{algo}_section.png` | 1D 反演真电阻率断面图 |
| `X线_inv_lci_section.png` | LCI 拟 2D 断面（左：LCI；右：独立 1D 对比） |
| `X线_current_summary.csv` / `_waveforms.png` | 发射电流 QC 汇总与波形图 |
| `cli_run_*.log` / `cli_summary_*.json` / `preprocess_report_*.json` | 运行日志与参数快照 |

## 许可证

本项目采用 [MIT License](LICENSE)。Copyright (c) 2026 TFM-TEM-Inversion contributors。

---

English README → [README_EN.md](README_EN.md)
