# TFM-TEM-Inversion

![CI](https://github.com/a1763545492/TFM-TEM-Inversion/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Version](https://img.shields.io/badge/v3.0.0-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Offline](https://img.shields.io/badge/100%25-Offline-brightgreen)

时频电磁法（TFM / TEM & IP）一体化处理系统 —— 对标商业 TEM 软件的**纯 Python** 实现。
覆盖从 TEM 原始二进制采集档（DAT）→ 数据体检 → 预处理 → 视电阻率 → 1D 电阻率反演
（12 种算法，含深度学习端到端）→ IP 极化率反演 → 多算法对比 QC → 标准化导出 →
HTML 报告的完整闭环。

> **离线声明**：本软件 100% 离线运行。所有数据仅在本地磁盘读写，绝不上传任何云端或远程服务器。

---

## v3.0.0 亮点

- **深度学习端到端反演**：1D U-Net（`dl`）+ PINN 物理信息 / MDN 混合密度（不确定度量化）/ TCN 时序卷积（`pinn/mdn/tcn`），秒级成像，torch 训练子进程隔离、numpy 推理部署免依赖
- **解析雅可比**：替代前向差分（26 次正演 → 1 次 + 矩阵运算），反演提速 5-10x
- **多测线并行**：`--n-line-jobs` ProcessPool 线级并行，异常隔离
- **数据体检前置**（Step0.5）：结构对齐 / dat 健康 / 坏数据早发现，`--check`
- **结果一键复现**：`--config args_snapshot_xxx.json` 参数快照反向加载 + 环境指纹对比
- **交付基础设施**：GitHub Actions CI 四矩阵、requirements.txt、PyInstaller 打包、CHANGELOG

## 功能特性

### 数据层
| 模块 | 功能 |
|---|---|
| 数据体检（`data_check`） | Excel 声明 vs 磁盘 dat 对齐、发射档缺失、文件截断/NaN/死通道/幅值越界，ERROR/WARN 分级阻断 |
| 发射电流处理（`current_proc`） | 大写校准档 CXXST01 波形分析：平台幅值、关断时刻 t_off、双极性纹波、桩间一致性 QC |
| 预处理（`preprocess`） | Inventory → DAT 加载 → Cube 对齐 → ADC 换算 → 去 DC → 25 时间门 → ρa → QC → Occam.dat |
| 增强预处理（`preprocess_plus`） | AATZero 关断精修、陷波、带通、MAD 野值剔除、衰减曲线提取 |

### 反演层（12 种算法）
| 类别 | 算法 |
|---|---|
| 迭代法（物理驱动） | TRF 信赖域（默认）/ LM / Occam 光滑反演（三算法对比）|
| 高级迭代 | 自适应 Occam（λ 自动寻优 + L 曲线）/ Nielsen 自适应 LM / CGLS / MCMC 后验不确定性 |
| 拟 2D | LCI 横向约束联合反演（Auken & Christiansen 2004）|
| 端到端（数据驱动） | DL 1D U-Net / PINN 物理信息 / MDN 混合密度 / TCN 时序卷积 —— 秒级，与迭代法交叉验证 |
| IP 极化率 | Pelton Cole-Cole 4 参数（ρ₀/m/τ/c），双残差 ρa+Vt |
| IP 谱 | 双 Cole-Cole、Debye 分解弛豫谱（NNLS）|

### 正演层
| 模块 | 功能 |
|---|---|
| 1D 双引擎 | Boerner 扩散深度近似（默认）+ 真频域严格正演（复电阻率层状递推 + 汉克尔/正弦 DLF）|
| 2D FEM（`fem2d_forward`） | 三角剖分 + PEFC 近源加密，单极-单极拟断面，均匀半空间误差 <12% |
| VTI 各向异性（`anisotropy`） | ρh/ρv λ 参数化，FEM 单元级各向异性正演 |

### 成果层
| 模块 | 功能 |
|---|---|
| 多算法对比 QC（`model_compare`） | 断面并排 / RMSE 统计表 / 逐站拟合曲线（GUI 第⑪页 + CLI Step7）|
| 标准化导出（`export_formats`） | CSV / Surfer XYZ / DSAA-GRD / GeoTIFF（rasterio 可选）|
| HTML 报告（`report_gen`） | 单文件自包含、图片 base64 内嵌，一键生成完整报告 |
| 多时态监测（`timelapse`） | 电阻率变化率断面 + 变化点统计 |
| 3D 可视化（`viz_3d`） | Plotly 交互式 HTML 三维显示 |
| 分辨率分析（`resolution`） | 灵敏度矩阵 / 层分辨率诊断 |

### 工程层
| 模块 | 功能 |
|---|---|
| 解析雅可比 | W 插值权重矩阵，5-10x 提速（`JAC_MODE` 可回退数值差分）|
| 并行（`parallel_utils`） | 站内 `--n-jobs` + 线间 `--n-line-jobs` 双级并行，逐站缓存断点续跑 |
| 快照复现 | 每次运行写 `args_snapshot_*.json`（参数+环境指纹），`--config` 反向加载 |
| CI | GitHub Actions：Ubuntu+Windows × Python 3.11/3.12，pytest 全量 + GUI 冒烟 + CLI help |
| 打包 | `tfm_app.spec` PyInstaller 单 exe（torch 排除，现场免 Python 环境）|

## 目录结构

```
TFM-TEM-Inversion/
  README.md / README_EN.md / CHANGELOG.md
  requirements.txt       # 依赖清单 (torch/rasterio 可选)
  run_tfm.bat            # Windows 一键启动 (GUI 默认 / --cli 无头)
  tfm_gui.py             # GUI 入口 (13 页签)
  tfm_cli_run.py         # CLI 无头入口 (Step0-11 完整流水线)
  make_demo_data.py      # 合成示例数据生成器
  tfm_app.spec           # PyInstaller 打包配置
  .github/workflows/ci.yml
  docs/                  # 技术文档 (cli_usage / dl_inversion / dl_advanced)
  tfm_engine/            # 处理引擎包 (35+ 模块)
    # 数据层: dat_parser excel_meta coord_loader preprocess preprocess_plus
    #          current_proc data_check
    # 反演层: inversion inv_algorithms adv_inversion lci_inversion
    #          ip_inversion debye_inversion dl_inversion dl_advanced
    # 正演层: forward fem_forward fem2d_forward anisotropy
    # 成果层: model_compare export_formats report_gen timelapse
    #          viz_3d resolution signal_viz static_correction denoise_ae
    #          nn_initial
    # 交互层: occam_bridge wlf_bridge project_db parallel_utils
  tests/                 # pytest 套件 (88 tests)
```

## 安装

- Python ≥ 3.10（本机开发环境 Anaconda Python 3.13）
- `pip install -r requirements.txt`
- 可选：`pip install torch`（DL 训练加速；推理纯 numpy 无需）、`pip install rasterio`（GeoTIFF 导出）

## 快速开始

### 1. 生成示例数据

```bat
python make_demo_data.py
```

生成 `demo_data/myfile_demo/`（1线 × 2 个 section，16 测站 S1-1..S1-16）。

### 2. 一键跑通全流程（CLI 无头）

```bat
:: 数据体检 → 预处理 → 反演 → 对比QC → 导出 → HTML报告
python tfm_cli_run.py -P demo_data\myfile_demo -O demo_out --check --current --inv --inv-algo trf --compare --export --report

:: 端到端快速成像 (DL 任选 dl/pinn/mdn/tcn)
python tfm_cli_run.py -P demo_data\myfile_demo -O demo_out_dl --inv --inv-algo mdn --report

:: 历史结果一键复现
python tfm_cli_run.py -P demo_data\myfile_demo -O demo_out_v2 --config demo_out\args_snapshot_xxx.json
```

### 3. GUI

双击 `run_tfm.bat` 或 `python tfm_gui.py`。13 个页签：

| # | 页签 | 功能 |
|---|---|---|
| ① | Basic Pipeline | 预处理 + 12 种反演算法主流水线 |
| ② | Layer Model | 层状模型设计器 |
| ③ | Forward Compare | 模型正演 vs 实测曲线叠加 |
| ④ | Signal Viewer | 4×4 衰减曲线 / 多站 Hx 信号检视 |
| ⑤ | IP Inversion | 全线 / 单站 极化率反演 |
| ⑥ | TEM 2D/3D | FMT24 / 2DINV / 3D tomo 桥接 |
| ⑦ | Tx Current | 发射电流档波形与 QC |
| ⑧-⑩ | 高级工具 | 去噪 / 静态校正 / Occam 桥接等 |
| ⑪ | Compare QC | 多算法断面并排 / RMSE / 拟合曲线 / 标准化导出 |
| ⑫ | Batch Queue | 多测线批量任务队列（排队/进度/后台执行）|
| ⑬ | Help | 内置快速上手 / 算法选型 / 版本信息 |

## CLI 流水线（Step0-11）

```
Step0   发射电流档处理          --current
Step0.5 数据体检                --check / --check-deep
Step1-2 预处理 (+增强)          --pre / --pre-plus
Step3   1D 反演 (12 算法)       --inv --inv-algo trf|...|mdn
Step4   Occam1D 桥接            --occam / --no-occam
Step5   HTML 报告               --report
Step6   多时态对比              --timelapse E1 E2
Step7   多算法对比 QC           --compare / --compare-algos
Step8   标准化导出              --export --export-formats csv,xyz,grd,geotiff
Step9   2D FEM 正演             --fem2d
Step10  VTI 各向异性对比        --vti
Step11  IP 极化率反演           --ip-inv
```

所有 Step 均可组合，失败不阻塞后续步骤。详见 [docs/cli_usage.md](docs/cli_usage.md)。

## 示例运行结果

合成数据（1线 × 2 section，16 测站）经 `tfm_cli_run.py --current --inv-algo all` 全流程产出。
预处理 16/16 站通过 QC，发射电流 QC 正常（Ex 双极性方波平台 1.90e6 counts ≈ 0.45 V，t_off = 19.99 ms）。

| 1D 反演算法 | 站数 | RMSE_log10 中位数 |
|---|---|---|
| TRF（信赖域，默认） | 16 | 0.827 |
| LM（Levenberg-Marquardt） | 16 | 0.827 |
| Occam（纯 Python 光滑反演） | 16 | 0.914 |

解读：TRF/LM 拟合略优且带边界约束更稳健；Occam 光滑模型作交叉验证。若需压低整体 RMSE，
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
  `Int16[4:15]` 为 ASCII 签名（大写电流档含 "SurveyRegion"，小写接收档含 "area"，每字符占一个 int16 低字节）
- 数据区 int32 LE，`reshape(4, 544859, Np, order='C')`，通道顺序 0=Ex 1=Ey 2=Hx 3=Hy
- 命名：大写档 `TXCAL01.DAT`（发射电流校准档）；小写档 `cXXXst00Y.dat`（接收档，Y=001/002/003）

### Excel 四表契约（每测线 `X线/发射接收备注信息.xlsx`）

1. **接收对应点位**：列=[日期, 排列, 点号, 通道, 仪器号, 备注]；ffill 语义（日期/排列/点号/仪器号仅首行出现）
2. **发射对应关系**：前 2 行表头；数据行列 2=发射档名，列 4..6=三台仪器接收文件名（行序=权威 section 顺序）
3. **发射时间**：列=[文件名, 发射日期, 开始时间, 结束时间, 电流大小]；电流（如 "30A"）为权威 I_tx
4. **坐标**：前 2 行多级表头；列 0..3=[仪器号, 线号, 点号, 磁棒号]，可含发射源角点大坐标

### 坐标文件契约（项目根 `坐标文件.xls`）

- **MN坐标**（无表头）：station_code, X, Y, Z（如 S1-1..S1-16）
- **源坐标**（无表头）：A1/B1/A2/B2 四个回线角点；奇号线用 A1B1 面积、偶号线用 A2B2 面积

## 算法说明

### 迭代反演

- **正演（扩散深度近似，默认）**：Boerner 风格 δ(t)=√(2ρ_ref·t/μ₀)/2，与 TEM Vt 量纲体系三向自洽
- **正演（真频域严格解）**：复电阻率层状递推 + Guptasarma 47 点 J₁ 汉克尔 DLF + Key 2012 正弦 DLF 频→时转换，
  晚期与 Ward & Hohmann t^-2.5 渐近精确一致；再现"良导中间层 S 控制"物理
- **TRF**：scipy least_squares（信赖域 + 边界 [0.1, 1e8] Ω·m），加权对数数据残差 + √λ·层间平滑
- **LM**：Levenberg-Marquardt 经典阻尼最小二乘
- **Occam（纯 Python）**：Constable 1987 光滑反演，外环迭代线性化 + λ 搜索使 χ²/N→1
- **自适应 Occam / LM / CGLS / MCMC**：λ 12 点网格寻优 + L 曲线 / Nielsen 阻尼 / 共轭梯度 / Metropolis-Hastings 后验
- **LCI**：Auken & Christiansen 2004，scipy.sparse 块对角雅可比 + lsmr，产出连续拟 2D 断面
- **解析雅可比**：W 插值权重矩阵显式构造，替代数值差分，提速 5-10x

### 深度学习端到端（详见 [docs/dl_advanced.md](docs/dl_advanced.md) / [docs/dl_inversion.md](docs/dl_inversion.md)）

- **训练域**：合成集 `make_synthetic_dataset`（块状/斜坡模型，θ=log₁₀ρ ∈ [0.3, 3.2]，25 对数层 5-1500 m，2% 噪声）
- **DL（U-Net）**：1D U-Net 跳连结构，torch 训练 spawn 子进程隔离（规避 MKL 冲突），numpy 推理降级
- **PINN**：物理信息 MLP，正演算子嵌入数据管线，物理一致性强、高噪声友好
- **MDN**：K=3 高斯混合三头输出（π/μ/σ），负对数似然训练，`predict_with_uncertainty` 输出置信区间
- **TCN**：因果膨胀卷积（d=1,2,4,8，感受野 31 覆盖 25 道）+ GAP，训练推理最快
- **DL 定位**：端到端法出**快速成像**（秒级），与迭代法**精细反演**经 Step7 交叉验证

### 2D 有限元 / VTI

- **fem2d_forward**：结构化三角网格 + PEFC 密度 CDF 逆映射近源加密，单极-单极拟断面（r1=a, r2=2a）
- **anisotropy**：VTI λ=√(ρv/ρh) 参数化，均匀 VTI 半空间地表 ρa=ρm（几何平均），FEM 单元级 ρv 支持

### IP 反演

- **Pelton Cole-Cole 4 参数**（ρ₀/m/τ/c）：双残差（ρa+Vt）+ 层间平滑；τ/c 全线共享稳健模式（默认）或逐层独立
- **双 Cole-Cole / Debye 分解**：两种弛豫机制分离 / 15 个对数等距弛豫 + NNLS 非负谱

## 调参纪律（单变量原则，少步长）

1. **RHOA_SCALE_CAL**（预处理，默认 1e6）：ρa 全局校准倍率，第一调参入口（`--rhoa-scale`）
2. **LAMBDA_SMOOTH**（1D 反演，默认 0.3）：层间平滑权重（`--lambda-smooth`）
3. **LAMBDA_LAT**（LCI，默认 1.0）：横向约束权重，按 0.1→1→10 扫描（`--lambda-lat`）
4. **LAMBDA_INIT**（Occam，默认 1.0）：初始正则化因子（`--lambda-init`）
5. **DL 系列**：`--dl-samples`（默认 4000）合成样本数；权重文件缺失自动训练，多算法并存需显式 `weights_path`

每次只动一个参数，观察 RMSE 与断面形态后再进下一步。

## 输出文件说明（out_dir/X线/）

| 文件 | 内容 |
|---|---|
| `X线_occam.dat` | OCCAMITER_FLEX / EMData_1.1 9 列数据 |
| `X线_rhoa_section.png` | 视电阻率 ρa(t) 剖面色带图 |
| `X线_pickup_xyz.csv` | 测点元数据表（坐标、QC 标志、ρa 中值等）|
| `X线_inv_{algo}_R.npy` / `_depth.npy` | 反演真电阻率张量与层中心深度 |
| `X线_inv_{algo}_summary.csv` | 逐站 RMSE / nfev / ρ 范围 |
| `X线_inv_{algo}_section.png` | 反演真电阻率断面图（12 种算法同构命名）|
| `X线_ip_{R,m,tau,c}.npy` | IP 反演四参数断面 |
| `X线_compare_*.png` / `all_lines_compare_stats.csv` | 多算法对比图与统计 |
| `all_lines_compare_stats.csv` → `report_*.html` | 一键 HTML 报告（自包含）|
| `export/` | CSV / XYZ / GRD / GeoTIFF 标准化导出 |
| `data_check_report.json` | 数据体检报告 |
| `args_snapshot_*.json` | 参数快照（`--config` 复现入口）|
| `cli_run_*.log` / `cli_summary_*.json` / `preprocess_report_*.json` | 运行日志与报告 |

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/cli_usage.md](docs/cli_usage.md) | CLI 完整使用指南（Step0-11 逐节详解）|
| [docs/dl_inversion.md](docs/dl_inversion.md) | DL U-Net 端到端反演技术文档 |
| [docs/dl_advanced.md](docs/dl_advanced.md) | PINN/MDN/TCN 高级 DL 算法技术文档 |
| [CHANGELOG.md](CHANGELOG.md) | 全版本变更追溯 |

## 许可证

本项目采用 [MIT License](LICENSE)。Copyright (c) 2026 TFM-TEM-Inversion contributors。

---

English README → [README_EN.md](README_EN.md)
