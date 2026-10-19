# TFM CLI 使用指南

> 命令行入口：`python tfm_cli_run.py`
> 适用版本：v2.6.0+
> 完全离线，数据绝不上传云端

---

## 1. 快速上手

```powershell
# 最简流水线（预处理 + 反演 + Occam 桥接 + 对比 QC）
python tfm_cli_run.py --project D:\MyProject\myfile --out D:\output

# 跳过预处理，只跑反演
python tfm_cli_run.py --project D:\MyProject\myfile --out D:\output --no-preprocess

# 完整生产流水线（含报告 + 导出）
python tfm_cli_run.py --project D:\MyProject\myfile --out D:\output --report --export
```

---

## 2. 流水线步骤总览

| 步骤 | 功能 | 默认开关 | 关键参数 |
|:---|:---|:---:|:---|
| Step0 | 发射电流档处理 | `--current` 关 | `--current` |
| Step1-2 | 预处理 (Inventory→Cube→ρa→Occam.dat) | `--preprocess` 开 | `--rhoa-scale`, `--n-gates` |
| Step3 | 1D 反演 | `--inv` 开 | `--inv-algo`, `--n-jobs`, `--n-line-jobs` |
| Step11 | IP 激发极化反演 | `--ip-inv` 关 | `--ip-share-tc`, `--ip-lambda` |
| Step4 | Occam1D 桥接文件生成 | `--occam` 开 | `--occam-exe`, `--lambda-init` |
| Step7 | **多算法对比 QC 仪表盘** | `--compare` 开 | `--compare-algos` |
| Step8 | 标准化导出 | `--export` 关 | `--export-formats`, `--export-epsg` |
| Step9 | 2D FEM 正演 | `--fem2d` 关 | `--fem2d-xlim`, `--fem2d-zmax` |
| Step10 | VTI 各向异性正演对比 | `--vti` 关 | `--vti-lambda` |
| Step5 | HTML 报告 | `--report` 关 | — |
| Step6 | 多时态对比 | `--timelapse` 关 | `--timelapse E1 E2` |

> 各步骤失败策略：Step0/1-2/3/4 失败会置 `rc` 并跳过后续；Step7-11 失败仅记录 warning，不阻塞报告生成。

---

## 3. Step7 多算法对比 QC 详解

### 3.1 功能

对同一测线下**多种反演算法**的结果进行系统性对比质检：

1. **断面并排对比图**：各算法 log10ρ 断面 + 差异断面（相对于基准算法）
2. **RMSE 统计表**：每算法每测线的拟合优度汇总（`{line}_compare_stats.csv`）
3. **逐站拟合曲线**：关键测站的观测 ρa vs 各算法正演回算曲线（`{line}_compare_fits.png`）

### 3.2 使用方法

```powershell
# 默认：跑完反演后自动对比（扫描 out_root 下所有已有算法结果）
python tfm_cli_run.py --project <数据根> --out <输出根> --inv

# 显式跳过对比
python tfm_cli_run.py ... --no-compare

# 仅对比指定算法（白名单）
python tfm_cli_run.py ... --compare-algos trf,lm,occam

# 配合 all 算法：先跑三算法反演，再自动对比
python tfm_cli_run.py ... --inv --inv-algo all --report
```

### 3.3 对比算法白名单

`--compare-algos` 接受逗号分隔的算法名，未指定时自动扫描 `out_root/{line}/` 下所有 `_inv_{algo}_summary.csv` 文件。

支持算法：`trf`, `lm`, `occam`, `occam_adaptive`, `lm_adaptive`, `cgls`, `mcmc`, `lci`, `dl`, `pinn`, `mdn`, `tcn`

### 3.4 产物文件

写入 `out_root/{line}/`：

| 文件 | 内容 |
|:---|:---|
| `{line}_compare_stats.csv` | 各算法 RMSE / 耗时 / 成功站数统计 |
| `{line}_compare_sections.png` | 多算法断面并排对比图 |
| `{line}_compare_fits.png` | 逐站拟合曲线对比 |

### 3.5 在流水线中的位置

```
Step3 反演完成
    ↓
Step7 对比 QC  ← 在这里
    ↓
Step8 导出 / Step9 FEM / Step10 VTI
    ↓
Step5 HTML 报告  ← 报告自动嵌入对比图
```

> **注意**：Step7 失败不会阻塞后续步骤，仅记录 warning。这样即使对比出问题，报告和导出仍可正常生成。

---

## 4. 各步骤详细参数

### Step0: 电流档处理

```powershell
python tfm_cli_run.py ... --current
```

处理发射电流波形（CXXST01），输出波形图、幅值统计、关断时间 QC。

---

### Step1-2: 预处理

```powershell
python tfm_cli_run.py ... --preprocess \
    --rhoa-scale 1.0e6 \
    --n-gates 25 \
    --t-min-us 2 --t-max-us 20000
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--rhoa-scale` | 1e6 | ρa 全局校准倍率 |
| `--g-adc` | 4194304 | ADC counts/V |
| `--n-gates` | 25 | 时间门数 |
| `--t-min-us` | 2 | 最早时间门 (μs) |
| `--t-max-us` | 20000 | 最晚时间门 (μs) |

---

### Step3: 1D 反演

```powershell
# 单算法
python tfm_cli_run.py ... --inv --inv-algo trf

# 三算法对比
python tfm_cli_run.py ... --inv --inv-algo all

# 高级算法
python tfm_cli_run.py ... --inv --inv-algo occam_adaptive
python tfm_cli_run.py ... --inv --inv-algo mcmc --mcmc-samples 5000

# LCI 横向约束
python tfm_cli_run.py ... --inv --inv-algo lci --lambda-lat 2.0

# DL 端到端
python tfm_cli_run.py ... --inv --inv-algo dl

# 高级 DL 算法（PINN/MDN/TCN，权重缺失时自动训练，需 torch）
python tfm_cli_run.py ... --inv --inv-algo pinn   # 物理信息神经网络，物理约束嵌入损失
python tfm_cli_run.py ... --inv --inv-algo mdn    # 混合密度网络，输出不确定度
python tfm_cli_run.py ... --inv --inv-algo tcn    # 时序卷积网络，快速成像

# 仅训练高级 DL 权重（独立入口，与流水线解耦）
python tfm_cli_run.py --dl-train-adv pinn --dl-samples 4000

# 并行加速
python tfm_cli_run.py ... --inv --n-jobs 4 --n-line-jobs 2
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--inv-algo` | trf | trf/lm/occam/all/occam_adaptive/lm_adaptive/cgls/mcmc/lci/dl/**pinn/mdn/tcn** |
| `--n-jobs` | 1 | 站内并行进程数 |
| `--n-line-jobs` | 1 | 测线并行进程数（0=自动） |
| `--use-cache` | False | 逐站缓存断点续跑 |
| `--lambda-smooth` | 0.3 | 层间平滑权重 |
| `--n-layers` | 25 | 反演层数 |
| `--lambda-lat` | 1.0 | LCI 横向约束权重 |
| `--mcmc-samples` | 2000 | MCMC 采样数 |
| `--dl-samples` | 4000 | DL/高级DL 合成训练样本数 |

> **高级 DL 算法选型**（详见 [dl_advanced.md](dl_advanced.md)）：pinn 物理一致性强、适合高噪声实测；
> mdn 唯一支持不确定度量化（置信区间）；tcn 训练推理最快、适合大批量数据。
> 三者产物命名与迭代法同构（`{line}_inv_{algo}_*`），自动进入 Step7 对比 / Step8 导出 / Step5 报告。

---

### Step11: IP 激发极化反演

#### 11.1 功能概述

IP（Induced Polarization，激发极化）反演基于 **Pelton Cole-Cole 模型**，
同时反演**电阻率 ρ₀** 和**极化率 m**（充电率），以及时间常数 τ、频率相关系数 c。

与 Step3 纯电阻率反演的核心区别：

| 维度 | Step3 纯电阻率 | Step11 IP |
|:---|:---|:---|
| 物理模型 | 纯欧姆导体 | Cole-Cole 复电阻率 |
| 待求参数 | 每层 1 个 (ρ) | 每层 2~4 个 (ρ₀, m, τ, c) |
| 观测数据 | ρa 视电阻率 | **ρa + Vt 双观测** |
| 输出产物 | ρ 断面 | **ρ₀ / m / τ / c 四组断面** |

#### 11.2 模型参数

每层 4 个物理量（Pelton Cole-Cole）：

| 参数 | 符号 | 物理意义 | 默认约束 |
|:---|:---|:---|:---|
| ρ₀ | rho0 | 零频电阻率 (Ω·m) | 0.1 ~ 10⁸ |
| **m** | chargeability | **极化率/充电率** (无量纲) | 0.001 ~ 0.8 |
| τ | tau | 时间常数 (s) | 10⁻⁶ ~ 30 |
| c | c_exp | 频率相关系数 | 0.1 ~ 0.95 |

#### 11.3 两种参数化模式

```powershell
# A) 共享 τ,c（默认，2N+2 参数，稳健推荐）
#    所有层共用一组 τ,c，只反演每层的 ρ₀ 和 m
python tfm_cli_run.py ... --ip-inv

# B) 每层独立 τ,c（4N 参数，高自由度，需高质量数据）
#    每层独立反演 ρ₀, m, τ, c
python tfm_cli_run.py ... --ip-inv --ip-no-share-tc
```

| 模式 | 参数数 | 适用场景 | 推荐 |
|:---|:---|:---|:---:|
| 共享 τ,c | 2N+2 | 常规数据，快速稳健 | ✓ 推荐 |
| 每层独立 | 4N | 高品质数据，精细地层刻画 | 数据质量好时用 |

#### 11.4 残差设计

采用**双残差 + 层间平滑**架构：

```
r_data = [ log10(ρa_pred/ρa_obs)/0.05 ,   ← ρa 残差 (Ng 项)
           log10(Vt_pred/Vt_obs)/0.08 ]   ← Vt 残差 (Ng 项)

r_smooth = √λ · diff(ρ₀_per_layer)         ← ρ₀ 层间平滑
         + √λ·0.7 · diff(m_per_layer)      ← m 层间平滑
```

#### 11.5 双数据路径

IP 反演需要同时有 ρa 和 Vt，系统根据数据可用性自动选择路径：

**路径 A：Vt-only 工业模式（优先，推荐）**
- 来源：`{line}_decay_curves.npz`（预处理 S4 完整衰减曲线）
- 特点：直接用真实 Vt 观测（含 IP 弛豫尾部），只拟合 Vt 残差
- 要求：decay_curves.npz 存在且 `prefer_vt_only=True`

**路径 B：双残差回退模式**
- 来源：`{line}_occam.dat` 中的 ρa + 由 ρa 反推的 Vt
- 特点：ρa 和 Vt 同时进入残差
- 触发：decay NPZ 不存在，或用户关闭 `prefer_vt_only`

> **注意**：若数据在预处理阶段已做 IP 去除（如标准 ρa 处理流程），
> ρa 中的 IP 信息可能丢失，此时路径 A（直接 Vt）更为可靠。

#### 11.6 使用方法

```powershell
# 基础：默认共享 τ,c，使用 decay NPZ（若存在）
python tfm_cli_run.py --project <数据根> --out <输出根> --ip-inv

# 全流程：预处理 + 电阻率反演 + IP 反演 + 报告
python tfm_cli_run.py --project <数据根> --out <输出根> \
    --inv --inv-algo trf --ip-inv --report

# 每层独立 τ,c（高自由度）
python tfm_cli_run.py ... --ip-inv --ip-no-share-tc

# 自定义 IP 平滑权重
python tfm_cli_run.py ... --ip-inv --ip-lambda 0.5
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--ip-inv` | 关 | 启用 IP 反演 |
| `--ip-share-tc` | True | 全局共享 τ,c（2N+2 参数，稳健） |
| `--ip-no-share-tc` | — | 每层独立 τ,c（4N 参数，高自由度） |
| `--ip-lambda` | 0.3 | IP 层间平滑权重（与 1D ρ 反演一致） |

#### 11.7 输入依赖

| 文件 | 来源 | 用途 |
|:---|:---|:---|
| `{line}_occam.dat` | 预处理 Step2 | ρa 观测（初值估计 + 路径 B） |
| `{line}_decay_curves.npz` | 预处理 Step4（可选） | 真实 Vt 观测（路径 A 优先） |

#### 11.8 输出产物

写入 `out_root/{line}/`：

| 文件 | 内容 | 色标 |
|:---|:---|:---|
| `{line}_ip_R_rho.npy` | 电阻率 ρ₀ 断面 (n_st, n_l) | — |
| `{line}_ip_R_m.npy` | **极化率 m 断面** (n_st, n_l) | — |
| `{line}_ip_R_tau.npy` | 时间常数 τ 断面 (n_st, n_l) | — |
| `{line}_ip_R_c.npy` | 频率相关系数 c 断面 (n_st, n_l) | — |
| `{line}_ip_depth.npy` | 层中心深度 | — |
| `{line}_ip_inversion_summary.csv` | 每站 RMSE_rhoa / RMSE_vt / nfev / vt_only | — |
| `{line}_ip_rho_section.png` | ρ₀ 断面图 | jet |
| `{line}_ip_m_section.png` | **m 断面图** | plasma (vmax=0.5) |
| `{line}_ip_tau_section.png` | log10τ 断面图 | cividis |

#### 11.9 在流水线中的位置

```
Step3  1D电阻率反演完成
    ↓
Step11 IP反演  ← 在这里（独立步骤，不依赖 Step3 结果）
    ↓
Step4  Occam → Step7 对比 → Step5 报告
```

> IP 反演失败不阻塞主流程，仅记录 warning。因为 IP 是"增值分析"，
> 即使失败，电阻率反演结果仍完整可用。

---

### Step4: Occam 桥接

```powershell
python tfm_cli_run.py ... --occam --occam-exe D:\Tools\Occam1D.exe --lambda-init 1.0
```

生成 Occam1D 输入文件（cfg/model-p/BAT），支持 DRY-RUN（无 exe 时仅生成模板）。

---

### Step8: 标准化导出

```powershell
# 默认 CSV+XYZ+GRD
python tfm_cli_run.py ... --export

# 仅 CSV
python tfm_cli_run.py ... --export --export-formats csv

# 含 GeoTIFF（需 rasterio）
python tfm_cli_run.py ... --export --export-formats csv,xyz,grd,geotiff --export-epsg 4547

# log10 标度
python tfm_cli_run.py ... --export --export-scale log10
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--export` | 关 | 启用导出 |
| `--export-formats` | csv,xyz,grd | 格式逗号分隔 |
| `--export-algo` | 同 inv-algo | 源断面算法 |
| `--export-epsg` | None | GeoTIFF 坐标系 EPSG |
| `--export-scale` | linear | linear/log10 |

---

### Step9: 2D FEM 正演

基于反演断面做 2D 有限元直流电法正演，评估布极方案。

```powershell
python tfm_cli_run.py ... --fem2d \
    --fem2d-xlim "-300,300" \
    --fem2d-zmax 300 \
    --fem2d-nx 160 --fem2d-nz 70 \
    --fem2d-spacings "5,10,20,40"
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--fem2d` | 关 | 启用 FEM 正演 |
| `--fem2d-algo` | 同 inv-algo | 源断面算法 |
| `--fem2d-xlim` | -300,300 | X 范围 |
| `--fem2d-zmax` | 300 | 最大深度 |
| `--fem2d-nx` / `--fem2d-nz` | 160 / 70 | 网格节点数 |
| `--fem2d-spacings` | 5,10,20,40 | 极距列表 |

产物：`{line}_fem2d_pseudo.csv` + `{line}_fem2d_pseudo.png`

---

### Step10: VTI 各向异性正演对比

```powershell
python tfm_cli_run.py ... --vti --vti-lambda 1.5
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--vti` | 关 | 启用 VTI 对比 |
| `--vti-lambda` | 1.5 | λ=√(ρv/ρh)，λ=1 为各向同性 |
| `--vti-algo` | 同 inv-algo | 源断面算法 |

产物：`{line}_vti_compare.csv` + `{line}_vti_compare.png`（三联图：ISO / VTI / 比值）

---

### Step5: HTML 报告

```powershell
python tfm_cli_run.py ... --report
```

一键生成自包含 HTML 报告，嵌入预处理 QC、反演断面、对比图、拟合曲线、统计摘要。

---

### Step6: 多时态对比

```powershell
python tfm_cli_run.py ... --timelapse D:\output_epoch1 D:\output_epoch2 --timelapse-labels "2024,2025"
```

对比两期观测的电阻率变化，输出变化率断面和统计表。

---

## 5. 常用组合命令

```powershell
# 生产流水线：预处理 + 三算法反演 + 对比 + 报告
python tfm_cli_run.py -P D:\data\myfile -O D:\output --inv-algo all --report

# 快速 DL 成像（先训练权重，再反演）
python tfm_cli_run.py -P D:\data\myfile -O D:\output --dl-train --dl-samples 4000
python tfm_cli_run.py -P D:\data\myfile -O D:\output --inv --inv-algo dl --report

# 高级 DL 成像（PINN/MDN/TCN 任选其一）
python tfm_cli_run.py -P D:\data\myfile -O D:\output --dl-train-adv mdn
python tfm_cli_run.py -P D:\data\myfile -O D:\output --inv --inv-algo mdn --report

# 全流程含 IP + 导出 + FEM
python tfm_cli_run.py -P D:\data\myfile -O D:\output \
    --inv --inv-algo trf --ip-inv --export --fem2d --report

# 仅补跑对比和报告（反演结果已存在）
python tfm_cli_run.py -P D:\data\myfile -O D:\output \
    --no-preprocess --no-inv --no-occam --compare --report
```

---

## 6. 返回值约定

| rc | 含义 |
|:---|:---|
| 0 | 全部成功 |
| 10 | 数据根目录不存在 |
| 15 | Step0 电流档失败 |
| 20 | Step1-2 预处理失败 |
| 30 | Step3 反演失败 |
| 40 | Step4 Occam 桥接失败 |
| 50 | Step5 报告失败 |
| 60 | Step6 多时态对比失败 |

> Step7-11 失败仅记录 warning，不修改 rc。

---

## 7. 日志与摘要

每次运行生成两个文件在 `out_root/`：

- `cli_run_YYYYMMDD_HHMMSS.log` — 完整控制台日志
- `cli_summary_YYYYMMDD_HHMMSS.json` — 结构化摘要（各步骤产物路径、统计、warning 列表）
