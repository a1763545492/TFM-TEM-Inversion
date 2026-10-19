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

支持算法：`trf`, `lm`, `occam`, `occam_adaptive`, `lm_adaptive`, `cgls`, `mcmc`, `lci`, `dl`

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

# 并行加速
python tfm_cli_run.py ... --inv --n-jobs 4 --n-line-jobs 2
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--inv-algo` | trf | trf/lm/occam/all/occam_adaptive/lm_adaptive/cgls/mcmc/lci/dl |
| `--n-jobs` | 1 | 站内并行进程数 |
| `--n-line-jobs` | 1 | 测线并行进程数（0=自动） |
| `--use-cache` | False | 逐站缓存断点续跑 |
| `--lambda-smooth` | 0.3 | 层间平滑权重 |
| `--n-layers` | 25 | 反演层数 |
| `--lambda-lat` | 1.0 | LCI 横向约束权重 |
| `--mcmc-samples` | 2000 | MCMC 采样数 |

---

### Step11: IP 激发极化反演

```powershell
# 默认共享 τ,c（稳健，推荐）
python tfm_cli_run.py ... --ip-inv

# 每层独立 τ,c（高自由度，需好数据）
python tfm_cli_run.py ... --ip-inv --ip-no-share-tc

# 自定义平滑权重
python tfm_cli_run.py ... --ip-inv --ip-lambda 0.5
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--ip-inv` | 关 | 启用 IP 反演 |
| `--ip-share-tc` | True | 全局共享 τ,c（2N+2 参数） |
| `--ip-no-share-tc` | — | 每层独立 τ,c（4N 参数） |
| `--ip-lambda` | 0.3 | IP 层间平滑权重 |

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
