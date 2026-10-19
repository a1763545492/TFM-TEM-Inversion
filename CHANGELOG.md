# 更新日志 (Changelog)

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
格式：`[版本] - 日期`，分 **新增 / 变更 / 修复** 三节。

---

## [3.0.0] - 2026-10-19

工程化里程碑版本：在 v2.x 功能全集基础上补齐交付基础设施。

### 新增
- **高级 DL 反演算法** `dl_advanced.py`：PINN（物理信息神经网络）、MDN（混合密度网络，支持不确定度量化）、TCN（时序卷积网络），torch 训练 spawn 子进程隔离 / numpy 推理，CLI `--inv-algo pinn|mdn|tcn` + `--dl-train-adv`
- **数据体检前置** `data_check.py`：Step0.5 站点完整性/NaN/量纲检查，坏数据早发现
- **配置反向加载**：CLI `--config <snapshot.json>` 从配置快照一键复现历史反演，快照内嵌环境指纹（Python/numpy/scipy 版本、平台）
- **GUI 帮助页**（第 13 Tab）：内置快速上手、算法选型指引、版本信息
- **GitHub Actions CI**：Ubuntu + Windows × Python 3.11/3.12 四矩阵，pytest 全量 + GUI 冒烟 + CLI help 检查
- **requirements.txt**：主依赖清单（含可选依赖说明 torch/rasterio）
- **PyInstaller 打包配置** `tfm_app.spec`：单 exe 分发现场部署

### 变更
- `model_compare` 算法注册表扩至 12 种（新增 pinn/mdn/tcn）
- GUI 基础页与批量队列算法下拉框同步新算法
- 版本号 2.5.0 → 3.0.0（自 2.5 后新增 15+ 模块，跨度达大版本）

---

## [2.5.x] - 2025-10 ~ 2026-06

v7/v6 扩展周期，12 个 GUI Tab 全部成形。

### 新增（v7, 2026-06）
- **解析雅可比矩阵**（`inv_algorithms`/`adv_inversion`/`lci_inversion`）：W 插值权重矩阵替代前向差分，反演提速 5-10x，`JAC_MODE` 开关可回退数值差分
- **多测线并行批处理** `parallel_utils.py`：ProcessPool 线级并行，异常隔离，CLI `--n-line-jobs`
- **多算法对比 QC 仪表盘** `model_compare.py` + GUI 第 11 Tab：断面并排/RMSE 统计/逐站拟合
- **标准化导出** `export_formats.py`：CSV / Surfer XYZ / DSAA-GRD / GeoTIFF（rasterio 可选）
- **HTML 报告生成器** `report_gen.py`：单文件自包含、图片 base64 内嵌、CLI `--report` / GUI 一键生成
- **多时态监测对比** `timelapse.py`：电阻率变化率断面 + 变化点统计，CLI `--timelapse`
- **GUI 批量任务队列**（第 12 Tab）：多测线排队、后台线程、进度跟踪
- **2D 有限元 DC 正演** `fem2d_forward.py`：三角剖分 + PEFC 近源加密，均匀半空间误差 <12%
- **VTI 各向异性** `anisotropy.py`：ρh/ρv λ 参数化，FEM 单元级各向异性
- **DL 端到端 1D 反演** `dl_inversion.py`：1D U-Net，torch 训练子进程 / numpy 推理降级

### 新增（v6, 2025-10）
- 频率域 1D 正演 `fem_forward`、静态效应校正 `static_correction`、
  分辨率矩阵 `resolution`、NN 初始模型 `nn_initial`、去噪自编码器 `denoise_ae`、
  3D 可视化 `viz_3d`、SQLite 项目库 `project_db`
- CLI 流水线 Step7-11：对比 QC / 导出 / FEM2D / VTI / IP 反演

---

## [2.0.0] - 2025 上半年

- LCI 横向约束联合反演 `lci_inversion`（Auken & Christiansen，拟 2D 断面）
- 高级 1D 反演 `adv_inversion`：Occam 自适应 λ（L 曲线）、LM 自适应（Nielsen 阻尼）、CGLS、MCMC 后验
- IP 激发极化反演 `ip_inversion`（Pelton Cole-Cole，ρ₀/m/τ/c 四参数）
- 双 Cole-Cole / Debye 谱反演 `debye_inversion`
- 发射电流档处理 `current_proc`（CXXST01 波形/幅值/关断 QC）
- 基础反演算法库 `inv_algorithms`（TRF/LM/Occam1D 桥接）

## [1.0.0] - 初始版本

- WLF dat 二进制解析 `dat_parser`、Excel 元数据 `excel_meta`、测站坐标 `coord_loader`
- 预处理流水线 `preprocess`（时间门/ρa/叠加）
- 1D 反演核心 `inversion`（前向 `forward_rhoa`、层中心网格）
- Occam1D 商业软件桥接 `occam_bridge`、信号可视化 `signal_viz`
