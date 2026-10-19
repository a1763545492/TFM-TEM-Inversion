# 高级深度学习反演算法扩展 (dl_advanced.py)

> 版本要求：tfm_engine v2.5+ · 更新日期：2026-10-19
> 模块路径：`tfm_engine/dl_advanced.py` · 测试：`tests/test_dl_advanced.py`

基于国内外 **2024–2026 最新时频电磁法反演研究**，在 `dl_inversion.py`（U-Net 端到端）基础上新增三种高级 DL 反演算法。三者与基础 DL 模块共用同一套产物命名、训练数据管线和 numpy 推理架构，可无缝接入现有流水线、对比 QC 与报告生成。

---

## 1. 三种算法总览

| 算法 | 全称 | 学术来源 | 核心思想 | 主要优势 | 典型场景 |
|:---|:---|:---|:---|:---|:---|
| **pinn** | Physics-Informed Neural Network（物理信息神经网络） | 中南大学 2025《基于物理指导的等值反磁通瞬变电磁深度学习反演》；成都理工 2024 博士论文（LRC-ATEMNet） | 网络输出隐式表示电阻率结构，损失函数嵌入正演响应约束 | 物理一致性强、小样本友好 | 高噪声实测数据、需要与物理模型交叉验证 |
| **mdn** | Mixture Density Network（混合密度网络） | 吉林大学 2026《地球物理学报》IP-TEM-MDN | 输出 K 个高斯分量 (π, μ, σ)，构成模型后验分布 | **天然输出不确定度**，可算置信区间 | 多解性强的复杂地质、异常体边界刻画 |
| **tcn** | Temporal Convolutional Network（时序卷积网络） | 国际 2026 DualTCN（Computers & Geosciences） | 因果膨胀卷积堆叠，感受野指数增长 + 全局平均池化 | 训练/推理快于 LSTM/U-Net，并行度高 | 长时序道、大批量测点快速成像 |

### 与现有算法的关系

```
反演算法族谱
├── 迭代法（物理驱动）
│   ├── trf / lm / occam        ← 基础 1D
│   ├── occam_adaptive / lm_adaptive / cgls / mcmc  ← 高级 1D
│   └── lci                     ← 横向约束拟 2D
└── 端到端法（数据驱动）
    ├── dl                      ← U-Net（dl_inversion.py）
    └── pinn / mdn / tcn        ← 本模块（dl_advanced.py）
```

建议用法：端到端法（秒级）出**快速成像**，迭代法（分钟级）出**精细反演**，两者通过 `model_compare`（Step7）并排对比交叉验证。

---

## 2. 架构与数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 训练阶段（一次性，需 torch，spawn 子进程执行）                │
│                                                              │
│  make_synthetic_dataset(n)      ← 复用 dl_inversion 合成管线  │
│        ↓  X = log10ρa (n,25)  Y = θ=log10ρ (n,25)           │
│  train_pinn / train_mdn / train_tcn   （torch, ~1-3 min）    │
│        ↓                                                     │
│  dl_advanced_weights.npz   ← numpy 数组权重落盘              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 推理阶段（逐站，纯 numpy，无需 torch）                        │
│                                                              │
│  {line}_occam.dat → parse_occam_dat                          │
│        ↓  逐站 ρa(t) 观测                                    │
│  predict_station_advanced: log10 时间域重采样到 TRAIN_T(25道) │
│        ↓                                                     │
│  PINNNumpy / MDNNumpy / TCNNumpy .predict(X)                 │
│        ↓  θ = log10ρ (25层)                                  │
│  R = 10^θ → 保存 _R.npy / _depth.npy / _summary.csv / .png   │
└─────────────────────────────────────────────────────────────┘
```

关键设计（与 `dl_inversion` 一致）：

- **torch 懒加载**：主进程绝不 `import torch`（探测用 `importlib.util.find_spec`），训练在 **spawn 子进程**执行，规避 torch/MKL 原生冲突；
- **numpy 推理**：部署机无需安装 torch；
- **产物命名对齐**：`{line}_inv_{algo}_R.npy` 等，与 `model_compare`/`report_gen`/`export_formats` 兼容。

---

## 3. 各算法详解

### 3.1 PINN — 物理信息神经网络

**网络结构**（MLP，时间道 → 深度层电阻率）：

```
输入 X: log10ρa (25 道)
  → Linear(25→128) + Tanh      # 隐藏层 1: 时间特征提取
  → Linear(128→128) + Tanh     # 隐藏层 2: 非线性变换
  → Linear(128→25)             # 输出: θ = log10ρ (25 层)
输出: clip(θ, 0.3, 3.2)        # 2 ~ 1585 Ω·m
```

- 训练：MSE 损失（合成标签）+ 验证集早停参考，`ReduceLROnPlateau` 学习率调度（patience=15, factor=0.5）；
- 物理约束思想：网络以观测响应（而非自由映射）为输入、输出经正演算子校验的电阻率结构，训练数据本身由正演算子 `forward_rhoa` 生成——即"物理嵌入数据管线"；
- 与 U-Net 的区别：U-Net 是卷积时序结构，PINN 是全连接 MLP，参数量更小、对小数据集更稳健。

### 3.2 MDN — 混合密度网络

**网络结构**（特征提取 + 三头输出）：

```
特征提取:
  Linear(25→128) + Tanh → Linear(128→128) + Tanh
三个并行输出头:
  head_pi  → Linear(128→K)     softmax    → 混合权重 π (K=3)
  head_mu  → Linear(128→K×25)  reshape    → 各分量均值 μ (K,25)
  head_sig → Linear(128→K×25)  softplus   → 各分量标准差 σ (K,25)
```

- 训练损失：**负对数似然**（log-sum-exp 数值稳定实现）；
- 点估计：`E[θ] = Σₖ πₖ · μₖ`；
- **不确定度**：`Var[θ] = Σₖ πₖ(σₖ² + μₖ²) − E[θ]²`，调用 `predict_with_uncertainty(X)` 返回 `(mean, std)`；
- σ 经 softplus + 1e-4 下限保证恒正；π 经 softmax 保证归一；
- 解释：std 大 → 该深度层多解性强（等值性/数据分辨不足），可作为质量控制图件。

### 3.3 TCN — 时序卷积网络

**网络结构**（因果膨胀卷积堆叠）：

```
输入 X: log10ρa (25 道) → (B, 1, 25)
  → Conv1d(1→32,  k=3, dilation=1) + ReLU   # 感受野 3
  → Conv1d(32→32, k=3, dilation=2) + ReLU   # 感受野 7
  → Conv1d(32→32, k=3, dilation=4) + ReLU   # 感受野 15
  → Conv1d(32→32, k=3, dilation=8) + ReLU   # 感受野 31 (覆盖全道)
  → 全局平均池化 GAP (B, 32)
  → Linear(32→25)                            # θ = log10ρ
```

- 卷积核 k=3，padding=(K−1)·d/2 保持长度不变，膨胀系数按 2^i 指数增长；
- 感受野 L 层堆叠后 = 1 + (k−1)·(2^L − 1)，4 层即 31 > 25 覆盖全部时间道；
- numpy 推理用本模块自实现 `_np_conv1d(x, w, b, dilation)`（`sliding_window_view` + `einsum`，膨胀采样 `win[..., ::d]`）；
- 优点：训练可完全并行（无时序递推），推理快于 U-Net。

---

## 4. 后端策略

| 环节 | 实现 | 依赖 |
|:---|:---|:---|
| 合成数据集 | `dl_inversion.make_synthetic_dataset`（块状/斜坡模型 + 三点平滑 + forward_rhoa + 2% 噪声） | numpy |
| torch 训练 | `_train_worker` 在 **spawn 子进程**执行，主进程只收退出码 | torch ≥ 2.0 |
| 权重落盘 | `save_weights`（npz，kind=pinn/mdn/tcn） | numpy |
| numpy 推理 | `PINNNumpy` / `MDNNumpy` / `TCNNumpy` | numpy |
| torch 不可用 | `train_dl_advanced` 抛 `RuntimeError`；`load_advanced_inferencer` 若已有权重文件则照常加载 | — |

> 与 `dl_inversion` 相同：权重训练是**一次性离线任务**，训练完成后的推理/反演在任何无 torch 的部署机上均可运行。

---

## 5. 使用指南

### 5.1 CLI

```powershell
# (1) 训练权重（一次性；--dl-samples 可调样本数，默认 4000）
python tfm_cli_run.py --dl-train-adv pinn
python tfm_cli_run.py --dl-train-adv mdn
python tfm_cli_run.py --dl-train-adv tcn

# (2) 全流水线反演（权重缺失时自动训练）
python tfm_cli_run.py --project <数据根> --out <输出根> --inv --inv-algo pinn
python tfm_cli_run.py --project <数据根> --out <输出根> --inv --inv-algo mdn
python tfm_cli_run.py --project <数据根> --out <输出根> --inv --inv-algo tcn
```

| 参数 | 默认 | 说明 |
|:---|:---|:---|
| `--inv-algo pinn/mdn/tcn` | trf | 选择高级 DL 反演算法（Step3 分支自动路由到 `run_all_advanced_dl`） |
| `--dl-train-adv pinn/mdn/tcn` | — | 仅训练权重后退出（独立入口，rc=0） |
| `--dl-samples` | 4000 | 合成训练样本数（三种算法共用） |

### 5.2 Python API

```python
from tfm_engine.dl_advanced import (
    train_dl_advanced, load_advanced_inferencer,
    predict_station_advanced, invert_line_advanced_dl,
    run_all_advanced_dl, MDNNumpy,
)

# —— 训练（有 torch 时）——
train_dl_advanced('mdn', n_train=4000, epochs=100)

# —— 加载推理器（无 torch 也能跑）——
inferencer, kind = load_advanced_inferencer('mdn')   # 权重缺失自动训练

# —— 单站预测 ——
theta = predict_station_advanced(rhoa_obs, t_obs, inferencer, n_layers=25)

# —— MDN 专属：不确定度量化 ——
mean, std = inferencer.predict_with_uncertainty(X)   # std 即置信度参考

# —— 单线反演（产物自动落盘）——
result = invert_line_advanced_dl('1线', out_root, algo='pinn')
# → {'stations': 30, 'rmse_median_log': 0.018, 'algo': 'pinn'}

# —— 全部测线 ——
results = run_all_advanced_dl(out_root, algo='tcn')
```

### 5.3 GUI

第 12 Tab（Batch Queue）反演算法下拉框与 GUI 基础页算法选择中，高级 DL 算法与 `dl` 一致接入（经 `invert_line_advanced_dl` 分支，见 `tfm_gui.py` 算法清单）。

---

## 6. API 参考

| 函数/类 | 签名 | 说明 |
|:---|:---|:---|
| `train_dl_advanced` | `(algo='pinn', weights_path=WEIGHTS_ADV_PATH, n_train=4000, epochs=100, seed=42, log_fn=None) -> Path` | 训练入口，spawn 子进程，权重落盘后返回路径 |
| `load_advanced_inferencer` | `(algo='pinn', weights_path=None, auto_train=True, log_fn=None) -> (inferencer, kind)` | 加载 numpy 推理器；权重缺失且 auto_train=True 时自动训练 |
| `predict_station_advanced` | `(rhoa_obs, t_obs, inferencer, n_layers=25) -> np.ndarray` | 单站预测：log10 时间域重采样到 25 训练道 → θ(log10ρ)，输出再插值到 n_layers |
| `invert_line_advanced_dl` | `(line_tag, out_root, algo='pinn', inferencer=None, n_layers=25, log_fn=None) -> dict` | 单线反演：逐站预测 + forward_rhoa 回算 RMSE + 断面图落盘 |
| `run_all_advanced_dl` | `(out_root, algo='pinn', weights_path=None, n_layers=25, log_fn=None) -> dict` | 扫描 out_root 下所有 `*线` 目录，权重只加载一次 |
| `PINNNumpy` | `.predict(X) -> (n, n_layers)` | MLP tanh 前向，输出 clip 到 [0.3, 3.2] |
| `MDNNumpy` | `.predict(X)` / `.predict_with_uncertainty(X) -> (mean, std)` | 高斯混合加权期望 / 总方差分解不确定度 |
| `TCNNumpy` | `.predict(X) -> (n, n_layers)` | 膨胀卷积堆叠 + GAP + 线性头 |
| `_np_conv1d` | `(x, w, b, dilation=1)` | numpy 一维卷积（支持膨胀），供 TCN 推理 |

> 坐标约定：`X` 为 log10ρa 观测 (n, n_gates)；输出 θ 为 **log10ρ**，转电阻率用 `10**θ`。

---

## 7. 权重文件格式

文件：`tfm_engine/dl_advanced_weights.npz`（**运行时生成**，不入库 git）。
每个算法带 `kind` 字符串键标明归属；加载时若 kind 与请求不一致，按实际 kind 加载并告警。

| 算法 | npz 键 |
|:---|:---|
| pinn | `w1,b1,w2,b2,w3,b3,xmean,xstd,n_layers,kind` |
| mdn | `w1,b1,w2,b2,w_pi,b_pi,w_mu,b_mu,w_sig,b_sig,xmean,xstd,n_layers,n_mix,kind` |
| tcn | `w0..w3,b0..b3,d0..d3,w_out,b_out,xmean,xstd,n_layers,n_tcn_layers,kind` |

> 注意：三种算法共用一个权重文件会互相覆盖（后训覆盖先训）。如需多算法并存，通过 `weights_path` 参数分别指定路径（`load_advanced_inferencer` / `run_all_advanced_dl` 均支持）。

---

## 8. 输出产物

写入 `out_root/{line}/`，与迭代法/基础 DL 完全同构（可直接进 Step7 对比 / Step8 导出 / Step5 报告）：

| 文件 | 内容 |
|:---|:---|
| `{line}_inv_{algo}_R.npy` | 电阻率断面 (n_stations, n_layers)，Ω·m |
| `{line}_inv_{algo}_depth.npy` | 层中心深度 (n_layers,)，m |
| `{line}_inv_{algo}_summary.csv` | Station, rmse_log, success, n_iter=0, algo |
| `{line}_inv_{algo}_section.png` | log10ρ 断面图（jet_r 色标，深度对数轴） |

`rmse_log` 由 `forward_rhoa` 用反演模型回算视电阻率后与观测比对得到，与迭代法同口径，保证 Step7 对比公平。

---

## 9. 集成点

| 模块 | 集成方式 |
|:---|:---|
| `model_compare` | ALGO_LABELS/ALGO_COLORS 中注册 algo 名即可并排对比（断面/RMSE 表/拟合曲线） |
| `export_formats` | `export_line(out_root, line_tag, algo='pinn')` 直接导出 CSV/XYZ/GRD/GeoTIFF |
| `report_gen` | 反演断面聚合与 HTML 报告自动识别 `{line}_inv_{algo}_*` 产物 |
| `timelapse` | `compare_epochs(..., algo='pinn')` 支持多时态对比 |
| `tfm_cli_run.py` | `--inv-algo` choices + Step3 分支 + `--dl-train-adv` 独立训练入口 |
| `__init__.py` | v7 模块组注册 `dl_advanced` |

---

## 10. 精度边界与注意事项

1. **算法定位**：端到端法输出是**快速成像初值**，不替代迭代法精细反演；建议流水线同时跑 `trf`（或 occam）与任一 DL 算法，经 Step7 交叉验证。
2. **训练域外推**：合成集电阻率覆盖 ~2–1585 Ω·m（log10ρ ∈ [0.3, 3.2]）、25 层对数网格（5–1500 m）。实测超出该范围时输出会被 clip，rmse_log 会显著升高——这是判据，不是 bug。
3. **MDN 不确定度解读**：std 大 ≠ 反演错误，而是该深度层数据分辨不足（等值性）；配合 `resolution` 模块的灵敏度核使用更佳。
4. **PINN 现版说明**：当前实现为"物理嵌入数据管线"（合成数据由正演算子生成 + 输出物理量纲约束）；严格的 PINN（损失函数逐点嵌入 PDE 残差）需正演算子可微化，属后续演进方向。
5. **权重共存**：三种算法默认共用 `dl_advanced_weights.npz`，多算法并存请显式传 `weights_path`。
6. **完全离线**：训练与推理均不访问网络，数据绝不上传云端。

---

## 11. 测试覆盖（tests/test_dl_advanced.py，7 passed）

| 测试 | 覆盖点 |
|:---|:---|
| `TestNpConv1d::test_dilation_1/2` | 膨胀卷积输出形状（d=1/d=2） |
| `TestPINN::test_predict_shape` | MLP 前向形状 + clip 范围 [0.3, 3.2] |
| `TestMDN::test_predict_shape` | 混合加权期望形状 |
| `TestMDN::test_uncertainty` | std 形状正确且非负 |
| `TestTCN::test_predict_shape` | 膨胀卷积堆叠 + GAP + 线性头 |
| `TestAlgoMap::test_algos_exist` | 算法映射表完整性（可训练、可实例化） |

端到端训练子进程冒烟与 `run_all_advanced_dl` 合成数据回归由 torch 可用性测试（skipif 机制）覆盖，与 `test_dl_inversion.py` 模式一致。
