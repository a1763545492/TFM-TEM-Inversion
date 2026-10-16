# DL 端到端反演模块技术文档（`dl_inversion`）

> 模块文件：`tfm_engine/dl_inversion.py`
> 适用版本：v2.6.0 及以上
> 依赖：numpy / pandas / matplotlib（必需）；torch（**可选**，仅训练 U-Net 时需要）

---

## 1. 概述

`dl_inversion` 实现了**深度学习端到端 1D 反演**：将观测视电阻率时间道
`log10(ρa)(t)` 直接映射为层模型参数 `θ = log10(ρ)`，**全程无迭代**（`n_iter=0`），
单站推理耗时毫秒级，适合大批量数据快速成像与迭代法（TRF/LM/Occam 等）的
先验初值或交叉验证参照。

与其他反演算法一致，产物命名对齐 `{line}_inv_dl_R.npy / _depth.npy /
_summary.csv / _section.png`，因此可直接被 `model_compare`（多算法对比）、
`report_gen`（HTML 报告）、`timelapse`（多时态对比）等下游模块消费。

---

## 2. 设计动机：端到端 vs 迭代反演

| 维度 | 迭代法（TRF/LM/Occam/LCI） | DL 端到端 |
|---|---|---|
| 单站耗时 | 秒级（数十次正演+雅可比） | 毫秒级（一次矩阵/卷积运算） |
| 需要初值 | 是（敏感性大） | 否 |
| 不确定性刻画 | MCMC/L 曲线等 | 无（仅确定性映射） |
| 物理一致性 | 正演拟合约束 | 由训练分布隐式保证 |
| 定位 | 精反演 / 成果解释 | 快速成像 / 初值 / 质量互检 |

**推荐工作流**：DL 先跑全线秒级出断面 → RMSE 异常大的站再用迭代法精反演。

---

## 3. 总体架构与数据流

```
                ┌──────────────────────────────────────────┐
                │  合成训练集 make_synthetic_dataset        │
                │  random_layer_model → forward_rhoa 正演   │
                │  X = 含噪 log10ρa (n×25)                  │
                │  Y = 真值 log10ρ  (n×25)                  │
                └───────────────┬──────────────────────────┘
                                │ train_dl (backend='auto')
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   torch 可用（spawn 子进程）                 无 torch / backend='linear'
   UNet1D 训练 → 权重 dict                   岭回归闭式解 train_linear
              │                                    │
              └──────── save_weights → dl_weights.npz ────────┘
                                │
                                ▼  load_inferencer（按 kind 字段分发）
              ┌─────────────────────────────────┐
              │ UNetNumpy（numpy 复现前向）      │  或  LinearNumpy
              └───────────────┬─────────────────┘
                              ▼
        真实数据逐站 invert_line_dl：
        {line}_occam.dat → predict_station（log10 时间域插值到 TRAIN_T）
        → θ → forward_rhoa 回算 rmse_log → 产物 4 件套
```

---

## 4. 双后端详解

### 4.1 合成训练集（物理正演驱动，无需真实标签）

- `random_layer_model(rng, n_layers=25)`：随机层状模型生成器
  - 基值 `log10ρ ∈ [0.7, 2.6]`（约 5–400 Ω·m）；
  - 75% 概率生成 **2–4 段块状层**（段间扰动 ±1.2），否则生成**线性渐变层**
    （斜率 ±0.04/层 + 随机截距）；
  - 50% 概率做 `[0.2, 0.6, 0.2]` 三点卷积平滑，模拟真实地层过渡；
  - 最终裁剪到 `[0.3, 3.2]`（约 2–1585 Ω·m）。
- `make_synthetic_dataset(n=4000, n_gates=25, n_layers=25, noise=0.02, seed=42)`
  - 时间道固定为 `TRAIN_T = geomspace(2e-6, 2e-2, 25)`（10 μs 级 – 20 ms）；
  - 深度层中心 `depth_layer_centers(25, 5.0, 1500.0)`（5–1500 m 对数分布）；
  - `X[i] = log10(ρa) + N(0, 0.02)`，`Y[i] = θ`。

### 4.2 torch U-Net 后端（训练）

`train_unet(X, Y, epochs=60, batch=128, lr=1e-3, seed=42, val_frac=0.1)`

网络为**单层级联 1D U-Net**（同长度输出，参数量约 4 万）：

```
输入 (B,1,L=25)
 ├─ cv1: Conv1d(1→32, k=5, same) + ReLU         → h (B,32,25)
 ├─ MaxPool1d(2)                                 → (B,32,12)
 ├─ cv2: Conv1d(32→64, k=5, same) + ReLU        → b (B,64,12)
 ├─ repeat_interleave ×2 上采样                  → (B,64,24)
 ├─ 奇数长度边缘复制补齐(replicate pad)          → (B,64,25)
 ├─ cv3: Conv1d(64+32→32, k=5, same) + ReLU     ← 跳连 concat h
 └─ cv4: Conv1d(32→1, k=1)                       → 输出 (B,1,25)
```

- Adam + MSE，末尾 10% 样本作验证集，每 10 轮打印 `train/val` 损失；
- 训练完成即导出 `w1..w4 / b1..b4` numpy 权重 dict——**torch 只在训练时使用**。

### 4.3 numpy 推理后端（部署）

`UNetNumpy(par)` 用纯 numpy 逐层复现上述前向：

| 算子 | numpy 实现 |
|---|---|
| Conv1d(same) | `sliding_window_view` + `einsum('bclk,ock->bol')` |
| MaxPool1d(2) | reshape 成对后 `max(axis=-1)` |
| 上采样 | `np.repeat(·, 2, axis=-1)` + `mode='edge'` 补齐奇数长度 |
| 跳连 | `np.concatenate([u, h], axis=1)` |

推理前按权重文件中的 `xmean/xstd` 标准化输入。随机权重下与 torch 前向
一致（`atol=1e-5`，测试保证），**部署机无需安装 torch**。

### 4.4 线性降级后端（保底）

`train_linear(X, Y, alpha=1e-2)`：标准化输入加偏置列后做**岭回归闭式解**
（`solve(XbᵀXb + αI, XbᵀY)`），秒级训练、无任何神经网络依赖。
线性后端精度低于 U-Net，但足以捕捉"高阻/低阻趋势"，适合无 torch 环境
或快速粗成像。

### 4.5 后端选择逻辑

| 场景 | 行为 |
|---|---|
| `train_dl(backend='auto')` + 环境有 torch | spawn 子进程训练 U-Net |
| `train_dl(backend='auto')` + 无 torch | 直接训练线性后端 |
| `train_dl(backend='torch')` + 无 torch | 打印提示后回退线性 |
| `load_inferencer(...)` 权重缺失 + `auto_train=True` | 自动训练线性并落盘（n=3000, seed=7） |
| `load_inferencer(...)` 权重缺失 + `auto_train=False` | 抛 `FileNotFoundError` |

---

## 5. 关键工程决策

### 5.1 torch 懒加载与 MKL 冲突规避（重要）

**问题**：Anaconda 环境下 `import torch` 后再调用 scipy LAPACK（如
`least_squares` 内的 SVD）会触发原生层 `Fatal Python error: Aborted`
（torch 的 OpenMP/CUDA 运行库与 MKL 冲突）。这会波及整个主流水线。

**方案**（三重防护）：
1. **探测不加载**：`torch_available()` 用 `importlib.util.find_spec`
   探测，绝不 `import torch`；模块级 `_HAS_TORCH` 仅为兼容保留。
2. **懒加载**：`_load_torch()` 仅在真正训练时 import。
3. **spawn 子进程训练**：`train_dl` 用
   `multiprocessing.get_context('spawn').Process` 在**独立进程**中完成
   torch 训练并落盘权重；主进程随后只加载 numpy 权重，永不加载 torch。

> 因此：主程序（GUI/CLI/pytest/其他反演算法）所在进程**零 torch 依赖**。

### 5.2 U-Net 奇数长度处理

`L=25` 经 MaxPool(2) 后变 12，`repeat_interleave×2` 上采样回 24，与跳连
`h`（25）拼接时维度不匹配。修正为：不足处用**边缘复制 padding**
（torch `mode='replicate'` / numpy `mode='edge'`）补齐最后 1 个通道，
双端实现严格一致。

### 5.3 真实数据的时间道重采样

实测时间门与训练道 `TRAIN_T`（25 道）不一致。`predict_station`：
1. 按 `t` 升序排序，取 `log10(t)` / `log10(ρa)`；
2. `np.interp` 插值到 `log10(TRAIN_T)`（对数域线性插值）；
3. 推理得到 25 层 θ；若与 `n_layers` 不符，再按归一化深度线性插值；
4. 裁剪到 `[0.3, 3.2]`，与训练分布一致。

---

## 6. 使用指南

### 6.1 命令行（CLI）

```powershell
# ① 仅训练 DL 权重（torch 可用时自动 U-Net，否则线性；训练后退出）
python tfm_cli_run.py --project <数据根> --out <输出根> --dl-train --dl-samples 4000

# ② 完整流水线，反演阶段使用 dl
python tfm_cli_run.py --project <数据根> --out <输出根> --inv --inv-algo dl

# ③ 显式指定层数（默认取 N_LAYERS=25）
python tfm_cli_run.py ... --inv --inv-algo dl --n-layers 25
```

### 6.2 GUI

- **基础处理页**：`1D 反演算法` 下拉框选择 `dl`，勾选反演后执行；
- **批量任务队列（第 12 Tab）**：反演阶段算法下拉框同样支持 `dl`，
  可与其他阶段（预处理/对比 QC/导出/HTML 报告）组队排队执行。

### 6.3 Python API

```python
from tfm_engine.dl_inversion import train_dl, load_inferencer, invert_line_dl

# 1) 训练（可选；权重缺失时 load_inferencer 会自动训练线性降级）
train_dl(n_train=4000, epochs=60, backend='auto')

# 2) 加载推理器（kind == 'unet' 或 'linear'）
inf = load_inferencer()                      # 默认 tfm_engine/dl_weights.npz

# 3) 单线反演（输入为预处理产物 {line}_occam.dat）
res = invert_line_dl('1线', 'out_root', inferencer=inf)
print(res)   # {'stations': 20, 'rmse_median_log': 0.11, 'algo': 'dl'}
```

---

## 7. API 参考

| 函数/类 | 说明 |
|---|---|
| `torch_available() -> bool` | 探测 torch 是否安装（不 import） |
| `random_layer_model(rng, n_layers=25)` | 随机层状 log10ρ 模型（块状/渐变+平滑） |
| `make_synthetic_dataset(n=4000, n_gates=25, n_layers=25, noise=0.02, seed=42)` | 生成训练对 `X=log10ρa(含噪)`, `Y=log10ρ` |
| `train_unet(X, Y, epochs=60, batch=128, lr=1e-3, seed=42, val_frac=0.1)` | 训练 U-Net，返回权重 dict（需 torch） |
| `train_linear(X, Y, alpha=1e-2)` | 岭回归闭式解，返回 `{'W','xmean','xstd'}` |
| `save_weights(path, params, kind)` | 落盘 .npz，自动附加 `kind` 与 `train_t` |
| `load_inferencer(path=None, auto_train=True, log_fn=None)` | 加载推理器；缺失时自动训练线性并落盘 |
| `train_dl(weights_path=WEIGHTS_PATH, n_train=4000, epochs=60, backend='auto', seed=42, log_fn=None)` | 训练入口；torch 走 spawn 子进程 |
| `predict_station(rhoa_obs, t_obs, inferencer, n_layers=25)` | 单站：时间道重采样→推理→裁剪 |
| `invert_line_dl(line_tag, out_root, inferencer=None, n_layers=25, log_fn=None)` | 单线反演，写产物 4 件套；返回 `{'stations','rmse_median_log','algo'}` |
| `run_all_dl(out_root, weights_path=None, n_layers=25, log_fn=None)` | 遍历 `out_root` 下 `[1-5]线` 目录逐线反演；权重只加载一次全线复用 |
| `UNetNumpy(par)` / `LinearNumpy(par)` | numpy 推理器，均有 `.kind` 与 `.predict(X)->(n,Nl)` |

---

## 8. 权重文件格式（`tfm_engine/dl_weights.npz`）

| 键 | U-Net | 线性 | 说明 |
|---|:-:|:-:|---|
| `kind` | ✔ | ✔ | `'unet'` / `'linear'`，`load_inferencer` 分发依据 |
| `train_t` | ✔ | ✔ | 训练时间道 `TRAIN_T`（25 道，geomspace） |
| `xmean`/`xstd` | ✔ | ✔ | 输入标准化常数 |
| `w1..w4` / `b1..b4` | ✔ | — | 四层卷积权重/偏置（numpy 可直接加载） |
| `W` | — | ✔ | `(Ng+1, Nl)`岭回归系数（末行为偏置） |

> 权重文件由训练生成，**不纳入 git 版本管理**；换数据类型/噪声水平建议重训。

---

## 9. 输出产物（写入 `out_root/{line}/`）

| 文件 | 内容 |
|---|---|
| `{line}_inv_dl_R.npy` | `(n_st, 25)` **线性**电阻率 `10**θ` |
| `{line}_inv_dl_depth.npy` | 25 个对数层中心深度（5–1500 m） |
| `{line}_inv_dl_summary.csv` | `Station, rmse_log, success, n_iter=0, algo='dl'` |
| `{line}_inv_dl_section.png` | log10ρ 断面图（jet_r，深度 log 轴，标题含 RMSE 中位数） |

`rmse_log` 定义：用模型 θ 经 `forward_rhoa` 回算**实测时间门**上的 ρa，
与观测取 `log10` 后的均方根误差——与其他算法的 `rmse_log` 统计口径一致，
可直接进入 `model_compare` 的 RMSE 对比表。

---

## 10. 与系统其他模块的集成

| 模块 | 集成点 |
|---|---|
| `inversion` | 复用 `depth_layer_centers` / `forward_rhoa` / `parse_occam_dat` |
| `model_compare` | 第 9 种算法：`ALGO_LABELS['dl']='DL端到端'`（黑色 `#000000`） |
| `report_gen` | 自动聚合 `_inv_dl_summary.csv` 与断面图进 HTML 报告 |
| `timelapse` | 算法白名单含 `dl`，可做多时态 DL 断面对比 |
| CLI `tfm_cli_run` | `--inv-algo dl`、`--dl-train`、`--dl-samples` |
| GUI `tfm_gui` | 基础页算法下拉框 + 第 12 Tab 批量队列算法列表 |
| 下游输入 | 依赖预处理产物 `{line}_occam.dat`（与其他算法相同） |

---

## 11. 精度与适用性说明

- 训练分布限定：`log10ρ ∈ [0.3, 3.2]`、5–1500 m、25 层、`TRAIN_T` 道集。
  实测数据超出该范围（如极高阻基底、超深目标）时预测被裁剪，需谨慎解读；
- DL 结果**不含不确定性**，也不能替代正演拟合检查——务必参考
  `rmse_log` 与断面连续性；
- 合成集为层状模型（无 2D/3D 响应），对强横向变化地电结构的绝对精度有限，
  推荐定位为**快速成像 + 迭代法交叉验证**。

## 12. 常见问题（FAQ）

**Q1：没有 torch 能用吗？**
可以。`load_inferencer` 在权重缺失时自动训练线性后端并落盘；若目录里已有
U-Net 权重（在别处训练），numpy 推理器照样加载运行。

**Q2：训练时程序崩溃 / exitcode 非 0？**
torch 训练在 spawn 子进程执行，崩溃不会拖垮主程序，`train_dl` 会抛
`RuntimeError`。子进程日志以 `[DL-child]` 前缀打印，便于定位。

**Q3：如何重训并覆盖权重？**
直接再调 `train_dl(...)`（或 CLI `--dl-train`）即可，同路径覆盖写。

**Q4：为什么反演全部站 `success=True, n_iter=0`？**
端到端映射无迭代概念，统一填常量以对齐 summary 表结构。

**Q5：能否自定义时间道/层数？**
改模块常量 `TRAIN_T` / `NL_DEFAULT` / `RHO_REF` 后**必须重训权重**，
否则训练-推理分布不匹配。

## 13. 测试覆盖（`tests/test_dl_inversion.py`）

| 测试 | 验证点 |
|---|---|
| `test_dataset_shapes` | 合成集形状/有限性/数值范围 |
| `test_linear_beats_mean` | 线性后端 MSE 显著优于均值基线，逐样本相关系数中位数 >0.5 |
| `test_auto_train_when_missing` | 权重缺失自动训练+二次加载幂等 |
| `test_numpy_parity_random_weights` | 随机权重下 torch 与 numpy 前向一致 atol=1e-5（spawn 子进程） |
| `test_train_and_save` | `train_dl` 子进程训练 U-Net → 主进程 numpy 加载推理 |
| `test_run_all_dl_linear` | 合成 2 线全流程：5 站/线、产物齐全、`rmse_median_log<0.25` |

运行：`python -m pytest tests/test_dl_inversion.py -q`
