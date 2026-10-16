# -*- coding: utf-8 -*-
"""
dl_inversion.py — 深度学习端到端 1D 反演 (U-Net, 可离线; torch 可选)
======================================================================
观测 log10(ρa) 时间道 → 直接映射到模型 log10(ρ) 层参数, 无迭代:

  后端 (按可用性自动选择):
    1) torch 后端: 单层级联 1D U-Net (32/64 通道, 同尺寸卷积+跳连),
       Adam/MSE, 合成数据集 (forward_rhoa 生成随机层状模型 + 2% 噪声)
    2) 纯 numpy 降级:
       a. U-Net **推理器** — 仅依赖 numpy, 读取 torch 导出的 .npz 权重,
          conv1d/maxpool/最近上采样/跳连全部 numpy 实现, 部署机无需 torch
       b. 无权重文件时 — 解析岭回归线性端到端映射 (合成数据闭式解, 秒级)

  权重文件: tfm_engine/dl_weights.npz (随模型保存, 可重复训练覆盖)
  产物与其他算法一致: {line}_inv_dl_R.npy / _depth.npy / _summary.csv / _section.png
"""
from __future__ import annotations
import re
import importlib.util
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
rcParams['axes.unicode_minus'] = False

from .inversion import (depth_layer_centers, forward_rhoa, parse_occam_dat)

WEIGHTS_PATH = Path(__file__).with_name('dl_weights.npz')
TRAIN_T = np.geomspace(2e-6, 2e-2, 25)
NL_DEFAULT = 25
RHO_REF = 100.0


def torch_available() -> bool:
    """探测 torch 是否安装 (不 import, 避免 torch/MKL 与 scipy LAPACK 原生冲突)。"""
    try:
        return importlib.util.find_spec('torch') is not None
    except Exception:
        return False


_HAS_TORCH = torch_available()


def _load_torch():
    """训练时才真正 import torch (懒加载)。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ============================================================
# 1) 合成训练集 (物理正演驱动, 无需真实标签)
# ============================================================
def random_layer_model(rng: np.random.Generator,
                       n_layers: int = NL_DEFAULT) -> np.ndarray:
    """随机层状 log10ρ 模型: 块状层 / 渐变层 + 轻微平滑。"""
    base = rng.uniform(0.7, 2.6)
    theta = np.full(n_layers, base)
    if rng.random() < 0.75:
        k = int(rng.integers(2, 5))
        bounds = np.sort(rng.integers(1, n_layers, size=k - 1))
        segs = np.split(np.arange(n_layers), bounds)
        for seg in segs:
            theta[seg] = base + rng.uniform(-1.2, 1.2)
    else:
        slope = rng.uniform(-0.04, 0.04)
        theta = base + slope * np.arange(n_layers) + rng.uniform(-0.3, 0.3)
    # 轻微三点平滑 (类真实地层过渡)
    if rng.random() < 0.5:
        theta = np.convolve(theta, np.array([0.2, 0.6, 0.2]), mode='same')
    return np.clip(theta, 0.3, 3.2)


def make_synthetic_dataset(n: int = 4000, n_gates: int = 25,
                           n_layers: int = NL_DEFAULT,
                           noise: float = 0.02,
                           seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """X(n,Ng)=含噪 log10ρa, Y(n,Nl)=log10ρ 真值。时间道 TRAIN_T。"""
    rng = np.random.default_rng(seed)
    z_c = depth_layer_centers(n_layers, 5.0, 1500.0)
    t = np.geomspace(2e-6, 2e-2, n_gates)
    X = np.empty((n, n_gates)); Y = np.empty((n, n_layers))
    for i in range(n):
        theta = random_layer_model(rng, n_layers)
        rhoa = forward_rhoa(theta, t, z_c, RHO_REF)
        X[i] = np.log10(np.maximum(rhoa, 1e-6)) \
            + rng.normal(0.0, noise, n_gates)
        Y[i] = theta
    return X, Y


# ============================================================
# 2) torch U-Net (懒加载: 仅训练时 import torch)
# ============================================================
def train_unet(X: np.ndarray, Y: np.ndarray,
               epochs: int = 60, batch: int = 128, lr: float = 1e-3,
               seed: int = 42, log_fn: Optional[Callable] = None,
               val_frac: float = 0.1) -> dict:
    """训练 1D U-Net, 返回可被 UNetNumpy 加载的权重 dict。"""
    if not torch_available():
        raise RuntimeError('当前环境未安装 torch, 请改用 linear 后端')
    torch, nn, F = _load_torch()
    log_fn = log_fn or (lambda s: print(s))

    class UNet1D(nn.Module):
        """单层级联 U-Net: 1→32→(pool)→64→(上采样+跳连)→32→1, 同长度。"""
        def __init__(self, c1: int = 32, c2: int = 64, k: int = 5):
            super().__init__()
            self.cv1 = nn.Conv1d(1, c1, k, padding=k // 2)
            self.cv2 = nn.Conv1d(c1, c2, k, padding=k // 2)
            self.cv3 = nn.Conv1d(c2 + c1, c1, k, padding=k // 2)
            self.cv4 = nn.Conv1d(c1, 1, 1)

        def forward(self, x):
            h = F.relu(self.cv1(x))                 # B,c1,L
            p = F.max_pool1d(h, 2)                  # B,c1,L//2
            b = F.relu(self.cv2(p))                 # B,c2,L//2
            u = b.repeat_interleave(2, dim=-1)
            if u.shape[-1] < h.shape[-1]:           # 奇数长度边缘复制补齐
                u = F.pad(u, (0, h.shape[-1] - u.shape[-1]), mode='replicate')
            d = F.relu(self.cv3(torch.cat([u, h], dim=1)))
            return self.cv4(d)                      # B,1,L

    torch.manual_seed(seed); np.random.seed(seed)
    n_val = max(1, int(len(X) * val_frac))
    Xv, Yv = X[-n_val:], Y[-n_val:]
    Xt, Yt = X[:-n_val], Y[:-n_val]
    net = UNet1D()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    xt = torch.tensor(Xt[:, None, :], dtype=torch.float32)
    yt = torch.tensor(Yt[:, None, :], dtype=torch.float32)
    xv = torch.tensor(Xv[:, None, :], dtype=torch.float32)
    yv = torch.tensor(Yv[:, None, :], dtype=torch.float32)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        net.train()
        for i0 in range(0, len(xt), batch):
            idx = perm[i0:i0 + batch]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward(); opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(xv), yv))
            log_fn(f'[DL] epoch {ep+1:3d}/{epochs} train={float(loss):.4f} '
                  f'val={vl:.4f}')
    net.eval()
    w = {f'w{i}': net.get_submodule(n).weight.detach().cpu().numpy()
         for i, n in [(1, 'cv1'), (2, 'cv2'), (3, 'cv3'), (4, 'cv4')]}
    b = {f'b{i}': net.get_submodule(n).bias.detach().cpu().numpy()
         for i, n in [(1, 'cv1'), (2, 'cv2'), (3, 'cv3'), (4, 'cv4')]}
    return {**w, **b}


# ============================================================
# 3) 纯 numpy 推理器
# ============================================================
def _np_conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x(B,C,L) * w(O,C,K) 同尺寸零填充互相关。"""
    B, C, L = x.shape
    O, _, K = w.shape
    pad = K // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, K, axis=-1)  # B,C,L,K
    return np.einsum('bclk,ock->bol', win, w, optimize=True) \
        + b[None, :, None]


def _np_maxpool(x: np.ndarray) -> np.ndarray:
    B, C, L = x.shape
    L2 = L // 2
    return x[:, :, :2 * L2].reshape(B, C, L2, 2).max(axis=3)


class UNetNumpy:
    kind = 'unet'

    def __init__(self, par: dict):
        self.w = [par[f'w{i}'] for i in range(1, 5)]
        self.b = [par[f'b{i}'] for i in range(1, 5)]
        self.xmean = float(par['xmean'])
        self.xstd = float(par['xstd'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = ((X - self.xmean) / max(self.xstd, 1e-9))[:, None, :]
        h = np.maximum(_np_conv1d(x, self.w[0], self.b[0]), 0.0)
        p = _np_maxpool(h)
        bb = np.maximum(_np_conv1d(p, self.w[1], self.b[1]), 0.0)
        u = np.repeat(bb, 2, axis=-1)
        if u.shape[-1] < h.shape[-1]:               # 奇数长度边缘复制
            u = np.pad(u, ((0, 0), (0, 0), (0, h.shape[-1] - u.shape[-1])),
                       mode='edge')
        d = np.maximum(_np_conv1d(np.concatenate([u, h], axis=1),
                                  self.w[2], self.b[2]), 0.0)
        return _np_conv1d(d, self.w[3], self.b[3])[:, 0, :]


class LinearNumpy:
    """闭式岭回归降级后端 (合成数据秒级训练)。"""
    kind = 'linear'

    def __init__(self, par: dict):
        self.W = par['W']                    # (Ng+1, Nl)
        self.xmean = float(par['xmean'])
        self.xstd = float(par['xstd'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        xn = (X - self.xmean) / max(self.xstd, 1e-9)
        Xb = np.column_stack([xn, np.ones(len(xn))])
        return Xb @ self.W


def train_linear(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-2) -> dict:
    xm, xs = float(X.mean()), float(X.std())
    xn = (X - xm) / max(xs, 1e-9)
    Xb = np.column_stack([xn, np.ones(len(xn))])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Y)
    return {'W': W, 'xmean': xm, 'xstd': xs}


def save_weights(path: str | Path, params: dict, kind: str):
    params = dict(params)
    params['kind'] = np.array(kind)
    params['train_t'] = TRAIN_T
    np.savez(path, **params)


def load_inferencer(path: Optional[str | Path] = None,
                    auto_train: bool = True,
                    log_fn: Optional[Callable] = None):
    """加载权重; 缺失时自动训练线性降级模型。"""
    log_fn = log_fn or (lambda s: None)
    path = Path(path) if path else WEIGHTS_PATH
    if path.is_file():
        par = dict(np.load(path, allow_pickle=True))
        kind = str(par['kind'])
        if kind == 'unet':
            log_fn('[DL] 加载 U-Net 权重 (numpy 推理)')
            return UNetNumpy(par)
        log_fn('[DL] 加载线性权重')
        return LinearNumpy(par)
    if not auto_train:
        raise FileNotFoundError(f'DL 权重不存在: {path}')
    log_fn('[DL] 权重缺失, 训练线性降级后端 (合成数据) ...')
    X, Y = make_synthetic_dataset(n=3000, seed=7)
    par = train_linear(X, Y)
    save_weights(path, par, 'linear')
    log_fn(f'[DL] 线性权重已保存 → {path}')
    return LinearNumpy(par)


def _torch_train_worker(weights_path: str, n_train: int,
                        epochs: int, seed: int) -> None:
    """spawn 子进程入口: 在独立进程 import torch 完成 U-Net 训练并落盘。

    规避 torch(CUDA/OpenMP) 与 Anaconda scipy-MKL 同进程加载时
    LAPACK gelss 原生 abort 的冲突。
    """
    import traceback
    try:
        X, Y = make_synthetic_dataset(n=n_train, seed=seed)
        params = train_unet(X, Y, epochs=epochs, seed=seed,
                            log_fn=lambda s: print(s, flush=True))
        params['xmean'] = float(X.mean()); params['xstd'] = float(X.std())
        save_weights(weights_path, params, 'unet')
        print(f'[DL-child] 权重 → {weights_path}', flush=True)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


def _torch_parity_worker() -> None:
    """子进程自检: 随机权重下 torch 前向与 numpy 推理一致。"""
    torch, nn, F = _load_torch()
    torch.manual_seed(0)

    class UNet1D(nn.Module):
        def __init__(self, c1=32, c2=64, k=5):
            super().__init__()
            self.cv1 = nn.Conv1d(1, c1, k, padding=k // 2)
            self.cv2 = nn.Conv1d(c1, c2, k, padding=k // 2)
            self.cv3 = nn.Conv1d(c2 + c1, c1, k, padding=k // 2)
            self.cv4 = nn.Conv1d(c1, 1, 1)

        def forward(self, x):
            h = F.relu(self.cv1(x))
            p = F.max_pool1d(h, 2)
            b = F.relu(self.cv2(p))
            u = b.repeat_interleave(2, dim=-1)
            if u.shape[-1] < h.shape[-1]:
                u = F.pad(u, (0, h.shape[-1] - u.shape[-1]), mode='replicate')
            d = F.relu(self.cv3(torch.cat([u, h], dim=1)))
            return self.cv4(d)

    net = UNet1D(); net.eval()
    x = torch.randn(4, 1, 25)
    with torch.no_grad():
        yt = net(x).numpy()[:, 0, :]
    params = {}
    for i, nm in [(1, 'cv1'), (2, 'cv2'), (3, 'cv3'), (4, 'cv4')]:
        params[f'w{i}'] = net.get_submodule(nm).weight.detach().numpy()
        params[f'b{i}'] = net.get_submodule(nm).bias.detach().numpy()
    params['xmean'] = 0.0; params['xstd'] = 1.0
    inf = UNetNumpy(params)
    yn = inf.predict(x.numpy()[:, 0, :])
    if not np.allclose(yt, yn, atol=1e-5):
        print('[DL-child] parity FAILED', np.abs(yt - yn).max())
        raise SystemExit(2)
    print('[DL-child] parity OK')


def train_dl(weights_path: str | Path = WEIGHTS_PATH,
             n_train: int = 4000, epochs: int = 60,
             backend: str = 'auto', seed: int = 42,
             log_fn: Optional[Callable] = None) -> Path:
    """训练并保存权重。backend='auto'|'torch'|'linear'。

    torch 训练在 spawn 子进程执行 (torch/MKL 同进程冲突规避)。
    """
    import multiprocessing as mp
    log_fn = log_fn or (lambda s: print(s))
    weights_path = Path(weights_path)
    use_torch = torch_available() and backend in ('auto', 'torch')
    if use_torch:
        log_fn(f'[DL] torch 后端 U-Net 训练 (spawn 子进程) '
               f'n={n_train} epochs={epochs}')
        ctx = mp.get_context('spawn')
        proc = ctx.Process(target=_torch_train_worker,
                           args=(str(weights_path), n_train, epochs, seed))
        proc.start(); proc.join()
        if proc.exitcode != 0 or not weights_path.is_file():
            raise RuntimeError(f'torch 子进程训练失败 exitcode={proc.exitcode}')
    else:
        if backend == 'torch':
            log_fn('[DL] torch 不可用, 回退线性后端')
        X, Y = make_synthetic_dataset(n=n_train, seed=seed)
        log_fn(f'[DL] 线性后端训练 n={n_train}')
        params = train_linear(X, Y)
        save_weights(weights_path, params, 'linear')
    log_fn(f'[DL] 权重 → {weights_path}')
    return weights_path


# ============================================================
# 4) 逐站 / 全线反演 (产物命名对齐 inv_algorithms)
# ============================================================
def predict_station(rhoa_obs: np.ndarray, t_obs: np.ndarray,
                    inferencer, n_layers: int = NL_DEFAULT) -> np.ndarray:
    """真实时间道重采样到训练道 → θ(log10ρ, n_layers)。"""
    order = np.argsort(t_obs)
    lg_t, lg_r = np.log10(np.asarray(t_obs)[order]), \
        np.log10(np.maximum(np.asarray(rhoa_obs)[order], 1e-6))
    x = np.interp(np.log10(TRAIN_T), lg_t, lg_r)[None, :]
    theta = inferencer.predict(x)[0]
    if len(theta) != n_layers:
        theta = np.interp(np.linspace(0, 1, n_layers),
                          np.linspace(0, 1, len(theta)), theta)
    return np.clip(theta, 0.3, 3.2)


def invert_line_dl(line_tag: str, out_root: str | Path,
                   inferencer=None,
                   n_layers: int = NL_DEFAULT,
                   log_fn: Optional[Callable] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    out_base = Path(out_root) / line_tag
    dat = out_base / f'{line_tag}_occam.dat'
    if not dat.is_file():
        log_fn(f'  [SKIP] {line_tag}: 无 {dat.name}')
        return {}
    if inferencer is None:
        inferencer = load_inferencer(log_fn=log_fn)
    df = parse_occam_dat(dat)
    stations = sorted(df.Station.unique())
    z_c = depth_layer_centers(n_layers, 5.0, 1500.0)
    R_log = np.zeros((len(stations), n_layers))
    summary = []
    for idx, st in enumerate(stations):
        sub = df[df.Station == st].sort_values('Time_s')
        theta = predict_station(sub.AppRes.values, sub.Time_s.values,
                                inferencer, n_layers)
        R_log[idx] = theta
        pred = forward_rhoa(theta, sub.Time_s.values, z_c, RHO_REF)
        rmse = float(np.sqrt(np.mean((np.log10(np.maximum(pred, 1e-6))
                                      - np.log10(np.maximum(sub.AppRes.values,
                                                            1e-6))) ** 2)))
        summary.append({'Station': st, 'rmse_log': rmse, 'success': True,
                        'n_iter': 0, 'algo': 'dl'})
    R_lin = 10.0 ** R_log
    np.save(out_base / f'{line_tag}_inv_dl_R.npy', R_lin)
    np.save(out_base / f'{line_tag}_inv_dl_depth.npy', z_c)
    pd.DataFrame(summary).to_csv(out_base / f'{line_tag}_inv_dl_summary.csv',
                                 index=False, encoding='utf-8-sig')
    rmse_med = float(np.median([s['rmse_log'] for s in summary]))
    if len(stations) >= 2:
        fig, ax = plt.subplots(figsize=(max(8, len(stations) * 0.4 + 2), 6))
        Xm, Ym = np.meshgrid(np.arange(len(stations)),
                             np.log10(z_c), indexing='ij')
        im = ax.pcolormesh(Xm, Ym, R_log, shading='auto', cmap='jet_r')
        ax.invert_yaxis()
        ax.set_xlabel(f'测点 ({len(stations)})'); ax.set_ylabel('深度 (m, log)')
        ax.set_title(f'{line_tag} DL端到端反演 ρ断面  RMSE_log10_med={rmse_med:.3f}')
        fig.colorbar(im, ax=ax).set_label('log10(ρ/Ω·m)')
        fig.tight_layout()
        fig.savefig(out_base / f'{line_tag}_inv_dl_section.png', dpi=180)
        plt.close(fig)
    log_fn(f'  [DL] {line_tag}: {len(stations)}站 RMSE_log10_med={rmse_med:.3f}')
    return {'stations': len(stations), 'rmse_median_log': rmse_med, 'algo': 'dl'}


def run_all_dl(out_root: str | Path, weights_path: Optional[str | Path] = None,
               n_layers: int = NL_DEFAULT,
               log_fn: Optional[Callable] = None) -> dict:
    log_fn = log_fn or (lambda s: print(s))
    out_root = Path(out_root)
    inferencer = load_inferencer(weights_path, log_fn=log_fn)
    res = {}
    for d in sorted(out_root.iterdir()):
        if d.is_dir() and re.fullmatch(r'[1-5]线', d.name):
            res[d.name] = invert_line_dl(d.name, out_root, inferencer,
                                         n_layers, log_fn)
    return res
