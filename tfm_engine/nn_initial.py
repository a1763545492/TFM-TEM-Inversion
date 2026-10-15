# -*- coding: utf-8 -*-
"""
tfm_engine.nn_initial (M16)
=============================
神经网络初始模型预测。

用正演批量生成合成样本，训练小型全连接网络从 ρa 曲线预测初始模型参数，
加速反演收敛并帮助跳出局部极小。

网络结构 (纯 numpy, 无外部 DL 依赖):
  输入: log10(ρa) 曲线 (n_gates,)
  隐藏层: 2 层全连接 + ReLU
  输出: log10(ρ) 各层初始值 (n_layers,)

训练数据生成:
  随机采样模型参数 → 正演计算 ρa → 作为输入
  真实模型参数 → 作为标签

使用方式:
  nn = NNInitialModel(n_gates=25, n_layers=25)
  nn.train_from_synthetic(n_samples=5000, ...)
  rho_init = nn.predict(rhoa_curve)

完全离线，纯 numpy。
"""
from __future__ import annotations
from typing import Optional, Callable
import numpy as np
from pathlib import Path
import json
import pickle


# ============================================================
# 极简全连接神经网络 (numpy 实现)
# ============================================================

class SimpleNN:
    """2 层全连接网络，ReLU 激活，MSE 损失，Adam 优化器。"""

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 lr: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He 初始化
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, n_hidden)) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_hidden)
        self.W3 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
        self.b3 = np.zeros(n_out)
        self.lr = lr
        # Adam 状态
        self.m = [np.zeros_like(w) for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]]
        self.v = [np.zeros_like(w) for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]]
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def forward(self, X: np.ndarray) -> tuple:
        """X: (batch, n_in) → 返回 (输出, 缓存)"""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)  # ReLU
        z3 = a2 @ self.W3 + self.b3
        return z3, (X, z1, a1, z2, a2)

    def backward(self, cache, dL_dz3: np.ndarray) -> list:
        """反向传播，返回各层梯度"""
        X, z1, a1, z2, a2 = cache
        batch = X.shape[0]

        dL_dW3 = a2.T @ dL_dz3 / batch
        dL_db3 = np.sum(dL_dz3, axis=0) / batch
        dL_da2 = dL_dz3 @ self.W3.T
        dL_dz2 = dL_da2 * (z2 > 0)  # ReLU 导数
        dL_dW2 = a1.T @ dL_dz2 / batch
        dL_db2 = np.sum(dL_dz2, axis=0) / batch
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = X.T @ dL_dz1 / batch
        dL_db1 = np.sum(dL_dz1, axis=0) / batch

        return [dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3]

    def adam_update(self, grads: list):
        """Adam 优化器更新"""
        self.t += 1
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        for i in range(6):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params

    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """单步训练，返回损失"""
        pred, cache = self.forward(X)
        batch = X.shape[0]
        # MSE 损失
        loss = np.sum((pred - Y) ** 2) / (2 * batch)
        dL_dz3 = (pred - Y) / batch
        grads = self.backward(cache, dL_dz3)
        self.adam_update(grads)
        return float(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred, _ = self.forward(X)
        return pred

    def save(self, path: Path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, lr=self.lr)

    def load(self, path: Path):
        data = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = data['W1'], data['b1'], data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        self.lr = float(data['lr'])


# ============================================================
# 训练数据生成 + 训练管理
# ============================================================

class NNInitialModel:
    """神经网络初始模型预测器。"""

    def __init__(self, n_gates: int, n_layers: int,
                 n_hidden: int = 64, lr: float = 1e-3):
        self.n_gates = n_gates
        self.n_layers = n_layers
        self.nn = SimpleNN(n_gates, n_hidden, n_layers, lr)
        self.trained = False

    def generate_synthetic_data(self, n_samples: int = 5000,
                                 forward_fn: Optional[Callable] = None,
                                 z_centers: Optional[np.ndarray] = None,
                                 t_gates: Optional[np.ndarray] = None,
                                 rho_range: tuple = (1.0, 1000.0),
                                 thick_range: tuple = (5.0, 200.0),
                                 seed: int = 42,
                                 log_fn: Optional[Callable] = None) -> tuple:
        """批量生成合成训练样本。

        Parameters
        ----------
        n_samples : int
        forward_fn : callable(theta_log10, t_gates, z_centers, rho_ref) → rhoa
            若 None，用默认扩散深度近似
        z_centers : (n_layers,)
        t_gates : (n_gates,)
        rho_range : tuple
        thick_range : tuple
        seed : int

        Returns
        -------
        X : (n_samples, n_gates) log10(ρa)
        Y : (n_samples, n_layers) log10(ρ)
        """
        if z_centers is None:
            z_centers = np.geomspace(5.0, 1500.0, self.n_layers)
        if t_gates is None:
            t_gates = np.geomspace(2e-6, 20e-3, self.n_gates)

        rng = np.random.default_rng(seed)
        X = np.zeros((n_samples, self.n_gates))
        Y = np.zeros((n_samples, self.n_layers))

        # 默认正演（扩散深度近似）
        if forward_fn is None:
            from .inversion import forward_rhoa
            forward_fn = forward_rhoa

        for i in range(n_samples):
            # 随机模型：1-3 层
            n_l = rng.integers(1, 4)
            thick = np.concatenate([
                rng.uniform(thick_range[0], thick_range[1], n_l - 1),
                [np.inf]
            ])
            rho = rng.uniform(rho_range[0], rho_range[1], n_l)
            theta = np.log10(rho)

            # 扩展到 n_layers（半空间填充）
            theta_full = np.full(self.n_layers, np.log10(rho[-1]))
            theta_full[:n_l] = theta

            # 正演
            try:
                rhoa = forward_fn(theta_full, t_gates, z_centers, 100.0)
                X[i] = np.log10(np.maximum(rhoa, 1e-3))
                Y[i] = theta_full
            except Exception:
                # 失败样本用均匀半空间替代
                X[i] = np.full(self.n_gates, np.log10(rho[0]))
                Y[i] = theta_full

            if log_fn and (i + 1) % 1000 == 0:
                log_fn(f'  [NN数据] 已生成 {i+1}/{n_samples} 样本')

        return X, Y

    def train(self, X: np.ndarray, Y: np.ndarray,
              n_epochs: int = 100, batch_size: int = 64,
              val_frac: float = 0.1,
              log_fn: Optional[Callable] = None) -> dict:
        """训练网络。

        Returns
        -------
        history : dict with 'train_loss', 'val_loss'
        """
        n = len(X)
        n_val = int(n * val_frac)
        idx = np.random.permutation(n)
        X_train, Y_train = X[idx[n_val:]], Y[idx[n_val:]]
        X_val, Y_val = X[idx[:n_val]], Y[idx[:n_val]]

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(n_epochs):
            # 训练
            perm = np.random.permutation(len(X_train))
            train_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                loss = self.nn.train_step(X_train[batch_idx], Y_train[batch_idx])
                train_loss += loss
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # 验证
            val_pred = self.nn.predict(X_val)
            val_loss = np.mean((val_pred - Y_val) ** 2)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if log_fn and (epoch + 1) % 10 == 0:
                log_fn(f'  [NN训练] Epoch {epoch+1}/{n_epochs} '
                       f'train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

        self.trained = True
        return history

    def train_from_synthetic(self, n_samples: int = 5000, **kwargs) -> dict:
        """从合成数据训练（一站式）。"""
        log_fn = kwargs.get('log_fn')
        X, Y = self.generate_synthetic_data(n_samples, **kwargs)
        return self.train(X, Y, **{k: v for k, v in kwargs.items()
                                   if k in ['n_epochs', 'batch_size', 'val_frac', 'log_fn']})

    def predict_initial(self, rhoa_curve: np.ndarray) -> np.ndarray:
        """预测初始模型。

        Parameters
        ----------
        rhoa_curve : (n_gates,) 观测视电阻率

        Returns
        -------
        rho_init : (n_layers,) 初始电阻率 [Ω·m]
        """
        if not self.trained:
            raise RuntimeError('模型未训练，请先调用 train() 或 train_from_synthetic()')
        x = np.log10(np.maximum(rhoa_curve, 1e-3)).reshape(1, -1)
        theta_pred = self.nn.predict(x)[0]
        return 10.0 ** theta_pred

    def save(self, path: Path):
        self.nn.save(path)
        meta = {'n_gates': self.n_gates, 'n_layers': self.n_layers,
                'n_hidden': self.nn.W1.shape[1], 'trained': self.trained}
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f)

    def load(self, path: Path):
        with open(path.with_suffix('.json')) as f:
            meta = json.load(f)
        self.n_gates = meta['n_gates']
        self.n_layers = meta['n_layers']
        self.nn = SimpleNN(meta['n_gates'], meta['n_hidden'], meta['n_layers'])
        self.nn.load(path)
        self.trained = meta['trained']


# ============================================================
# 与反演集成
# ============================================================

def create_warm_start_inversion(nn_model: NNInitialModel,
                                 rhoa_curve: np.ndarray,
                                 z_centers: np.ndarray) -> np.ndarray:
    """用 NN 预测初始模型，作为反演热启动。

    Returns
    -------
    theta0 : (n_layers,) log10(ρ) 初始参数
    """
    rho_init = nn_model.predict_initial(rhoa_curve)
    # 确保在合理范围内
    rho_init = np.clip(rho_init, 0.1, 1e6)
    return np.log10(rho_init)


__all__ = [
    'SimpleNN',
    'NNInitialModel',
    'create_warm_start_inversion',
]
