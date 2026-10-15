# -*- coding: utf-8 -*-
"""
tfm_engine.denoise_ae (M17)
=============================
去噪自编码器 — 衰减曲线进反演前先压制晚期噪声。

原理:
  - 自编码器学习干净衰减曲线的低维流形
  - 编码器: 衰减曲线 → 潜在空间 (n_latent << n_gates)
  - 解码器: 潜在空间 → 重建曲线
  - 训练: 干净正演曲线 → 加噪 → 网络学习去除噪声

网络结构 (纯 numpy, 与 nn_initial 一致):
  编码器: n_gates → 64 → 32 → n_latent (ReLU)
  解码器: n_latent → 32 → 64 → n_gates (ReLU, 输出层线性)

训练数据:
  用正演批量生成干净衰减曲线
  加噪: 高斯白噪声 + 晚期噪声增强（模拟真实数据晚期 SNR 下降）

使用方式:
  ae = DenoisingAutoencoder(n_gates=25, n_latent=8)
  ae.train_from_synthetic(n_samples=5000, ...)
  clean_curve = ae.denoise(noisy_curve)

完全离线，纯 numpy。
"""
from __future__ import annotations
from typing import Optional, Callable
import numpy as np
from pathlib import Path
import json


# ============================================================
# 自编码器网络 (numpy 实现)
# ============================================================

class Autoencoder:
    """对称自编码器，ReLU 隐藏层，线性输出。"""

    def __init__(self, n_in: int, n_hidden: int, n_latent: int,
                 lr: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)
        # 编码器
        self.We1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
        self.be1 = np.zeros(n_hidden)
        self.We2 = rng.standard_normal((n_hidden, n_hidden // 2)) * np.sqrt(2.0 / n_hidden)
        self.be2 = np.zeros(n_hidden // 2)
        self.We3 = rng.standard_normal((n_hidden // 2, n_latent)) * np.sqrt(2.0 / (n_hidden // 2))
        self.be3 = np.zeros(n_latent)
        # 解码器
        self.Wd1 = rng.standard_normal((n_latent, n_hidden // 2)) * np.sqrt(2.0 / n_latent)
        self.bd1 = np.zeros(n_hidden // 2)
        self.Wd2 = rng.standard_normal((n_hidden // 2, n_hidden)) * np.sqrt(2.0 / (n_hidden // 2))
        self.bd2 = np.zeros(n_hidden)
        self.Wd3 = rng.standard_normal((n_hidden, n_in)) * np.sqrt(2.0 / n_hidden)
        self.bd3 = np.zeros(n_in)

        self.lr = lr
        self._init_adam()

    def _init_adam(self):
        params = self._get_params()
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def _get_params(self):
        return [self.We1, self.be1, self.We2, self.be2, self.We3, self.be3,
                self.Wd1, self.bd1, self.Wd2, self.bd2, self.Wd3, self.bd3]

    def _set_params(self, params):
        (self.We1, self.be1, self.We2, self.be2, self.We3, self.be3,
         self.Wd1, self.bd1, self.Wd2, self.bd2, self.Wd3, self.bd3) = params

    def encode(self, X: np.ndarray) -> np.ndarray:
        z1 = np.maximum(X @ self.We1 + self.be1, 0)
        z2 = np.maximum(z1 @ self.We2 + self.be2, 0)
        z3 = z2 @ self.We3 + self.be3  # 潜在层，线性
        return z3

    def decode(self, Z: np.ndarray) -> np.ndarray:
        z1 = np.maximum(Z @ self.Wd1 + self.bd1, 0)
        z2 = np.maximum(z1 @ self.Wd2 + self.bd2, 0)
        z3 = z2 @ self.Wd3 + self.bd3  # 输出层，线性
        return z3

    def forward(self, X: np.ndarray) -> tuple:
        z_e1 = X @ self.We1 + self.be1
        a_e1 = np.maximum(z_e1, 0)
        z_e2 = a_e1 @ self.We2 + self.be2
        a_e2 = np.maximum(z_e2, 0)
        z_e3 = a_e2 @ self.We3 + self.be3
        # 解码器
        z_d1 = z_e3 @ self.Wd1 + self.bd1
        a_d1 = np.maximum(z_d1, 0)
        z_d2 = a_d1 @ self.Wd2 + self.bd2
        a_d2 = np.maximum(z_d2, 0)
        z_d3 = a_d2 @ self.Wd3 + self.bd3
        cache = (X, z_e1, a_e1, z_e2, a_e2, z_e3,
                 z_d1, a_d1, z_d2, a_d2)
        return z_d3, cache

    def backward(self, cache, dL_dout: np.ndarray) -> list:
        X, z_e1, a_e1, z_e2, a_e2, z_e3, z_d1, a_d1, z_d2, a_d2 = cache
        batch = X.shape[0]
        grads = []

        # 解码器反向
        dL_dWd3 = a_d2.T @ dL_dout / batch
        dL_dbd3 = np.sum(dL_dout, axis=0) / batch
        dL_da_d2 = dL_dout @ self.Wd3.T
        dL_dz_d2 = dL_da_d2 * (z_d2 > 0)
        dL_dWd2 = a_d1.T @ dL_dz_d2 / batch
        dL_dbd2 = np.sum(dL_dz_d2, axis=0) / batch
        dL_da_d1 = dL_dz_d2 @ self.Wd2.T
        dL_dz_d1 = dL_da_d1 * (z_d1 > 0)
        dL_dWd1 = z_e3.T @ dL_dz_d1 / batch
        dL_dbd1 = np.sum(dL_dz_d1, axis=0) / batch
        dL_dz_e3 = dL_dz_d1 @ self.Wd1.T

        # 编码器反向
        dL_dWe3 = a_e2.T @ dL_dz_e3 / batch
        dL_dbe3 = np.sum(dL_dz_e3, axis=0) / batch
        dL_da_e2 = dL_dz_e3 @ self.We3.T
        dL_dz_e2 = dL_da_e2 * (z_e2 > 0)
        dL_dWe2 = a_e1.T @ dL_dz_e2 / batch
        dL_dbe2 = np.sum(dL_dz_e2, axis=0) / batch
        dL_da_e1 = dL_dz_e2 @ self.We2.T
        dL_dz_e1 = dL_da_e1 * (z_e1 > 0)
        dL_dWe1 = X.T @ dL_dz_e1 / batch
        dL_dbe1 = np.sum(dL_dz_e1, axis=0) / batch

        return [dL_dWe1, dL_dbe1, dL_dWe2, dL_dbe2, dL_dWe3, dL_dbe3,
                dL_dWd1, dL_dbd1, dL_dWd2, dL_dbd2, dL_dWd3, dL_dbd3]

    def adam_update(self, grads: list):
        self.t += 1
        params = self._get_params()
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self._set_params(params)

    def train_step(self, X_noisy: np.ndarray, X_clean: np.ndarray) -> float:
        pred, cache = self.forward(X_noisy)
        batch = X_noisy.shape[0]
        loss = np.sum((pred - X_clean) ** 2) / (2 * batch)
        dL_dout = (pred - X_clean) / batch
        grads = self.backward(cache, dL_dout)
        self.adam_update(grads)
        return float(loss)

    def denoise(self, X: np.ndarray) -> np.ndarray:
        pred, _ = self.forward(X)
        return pred

    def save(self, path: Path):
        params = self._get_params()
        np.savez(path, **{f'p{i}': p for i, p in enumerate(params)},
                 lr=self.lr, n_in=self.We1.shape[0],
                 n_hidden=self.We1.shape[1], n_latent=self.We3.shape[1])

    def load(self, path: Path):
        data = np.load(path)
        params = [data[f'p{i}'] for i in range(12)]
        self._set_params(params)
        self.lr = float(data['lr'])
        self._init_adam()


# ============================================================
# 训练数据生成 + 训练管理
# ============================================================

class DenoisingAutoencoder:
    """去噪自编码器管理器。"""

    def __init__(self, n_gates: int, n_latent: int = 8,
                 n_hidden: int = 64, lr: float = 1e-3):
        self.n_gates = n_gates
        self.n_latent = n_latent
        self.ae = Autoencoder(n_gates, n_hidden, n_latent, lr)
        self.trained = False

    def generate_clean_data(self, n_samples: int = 5000,
                             forward_fn: Optional[Callable] = None,
                             t_gates: Optional[np.ndarray] = None,
                             rho_range: tuple = (1.0, 1000.0),
                             thick_range: tuple = (5.0, 200.0),
                             seed: int = 42,
                             log_fn: Optional[Callable] = None) -> np.ndarray:
        """生成干净衰减曲线（正演合成）。

        Returns
        -------
        X_clean : (n_samples, n_gates) 干净 Vt 曲线（log 域）
        """
        if t_gates is None:
            t_gates = np.geomspace(2e-6, 20e-3, self.n_gates)

        rng = np.random.default_rng(seed)
        X_clean = np.zeros((n_samples, self.n_gates))

        if forward_fn is None:
            from .forward import tem_forward_1d, LayeredModel
            forward_fn = lambda theta, t, z, ref: tem_forward_1d(
                LayeredModel(np.array([np.inf]), np.array([10**theta[0]])),
                1e6, 30.0, t)[1]

        for i in range(n_samples):
            # 随机均匀半空间模型（简化）
            rho = rng.uniform(rho_range[0], rho_range[1])
            try:
                vt = forward_fn(np.array([np.log10(rho)]), t_gates, None, 100.0)
                X_clean[i] = np.log10(np.maximum(vt, 1e-35))
            except Exception:
                X_clean[i] = np.full(self.n_gates, np.log10(1e-6))

            if log_fn and (i + 1) % 1000 == 0:
                log_fn(f'  [AE数据] 已生成 {i+1}/{n_samples} 干净样本')

        return X_clean

    def add_noise(self, X_clean: np.ndarray,
                  noise_level: float = 0.1,
                  late_boost: float = 2.0,
                  seed: int = 42) -> np.ndarray:
        """加噪：高斯白噪声 + 晚期噪声增强。

        Parameters
        ----------
        X_clean : (n_samples, n_gates) log 域曲线
        noise_level : float, 基础噪声标准差
        late_boost : float, 晚期噪声增强倍数
        seed : int

        Returns
        -------
        X_noisy : (n_samples, n_gates)
        """
        rng = np.random.default_rng(seed)
        n_samples, n_gates = X_clean.shape
        noise = rng.standard_normal((n_samples, n_gates)) * noise_level
        # 晚期噪声增强（后 1/3 门）
        late_start = n_gates * 2 // 3
        noise[:, late_start:] *= late_boost
        return X_clean + noise

    def train(self, X_clean: np.ndarray,
              noise_level: float = 0.1,
              n_epochs: int = 100,
              batch_size: int = 64,
              val_frac: float = 0.1,
              log_fn: Optional[Callable] = None) -> dict:
        """训练自编码器。"""
        n = len(X_clean)
        n_val = int(n * val_frac)
        idx = np.random.permutation(n)
        X_train = X_clean[idx[n_val:]]
        X_val = X_clean[idx[:n_val]]

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(n_epochs):
            # 训练
            perm = np.random.permutation(len(X_train))
            train_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch_clean = X_train[batch_idx]
                X_batch_noisy = self.add_noise(X_batch_clean, noise_level,
                                               seed=hash((epoch, i)) % (2**32))
                loss = self.ae.train_step(X_batch_noisy, X_batch_clean)
                train_loss += loss
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # 验证
            X_val_noisy = self.add_noise(X_val, noise_level, seed=epoch)
            val_pred = self.ae.denoise(X_val_noisy)
            val_loss = np.mean((val_pred - X_val) ** 2)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if log_fn and (epoch + 1) % 10 == 0:
                log_fn(f'  [AE训练] Epoch {epoch+1}/{n_epochs} '
                       f'train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

        self.trained = True
        return history

    def train_from_synthetic(self, n_samples: int = 5000, **kwargs) -> dict:
        """从合成数据训练（一站式）。"""
        log_fn = kwargs.get('log_fn')
        X_clean = self.generate_clean_data(n_samples, **kwargs)
        return self.train(X_clean, **{k: v for k, v in kwargs.items()
                                       if k in ['noise_level', 'n_epochs', 'batch_size', 'val_frac', 'log_fn']})

    def denoise(self, noisy_curve: np.ndarray) -> np.ndarray:
        """去噪。

        Parameters
        ----------
        noisy_curve : (n_gates,) 或 (n_samples, n_gates) 含噪曲线

        Returns
        -------
        clean_curve : 同形状 去噪后曲线
        """
        if not self.trained:
            raise RuntimeError('模型未训练，请先调用 train() 或 train_from_synthetic()')
        single = noisy_curve.ndim == 1
        if single:
            noisy_curve = noisy_curve.reshape(1, -1)
        # 输入应该是 log 域
        x = np.log10(np.maximum(noisy_curve, 1e-35))
        clean = self.ae.denoise(x)
        clean = 10.0 ** clean
        if single:
            clean = clean[0]
        return clean

    def save(self, path: Path):
        self.ae.save(path)
        meta = {'n_gates': self.n_gates, 'n_latent': self.n_latent,
                'n_hidden': self.ae.We1.shape[1], 'trained': self.trained}
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f)

    def load(self, path: Path):
        with open(path.with_suffix('.json')) as f:
            meta = json.load(f)
        self.n_gates = meta['n_gates']
        self.n_latent = meta['n_latent']
        self.ae = Autoencoder(meta['n_gates'], meta['n_hidden'], meta['n_latent'])
        self.ae.load(path)
        self.trained = meta['trained']


__all__ = [
    'Autoencoder',
    'DenoisingAutoencoder',
]
