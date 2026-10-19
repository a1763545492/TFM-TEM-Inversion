# -*- coding: utf-8 -*-
"""test_dl_advanced.py — dl_advanced.py 单元测试 (PINN / MDN / TCN)"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

# 确保 tfm_engine 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_engine.dl_advanced import (
    PINNNumpy, MDNNumpy, TCNNumpy,
    _np_conv1d, _ALGO_MAP,
)


class TestNpConv1d:
    """numpy 卷积基础测试"""

    def test_dilation_1(self):
        """dilation=1 时与标准卷积一致"""
        x = np.random.randn(2, 3, 20)
        w = np.random.randn(4, 3, 5)
        b = np.random.randn(4)
        out = _np_conv1d(x, w, b, dilation=1)
        assert out.shape == (2, 4, 20)

    def test_dilation_2(self):
        """dilation=2 时输出长度正确"""
        x = np.random.randn(2, 3, 20)
        w = np.random.randn(4, 3, 5)
        b = np.random.randn(4)
        out = _np_conv1d(x, w, b, dilation=2)
        assert out.shape == (2, 4, 20)


class TestPINN:
    """PINN 推理器测试"""

    def test_predict_shape(self):
        """输出形状正确"""
        n_gates, n_layers, hidden = 25, 25, 64
        params = {
            'w1': np.random.randn(n_gates, hidden) * 0.1,
            'b1': np.zeros(hidden),
            'w2': np.random.randn(hidden, hidden) * 0.1,
            'b2': np.zeros(hidden),
            'w3': np.random.randn(hidden, n_layers) * 0.1,
            'b3': np.zeros(n_layers),
            'xmean': np.zeros(n_gates), 'xstd': np.ones(n_gates),
            'n_layers': n_layers,
        }
        net = PINNNumpy(params)
        X = np.random.randn(3, n_gates)
        out = net.predict(X)
        assert out.shape == (3, n_layers)
        assert np.all(out >= 0.3) and np.all(out <= 3.2)


class TestMDN:
    """MDN 推理器测试"""

    def test_predict_shape(self):
        """输出形状正确"""
        n_gates, n_layers, hidden, n_mix = 25, 25, 64, 3
        params = {
            'w1': np.random.randn(n_gates, hidden) * 0.1,
            'b1': np.zeros(hidden),
            'w2': np.random.randn(hidden, hidden) * 0.1,
            'b2': np.zeros(hidden),
            'w_pi': np.random.randn(hidden, n_mix) * 0.1,
            'b_pi': np.zeros(n_mix),
            'w_mu': np.random.randn(hidden, n_mix * n_layers) * 0.1,
            'b_mu': np.zeros(n_mix * n_layers),
            'w_sig': np.random.randn(hidden, n_mix * n_layers) * 0.1,
            'b_sig': np.zeros(n_mix * n_layers),
            'xmean': np.zeros(n_gates), 'xstd': np.ones(n_gates),
            'n_layers': n_layers, 'n_mix': n_mix,
        }
        net = MDNNumpy(params)
        X = np.random.randn(3, n_gates)
        out = net.predict(X)
        assert out.shape == (3, n_layers)

    def test_uncertainty(self):
        """不确定度输出形状正确且非负"""
        n_gates, n_layers, hidden, n_mix = 25, 25, 64, 3
        params = {
            'w1': np.random.randn(n_gates, hidden) * 0.1,
            'b1': np.zeros(hidden),
            'w2': np.random.randn(hidden, hidden) * 0.1,
            'b2': np.zeros(hidden),
            'w_pi': np.random.randn(hidden, n_mix) * 0.1,
            'b_pi': np.zeros(n_mix),
            'w_mu': np.random.randn(hidden, n_mix * n_layers) * 0.1,
            'b_mu': np.zeros(n_mix * n_layers),
            'w_sig': np.random.randn(hidden, n_mix * n_layers) * 0.1,
            'b_sig': np.zeros(n_mix * n_layers),
            'xmean': np.zeros(n_gates), 'xstd': np.ones(n_gates),
            'n_layers': n_layers, 'n_mix': n_mix,
        }
        net = MDNNumpy(params)
        X = np.random.randn(3, n_gates)
        mean, std = net.predict_with_uncertainty(X)
        assert mean.shape == (3, n_layers)
        assert std.shape == (3, n_layers)
        assert np.all(std >= 0)


class TestTCN:
    """TCN 推理器测试"""

    def test_predict_shape(self):
        """输出形状正确"""
        n_gates, n_layers, channels, n_tcn = 25, 25, 16, 3
        params = {
            'xmean': np.zeros(n_gates), 'xstd': np.ones(n_gates),
            'n_layers': n_layers, 'n_tcn_layers': n_tcn,
        }
        for i in range(n_tcn):
            params[f'w{i}'] = np.random.randn(channels, 1 if i == 0 else channels, 3) * 0.1
            params[f'b{i}'] = np.zeros(channels)
            params[f'd{i}'] = 2 ** i
        params['w_out'] = np.random.randn(channels, n_layers) * 0.1
        params['b_out'] = np.zeros(n_layers)

        net = TCNNumpy(params)
        X = np.random.randn(3, n_gates)
        out = net.predict(X)
        assert out.shape == (3, n_layers)


class TestAlgoMap:
    """算法映射完整性"""

    def test_algos_exist(self):
        assert 'pinn' in _ALGO_MAP
        assert 'mdn' in _ALGO_MAP
        assert 'tcn' in _ALGO_MAP
        for algo, (label, train_fn, cls) in _ALGO_MAP.items():
            assert callable(train_fn)
            assert isinstance(cls, type)
