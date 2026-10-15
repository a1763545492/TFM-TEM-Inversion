# -*- coding: utf-8 -*-
"""
test_inversion.py — 反演模块单元测试
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import depth_layer_centers, forward_rhoa, DEFAULT_INV_CFG
from tfm_engine.inv_algorithms import _invert_trf, _invert_lm, _invert_occam
from tfm_engine.adv_inversion import _invert_occam_adaptive, _invert_cgls


class TestForwardRhoa:
    def test_uniform_halfspace(self):
        """均匀半空间正演应返回恒定 ρa"""
        z_c = depth_layer_centers(25, 5.0, 1500.0)
        theta = np.full(25, np.log10(100.0))
        t = np.geomspace(2e-6, 20e-3, 25)
        rhoa = forward_rhoa(theta, t, z_c, 100.0)
        # 均匀模型，ρa 应接近 100（扩散深度近似有波动）
        assert np.all(rhoa > 50)
        assert np.all(rhoa < 200)

    def test_two_layer(self):
        """两层模型正演"""
        z_c = depth_layer_centers(25, 5.0, 1500.0)
        theta = np.concatenate([np.full(5, np.log10(50.0)),
                                np.full(20, np.log10(500.0))])
        t = np.geomspace(2e-6, 20e-3, 25)
        rhoa = forward_rhoa(theta, t, z_c, 100.0)
        assert len(rhoa) == 25
        assert np.all(np.isfinite(rhoa))


class TestInvAlgorithms:
    @pytest.fixture
    def synthetic_data(self):
        """生成合成反演测试数据"""
        z_c = depth_layer_centers(25, 5.0, 1500.0)
        theta_true = np.concatenate([np.full(5, np.log10(50.0)),
                                     np.full(20, np.log10(500.0))])
        t = np.geomspace(2e-6, 20e-3, 25)
        rhoa_obs = forward_rhoa(theta_true, t, z_c, 100.0)
        err_obs = rhoa_obs * 0.05
        return t, rhoa_obs, err_obs, z_c, theta_true

    def test_trf(self, synthetic_data):
        t, r, e, z_c, theta_true = synthetic_data
        w = 1.0 / np.maximum(np.abs(np.log10(e / r + 1e-10)), 0.05)
        theta_est, info = _invert_trf(t, r, w, z_c, DEFAULT_INV_CFG)
        assert theta_est.shape == (25,)
        # 反演结果应接近真实值（前 5 层）
        assert np.mean(np.abs(theta_est[:5] - theta_true[:5])) < 1.0

    def test_lm(self, synthetic_data):
        t, r, e, z_c, theta_true = synthetic_data
        w = 1.0 / np.maximum(np.abs(np.log10(e / r + 1e-10)), 0.05)
        theta_est, info = _invert_lm(t, r, w, z_c, DEFAULT_INV_CFG)
        assert theta_est.shape == (25,)

    def test_occam(self, synthetic_data):
        t, r, e, z_c, theta_true = synthetic_data
        w = 1.0 / np.maximum(np.abs(np.log10(e / r + 1e-10)), 0.05)
        occfg = {'TARGET_MISFIT': 1.0, 'LAMBDA_INIT': 1.0, 'MAX_ITER': 10,
                 'TOL_MISFIT': 0.05, 'MIN_STEP': 1e-4}
        theta_est, info = _invert_occam(t, r, w, z_c, DEFAULT_INV_CFG, occfg)
        assert theta_est.shape == (25,)


class TestAdvancedAlgorithms:
    @pytest.fixture
    def synthetic_data(self):
        z_c = depth_layer_centers(25, 5.0, 1500.0)
        theta_true = np.concatenate([np.full(5, np.log10(50.0)),
                                     np.full(20, np.log10(500.0))])
        t = np.geomspace(2e-6, 20e-3, 25)
        rhoa_obs = forward_rhoa(theta_true, t, z_c, 100.0)
        err_obs = rhoa_obs * 0.05
        return t, rhoa_obs, err_obs, z_c

    def test_occam_adaptive(self, synthetic_data):
        t, r, e, z_c = synthetic_data
        w = 1.0 / np.maximum(np.abs(np.log10(e / r + 1e-10)), 0.05)
        cfg = dict(DEFAULT_INV_CFG)
        cfg.update({'TARGET_MISFIT': 1.0, 'MAX_ITER': 10})
        theta_est, info = _invert_occam_adaptive(t, r, w, z_c, cfg)
        assert theta_est.shape == (25,)

    def test_cgls(self, synthetic_data):
        t, r, e, z_c = synthetic_data
        w = 1.0 / np.maximum(np.abs(np.log10(e / r + 1e-10)), 0.05)
        cfg = dict(DEFAULT_INV_CFG)
        cfg.update({'CGLS_ITER': 20})
        theta_est, info = _invert_cgls(t, r, w, z_c, cfg)
        assert theta_est.shape == (25,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
