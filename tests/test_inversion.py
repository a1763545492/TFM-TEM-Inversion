# -*- coding: utf-8 -*-
"""
test_inversion.py — 反演模块单元测试
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import (depth_layer_centers, forward_rhoa, DEFAULT_INV_CFG,
                                  interp_weight_matrix, analytic_jacobian)
from tfm_engine.inv_algorithms import (_invert_trf, _invert_lm, _invert_occam,
                                       _weights, _numeric_jacobian)
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


class TestAnalyticJacobian:
    """解析雅可比 (插值权重矩阵) 正确性 + 与数值差分一致性"""

    @pytest.fixture
    def prob(self):
        rng = np.random.default_rng(7)
        z_c = depth_layer_centers(25, 5.0, 1500.0)
        theta = rng.normal(2.0, 0.5, 25)
        t = np.geomspace(2e-6, 20e-3, 25)
        rhoa = forward_rhoa(theta, t, z_c, 100.0)
        err = rhoa * 0.05
        w = _weights(rhoa, err, 1e-3)
        return t, z_c, theta, rhoa, w

    def test_weight_identity(self, prob):
        """W @ θ == log10(forward_rhoa(θ)) 机器精度恒等"""
        t, z_c, theta, _, _ = prob
        W = interp_weight_matrix(t, z_c, 100.0)
        assert W.shape == (25, 25)
        assert np.allclose(W @ theta, np.log10(forward_rhoa(theta, t, z_c, 100.0)),
                           atol=1e-12)

    def test_weight_structure(self, prob):
        """每行权重和=1, 至多2个非零元 (区间外钳端点=1个)"""
        t, z_c, _, _, _ = prob
        W = interp_weight_matrix(t, z_c, 100.0)
        assert np.allclose(W.sum(axis=1), 1.0)
        assert (W != 0).sum(axis=1).max() <= 2
        assert (W >= 0).all()

    def test_analytic_vs_numeric(self, prob):
        """解析雅可比 ≈ 前向差分数值雅可比"""
        t, z_c, theta, rhoa, w = prob
        Ja = analytic_jacobian(theta, t, z_c, 100.0, w)
        Jn = _numeric_jacobian(theta, t, z_c, 100.0, w)
        assert Ja.shape == Jn.shape
        assert np.allclose(Ja, Jn, atol=1e-5)

    def test_trf_analytic_equals_numeric(self, prob):
        """TRF 解析/数值两条路径反演结果一致"""
        t, z_c, _, rhoa, w = prob
        err = rhoa * 0.05
        th_a, _ = _invert_trf(t, rhoa, w, z_c, dict(DEFAULT_INV_CFG))
        th_n, _ = _invert_trf(t, rhoa, w, z_c,
                              dict(DEFAULT_INV_CFG, JAC_MODE='numeric'))
        assert np.allclose(th_a, th_n, atol=1e-6)

    def test_lci_global_jacobian(self):
        """LCI 全局稀疏雅可比: 独立参考残差的中心差分方向导数一致"""
        from tfm_engine.lci_inversion import (_build_global_jac_analytic,
                                              _build_lat_weight_matrix)
        rng = np.random.default_rng(3)
        n_st, n_l, ng = 4, 10, 12
        z_c = depth_layer_centers(n_l, 5.0, 1500.0)
        t = np.geomspace(2e-6, 20e-3, ng)
        t_list, r_list, e_list = [], [], []
        theta_cols = []
        for _ in range(n_st):
            th = rng.normal(2.0, 0.4, n_l)
            theta_cols.append(th)
            ra = forward_rhoa(th, t, z_c, 100.0)
            t_list.append(t); r_list.append(ra); e_list.append(ra * 0.05)
        lam_s, lam_l = 0.3, 1.0
        w_lat = np.ones(n_st - 1)
        R = _build_lat_weight_matrix(n_l, 'uniform')
        cfg = dict(DEFAULT_INV_CFG, LOG_EPS=1e-3)
        J = _build_global_jac_analytic(t_list, r_list, e_list, n_l, z_c,
                                       100.0, lam_s, lam_l, w_lat, R, cfg)
        # 独立线性参考残差 (与模块内 global_resid 同数学定义, 独立书写)
        def ref_resid(th_all):
            rs = []
            for i in range(n_st):
                thi = th_all[i * n_l:(i + 1) * n_l]
                pred = W_list[i] @ thi
                rs.append((pred - np.log10(np.maximum(r_list[i], 1e-3))) * w_list[i])
            for i in range(n_st):
                thi = th_all[i * n_l:(i + 1) * n_l]
                rs.append(np.sqrt(lam_s) * np.diff(thi))
            for i in range(n_st - 1):
                thi = th_all[i * n_l:(i + 1) * n_l]
                thj = th_all[(i + 1) * n_l:(i + 2) * n_l]
                rs.append(np.sqrt(lam_l * w_lat[i]) * R @ (thi - thj))
            return np.concatenate(rs)
        W_list = [interp_weight_matrix(t, z_c, 100.0) for _ in range(n_st)]
        w_list = [_weights(r_list[i], e_list[i], 1e-3) for i in range(n_st)]
        th = np.concatenate(theta_cols)
        v = rng.normal(0, 1, th.size)
        eps = 1e-6
        fd = (ref_resid(th + eps * v) - ref_resid(th - eps * v)) / (2 * eps)
        assert np.allclose(J @ v, fd, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
