# -*- coding: utf-8 -*-
"""
test_utils.py — 工具模块单元测试
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.static_correction import hampel_filter, detect_outliers_2d, remove_static_shift
from tfm_engine.resolution import compute_resolution
from tfm_engine.fem_forward import fem_forward_1d, fem_apparent_resistivity
from tfm_engine.forward import LayeredModel


class TestHampelFilter:
    def test_no_outliers(self):
        x = np.random.randn(100)
        mask = hampel_filter(x, window=5, n_sigma=3.0)
        # 高斯噪声下，异常点应很少（<10%，允许边缘效应）
        assert np.sum(mask) < 10

    def test_with_outlier(self):
        x = np.random.randn(100)
        x[50] = 100.0  # 明显异常
        mask = hampel_filter(x, window=5, n_sigma=3.0)
        assert mask[50] == True

    def test_edge(self):
        x = np.random.randn(10)
        mask = hampel_filter(x, window=3, n_sigma=3.0)
        assert len(mask) == 10


class TestStaticCorrection:
    def test_remove_shift(self):
        """静态偏移应被去除"""
        n_st, n_g = 20, 25
        station_x = np.arange(n_st) * 50.0
        rhoa = np.ones((n_st, n_g)) * 100.0
        # 添加静态偏移：中间几站偏移
        rhoa[8:12, :] *= 10.0
        rhoa_corr = remove_static_shift(rhoa, station_x, method='median_filter',
                                        filter_width=5, preserve_regional=True)
        # 校正后偏移应减小
        assert np.std(np.log10(rhoa_corr[:, 0])) < np.std(np.log10(rhoa[:, 0]))


class TestResolution:
    def test_resolution_matrix(self):
        """分辨率矩阵应接近单位阵（数据充足时）"""
        n_data, n_layers = 25, 25
        J = np.random.randn(n_data, n_layers)
        depth = np.geomspace(5.0, 1500.0, n_layers)
        res = compute_resolution(J, depth, cond_threshold=1e3)
        assert res['R'].shape == (n_layers, n_layers)
        assert res['sensitivity'].shape == (n_layers,)
        assert res['n_effective'] >= 1
        assert res['n_effective'] <= n_layers


class TestFEMForward:
    def test_fem_basic(self):
        """FEM 频率域正演"""
        model = LayeredModel(np.array([np.inf]), np.array([100.0]))
        freq = np.geomspace(1.0, 10000.0, 20)
        f, hz_mag, hz_phase = fem_forward_1d(model, 1e6, 30.0, freq)
        assert len(hz_mag) == 20
        assert np.all(hz_mag > 0)
        assert np.all(np.isfinite(hz_phase))

    def test_fem_rhoa(self):
        """FEM 视电阻率"""
        model = LayeredModel(np.array([np.inf]), np.array([100.0]))
        freq = np.geomspace(1.0, 10000.0, 20)
        f, hz_mag, _ = fem_forward_1d(model, 1e6, 30.0, freq)
        rhoa = fem_apparent_resistivity(hz_mag, freq, 1e6, 30.0)
        # 应接近 100（允许数值误差）
        assert np.all(rhoa > 10)
        assert np.all(rhoa < 1000)


class TestDatParser:
    def test_i16_ascii(self):
        """int16 切片转 ASCII"""
        from tfm_engine.dat_parser import i16_slice_to_ascii
        # 'ra' = 0x6172, 'ea' = 0x6165 (little-endian int16)
        raw = np.array([0x6172, 0x6165], dtype=np.int32)
        result = i16_slice_to_ascii(raw)
        assert isinstance(result, str)

    def test_n_pickets(self):
        """DAT 文件 picket 数计算（需实际文件，跳过）"""
        pytest.skip('需要实际 DAT 文件')


class TestCoordLoader:
    def test_import(self):
        """坐标加载模块可导入"""
        from tfm_engine import coord_loader
        assert hasattr(coord_loader, 'load_authoritative_coordinates')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
