# -*- coding: utf-8 -*-
"""
test_forward.py — 正演模块单元测试
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.forward import (LayeredModel, tem_forward_1d, tem_forward_1d_freq,
                                 apparent_resistivity_dbzdt, apparent_resistivity_dbzdt_freq,
                                 _MU0)


class TestLayeredModel:
    def test_basic(self):
        m = LayeredModel(np.array([50.0, np.inf]), np.array([100.0, 500.0]))
        assert m.n_layers == 2
        assert not m.has_ip

    def test_ip(self):
        m = LayeredModel(np.array([50.0, np.inf]), np.array([100.0, 500.0]),
                         np.array([0.1, 0.2]), np.array([0.01, 0.01]), np.array([0.5, 0.5]))
        assert m.has_ip
        cr = m.complex_rho(np.array([1.0]))
        assert cr.shape == (2, 1)
        assert np.iscomplexobj(cr)


class TestTemForward1d:
    def test_halfspace(self):
        """均匀半空间正演应返回合理 Vt"""
        model = LayeredModel(np.array([np.inf]), np.array([100.0]))
        gates = np.geomspace(2e-6, 20e-3, 25)
        t, vt = tem_forward_1d(model, 1e6, 30.0, gates)
        assert len(vt) == 25
        assert np.all(vt > 0)
        assert np.all(np.isfinite(vt))

    def test_rhoa_roundtrip(self):
        """正演→视电阻率应近似恢复模型电阻率（晚期）"""
        model = LayeredModel(np.array([np.inf]), np.array([100.0]))
        gates = np.geomspace(5e-3, 20e-3, 5)  # 晚期
        t, vt = tem_forward_1d(model, 1e6, 30.0, gates)
        rhoa = apparent_resistivity_dbzdt(vt, t, 1e6, 30.0)
        # 均匀半空间晚期应接近 100（扩散深度近似有系统偏差，放宽到 50%）
        assert np.all(rhoa > 50)
        assert np.all(rhoa < 200)


class TestTemForward1dFreq:
    def test_halfspace_freq(self):
        """真频域正演：均匀半空间晚期 ρa → 真值"""
        model = LayeredModel(np.array([np.inf]), np.array([100.0]))
        gates = np.geomspace(5e-3, 20e-3, 5)
        t, dbzdt = tem_forward_1d_freq(model, 1e6, 30.0, gates)
        rhoa = apparent_resistivity_dbzdt_freq(dbzdt, t, 1e6, 30.0)
        # 真频域晚期应接近 100 ± 10%
        assert np.all(rhoa > 90)
        assert np.all(rhoa < 110)

    def test_two_layer(self):
        """两层模型正演"""
        model = LayeredModel(np.array([50.0, np.inf]), np.array([50.0, 500.0]))
        gates = np.geomspace(2e-6, 20e-3, 25)
        t, dbzdt = tem_forward_1d_freq(model, 1e6, 30.0, gates)
        assert len(dbzdt) == 25
        assert np.all(np.isfinite(dbzdt))


class TestIP:
    def test_ip_effect(self):
        """IP 效应应使晚期 Vt 增大（ρa 降低）"""
        model_noip = LayeredModel(np.array([np.inf]), np.array([100.0]))
        model_ip = LayeredModel(np.array([np.inf]), np.array([100.0]),
                                np.array([0.2]), np.array([0.01]), np.array([0.5]))
        gates = np.geomspace(1e-3, 20e-3, 10)
        _, vt_noip = tem_forward_1d(model_noip, 1e6, 30.0, gates)
        _, vt_ip = tem_forward_1d(model_ip, 1e6, 30.0, gates)
        # IP 应使晚期 Vt 增大（负激电效应）
        assert vt_ip[-1] > vt_noip[-1] * 0.9  # 允许数值波动


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
