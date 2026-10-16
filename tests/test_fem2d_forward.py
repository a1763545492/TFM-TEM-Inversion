# -*- coding: utf-8 -*-
"""test_fem2d_forward.py — 2D 有限元正演测试"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.fem2d_forward import (build_mesh, tri_geometry, assemble,
                                      source_vector, solve, element_rho,
                                      rho_from_section,
                                      apparent_rho_halfspace,
                                      apparent_rho_fullspace,
                                      polepole_pseudosection,
                                      plot_pseudosection)
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix


class TestMesh:
    def test_structured_counts(self):
        nodes, tris, surf, xs, zs = build_mesh((-10, 10), (0, 10),
                                               nx=20, nz=10)
        assert nodes.shape == (21 * 11, 2)
        assert tris.shape == (20 * 10 * 2, 3)
        assert len(surf) == 21
        A, _, _ = tri_geometry(nodes, tris)
        assert np.all(A > 0)
        assert np.allclose(xs[[0, -1]], [-10, 10])

    def test_pefc_refinement(self):
        # 聚焦点附近步长应明显小于均匀网格
        n, t, s, xs, zs = build_mesh((-100, 100), (0, 100), nx=100, nz=10,
                                     focus_x=(0.5,), focus_width=0.05,
                                     focus_strength=4.0)
        dx = np.diff(xs)
        center = np.argmin(np.abs(xs))
        assert dx[center] < 0.5 * dx[len(dx) // 4]
        # 单调
        assert np.all(xs[1:] > xs[:-1])

    def test_surface_refinement(self):
        _, _, _, _, zs = build_mesh((0, 100), (0, 100), nx=4, nz=40,
                                    surface_power=2.0)
        dz = np.diff(zs)
        assert dz[0] < 0.3 * dz[-1]


class TestPhysics:
    def test_assembly_symmetric(self):
        nodes, tris, surf, _, _ = build_mesh((-50, 50), (0, 50), 30, 20)
        rho = np.full(tris.shape[0], 100.0)
        K = assemble(nodes, tris, rho)
        d = np.abs(K - K.T).data
        assert np.all(d < 1e-10)

    def test_homogeneous_halfspace(self):
        """半空间均匀介质 ρa 应回到真值 (近间距, 误差容 12%)。"""
        rho0 = 100.0
        df = polepole_pseudosection(
            lambda x, z: rho0, elec_x=[-60, -30, 0, 30, 60],
            xlim=(-300, 300), zmax=300, nx=160, nz=70,
            spacings=[5.0, 10.0], log_fn=lambda s: None)
        # 小间距 (受点源奇异性与边界影响最小)
        near = df[df['spacing'] == 5.0]
        assert len(near) == 5
        assert np.median(near['rhoa']) == pytest.approx(rho0, rel=0.12)

    def test_homogeneous_fullspace(self):
        """全空间 (四边接地) 内部点源: 2π 因子公式。"""
        nodes, tris, surf, _, _ = build_mesh((-300, 300), (-300, 300),
                                             160, 120, focus_x=(0.5,))
        rho0 = 50.0
        x, z = nodes[:, 0], nodes[:, 1]
        edge = (np.abs(np.abs(x) - 300) < 1e-9) | (np.abs(np.abs(z) - 300) < 1e-9)
        gnd = np.where(edge)[0]
        K2 = assemble(nodes, tris, np.full(tris.shape[0], rho0)).tolil()
        K2[gnd, :] = 0.0; K2[gnd, gnd] = 1.0
        b = source_vector(nodes, tris, (0.0, 0.0), 1.0)
        b[gnd] = 0.0
        u = splu(csc_matrix(K2)).solve(b)
        for r1, r2 in [(8.0, 16.0), (10.0, 20.0)]:
            i1 = np.argmin((x - r1) ** 2 + z ** 2)
            i2 = np.argmin((x - r2) ** 2 + z ** 2)
            ra1, ra2 = np.hypot(x[i1], z[i1]), np.hypot(x[i2], z[i2])
            rhoa = apparent_rho_fullspace(u[i1], u[i2], ra1, ra2)
            assert abs(rhoa - rho0) / rho0 < 0.12

    def test_conductive_layer_decreases_rhoa(self):
        """浅层高阻 100, 深部良导 10: 大间距 ρa 应明显低于小间距。"""
        def rho_fn(x, z):
            return 10.0 if z > 25 else 100.0
        df = polepole_pseudosection(
            rho_fn, elec_x=[-40, -20, 0, 20, 40],
            xlim=(-300, 300), zmax=300, nx=150, nz=70,
            spacings=[5.0, 40.0], log_fn=lambda s: None)
        small = df[df['spacing'] == 5.0]['rhoa'].median()
        large = df[df['spacing'] == 40.0]['rhoa'].median()
        assert small > 70.0          # 浅部主要感受 100
        assert large < small * 0.7  # 大间距被低阻层拉低


class TestSectionModel:
    def test_rho_from_section(self):
        z = np.array([10.0, 30.0, 60.0])
        ch = np.array([0.0, 100.0])
        R = np.array([[100, 50, 20],
                      [100, 50, 20]], dtype=float)
        fn = rho_from_section(R, ch, z)
        assert fn(0.0, 10.0) == pytest.approx(100.0)
        assert fn(100.0, 60.0) == pytest.approx(20.0)
        # 深度插值: log 域线性
        assert fn(0.0, 20.0) == pytest.approx(np.sqrt(100 * 50), rel=1e-6)
        # 水平越界回退端部
        assert fn(-50.0, 10.0) == pytest.approx(100.0)

    def test_pseudosection_outputs(self, tmp_path):
        fn = rho_from_section(
            np.full((8, 20), 100.0),
            np.linspace(0, 350, 8),
            np.geomspace(5, 200, 20))
        df = polepole_pseudosection(fn, elec_x=np.linspace(0, 200, 6),
                                    xlim=(-300, 600), zmax=300,
                                    spacings=[5, 10], nx=120, nz=50,
                                    log_fn=lambda s: None)
        assert {'src_x', 'spacing', 'x_mid', 'z_eff', 'rhoa'} <= set(df.columns)
        assert len(df) > 0
        p = plot_pseudosection(df, tmp_path / 'pseudo.png')
        assert p.is_file() and p.stat().st_size > 10000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
