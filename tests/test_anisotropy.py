# -*- coding: utf-8 -*-
"""test_anisotropy.py — VTI 各向异性测试"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.fem2d_forward import (build_mesh, assemble,
                                      polepole_pseudosection)
from tfm_engine.anisotropy import (vti_fields, lam_from_rho,
                                   vti_pseudosection, compare_iso_vti,
                                   vti_section_fields)

KW = dict(xlim=(-300, 300), zmax=300, nx=150, nz=65,
          spacings=[5.0, 10.0], log_fn=lambda s: None)
ELEC = [-60, -30, 0, 30, 60]


class TestVTI:
    def test_parametrization(self):
        rh, rv = vti_fields(lambda x, z: 100.0, lambda x, z: 2.0)
        assert rh(0, 0) == pytest.approx(50.0)
        assert rv(0, 0) == pytest.approx(200.0)
        assert np.sqrt(rh(0, 0) * rv(0, 0)) == pytest.approx(100.0)
        lam = lam_from_rho(np.array([50.0]), np.array([200.0]))
        assert lam[0] == pytest.approx(2.0)

    def test_assembly_iso_equals_vti_unit_lambda(self):
        nodes, tris, surf, _, _ = build_mesh((-50, 50), (0, 50), 30, 20)
        rho = np.random.default_rng(1).uniform(20, 300, tris.shape[0])
        K1 = assemble(nodes, tris, rho)
        K2 = assemble(nodes, tris, rho, rho)
        assert np.abs(K1 - K2).max() < 1e-9

    def test_homogeneous_vti_geometric_mean(self):
        """均匀 VTI 半空间: 地表 ρa = √(ρhρv)。"""
        rho_m, lam = 80.0, 1.5
        rh, rv = vti_fields(lambda x, z: rho_m, lambda x, z: lam)
        df = vti_pseudosection(rh, rv, ELEC, **KW)
        assert np.median(df['rhoa']) == pytest.approx(rho_m, rel=0.12)

    def test_vti_matches_isotropic_when_lambda1(self, tmp_path):
        df = compare_iso_vti(lambda x, z: 100.0, lambda x, z: 1.0,
                             ELEC, tmp_path / 'cmp.png', **KW)[0]
        assert np.allclose(df['rhoa_vti'], df['rhoa_iso'], rtol=1e-10)
        assert (tmp_path / 'cmp.png').is_file()

    def test_vertical_conductor_deep_response(self):
        """下层 ρv 低 (垂向良导) 时大间距 ρa 应比 ρv=ρh 更低。"""
        def rh_fn(x, z):
            return 100.0 if z < 25 else 100.0
        rv_lo = lambda x, z: 100.0 if z < 25 else 5.0
        rv_eq = lambda x, z: 100.0
        df_lo = vti_pseudosection(rh_fn, rv_lo, ELEC,
                                  xlim=(-300, 300), zmax=300, nx=150, nz=65,
                                  spacings=[40.0], log_fn=lambda s: None)
        df_eq = vti_pseudosection(rh_fn, rv_eq, ELEC,
                                  xlim=(-300, 300), zmax=300, nx=150, nz=65,
                                  spacings=[40.0], log_fn=lambda s: None)
        assert df_lo['rhoa'].median() < df_eq['rhoa'].median() * 0.85

    def test_vti_section_fields(self):
        z = np.array([10.0, 30.0])
        ch = np.array([0.0, 100.0])
        Rm = np.array([[100.0, 50.0], [100.0, 50.0]])
        Lam = np.full_like(Rm, 2.0)
        rh, rv = vti_section_fields(Rm, Lam, ch, z)
        assert rh(0.0, 10.0) == pytest.approx(50.0)
        assert rv(100.0, 30.0) == pytest.approx(100.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
