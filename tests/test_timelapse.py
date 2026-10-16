# -*- coding: utf-8 -*-
"""test_timelapse.py — 多时态对比测试"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import depth_layer_centers
from tfm_engine.timelapse import (align_epochs, compare_epochs,
                                  compare_epochs_all_lines)
from tfm_engine.export_formats import load_section


def _build_epoch(root: Path, tag: str, R: np.ndarray, z: np.ndarray,
                 codes=None, n_stations=None):
    """直接构造 occam.dat + 反演结果 npy (差异精确可控)。"""
    d = root / tag; d.mkdir(parents=True)
    n_st = R.shape[0]
    if codes is None:
        codes = [f'{tag}-{i+1:02d}' for i in range(n_st)]
    rows = ['# syn']
    t = np.geomspace(2e-6, 2e-2, 25)
    for i, code in enumerate(codes):
        for g in range(25):
            rows.append(f'{i*50:14.4f} 0 0 {t[g]:12.4e} 100 5 '
                        f'"{code}" "017" "001"')
    (d / f'{tag}_occam.dat').write_text('\n'.join(rows), encoding='utf-8')
    np.save(d / f'{tag}_inv_trf_R.npy', R)
    np.save(d / f'{tag}_inv_trf_depth.npy', z)


@pytest.fixture
def grids():
    z = depth_layer_centers(20, 5.0, 1000.0)
    n = 10
    R1 = np.full((n, len(z)), 100.0)
    R1[:, 8:14] = 30.0           # 低阻层
    R2 = R1.copy()
    R2[:, 10:14] = 60.0          # 深部低阻层电阻率翻倍 (充水/注浆变化)
    return z, R1, R2


class TestTimelapse:
    def test_identical_epoch_no_change(self, tmp_path, grids):
        z, R1, _ = grids
        e1 = tmp_path / 'e1'; e2 = tmp_path / 'e2'
        _build_epoch(e1, '1线', R1, z)
        _build_epoch(e2, '1线', R1, z)
        out = compare_epochs(e1, e2, '1线', log_fn=lambda s: None)
        assert abs(out['dlog10_median']) < 1e-9
        assert out['frac_unchanged'] == pytest.approx(1.0)
        assert Path(out['png_dlog10']).is_file()
        assert Path(out['csv_changes']).is_file()

    def test_change_detected(self, tmp_path, grids):
        z, R1, R2 = grids
        e1 = tmp_path / 'e1'; e2 = tmp_path / 'e2'
        _build_epoch(e1, '1线', R1, z)
        _build_epoch(e2, '1线', R2, z)
        out = compare_epochs(e1, e2, '1线', label1='2025', label2='2026',
                             log_fn=lambda s: None)
        # 变化层 j=10..13 Δlog10 = log10(2) ≈ 0.301
        d = pd.read_csv(out['csv_changes'])
        changed = d[(d['depth_m'] > z[10]) & (d['depth_m'] < z[14])]
        assert (changed['dlog10rho'].groupby(d['station_code']).mean() > 0.25).all()
        assert out['frac_up_over_thr'] > 0.1
        assert out['align_mode'] == 'station_code'
        # 变化率 = +100%
        assert np.isclose(d['change_pct'].max(), 100.0, atol=1e-6)
        # 未变化层
        shallow = d[d['depth_m'] < z[5]]
        assert np.allclose(shallow['dlog10rho'], 0.0, atol=1e-9)

    def test_chainage_align_different_stations(self, tmp_path, grids):
        z, R1, _ = grids
        # 期2 用不同测站编码且 8 站 (触发里程重采样)
        codes2 = [f'X-{i+1:02d}' for i in range(8)]
        R2e = np.full((8, len(z)), 100.0)
        R2e[:, 8:14] = 50.0
        e1 = tmp_path / 'e1'; e2 = tmp_path / 'e2'
        _build_epoch(e1, '1线', R1, z)
        _build_epoch(e2, '1线', R2e, z, codes=codes2)
        sec1 = load_section(e1, '1线', 'trf')
        sec2 = load_section(e2, '1线', 'trf')
        aln = align_epochs(sec1, sec2)
        assert aln['mode'] == 'chainage_interp'
        assert aln['L1'].shape == aln['L2'].shape == (10, 20)

    def test_different_depth_grid(self, tmp_path, grids):
        z, R1, _ = grids
        z2 = depth_layer_centers(15, 5.0, 1000.0)
        R2 = np.full((10, 15), 200.0)
        e1 = tmp_path / 'e1'; e2 = tmp_path / 'e2'
        _build_epoch(e1, '1线', R1, z)
        _build_epoch(e2, '1线', R2, z2)
        sec1 = load_section(e1, '1线', 'trf')
        sec2 = load_section(e2, '1线', 'trf')
        aln = align_epochs(sec1, sec2)
        assert aln['L1'].shape == aln['L2'].shape == (10, 20)
        # 期2 均匀 200 → 对齐后处处 log10(200)
        assert np.allclose(aln['L2'], np.log10(200.0))

    def test_all_lines_summary(self, tmp_path, grids):
        z, R1, R2 = grids
        e1 = tmp_path / 'e1'; e2 = tmp_path / 'e2'
        for tag in ['1线', '2线']:
            _build_epoch(e1, tag, R1, z)
            _build_epoch(e2, tag, R2, z)
        df = compare_epochs_all_lines(e1, e2, log_fn=lambda s: None)
        assert set(df['line']) == {'1线', '2线'}
        assert (e2 / '_timelapse' / 'timelapse_summary.csv').is_file()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
