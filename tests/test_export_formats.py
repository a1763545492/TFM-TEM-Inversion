# -*- coding: utf-8 -*-
"""test_export_formats.py — 标准化导出测试"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import depth_layer_centers, forward_rhoa
from tfm_engine.inv_algorithms import run_all_algo
from tfm_engine import export_formats as ef


def _make_line(root: Path, tag: str, n_st: int = 6):
    z_c = depth_layer_centers(25, 5.0, 1500.0)
    t = np.geomspace(2e-6, 2e-2, 25)
    d = root / tag; d.mkdir(parents=True)
    rng = np.random.default_rng(6000 + int(tag[0]))
    rows = []
    for i in range(n_st):
        theta = np.full(25, 1.8 + 0.5 * np.sin(i / 3 + int(tag[0])))
        theta[8:16] += 0.7
        ra = forward_rhoa(theta, t, z_c, 1000.0) * (1 + rng.normal(0, 0.03, 25))
        er = ra * 0.05
        for g in range(25):
            rows.append(f'{i*50:14.4f} {i*10:10.2f} 0 {t[g]:12.4e} {ra[g]:12.4e} '
                        f'{er[g]:12.4e} "{tag}-{i+1:02d}" "017" "001"')
    (d / f'{tag}_occam.dat').write_text('# syn\n' + '\n'.join(rows), encoding='utf-8')


def _read_dsaa(path: Path):
    """回读 DSAA GRD, 返回 nx,ny,xlim,ylim,矩阵。"""
    lines = path.read_text(encoding='ascii').strip().split('\n')
    assert lines[0].strip() == 'DSAA'
    nx, ny = map(int, lines[1].split())
    x0, x1 = map(float, lines[2].split())
    y0, y1 = map(float, lines[3].split())
    z0, z1 = map(float, lines[4].split())
    mat = np.array([list(map(float, ln.split())) for ln in lines[5:5 + ny]])
    return nx, ny, (x0, x1), (y0, y1), (z0, z1), mat


@pytest.fixture
def ready_root(tmp_path):
    root = tmp_path / 'data'; root.mkdir()
    _make_line(root, '1线')
    run_all_algo(root, algo='trf', log_fn=lambda s: None)
    return root


class TestExport:
    def test_load_section_fallback_coords(self, ready_root):
        sec = ef.load_section(ready_root, '1线', 'trf')
        assert sec['R'].shape == (6, 25)
        assert len(sec['z_centers']) == 25
        # 无 pickup csv → 回退等间距 50 m
        assert np.allclose(sec['chainage'], np.arange(6) * 50.0)
        assert len(sec['stations']) == 6

    def test_csv(self, ready_root):
        sec = ef.load_section(ready_root, '1线', 'trf')
        p = ef.export_csv(sec, ready_root / '1线' / 'out.csv')
        import pandas as pd
        df = pd.read_csv(p)
        assert len(df) == 6 * 25
        assert {'station_code', 'X_m', 'depth_m', 'rho_ohmm',
                'log10_rho'} <= set(df.columns)
        assert np.isfinite(df['rho_ohmm']).all()

    def test_xyz(self, ready_root):
        sec = ef.load_section(ready_root, '1线', 'trf')
        p = ef.export_xyz(sec, ready_root / '1线' / 'out.xyz')
        data = np.loadtxt(p)
        assert data.shape == (6 * 25, 3)
        # y = -深度, 首行为最浅 (XYZ 文本保留 3 位小数)
        assert data[0, 1] == pytest.approx(-sec['z_centers'][0], abs=1e-3)
        # z 值与 R 对应
        assert data[0, 2] == pytest.approx(sec['R'][0, 0], rel=1e-4)

    def test_xyz_log10(self, ready_root):
        sec = ef.load_section(ready_root, '1线', 'trf')
        p = ef.export_xyz(sec, ready_root / '1线' / 'out_log.xyz',
                          value_scale='log10')
        data = np.loadtxt(p)
        assert data[0, 2] == pytest.approx(np.log10(sec['R'][0, 0]), abs=1e-4)

    def test_grd_dsaa_roundtrip(self, ready_root):
        sec = ef.load_section(ready_root, '1线', 'trf')
        p = ef.export_surfer_grd(sec, ready_root / '1线' / 'out.grd')
        nx, ny, xlim, ylim, zlim, mat = _read_dsaa(p)
        assert (nx, ny) == (6, 25)
        # 首行 = ymax 最浅 = R 的第 0 层
        assert np.allclose(mat[0], sec['R'][:, 0], rtol=1e-4)
        assert mat.shape == (25, 6)
        assert zlim[0] <= sec['R'].min() <= zlim[1]

    def test_export_line_batch(self, ready_root):
        done = ef.export_line(ready_root, '1线', 'trf',
                              formats=('csv', 'xyz', 'grd'),
                              log_fn=lambda s: None)
        assert set(done) == {'csv', 'xyz', 'grd'}
        assert all(p.is_file() for p in done.values())

    def test_geotiff_optional(self, ready_root):
        rasterio = pytest.importorskip('rasterio')  # 无 rasterio 则跳过
        sec = ef.load_section(ready_root, '1线', 'trf')
        p = ef.export_geotiff(sec, ready_root / '1线' / 'out.tif')
        assert p is not None and p.is_file()
        with rasterio.open(p) as ds:
            assert ds.count == 1
            assert ds.read(1).shape == (25, 6)

    def test_missing_result_raises(self, tmp_path):
        root = tmp_path / 'd'; root.mkdir(); (root / '2线').mkdir()
        with pytest.raises(FileNotFoundError):
            ef.load_section(root, '2线', 'trf')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
