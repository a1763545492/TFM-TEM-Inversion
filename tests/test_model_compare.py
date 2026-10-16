# -*- coding: utf-8 -*-
"""test_model_compare.py — 多算法对比 QC 模块测试"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import depth_layer_centers, forward_rhoa
from tfm_engine.inv_algorithms import run_all_algo
from tfm_engine.model_compare import (scan_algos, load_algo_result, build_stats,
                                      compare_line, compare_all_lines)


def _make_line(root: Path, tag: str, n_st: int = 8):
    z_c = depth_layer_centers(25, 5.0, 1500.0)
    t = np.geomspace(2e-6, 2e-2, 25)
    d = root / tag
    d.mkdir(parents=True)
    rng = np.random.default_rng(5000 + int(tag[0]))
    rows = []
    for i in range(n_st):
        theta = np.full(25, 2.0 + 0.5 * np.sin(i / 4 + int(tag[0])))
        theta[6:14] += 0.9
        ra = forward_rhoa(theta, t, z_c, 1000.0) * (1 + rng.normal(0, 0.03, 25))
        er = ra * 0.05
        for g in range(25):
            rows.append(f'{i*50:14.4f} 0 0 {t[g]:12.4e} {ra[g]:12.4e} {er[g]:12.4e} '
                        f'"{tag}-{i+1:02d}" "017" "001"')
    (d / f'{tag}_occam.dat').write_text('# syn\n' + '\n'.join(rows), encoding='utf-8')


@pytest.fixture
def ready_root(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    _make_line(root, '1线')
    run_all_algo(root, algo='trf', log_fn=lambda s: None)
    run_all_algo(root, algo='lm', log_fn=lambda s: None)
    return root


class TestModelCompare:
    def test_scan_and_load(self, ready_root):
        base = ready_root / '1线'
        algos = scan_algos(base, '1线')
        assert 'trf' in algos and 'lm' in algos
        r = load_algo_result(base, '1线', 'trf')
        assert r['R'].shape[1] == 25
        assert r['z'] is not None and r['summary'] is not None
        assert np.isfinite(r['rmse_med'])

    def test_stats_sorted(self, ready_root):
        base = ready_root / '1线'
        results = {a: load_algo_result(base, '1线', a)
                   for a in scan_algos(base, '1线')}
        df = build_stats(results)
        assert len(df) >= 2
        rmses = df['rmse_median_log'].dropna().values
        assert np.all(np.diff(rmses) >= -1e-9)  # 按 RMSE 升序

    def test_compare_line_outputs(self, ready_root):
        rows = compare_line(ready_root, '1线', log_fn=lambda s: None)
        assert len(rows) >= 2
        base = ready_root / '1线'
        for name in ('1线_compare_sections.png', '1线_compare_rmse.png',
                     '1线_compare_fits.png', '1线_compare_stats.csv'):
            assert (base / name).is_file(), name

    def test_compare_all_lines_total_csv(self, ready_root):
        _make_line(ready_root, '2线')
        run_all_algo(ready_root, algo='trf', log_fn=lambda s: None)
        per = compare_all_lines(ready_root, log_fn=lambda s: None)
        assert set(per) == {'1线', '2线'}
        assert (ready_root / 'all_lines_compare_stats.csv').is_file()

    def test_missing_line_empty(self, tmp_path):
        root = tmp_path / 'empty'; root.mkdir()
        (root / '1线').mkdir()
        assert compare_line(root, '1线', log_fn=lambda s: None) == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
