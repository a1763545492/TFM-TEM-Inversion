# -*- coding: utf-8 -*-
"""test_report_gen.py — HTML 报告生成测试"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import depth_layer_centers, forward_rhoa
from tfm_engine.inv_algorithms import run_all_algo
from tfm_engine.model_compare import compare_all_lines
from tfm_engine.report_gen import generate_report


def _make_line(root: Path, tag: str, n_st: int = 6):
    z_c = depth_layer_centers(25, 5.0, 1500.0)
    t = np.geomspace(2e-6, 2e-2, 25)
    d = root / tag; d.mkdir(parents=True)
    rng = np.random.default_rng(7000 + int(tag[0]))
    rows = []
    codes = []
    for i in range(n_st):
        code = f'{tag}-{i+1:02d}'
        codes.append(code)
        theta = np.full(25, 1.8 + 0.4 * np.sin(i / 3 + int(tag[0])))
        theta[8:16] += 0.7
        ra = forward_rhoa(theta, t, z_c, 1000.0) * (1 + rng.normal(0, 0.03, 25))
        er = ra * 0.05
        for g in range(25):
            rows.append(f'{i*50:14.4f} {i*10:10.2f} 0 {t[g]:12.4e} {ra[g]:12.4e} '
                        f'{er[g]:12.4e} "{code}" "017" "001"')
    (d / f'{tag}_occam.dat').write_text('# syn\n' + '\n'.join(rows), encoding='utf-8')
    # 模拟预处理产物
    pd.DataFrame({'station_code': codes,
                  'X_m': np.arange(n_st) * 50.0,
                  'Y_m': 0.0, 'Z_m': 0.0,
                  'qc_drop': False, 'qc_dead': False,
                  'rhoa_median_Ohm': rng.uniform(40, 120, n_st)}
                 ).to_csv(d / f'{tag}_pickup_xyz.csv', index=False,
                          encoding='utf-8-sig')


@pytest.fixture
def ready_root(tmp_path):
    root = tmp_path / 'data'; root.mkdir()
    _make_line(root, '1线')
    _make_line(root, '2线')
    (root / 'preprocess_report_20260101_000000.json').write_text(
        json.dumps({'summary': {'1线': {'stations_total': 6}}}))
    run_all_algo(root, algo='trf', log_fn=lambda s: None)
    run_all_algo(root, algo='lm', log_fn=lambda s: None)
    compare_all_lines(root, log_fn=lambda s: None)
    return root


class TestReport:
    def test_report_self_contained(self, ready_root):
        p = generate_report(ready_root, log_fn=lambda s: None)
        assert p.is_file()
        text = p.read_text(encoding='utf-8')
        assert text.lstrip().startswith('<!DOCTYPE html>')
        # 章节齐全
        for key in ('① 预处理 QC', '② 反演断面', '③ 多算法对比',
                    '④ 其他成果', '⑤ 参数快照'):
            assert key in text
        # 图片 base64 内嵌 (反演断面 + 对比断面至少各一)
        assert text.count('data:image/png;base64,') >= 3
        # RMSE 统计表内容
        assert 'TRF' in text and 'LM' in text

    def test_report_empty_root(self, tmp_path):
        root = tmp_path / 'empty'; root.mkdir()
        p = generate_report(root, log_fn=lambda s: None)
        text = p.read_text(encoding='utf-8')
        assert '未发现反演断面图' in text
        assert '未发现预处理产物' in text

    def test_custom_title_path(self, ready_root):
        custom = ready_root / 'my_report.html'
        p = generate_report(ready_root, html_path=custom, title='自定义标题X',
                            log_fn=lambda s: None)
        assert p == custom
        assert '自定义标题X' in p.read_text(encoding='utf-8')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
