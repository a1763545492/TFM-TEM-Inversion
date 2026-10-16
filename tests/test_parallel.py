# -*- coding: utf-8 -*-
"""
test_parallel.py — 多测线并行调度测试
串行/并行结果必须逐位一致 (反演本身确定性); 单线异常必须隔离。
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine.inversion import depth_layer_centers, forward_rhoa, run_all_inversions
from tfm_engine.parallel_utils import map_lines, auto_line_jobs


def _make_syn_line(root: Path, tag: str, n_st: int = 6):
    """生成合成 Occam.dat (固定种子, 保证两 root 数据完全一致)。"""
    z_c = depth_layer_centers(25, 5.0, 1500.0)
    t = np.geomspace(2e-6, 2e-2, 25)
    d = root / tag
    d.mkdir(parents=True)
    rng = np.random.default_rng(3000 + int(tag[0]))
    rows = []
    for i in range(n_st):
        theta = np.full(25, 2.0 + 0.5 * np.sin(i / 5 + int(tag[0])))
        theta[6:14] += 0.9
        ra = forward_rhoa(theta, t, z_c, 1000.0) * (1 + rng.normal(0, 0.03, 25))
        er = ra * 0.05
        for g in range(25):
            rows.append(f'{i*50:14.4f} 0 0 {t[g]:12.4e} {ra[g]:12.4e} {er[g]:12.4e} '
                        f'"{tag}-{i+1:02d}" "017" "001"')
    (d / f'{tag}_occam.dat').write_text('# syn\n' + '\n'.join(rows), encoding='utf-8')


def _line_ok_worker(line_tag: str):
    """测试用模块级 worker (spawn 可 pickle)。"""
    return line_tag, {'ok': True}, [f'{line_tag} done']


def _line_fail_worker(line_tag: str):
    if line_tag == '2线':
        raise RuntimeError('mock line failure')
    return line_tag, {'ok': True}, []


class TestParallelUtils:
    def test_auto_line_jobs(self):
        assert auto_line_jobs(1) == 1
        assert 1 <= auto_line_jobs(4) <= 4

    def test_serial_path(self):
        """n_line_jobs=1 串行, 日志实时, 结果完整"""
        logs = []
        out = map_lines(['1线', '2线'], _line_ok_worker, n_line_jobs=1,
                        log_fn=logs.append)
        assert out == {'1线': {'ok': True}, '2线': {'ok': True}}
        assert any('1线 done' in x for x in logs)

    def test_exception_isolation_serial(self):
        out = map_lines(['1线', '2线', '3线'], _line_fail_worker, n_line_jobs=1,
                        log_fn=lambda s: None)
        assert out['1线'] == {'ok': True}
        assert 'error' in out['2线']
        assert out['3线'] == {'ok': True}

    def test_exception_isolation_parallel(self):
        """进程池下单线崩溃不影响其他线"""
        out = map_lines(['1线', '2线', '3线'], _line_fail_worker, n_line_jobs=3,
                        log_fn=lambda s: None)
        assert out['1线'] == {'ok': True}
        assert 'error' in out['2线']
        assert out['3线'] == {'ok': True}


class TestParallelInversion:
    def test_serial_parallel_identical(self, tmp_path):
        """2 线合成数据: 串行与 2 进程反演结果逐位一致 + 输出文件齐全"""
        root_s = tmp_path / 'ser'
        root_p = tmp_path / 'par'
        root_s.mkdir(); root_p.mkdir()
        for r in (root_s, root_p):
            _make_syn_line(r, '1线')
            _make_syn_line(r, '2线')

        res_s = run_all_inversions(root_s, log_fn=lambda s: None, n_line_jobs=1)
        res_p = run_all_inversions(root_p, log_fn=lambda s: None, n_line_jobs=2)
        assert set(res_s) == {'1线', '2线'}
        assert set(res_p) == {'1线', '2线'}
        for tag in ('1线', '2线'):
            Rs = np.load(root_s / tag / f'{tag}_1d_inversion_R.npy')
            Rp = np.load(root_p / tag / f'{tag}_1d_inversion_R.npy')
            assert Rs.shape == Rp.shape
            assert np.array_equal(Rs, Rp), f'{tag} 串行/并行结果不一致'
            assert (root_p / tag / f'{tag}_1d_inversion_summary.csv').is_file()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
