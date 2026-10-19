# -*- coding: utf-8 -*-
"""test_data_check.py — Step0.5 数据体检 + --config 快照复现 单元测试"""
from __future__ import annotations
import sys
import json
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_engine.data_check import check_project, check_station_dat


class TestCheckProject:
    """结构级体检 (不依赖真实 dat 二进制)"""

    def test_empty_dir_rc2(self, tmp_path):
        """空目录: 无 [1-5]线 → inventory 失败 → rc=2 优雅降级"""
        r = check_project(tmp_path, out_root=tmp_path / 'out')
        assert r['rc'] == 2
        assert any('inventory' in e for e in r['errors'])
        # 报告落盘
        rp = tmp_path / 'out' / 'data_check_report.json'
        assert rp.is_file()
        assert json.loads(rp.read_text(encoding='utf-8'))['rc'] == 2

    def test_no_coord_file_rc2(self, tmp_path):
        """有测线目录但无坐标文件 → 报 ERROR"""
        (tmp_path / '1线').mkdir()
        # run_inventory 会因 xlsx 解析失败抛异常 → 顶层捕获 rc=2
        r = check_project(tmp_path, out_root=None)
        assert r['rc'] in (1, 2)
        assert r['errors'], '应至少有一条错误信息'


class TestSnapshot:
    """CLI --config 快照写/读往返 (纯 JSON 层)"""

    def _make_args(self, tmp_path):
        return Namespace(
            project=str(tmp_path / 'data'), out=str(tmp_path / 'out'),
            config_path=None, do_check=False, check_deep=False,
            inv_algo='mdn', dl_samples=1234, rhoa_scale=2e6,
            g_adc=4194304, n_gates=25,
            n_line_jobs=3, mcmc_samples=999, do_pre=True,
        )

    def test_roundtrip(self, tmp_path, monkeypatch):
        """写快照 → 改参数 → 读快照 → 参数恢复"""
        from tfm_cli_run import _write_args_snapshot, _load_args_snapshot
        out = tmp_path / 'out'
        out.mkdir()
        args = self._make_args(tmp_path)
        _write_args_snapshot(args, out)

        snaps = list(out.glob('args_snapshot_*.json'))
        assert len(snaps) == 1
        payload = json.loads(snaps[0].read_text(encoding='utf-8'))
        assert payload['__format__'] == 'tfm-cli-args-v1'
        assert 'numpy' in payload['environment']
        assert payload['args']['inv_algo'] == 'mdn'
        # 排除键
        assert 'project' not in payload['args']
        assert 'out' not in payload['args']

        # 篡改参数后恢复
        args2 = self._make_args(tmp_path)
        args2.inv_algo = 'trf'
        args2.dl_samples = 1
        args2.config_path = str(snaps[0])
        _load_args_snapshot(args2)
        assert args2.inv_algo == 'mdn'
        assert args2.dl_samples == 1234
        assert args2.n_line_jobs == 3

    def test_old_cfg_snapshot(self, tmp_path):
        """旧版 cfg_snapshot_*.json → 恢复预处理超参映射"""
        from tfm_cli_run import _load_args_snapshot
        old = tmp_path / 'cfg_snapshot_old.json'
        old.write_text(json.dumps({
            'RHOA_SCALE_CAL': 3e6, 'G_ADC_COUNTS_PER_VOLT': 2097152,
            'TEM_N_GATES': 20,
        }, ensure_ascii=False), encoding='utf-8')
        args = self._make_args(tmp_path)
        args.config_path = str(old)
        _load_args_snapshot(args)
        assert args.rhoa_scale == 3e6
        assert args.g_adc == 2097152
        assert args.n_gates == 20

    def test_missing_snapshot_raises(self, tmp_path):
        from tfm_cli_run import _load_args_snapshot
        args = self._make_args(tmp_path)
        args.config_path = str(tmp_path / 'nope.json')
        with pytest.raises(FileNotFoundError):
            _load_args_snapshot(args)


class TestCheckStationDat:
    """单文件体检: 非 dat 文件应返回 error 而非崩溃"""

    def test_garbage_file(self, tmp_path):
        fp = tmp_path / 'c001st001.dat'
        fp.write_bytes(b'\x00' * 128)
        r = check_station_dat(fp, log_fn=lambda s: None)
        assert r['status'] == 'error'
