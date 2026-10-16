# -*- coding: utf-8 -*-
"""test_dl_inversion.py — 深度学习端到端反演测试"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfm_engine import dl_inversion as dl
from tfm_engine.inversion import depth_layer_centers, forward_rhoa


@pytest.fixture
def dataset():
    X, Y = dl.make_synthetic_dataset(n=1500, seed=11)
    return X, Y


class TestSynthetic:
    def test_dataset_shapes(self, dataset):
        X, Y = dataset
        assert X.shape == (1500, 25) and Y.shape == (1500, 25)
        assert np.isfinite(X).all() and np.isfinite(Y).all()
        assert Y.min() >= 0.3 - 1e-9 and Y.max() <= 3.2 + 1e-9


class TestLinearBackend:
    def test_linear_beats_mean(self, dataset):
        X, Y = dataset
        par = dl.train_linear(X[:1200], Y[:1200])
        inf = dl.LinearNumpy(par)
        pred = inf.predict(X[1200:])
        mse = np.mean((pred - Y[1200:]) ** 2)
        base = np.mean((Y[1200:] - Y[:1200].mean()) ** 2)
        assert mse < 0.75 * base
        # 逐样本相关性总体为正 (常量样本相关系数 nan, 剔除)
        corrs = []
        for i in range(len(pred)):
            if Y[1200 + i].std() > 1e-9 and pred[i].std() > 1e-9:
                corrs.append(np.corrcoef(pred[i], Y[1200 + i])[0, 1])
        assert np.median(corrs) > 0.5

    def test_auto_train_when_missing(self, tmp_path):
        w = tmp_path / 'no_weights.npz'
        inf = dl.load_inferencer(w, auto_train=True, log_fn=lambda s: None)
        assert inf.kind == 'linear'
        assert w.is_file()
        again = dl.load_inferencer(w, log_fn=lambda s: None)
        assert again.kind == 'linear'


@pytest.mark.skipif(not dl.torch_available(), reason='torch 不可用')
class TestTorchUNet:
    def test_numpy_parity_random_weights(self):
        """torch 前向与 numpy 推理器一致 (spawn 子进程, 规避 torch/MKL 冲突)。"""
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        proc = ctx.Process(target=dl._torch_parity_worker)
        proc.start(); proc.join()
        assert proc.exitcode == 0, f'torch/numpy parity failed ({proc.exitcode})'

    def test_train_and_save(self, tmp_path):
        """train_dl 在 spawn 子进程训练 U-Net, 父进程 numpy 加载验证。"""
        w = tmp_path / 'w_unet.npz'
        dl.train_dl(w, n_train=600, epochs=3, backend='torch',
                    log_fn=lambda s: None)
        X, _ = dl.make_synthetic_dataset(n=10, seed=5)
        inf = dl.load_inferencer(w, log_fn=lambda s: None)
        assert inf.kind == 'unet'
        pred = inf.predict(X)
        assert pred.shape == (10, 25) and np.isfinite(pred).all()


class TestInvertLine:
    def _make_line(self, root: Path, tag: str, n_st: int = 5):
        z = depth_layer_centers(25, 5.0, 1500.0)
        t = np.geomspace(2e-6, 2e-2, 25)
        d = root / tag; d.mkdir(parents=True)
        rng = np.random.default_rng(900)
        rows = []
        for i in range(n_st):
            theta = np.full(25, 1.5 + 0.3 * np.sin(i / 2))
            theta[7:15] += 0.6
            ra = forward_rhoa(theta, t, z, 100.0)
            for g in range(25):
                rows.append(f'{i*50:10.2f} 0 0 {t[g]:12.4e} {ra[g]:12.4e} '
                            f'{ra[g]*0.05:12.4e} "{tag}-{i+1:02d}" "017" "001"')
        (d / f'{tag}_occam.dat').write_text('# s\n' + '\n'.join(rows))

    def test_run_all_dl_linear(self, tmp_path):
        root = tmp_path / 'out'
        self._make_line(root, '1线')
        self._make_line(root, '2线')
        w = tmp_path / 'lin.npz'
        X, Y = dl.make_synthetic_dataset(n=2500, seed=3)
        dl.save_weights(w, dl.train_linear(X, Y), 'linear')
        res = dl.run_all_dl(root, weights_path=w, log_fn=lambda s: None)
        assert res['1线']['stations'] == 5
        assert res['1线']['rmse_median_log'] < 0.25
        d = root / '1线'
        assert (d / '1线_inv_dl_R.npy').is_file()
        assert (d / '1线_inv_dl_section.png').is_file()
        summ = pd.read_csv(d / '1线_inv_dl_summary.csv')
        assert len(summ) == 5 and (summ['success'] == True).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
