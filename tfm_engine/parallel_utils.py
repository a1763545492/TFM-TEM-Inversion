# -*- coding: utf-8 -*-
"""
parallel_utils.py — 多测线并行批处理公共调度器
================================================================
各处理阶段 (预处理 / 1D反演 / 高级反演 / LCI) 原来逐线串行。
本模块提供统一的按测线 ProcessPoolExecutor 调度:

    worker 约定 (必须定义在模块顶层, spawn 可 pickle):
        worker(lt, *args) -> (lt, result, logs: list[str])

    - n_line_jobs <= 1 : 串行, 日志实时回显 (行为与旧版一致)
    - n_line_jobs >  1 : 多进程, 每条线的日志在其完成后整体回放
    - 异常隔离: 单线失败不影响其他线, 返回 {'error': str(e)}
    - 防超订: 线级并行时, 调用方应把站内 n_jobs 强制为 1

完全离线, 无任何网络/云端调用。
"""
from __future__ import annotations
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional, Sequence


def auto_line_jobs(n_lines: int) -> int:
    """默认并行度: min(测线数, CPU逻辑核数-1), 至少 1。"""
    cpu = os.cpu_count() or 1
    return max(1, min(n_lines, cpu - 1))


def map_lines(line_tags: Sequence[str],
              worker: Callable,
              worker_args: tuple = (),
              n_line_jobs: int = 1,
              log_fn: Optional[Callable[[str], None]] = None,
              label: str = '') -> dict:
    """按测线调度 worker, 返回 {line_tag: result}。

    Parameters
    ----------
    line_tags : 测线标签序列 (如 ['1线','2线','3线','4线'])
    worker    : 模块顶层函数, 签名 worker(lt, *worker_args)
                返回 (lt, result, logs[list[str]])
    n_line_jobs : <=1 串行(实时日志); >1 进程池; 0 由 auto_line_jobs 决定
    """
    log_fn = log_fn or (lambda s: print(s))
    line_tags = list(line_tags)
    if n_line_jobs in (None, 0):
        n_line_jobs = auto_line_jobs(len(line_tags))
    n_line_jobs = max(1, int(n_line_jobs))

    # ---------- 串行: 日志实时, 保持旧版行为 ----------
    if n_line_jobs <= 1 or len(line_tags) <= 1:
        results = {}
        for lt in line_tags:
            try:
                _, result, logs = worker(lt, *worker_args)
                for line in logs:
                    log_fn(line)
                results[lt] = result
            except Exception as e:  # 异常隔离, 与串行循环语义一致
                tb = traceback.format_exc()
                log_fn(f'  ✗ {lt}: 并行任务异常: {e}\n{tb}')
                results[lt] = {'error': str(e)}
        return results

    # ---------- 并行: 每线一个进程, 完成后回放日志 ----------
    log_fn(f'[并行] {label} 启动 ProcessPool: {n_line_jobs} 进程, '
           f'测线 {line_tags} (站内并行已自动关闭以防 CPU 超订)')
    results = {}
    with ProcessPoolExecutor(max_workers=n_line_jobs) as exe:
        fut_map = {exe.submit(worker, lt, *worker_args): lt for lt in line_tags}
        for fut in as_completed(fut_map):
            lt = fut_map[fut]
            try:
                lt_ret, result, logs = fut.result()
                log_fn(f'\n{"─" * 30} {lt_ret} 完成 {"─" * 30}')
                for line in logs:
                    log_fn(line)
                results[lt_ret] = result
            except Exception as e:
                tb = traceback.format_exc()
                log_fn(f'  ✗ {lt}: 并行任务异常: {e}\n{tb}')
                results[lt] = {'error': str(e)}
    return results
