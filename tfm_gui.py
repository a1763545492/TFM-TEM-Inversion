# -*- coding: utf-8 -*-
"""
tfm_gui.py (M6 Final Upgrade: WLF-Full-Feature)
=================================================
ttk.Notebook 6 Tabs 完整集成：
  [1] Basic Pipeline (原 GUI: 路径/Steps/HP 3★/开始处理)
  [2] Layer Model Designer (1D层状模型 Treeview: h/ρ/m/τ/c 编辑 + 保存 + 正演预览)
  [3] Forward Compare (选模型.dat + 选 Occam station → 正演叠加对比)
  [4] Signal Viewer (从 decay_curves.npz 选 station → 4×4 衰减曲线 FigureCanvas)
  [5] IP Inversion (share_τc / λ / N_Layers / 全线IP反演)
  [6] WLF 2D / 3D Bridge (TRANS24 / SWP24D / 3D tomo 一键桥接)
遵循:
  ★ 100% 离线（GUI 仅调本地 python/BAT/EXE，无网络）
  ★ 日志队列（后台线程 + after 200ms 轮询 + 磁盘落盘 gui_*.log）
  ★ 异常捕获 traceback
"""
from __future__ import annotations
import sys
import os
import threading
import queue
import time
import traceback
import datetime
import json
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.simpledialog  # noqa: 模块级导入, 用于 Tab2 双击编辑 / Tab5 单站选择 (避免仅 main() 内 import 导致 NameError)

import numpy as np

# =====================================================================
# matplotlib 嵌入 tkinter
# =====================================================================
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ---------------------------------------------------------------------
# 绝对导入 (工具目录已在 sys.path)
# ---------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from tfm_engine.preprocess import run_full_preprocess
from tfm_engine.inversion import run_all_inversions
from tfm_engine.inv_algorithms import (run_all_algo, run_all_algos_compare,
                                       ALGO_CHOICES)
from tfm_engine.adv_inversion import run_all_advanced, ALGO_ADVANCED
from tfm_engine.lci_inversion import run_all_lci
from tfm_engine.current_proc import run_all_current
from tfm_engine.occam_bridge import materialize_all
from tfm_engine.forward import (LayeredModel, tem_forward_1d,
                                apparent_resistivity_dbzdt)
from tfm_engine.preprocess_plus import run_full_preprocess_plus
from tfm_engine.ip_inversion import run_all_ip_inversions, invert_one_station_ip
from tfm_engine.signal_viz import (make_decay_4x4figure, batch_save_decay_pngs,
                                   plot_channel_overlay)
from tfm_engine.wlf_bridge import (run_trans24_convert, run_2d_inv_wlf,
                                   run_3d_inv_wlf)
# v6 新模块
from tfm_engine.fem_forward import fem_forward_1d, fem_apparent_resistivity
from tfm_engine.static_correction import correct_profile, plot_correction_comparison
from tfm_engine.resolution import compute_resolution_from_inversion, plot_resolution
from tfm_engine.nn_initial import NNInitialModel, create_warm_start_inversion
from tfm_engine.denoise_ae import DenoisingAutoencoder
from tfm_engine.viz_3d import (plot_2d_section_enhanced, plot_section_comparison,
                               export_3d_volume_html, export_section_interactive)
from tfm_engine.project_db import ProjectDB, create_project_db


# =====================================================================
# 主应用
# =====================================================================
class TFMMainApp:
    WINDOW_TITLE = "TFM 时频电磁法完整处理系统 v2.0 (WLF算法集·全离线)"
    WINDOW_SIZE = "1280x820"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.minsize(1024, 680)

        # ---------- 日志基础设施 ----------
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self._logger_path = _HERE.parent / "example_output" / "7_tfm_production"
        self._logger_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_fh = open(self._logger_path / f"gui_{ts}.log",
                            "a", encoding="utf-8", buffering=1)

        # ---------- Tk 样式 ----------
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabelframe.Label", font=("Microsoft YaHei", 10, "bold"))
        style.configure("Treeview", rowheight=24, font=("Consolas", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei", 9, "bold"))

        # ========== Notebook (6 Tabs) ==========
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.tab_basic = ttk.Frame(self.nb);        self.nb.add(self.tab_basic, text=" ① Basic Pipeline ")
        self.tab_model = ttk.Frame(self.nb);        self.nb.add(self.tab_model, text=" ② Layer Model ")
        self.tab_fwd   = ttk.Frame(self.nb);        self.nb.add(self.tab_fwd,   text=" ③ Forward Compare ")
        self.tab_sig   = ttk.Frame(self.nb);        self.nb.add(self.tab_sig,   text=" ④ Signal Viewer ")
        self.tab_ip    = ttk.Frame(self.nb);        self.nb.add(self.tab_ip,    text=" ⑤ IP Inversion ")
        self.tab_wlf   = ttk.Frame(self.nb);        self.nb.add(self.tab_wlf,   text=" ⑥ WLF 2D/3D ")
        self.tab_cur   = ttk.Frame(self.nb);        self.nb.add(self.tab_cur,   text=" ⑦ Tx Current ")
        self.tab_adv   = ttk.Frame(self.nb);        self.nb.add(self.tab_adv,   text=" ⑧ Advanced ")
        self.tab_ai    = ttk.Frame(self.nb);        self.nb.add(self.tab_ai,    text=" ⑨ AI & Viz ")
        self.tab_proj  = ttk.Frame(self.nb);        self.nb.add(self.tab_proj,  text=" ⑩ Project DB ")

        self._build_tab_basic()
        self._build_tab_model()
        self._build_tab_forward()
        self._build_tab_signal()
        self._build_tab_ip()
        self._build_tab_wlf()
        self._build_tab_current()
        self._build_tab_advanced()
        self._build_tab_ai_viz()
        self._build_tab_project()

        # ---------- 底部状态栏（路径 + 日志滚动） ----------
        status = ttk.LabelFrame(root, text="运行日志 (实时同步至磁盘 gui_*.log)")
        status.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.log_text = tk.Text(status, height=10, wrap="word",
                                bg="#1e1e1e", fg="#e0e0e0",
                                insertbackground="#e0e0e0",
                                selectbackground="#264f78",
                                font=("Consolas", 10), state="disabled")
        for tag, color in [("I", "#9cdcfe"), ("S", "#4ec9b0"),
                           ("W", "#ffd700"), ("E", "#f44747")]:
            self.log_text.tag_configure(tag, foreground=color)
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._drain_log_queue()

    # =================================================================
    # Common helpers
    # =================================================================
    def log(self, msg: str, level: str = "I"):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        self.log_queue.put(line)
        try:
            self._log_fh.write(line + "\n"); self._log_fh.flush()
        except Exception:
            pass

    def _drain_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                tag = line[9:10]  # [H:M:S] [X]
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line + "\n", tag if tag in "ISWE" else "")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self._drain_log_queue)

    def _start_worker(self, target, *args):
        t = threading.Thread(target=self._worker_wrapper,
                             args=(target, args), daemon=True)
        t.start()

    def _worker_wrapper(self, target, args):
        try:
            target(*args)
        except Exception as exc:
            tb = traceback.format_exc()
            self.log(f"FATAL: {exc}\n{tb}", level="E")
            try:
                messagebox.showerror("运行错误", f"{exc}\n\n详细日志见 gui_*.log")
            except Exception:
                pass

    # =================================================================
    # TAB 1: Basic Pipeline (原 GUI)
    # =================================================================
    def _build_tab_basic(self):
        p = self.tab_basic
        # --- 路径 ---
        lf_path = ttk.LabelFrame(p, text="① 路径设置")
        lf_path.pack(fill="x", padx=6, pady=6)
        ttk.Label(lf_path, text="项目根 (D:\\MyProject\\myfile):").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.var_project = tk.StringVar(value=r"D:\MyProject\myfile")
        ttk.Entry(lf_path, textvariable=self.var_project, width=70).grid(row=0, column=1, padx=4, pady=3)
        ttk.Button(lf_path, text="浏览…", command=self._browse_project).grid(row=0, column=2, padx=4)
        ttk.Label(lf_path, text="输出根:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.var_out = tk.StringVar(value=str(_HERE.parent / "example_output" / "7_tfm_production"))
        ttk.Entry(lf_path, textvariable=self.var_out, width=70).grid(row=1, column=1, padx=4, pady=3)
        ttk.Button(lf_path, text="浏览…", command=self._browse_out).grid(row=1, column=2, padx=4)

        # --- Steps ---
        lf_steps = ttk.LabelFrame(p, text="② 处理步骤 (勾选执行)")
        lf_steps.pack(fill="x", padx=6, pady=6)
        self.var_s12 = tk.BooleanVar(value=True);  self.var_s3 = tk.BooleanVar(value=True)
        self.var_s4  = tk.BooleanVar(value=True);  self.var_s_pp = tk.BooleanVar(value=True)
        self.var_s_ip = tk.BooleanVar(value=False); self.var_s_sig = tk.BooleanVar(value=True)
        rows = [("Step1-2 预处理 (权威Excel解析 → Cube→gates→ρa→Occam.dat)", self.var_s12),
                ("Step1-2+ 增强预处理 (陷波/带通/关断精修 + 衰减曲线NPZ+PNG)", self.var_s_pp),
                ("Step3   1D least_squares 电阻率反演 (25层断面图)", self.var_s3),
                ("Step3+  1D 激电(IP)极化率反演 (ρ0,m,τ,c 断面图×3)", self.var_s_ip),
                ("Step4   Occam1D 商业桥接 (cfg+model-p+BAT UTF-8)", self.var_s4),
                ("导出    所有站信号衰减曲线 PNG (4×4=16子图) ★★", self.var_s_sig)]
        for i, (t, v) in enumerate(rows):
            ttk.Checkbutton(lf_steps, text=t, variable=v).grid(row=i // 2, column=i % 2, sticky="w", padx=10, pady=2)

        # --- 超参数 HP ★首选3个红标 ---
        lf_hp = ttk.LabelFrame(p, text="③ 超参数 HP (★首选调参: 单变量, 步长×2)")
        lf_hp.pack(fill="x", padx=6, pady=6)
        self.var_hp_rhoa = tk.StringVar(value="1e6")
        self.var_hp_lam_s = tk.StringVar(value="0.3")
        self.var_hp_lam_i = tk.StringVar(value="1.0")
        self.var_hp_ngates = tk.StringVar(value="25")
        self.var_hp_nlayers = tk.StringVar(value="25")
        self.var_occam_exe = tk.StringVar(value="Occam1D.exe")
        hp = [("★ RHOA_SCALE_CAL (Ω·m 校准)", self.var_hp_rhoa, 0, 0),
              ("★ LAMBDA_SMOOTH (1D正则)",       self.var_hp_lam_s, 0, 2),
              ("★ LAMBDA_INIT (Occam)",           self.var_hp_lam_i, 1, 0),
              ("TEM_N_GATES (门数)",              self.var_hp_ngates, 1, 2),
              ("N_LAYERS (反演层数)",             self.var_hp_nlayers, 2, 0),
              ("Occam EXE 名",                    self.var_occam_exe, 2, 2)]
        for t, v, r, c in hp:
            lbl = ttk.Label(lf_hp, text=t)
            if t.startswith("★"):
                lbl.configure(foreground="#c00000", font=("Microsoft YaHei", 9, "bold"))
            lbl.grid(row=r, column=c, sticky="e", padx=6, pady=3)
            ttk.Entry(lf_hp, textvariable=v, width=14).grid(row=r, column=c + 1, padx=3, pady=3, sticky="w")

        # --- 1D 反演算法选择 (trf / lm / occam / 高级算法 / all=三算法对比) ---
        ttk.Label(lf_hp, text="1D 反演算法:",
                  foreground="#0a5cad", font=("Microsoft YaHei", 9, "bold")
                  ).grid(row=3, column=0, sticky="e", padx=6, pady=3)
        self.var_inv_algo = tk.StringVar(value="trf")
        ttk.Combobox(lf_hp, textvariable=self.var_inv_algo, width=18, state="readonly",
                     values=["trf", "lm", "occam",
                             "occam_adaptive", "lm_adaptive", "cgls", "mcmc",
                             "lci", "all(三算法对比)"]
                     ).grid(row=3, column=1, padx=3, pady=3, sticky="w")

        # --- 高级反演参数 (occam_adaptive / lm_adaptive / cgls / mcmc) ---
        self.var_n_jobs = tk.StringVar(value="1")
        self.var_use_cache = tk.BooleanVar(value=False)
        self.var_mcmc_samples = tk.StringVar(value="2000")
        ttk.Label(lf_hp, text="并行进程数:").grid(row=4, column=0, sticky="e", padx=6, pady=3)
        ttk.Spinbox(lf_hp, from_=1, to=32, textvariable=self.var_n_jobs,
                    width=12).grid(row=4, column=1, padx=3, pady=3, sticky="w")
        ttk.Checkbutton(lf_hp, text="启用缓存断点续跑",
                        variable=self.var_use_cache).grid(row=4, column=2, columnspan=2,
                                                          sticky="w", padx=6, pady=3)
        ttk.Label(lf_hp, text="MCMC采样数:").grid(row=5, column=0, sticky="e", padx=6, pady=3)
        ttk.Entry(lf_hp, textvariable=self.var_mcmc_samples,
                  width=14).grid(row=5, column=1, padx=3, pady=3, sticky="w")
        ttk.Label(lf_hp, text="LCI λ_lat:").grid(row=5, column=2, sticky="e", padx=6, pady=3)
        self.var_lambda_lat = tk.StringVar(value="1.0")
        ttk.Entry(lf_hp, textvariable=self.var_lambda_lat,
                  width=10).grid(row=5, column=3, padx=3, pady=3, sticky="w")

        # --- 执行按钮 ---
        bf = ttk.Frame(p); bf.pack(fill="x", padx=6, pady=8)
        ttk.Button(bf, text="▶ 开始处理", style="Accent.TButton",
                   command=self._on_run_basic, width=22).pack(side="left", padx=4)
        ttk.Button(bf, text="打开输出目录", width=14,
                   command=self._open_out).pack(side="left", padx=4)
        ttk.Button(bf, text="清空日志", width=10,
                   command=lambda: (self.log_text.configure(state="normal"),
                                    self.log_text.delete("1.0", "end"),
                                    self.log_text.configure(state="disabled"))
                   ).pack(side="right", padx=4)

    def _browse_project(self):
        p = filedialog.askdirectory(title="选择项目根目录 (含1/2/4/5线 + 坐标文件.xls)",
                                   initialdir=self.var_project.get())
        if p: self.var_project.set(p)
    def _browse_out(self):
        p = filedialog.askdirectory(title="选择输出根目录", initialdir=self.var_out.get())
        if p: self.var_out.set(p)
    def _open_out(self):
        try:
            os.startfile(self.var_out.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _collect_cfg(self):
        pp_cfg = dict(RHOA_SCALE_CAL=float(self.var_hp_rhoa.get()),
                      TEM_N_GATES=int(self.var_hp_ngates.get()))
        inv_cfg = dict(N_LAYERS=int(self.var_hp_nlayers.get()),
                       LAMBDA_SMOOTH=float(self.var_hp_lam_s.get()),
                       LAMBDA_LAT=float(self.var_lambda_lat.get()))
        return pp_cfg, inv_cfg

    def _on_run_basic(self):
        project, out = self.var_project.get(), self.var_out.get()
        if not Path(project).is_dir():
            messagebox.showerror("错误", f"项目根目录不存在: {project}"); return
        self._start_worker(self._worker_basic, project, out,
                           self.var_s12.get(), self.var_s_pp.get(),
                           self.var_s3.get(), self.var_s_ip.get(),
                           self.var_s4.get(), self.var_s_sig.get(),
                           *self._collect_cfg(),
                           self.var_occam_exe.get(),
                           float(self.var_hp_lam_i.get()),
                           self.var_inv_algo.get(),
                           int(self.var_n_jobs.get()),
                           self.var_use_cache.get(),
                           int(self.var_mcmc_samples.get()))

    def _worker_basic(self, project, out, do_s12, do_pp, do_inv, do_ip,
                      do_occam, do_sig, pp_cfg, inv_cfg, occam_exe, lam_init,
                      inv_algo="trf", n_jobs=1, use_cache=False,
                      mcmc_samples=2000):
        log = self.log
        t0 = time.time()
        log("=" * 64); log(" GUI 全流水线 启动 100% OFFLINE")
        log(f"  Project : {project}"); log(f"  Output  : {out}")
        Path(out).mkdir(parents=True, exist_ok=True)
        # S12 基础预处理
        if do_s12:
            log(">>> Step1-2 预处理")
            rep = run_full_preprocess(project, out, pp_cfg, log_fn=log)
            log(f"<<< Step1-2 完成  lines={len(rep.get('lines', {}))}")
        # S12+ 增强预处理 (衰减曲线 + PNG)
        if do_pp:
            log(">>> Step1-2+ 增强预处理 (衰减曲线提取)")
            rpp = run_full_preprocess_plus(project, out, pp_cfg, log_fn=log)
            lines = list(rpp['decay_lines'].keys())
            if do_sig:
                total_png = 0
                for lt in lines:
                    npz = Path(out) / lt / f"{lt}_decay_curves.npz"
                    if npz.exists():
                        total_png += batch_save_decay_pngs(npz, Path(out) / lt / "signal_pngs",
                                                           line_tag=lt, log_fn=log)
                # 通道叠加 Hx 对比
                for lt in lines:
                    try:
                        npz = np.load(Path(out) / lt / f"{lt}_decay_curves.npz", allow_pickle=True)
                        data = npz['data'].item()
                        plot_channel_overlay(data, channel='Hx', attr='bandpass',
                                             title=f'{lt} · Hx bandpass 关断窗口叠加 (WLF verify-kru)',
                                             out_png=Path(out)/lt/f'{lt}_Hx_verify_overlay.png')
                    except Exception as exc:
                        log(f"  [WARN] overlay {lt}: {exc}", level="W")
                log(f"  信号 PNG: {total_png} 张")
            log("<<< Step1-2+ 完成")
        # Step3 电阻率反演 (多算法选择, 含高级算法 occam_adaptive/lm_adaptive/cgls/mcmc)
        if do_inv:
            algo = (inv_algo or "trf").strip()
            if algo.startswith("all"):
                log(f">>> Step3 1D 电阻率反演 [trf+lm+occam 三算法对比]  n_jobs={n_jobs} cache={use_cache}")
                run_all_algos_compare(out, user_cfg=inv_cfg,
                                      occam_cfg={'LAMBDA_INIT': lam_init},
                                      n_jobs=n_jobs, use_cache=use_cache, log_fn=log)
            elif algo == 'lci':
                lam_lat = inv_cfg.get('LAMBDA_LAT', 1.0)
                log(f">>> Step3 LCI 横向约束联合反演 [λ_lat={lam_lat}]  n_jobs={n_jobs}")
                run_all_lci(out, user_cfg=inv_cfg, n_jobs=n_jobs, log_fn=log)
            elif algo in ALGO_ADVANCED:
                log(f">>> Step3 1D 高级电阻率反演 [algo={algo}]  "
                    f"n_jobs={n_jobs} cache={use_cache} mcmc_samples={mcmc_samples}")
                run_all_advanced(out, algo=algo, user_cfg=inv_cfg,
                                 n_jobs=n_jobs, use_cache=use_cache,
                                 mcmc_samples=mcmc_samples, log_fn=log)
            else:
                log(f">>> Step3 1D 电阻率反演 [algo={algo}]  n_jobs={n_jobs} cache={use_cache}")
                run_all_algo(out, algo=algo, user_cfg=inv_cfg,
                             occam_cfg={'LAMBDA_INIT': lam_init},
                             n_jobs=n_jobs, use_cache=use_cache, log_fn=log)
            log("<<< Step3 完成")
        # Step3+ IP 反演
        if do_ip:
            log(">>> Step3+ IP 极化率反演 (ρ0,m,τ,c)")
            ip = run_all_ip_inversions(out, inv_cfg, ip_cfg=None, log_fn=log)
            log(f"<<< Step3+ IP完成 stations_ok={ip.get('n_stations_ok', '?')}")
        # Step4 Occam 桥接
        if do_occam:
            log(">>> Step4 Occam1D 商业桥接")
            materialize_all(out, occam_exe=occam_exe, lambda_init=lam_init, log_fn=log)
            log("<<< Step4 完成")
        dt = time.time() - t0
        log(f"✓ 全部完成, 耗时 {dt:.1f}s"); log("=" * 64)
        try:
            messagebox.showinfo("完成", f"处理完毕!\n耗时 {dt:.1f} s\n输出目录: {out}")
        except Exception:
            pass

    # =================================================================
    # TAB 2: Layer Model Designer (Treeview 编辑 N 层 × 5参数)
    # =================================================================
    def _build_tab_model(self):
        p = self.tab_model
        top = ttk.Frame(p); top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="发射面积 (m²):").pack(side="left", padx=2)
        self.var_mdl_area = tk.StringVar(value="4452697")
        ttk.Entry(top, textvariable=self.var_mdl_area, width=12).pack(side="left", padx=2)
        ttk.Label(top, text="电流 (A):").pack(side="left", padx=2)
        self.var_mdl_cur = tk.StringVar(value="30")
        ttk.Entry(top, textvariable=self.var_mdl_cur, width=8).pack(side="left", padx=2)
        ttk.Label(top, text="gates (geom us):").pack(side="left", padx=2)
        self.var_mdl_gates = tk.StringVar(value="2,20000,25")
        ttk.Entry(top, textvariable=self.var_mdl_gates, width=14).pack(side="left", padx=2)
        ttk.Button(top, text="+添加层", width=10, command=self._mdl_add).pack(side="left", padx=8)
        ttk.Button(top, text="-删除行", width=10, command=self._mdl_del).pack(side="left", padx=2)
        ttk.Button(top, text="载入CSV", width=10, command=self._mdl_load).pack(side="left", padx=8)
        ttk.Button(top, text="保存CSV", width=10, command=self._mdl_save).pack(side="left", padx=2)
        ttk.Button(top, text="▶ 正演预览", style="Accent.TButton", width=14,
                   command=self._on_mdl_forward_preview).pack(side="right", padx=4)
        ttk.Button(top, text="25层模板", width=10, command=self._mdl_preset25).pack(side="right", padx=4)

        cols = ("h_m", "rho_ohm_m", "m_chargeability", "tau_s", "c_exp")
        self.mdl_tree = ttk.Treeview(p, columns=cols, show="headings", height=18)
        hdrs = [("h_m", "层厚 h (m)", 90),
                ("rho_ohm_m", "电阻率 ρ (Ω·m)", 140),
                ("m_chargeability", "极化率 m (0~0.8)", 140),
                ("tau_s", "时间常数 τ (s)", 120),
                ("c_exp", "频率指数 c (0.1~0.95)", 140)]
        for k, t, w in hdrs:
            self.mdl_tree.heading(k, text=t); self.mdl_tree.column(k, width=w, anchor="e")
        self.mdl_tree.pack(fill="both", expand=True, padx=6, pady=4)
        self.mdl_tree.bind("<Double-1>", self._mdl_edit_cell)
        # 预设 5 层
        self._mdl_preset5()

    def _mdl_rows(self, new_rows):
        for i in self.mdl_tree.get_children(): self.mdl_tree.delete(i)
        for r in new_rows:
            self.mdl_tree.insert("", "end", values=tuple(r))

    def _mdl_preset5(self):
        rows = [(20, 100, 0.05, 1e-3, 0.5),
                (40, 500, 0.03, 1e-3, 0.5),
                (80, 2000, 0.08, 2e-3, 0.5),
                (160, 50, 0.10, 5e-4, 0.5),
                (9999, 5000, 0.02, 1e-2, 0.5)]
        self._mdl_rows(rows)

    def _mdl_preset25(self):
        h = np.geomspace(5, 400, 24).tolist() + [9999]
        rho = np.logspace(1.5, 5, 25).tolist()
        rows = [(h[i], rho[i], 0.05, 1e-3, 0.5) for i in range(25)]
        self._mdl_rows(rows)

    def _mdl_add(self):
        sel = self.mdl_tree.selection()
        idx = self.mdl_tree.index(sel[0]) if sel else len(self.mdl_tree.get_children())
        self.mdl_tree.insert("", idx, values=(20, 1000, 0.05, 1e-3, 0.5))

    def _mdl_del(self):
        for i in self.mdl_tree.selection():
            self.mdl_tree.delete(i)

    def _mdl_edit_cell(self, evt=None):
        sel = self.mdl_tree.selection()
        if not sel: return
        item = sel[0]
        col_id = self.mdl_tree.identify_column(evt.x)
        col_index = int(col_id.replace("#", "")) - 1
        cols = ("h_m", "rho_ohm_m", "m_chargeability", "tau_s", "c_exp")
        val = list(self.mdl_tree.item(item, "values"))
        old = str(val[col_index])
        new = tk.simpledialog.askstring("编辑", f"{cols[col_index]} = ?", initialvalue=old)
        if new is None: return
        try:
            val[col_index] = float(new)
        except ValueError:
            messagebox.showerror("错误", "请输入数字"); return
        self.mdl_tree.item(item, values=val)

    def _mdl_collect(self):
        rows = []
        for iid in self.mdl_tree.get_children():
            rows.append([float(x) for x in self.mdl_tree.item(iid, "values")])
        arr = np.array(rows, dtype=np.float64)
        h = arr[:, 0]; rho = arr[:, 1]; m = arr[:, 2]; tau = arr[:, 3]; c = arr[:, 4]
        return LayeredModel(h, rho, m, tau, c)

    def _mdl_save(self):
        p = filedialog.asksaveasfilename(title="保存层状模型CSV",
                                         defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p: return
        import csv
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["h_m", "rho_ohm_m", "m_chargeability", "tau_s", "c_exp"])
            for iid in self.mdl_tree.get_children():
                w.writerow(self.mdl_tree.item(iid, "values"))
        self.log(f"模型已保存 → {p}")

    def _mdl_load(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p: return
        import csv
        rows = []
        with open(p, "r", encoding="utf-8-sig") as fh:
            rd = csv.reader(fh); next(rd, None)
            for r in rd:
                if len(r) >= 5:
                    rows.append([float(x) for x in r[:5]])
        self._mdl_rows(rows); self.log(f"载入模型 {p} ({len(rows)}层)")

    def _on_mdl_forward_preview(self):
        try:
            model = self._mdl_collect()
            A = float(self.var_mdl_area.get()); I = float(self.var_mdl_cur.get())
            parts = [float(x) for x in self.var_mdl_gates.get().split(",")]
            gates_us = np.geomspace(parts[0], parts[1], num=max(3, int(parts[2]) if len(parts) > 2 else 25))
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc)); return
        self._start_worker(self._worker_mdl_forward, model, A, I, gates_us)

    def _worker_mdl_forward(self, model, A, I, gates_us):
        log = self.log
        log(f"[MODEL] 正演 N={model.n_layers}  A={A:.2e}m²  I={I:.1f}A  gates={len(gates_us)}")
        t_s, vt = tem_forward_1d(model, A, I, gates_us * 1e-6, include_ip=model.has_ip)
        rhoa = apparent_resistivity_dbzdt(vt, t_s, A, I)
        log(f"  Vt ∈ [{vt.min():.2e}, {vt.max():.2e}]  ρa median={np.median(rhoa):.2e} Ω·m")
        # 画子图 (2×2): Vt(t) / ρa(t) / 模型ρ vs z / 模型m vs z
        try:
            self.root.after(0, lambda: self._show_mdl_preview_win(model, t_s, vt, rhoa))
        except Exception as exc:
            log(f"[MODEL] 绘图失败: {exc}", level="E")

    def _show_mdl_preview_win(self, model, t_s, vt, rhoa):
        win = tk.Toplevel(self.root); win.title("1D TEM 正演预览"); win.geometry("1020x720")
        fig = Figure(figsize=(10.1, 6.9), dpi=100)
        ax = fig.subplots(2, 2)
        # (0,0) Vt(t) loglog
        ax[0, 0].loglog(t_s * 1e6, np.abs(vt), "o-", color="#c00000",
                        linewidth=1.1, markersize=4.5, markerfacecolor="w")
        ax[0, 0].set_xlabel("Time (μs)"); ax[0, 0].set_ylabel("|dBz/dt| (V/m²)")
        ax[0, 0].set_title("瞬态电压 V(t) — gates")
        ax[0, 0].grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        # (0,1) ρa(t) semilogx
        ax[0, 1].semilogx(t_s * 1e6, rhoa, "s-", color="#1f4e79", markersize=5, markerfacecolor="w")
        ax[0, 1].set_xlabel("Time (μs)"); ax[0, 1].set_ylabel("ρa apparent (Ω·m)")
        ax[0, 1].set_title(f"Apparent Resistivity (median {np.median(rhoa):.2e})")
        ax[0, 1].grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        # (1,0) 模型 ρ vs z (深度)
        h = np.asarray(model.thickness_m, dtype=float); h[-1] = h[-2] * 3.0
        z = np.concatenate([[0], np.cumsum(h)])
        rho = model.rho_ohm_m
        ax[1, 0].step(rho, z[:-1], where="post", color="#2e75b5", linewidth=2)
        if model.has_ip and model.chargeability is not None:
            axt = ax[1, 0].twiny(); axt.step(model.chargeability, z[:-1], where="post",
                                             color="#c55a11", linewidth=1.6, linestyle="--")
            axt.set_xlabel("m chargeability", color="#c55a11")
            axt.tick_params(axis="x", labelcolor="#c55a11")
        ax[1, 0].invert_yaxis(); ax[1, 0].set_xscale("log")
        ax[1, 0].set_xlabel("ρ (Ω·m, log)"); ax[1, 0].set_ylabel("Depth (m)")
        ax[1, 0].set_title("Layer Model (ρ/m vs depth)")
        ax[1, 0].grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        # (1,1) IP τ / c
        if model.has_ip:
            ax[1, 1].step(model.tau_s, z[:-1], where="post", color="#7030a0", linewidth=1.8, label="τ (s)")
            axt2 = ax[1, 1].twiny(); axt2.step(model.c_exp, z[:-1], where="post",
                                                color="#375623", linewidth=1.8, label="c exponent")
            ax[1, 1].set_xscale("log"); ax[1, 1].invert_yaxis()
            ax[1, 1].set_xlabel("τ (s, log)", color="#7030a0")
            axt2.set_xlabel("c (0.1~0.95)", color="#375623")
            ax[1, 1].tick_params(axis="x", labelcolor="#7030a0")
            axt2.tick_params(axis="x", labelcolor="#375623")
            ax[1, 1].set_ylabel("Depth (m)"); ax[1, 1].set_title("Cole-Cole τ/c vs depth")
            ax[1, 1].grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        else:
            ax[1, 1].text(0.5, 0.5, "模型无IP参数\n(在Model里填写m/τ/c后重新正演)",
                          ha="center", va="center", transform=ax[1, 1].transAxes,
                          color="gray", fontsize=12)
            ax[1, 1].set_axis_off()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, win).update()

    # =================================================================
    # TAB 3: Forward Compare (模型正演 vs 实测叠加)
    # =================================================================
    def _build_tab_forward(self):
        p = self.tab_fwd
        top = ttk.Frame(p); top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="模型CSV:").pack(side="left", padx=2)
        self.var_fwd_csv = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_fwd_csv, width=48).pack(side="left", padx=2)
        ttk.Button(top, text="…", width=3,
                   command=lambda: self.var_fwd_csv.set(
                       filedialog.askopenfilename(filetypes=[("CSV模型", "*.csv")]))).pack(side="left")
        ttk.Label(top, text="Occam.dat:").pack(side="left", padx=2)
        self.var_fwd_occam = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_fwd_occam, width=48).pack(side="left", padx=2)
        ttk.Button(top, text="…", width=3,
                   command=lambda: self.var_fwd_occam.set(
                       filedialog.askopenfilename(filetypes=[("Occam dat", "*.dat")]))).pack(side="left")
        ttk.Label(top, text="Station:").pack(side="left", padx=4)
        self.var_fwd_station = ttk.Combobox(top, width=10, values=())
        self.var_fwd_station.pack(side="left", padx=2)
        ttk.Button(top, text="加载", command=self._fwd_load_stations, width=8).pack(side="left", padx=4)
        ttk.Button(top, text="▶ 正演叠加对比", style="Accent.TButton", width=18,
                   command=self._on_fwd_compare).pack(side="right", padx=4)
        # 画布
        self._fwd_fig = Figure(figsize=(12, 6.6), dpi=100)
        self._fwd_canvas = FigureCanvasTkAgg(self._fwd_fig, master=p)
        self._fwd_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        NavigationToolbar2Tk(self._fwd_canvas, p).update()

    def _fwd_load_stations(self):
        from tfm_engine.occam_bridge import parse_occam_dat
        try:
            df = parse_occam_dat(self.var_fwd_occam.get())
            sts = list(dict.fromkeys(df["station"].tolist()))
            self.var_fwd_station["values"] = sts
            if sts: self.var_fwd_station.set(sts[0])
            self.log(f"Forward: 载入 {len(sts)} stations")
        except Exception as exc:
            messagebox.showerror("错误", str(exc))

    def _on_fwd_compare(self):
        csv_p, occam_p, st = (self.var_fwd_csv.get(), self.var_fwd_occam.get(),
                              self.var_fwd_station.get())
        if not (csv_p and occam_p and st):
            messagebox.showwarning("警告", "请选CSV模型+Occam.dat+Station"); return
        self._start_worker(self._worker_fwd, csv_p, occam_p, st)

    def _worker_fwd(self, csv_p, occam_p, st):
        from tfm_engine.occam_bridge import parse_occam_dat
        import csv
        log = self.log
        log(f"[FORWARD] 对比 station={st}")
        # 模型
        rows = []
        with open(csv_p, "r", encoding="utf-8-sig") as fh:
            rd = csv.reader(fh); next(rd, None)
            for r in rd:
                if len(r) >= 5: rows.append([float(x) for x in r[:5]])
        arr = np.array(rows, dtype=float)
        model = LayeredModel(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
        # 实测
        df = parse_occam_dat(occam_p)
        sub = df[df["station"] == st].sort_values("t_sec")
        if len(sub) < 3:
            log(f"  [ERR] station {st} 实测数据不够 ({len(sub)} 行)", level="E"); return
        t_obs = sub["t_sec"].to_numpy(); rhoa_obs = sub["rhoa_ohm_m"].to_numpy()
        # Tx 元数据
        A_guess = 4.45e6 if any(x in csv_p for x in ["1", "5"]) else 1.72e6
        I_guess = 37 if "5" in Path(occam_p).parent.name else 30
        try:
            t_pred, vt_pred = tem_forward_1d(model, A_guess, I_guess, t_obs, include_ip=model.has_ip)
            rhoa_pred = apparent_resistivity_dbzdt(vt_pred, t_pred, A_guess, I_guess)
            # 绘图
            self.root.after(0, lambda: self._plot_fwd_compare(st, t_obs, rhoa_obs, rhoa_pred, vt_pred))
        except Exception as exc:
            log(f"[FORWARD] FAIL: {exc}", level="E")

    def _plot_fwd_compare(self, st, t_obs, rhoa_obs, rhoa_pred, vt_pred):
        self._fwd_fig.clear()
        ax1, ax2 = self._fwd_fig.subplots(1, 2)
        ax1.semilogx(t_obs * 1e6, rhoa_obs, "o", color="#c00000", label="Observed (Occam.dat)",
                     markersize=6, markerfacecolor="white", markeredgewidth=1.4)
        ax1.semilogx(t_obs * 1e6, rhoa_pred, "-", color="#1f4e79", linewidth=1.8,
                     label=f"Predicted  (median ratio {np.median(rhoa_pred/rhoa_obs):.2f})")
        ax1.set_xlabel("t (μs)"); ax1.set_ylabel("ρa (Ω·m, log)")
        ax1.set_title(f"ρa 对比 — {st}"); ax1.legend(); ax1.grid(linestyle=":", linewidth=0.4)
        # Vt(t)
        ax2.loglog(t_obs * 1e6, np.maximum(np.abs(vt_pred), 1e-30),
                   "o-", color="#2e75b5", linewidth=1.2, markersize=5, markerfacecolor="white")
        ax2.set_xlabel("t (μs)"); ax2.set_ylabel("|dBz/dt| predicted (V/m²)")
        ax2.set_title(f"V(t) 正演 — {st}"); ax2.grid(which="both", linestyle=":", linewidth=0.4)
        self._fwd_fig.suptitle("Forward vs Observed  (WLF Mod1 正演对比)", fontsize=13, y=0.98)
        self._fwd_fig.tight_layout(); self._fwd_canvas.draw_idle()

    # =================================================================
    # TAB 4: Signal Viewer (4×4 FigureCanvas)
    # =================================================================
    def _build_tab_signal(self):
        p = self.tab_sig
        top = ttk.Frame(p); top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="衰减曲线 NPZ:").pack(side="left", padx=2)
        self.var_sig_npz = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_sig_npz, width=60).pack(side="left", padx=2)
        ttk.Button(top, text="…", width=3,
                   command=self._sig_browse_npz).pack(side="left", padx=2)
        ttk.Label(top, text="Station:").pack(side="left", padx=6)
        self.var_sig_st = ttk.Combobox(top, width=10, values=())
        self.var_sig_st.pack(side="left", padx=2)
        self.var_sig_st.bind("<<ComboboxSelected>>", lambda e: self._sig_plot())
        ttk.Button(top, text="加载全部并绘图", width=18,
                   command=self._sig_load).pack(side="left", padx=6)
        ttk.Button(top, text="批量导出PNG", width=14,
                   command=self._sig_batch_png).pack(side="right", padx=4)
        ttk.Button(top, text="Hx叠加对比(verify)", width=22,
                   command=self._sig_overlay).pack(side="right", padx=4)
        self._sig_fig = Figure(figsize=(17, 10), dpi=96)
        self._sig_canvas = FigureCanvasTkAgg(self._sig_fig, master=p)
        self._sig_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        NavigationToolbar2Tk(self._sig_canvas, p).update()

    def _sig_browse_npz(self):
        p = filedialog.askopenfilename(filetypes=[("衰减曲线NPZ", "*.npz")])
        if p: self.var_sig_npz.set(p)

    def _sig_load(self):
        p = self.var_sig_npz.get()
        if not Path(p).exists():
            messagebox.showerror("错误", f"NPZ不存在: {p}"); return
        self._sig_data = np.load(p, allow_pickle=True)["data"].item()
        sts = list(self._sig_data.keys())
        self.var_sig_st["values"] = sts
        if sts:
            self.var_sig_st.set(sts[0]); self._sig_plot()
        self.log(f"Signal: 载入 {len(sts)} 个 stations → {p}")

    def _sig_plot(self):
        st = self.var_sig_st.get()
        if not st or not hasattr(self, "_sig_data"): return
        self._sig_fig.clear()
        _tmp = make_decay_4x4figure(self._sig_data[st], station_code=st)
        # 拷贝到 self._sig_fig
        for new_ax in _tmp.axes:
            pos = new_ax.get_position(original=False).bounds
            a = self._sig_fig.add_axes(pos); a.set_xticks(new_ax.get_xticks())
            a.set_yticks(new_ax.get_yticks())
            for line in new_ax.get_lines():
                a.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth(), color=line.get_color(),
                       marker=line.get_marker(), markersize=line.get_markersize(),
                       markerfacecolor=line.get_markerfacecolor(),
                       markeredgewidth=line.get_markeredgewidth(), alpha=line.get_alpha(),
                       label=line.get_label())
            a.set_xlim(new_ax.get_xlim()); a.set_ylim(new_ax.get_ylim())
            a.set_xlabel(new_ax.get_xlabel() or ""); a.set_ylabel(new_ax.get_ylabel() or "")
            a.set_title(new_ax.get_title() or "", fontdict=dict(new_ax.title.get_fontproperties()))
            if new_ax.get_yscale() == "log": a.set_yscale("log")
            if new_ax.get_xscale() == "log": a.set_xscale("log")
            if new_ax.get_legend_handles_labels()[0]:
                a.legend(fontsize=6, frameon=False)
            a.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        self._sig_fig.suptitle(_tmp._suptitle.get_text() if False else
                               getattr(_tmp, '_suptitle', None) and _tmp._suptitle.get_text() or "",
                               fontsize=13, y=0.995, weight="bold")
        import matplotlib; matplotlib.figure.Figure
        # 更直接：把 _tmp.suptitle 搬过来
        try:
            self._sig_fig.texts = _tmp.texts.copy()
        except Exception:
            pass
        self._sig_canvas.draw_idle()

    def _sig_batch_png(self):
        if not hasattr(self, "_sig_data"):
            messagebox.showwarning("提示", "请先加载NPZ"); return
        outd = filedialog.askdirectory(title="导出PNG到目录")
        if not outd: return
        cnt = batch_save_decay_pngs(self._sig_data, outd, line_tag=Path(self.var_sig_npz.get()).stem,
                                   log_fn=self.log)
        messagebox.showinfo("完成", f"导出 {cnt} 张 PNG → {outd}")

    def _sig_overlay(self):
        if not hasattr(self, "_sig_data"): return
        outp = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png")],
                                             initialfile="Hx_bandpass_verify_overlay.png")
        if not outp: return
        plot_channel_overlay(self._sig_data, channel="Hx", attr="bandpass", out_png=outp)
        self.log(f"Hx verify overlay 已导出 → {outp}")
        try: os.startfile(Path(outp).parent)
        except Exception: pass

    # =================================================================
    # TAB 5: IP Inversion (全线 + 单站)
    # =================================================================
    def _build_tab_ip(self):
        p = self.tab_ip
        lf = ttk.LabelFrame(p, text="IP 反演参数 (Pelton Cole-Cole 4参数: ρ0/m/τ/c)")
        lf.pack(fill="x", padx=6, pady=6)
        self.var_ip_share = tk.BooleanVar(value=False)
        self.var_ip_vtonly = tk.BooleanVar(value=True)  # vt_only_mode (WLF pcvogoss_pc) 默认开启, 已验证 RMSE_Vt 降66%
        self.var_ip_lam = tk.StringVar(value="0.3")
        self.var_ip_nfev = tk.StringVar(value="100")
        self.var_ip_m0 = tk.StringVar(value="0.05")
        self.var_ip_tau0 = tk.StringVar(value="0.001")
        self.var_ip_c0 = tk.StringVar(value="0.5")
        row_wid = [
            ("共享 τ, c (仅反 ρ0/m + 2全局)", self.var_ip_share, 0, 0),
            ("LAMBDA_SMOOTH 正则", self.var_ip_lam, 0, 2),
            ("MAX_NFEV 迭代次数", self.var_ip_nfev, 1, 0),
            ("初始极化率 m₀", self.var_ip_m0, 1, 2),
            ("初始时间常数 τ₀ (s)", self.var_ip_tau0, 2, 0),
            ("初始频率指数 c₀", self.var_ip_c0, 2, 2),
        ]
        for t, v, r, c in row_wid:
            if isinstance(v, tk.BooleanVar):
                ttk.Checkbutton(lf, text=t, variable=v).grid(row=r, column=c,
                                                              columnspan=3 if c == 0 else 1,
                                                              sticky="w", padx=8, pady=2)
            else:
                ttk.Label(lf, text=t).grid(row=r, column=c, sticky="e", padx=6, pady=2)
                ttk.Entry(lf, textvariable=v, width=12).grid(row=r, column=c+1, sticky="w", pady=2)
        # ★ 新增: vt_only_mode 勾选框 (WLF pcvogoss_pc 工业做法) 单独一行醒目
        ttk.Checkbutton(lf,
            text="★ Vt-only 模式 (WLF pcvogoss_pc 工业做法, 直接拟合 decay NPZ 衰减曲线 — 推荐, 已验证 RMSE_Vt 降66%)",
            variable=self.var_ip_vtonly).grid(row=3, column=0, columnspan=5, sticky="w", padx=8, pady=4)
        bf = ttk.Frame(p); bf.pack(fill="x", padx=6, pady=8)
        ttk.Button(bf, text="▶ 全线 IP 反演", style="Accent.TButton", width=22,
                   command=self._on_ip_all).pack(side="left", padx=4)
        ttk.Button(bf, text="▶ 单站 IP 反演", width=20,
                   command=self._on_ip_one).pack(side="left", padx=4)
        # IP 预览画布
        self._ip_fig = Figure(figsize=(12.5, 6.2), dpi=96)
        self._ip_canvas = FigureCanvasTkAgg(self._ip_fig, master=p)
        self._ip_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def _ip_cfg(self):
        return dict(SHARE_TAU_C=bool(self.var_ip_share.get()),
                    LAMBDA_SMOOTH=float(self.var_ip_lam.get()),
                    MAX_NFEV=int(self.var_ip_nfev.get()),
                    INIT_m=float(self.var_ip_m0.get()),
                    INIT_tau=float(self.var_ip_tau0.get()),
                    INIT_c=float(self.var_ip_c0.get()))

    def _on_ip_all(self):
        out = self.var_out.get() if hasattr(self, "var_out") else filedialog.askdirectory(title="输出根")
        if not Path(out).is_dir(): return
        cfg = self._ip_cfg()
        self._start_worker(self._worker_ip_all, out, cfg)

    def _worker_ip_all(self, out, ip_cfg):
        res = run_all_ip_inversions(out, user_cfg=None, ip_cfg=ip_cfg, log_fn=self.log,
                                    prefer_vt_only=bool(self.var_ip_vtonly.get()))
        self.log(f"IP 反演完成: 线路={res['n_lines']}  站点={res['n_stations_ok']}")
        try:
            messagebox.showinfo("完成", f"IP反演完毕\n{json.dumps({k: v for k, v in res.items() if k != 'summary_rows'}, ensure_ascii=False, indent=2)}")
        except Exception:
            pass

    def _on_ip_one(self):
        occam_p = filedialog.askopenfilename(filetypes=[("Occam.dat", "*.dat")])
        if not occam_p: return
        from tfm_engine.occam_bridge import parse_occam_dat
        try:
            df = parse_occam_dat(occam_p)
        except Exception as exc:
            messagebox.showerror("错误", str(exc)); return
        sts = list(dict.fromkeys(df["station"].tolist()))
        st = tk.simpledialog.askstring("单站选择",
                                        f"可选: {', '.join(sts[:8])}{'…' if len(sts) > 8 else ''}",
                                        initialvalue=sts[0] if sts else "")
        if not st: return
        self._start_worker(self._worker_ip_one, occam_p, st, self._ip_cfg())

    def _worker_ip_one(self, occam_p, st, ip_cfg):
        from tfm_engine.occam_bridge import parse_occam_dat
        df = parse_occam_dat(occam_p)
        sub = df[df["station"] == st].sort_values("t_sec")
        A = 4.45e6 if any(x in occam_p for x in ["1线", "5线"]) else 1.72e6
        I = 37 if "5线" in occam_p else 30
        rhoa = sub["rhoa_ohm_m"].to_numpy(); t_s = sub["t_sec"].to_numpy()
        vt_est = 1e-15 / np.maximum(t_s, 1e-12) ** 2.25 * np.geomspace(1, 0.01, len(t_s))
        r = invert_one_station_ip(rhoa, vt_est, t_s, A, I, user_cfg=ip_cfg, log_fn=self.log)
        if r["status"] != "OK":
            messagebox.showerror("失败", r.get("error", "unknown")); return
        # 绘图
        self.root.after(0, lambda: self._plot_ip_one(st, r, t_s, rhoa, vt_est))

    def _plot_ip_one(self, st, r, t_s, rhoa_obs, vt_obs):
        self._ip_fig.clear()
        ax = self._ip_fig.subplots(1, 3)
        # ρa 拟合
        ax[0].loglog(t_s * 1e6, rhoa_obs, "o", color="#c00000", label="obs",
                     markersize=6, markerfacecolor="white", markeredgewidth=1.4)
        ax[0].loglog(t_s * 1e6, r["rhoa_fit"], "-", color="#1f4e79", linewidth=1.8,
                      label=f"fit  RMSE_logρ={r['rmse_log10_rhoa']:.3f}")
        ax[0].set_xlabel("t (μs)"); ax[0].set_ylabel("ρa (Ω·m)"); ax[0].legend()
        ax[0].grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        ax[0].set_title(f"ρa(t) fit  Station {st}")
        # Vt 拟合
        ax[1].loglog(t_s * 1e6, np.maximum(np.abs(vt_obs), 1e-30), "o", color="#c00000", label="obs(est)",
                     markersize=6, markerfacecolor="white", markeredgewidth=1.4)
        ax[1].loglog(t_s * 1e6, np.maximum(np.abs(r["vt_fit"]), 1e-30), "-",
                     color="#2e75b5", linewidth=1.8,
                     label=f"fit  RMSE_logVt={r['rmse_log10_vt']:.3f}")
        ax[1].set_xlabel("t (μs)"); ax[1].set_ylabel("|V(t)| (V/m²)"); ax[1].legend()
        ax[1].grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        ax[1].set_title(f"V(t) fit  (nfev={r['nfev']})")
        # 模型: ρ / m / τ / c vs depth
        mdl: LayeredModel = r["model"]
        h = np.asarray(mdl.thickness_m); h[-1] = h[-2] * 3
        z = np.concatenate([[0], np.cumsum(h)])
        ax2 = ax[2]
        ax2.step(mdl.rho_ohm_m, z[:-1], where="post", color="#1f4e79", linewidth=1.9, label="ρ (Ω·m)")
        ax2.set_xscale("log"); ax2.invert_yaxis()
        ax2.set_xlabel("ρ (Ω·m, log)", color="#1f4e79")
        ax2.tick_params(axis="x", labelcolor="#1f4e79")
        axt = ax2.twiny(); axt.step(mdl.chargeability, z[:-1], where="post",
                                    color="#c55a11", linewidth=1.7, label="m chargeability")
        axt.set_xlabel("m (Pelton)", color="#c55a11")
        axt.tick_params(axis="x", labelcolor="#c55a11")
        ax2.set_ylabel("Depth (m)")
        ax2.set_title(f"Inverted Model  n_layers={mdl.n_layers}")
        ax2.grid(which="both", linestyle=":", linewidth=0.4, alpha=0.6)
        self._ip_fig.suptitle(f"IP 1D 反演结果 · Station {st}", fontsize=13, y=0.98)
        self._ip_fig.tight_layout(); self._ip_canvas.draw_idle()

    # =================================================================
    # TAB 6: WLF 2D/3D Bridge (TRANS24 / SWP24D / 3D tomo)
    # =================================================================
    def _build_tab_wlf(self):
        p = self.tab_wlf
        # 6.1 TRANS24
        lf = ttk.LabelFrame(p, text="① TRANS24.exe — 格式互转 (.dat ↔ .cur ↔ .mod ↔ .grid3D)")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Label(lf, text="Occam.dat:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.var_t_in = tk.StringVar()
        ttk.Entry(lf, textvariable=self.var_t_in, width=60).grid(row=0, column=1, padx=4, pady=3)
        ttk.Button(lf, text="…", width=3,
                   command=lambda: self.var_t_in.set(
                       filedialog.askopenfilename(filetypes=[("Occam dat", "*.dat")]))
                   ).grid(row=0, column=2, padx=3)
        ttk.Label(lf, text="输出格式:").grid(row=0, column=3, sticky="e", padx=4)
        self.var_t_fmt = ttk.Combobox(lf, values=["wlf_cur", "wlf_mod", "occam"], width=10, state="readonly")
        self.var_t_fmt.set("wlf_cur"); self.var_t_fmt.grid(row=0, column=4, padx=3)
        ttk.Button(lf, text="▶ 运行 TRANS24", width=20,
                   command=self._on_trans24).grid(row=0, column=5, padx=8, pady=3)

        # 6.2 SWP24 2D
        lf2 = ttk.LabelFrame(p, text="② SWP24D.EXE — WLF 2D TEM Occam 反演 (按carmod2_kru+NewLimModR_reg)")
        lf2.pack(fill="x", padx=6, pady=4)
        ttk.Label(lf2, text="Occam.dat (线):").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.var_2d_in = tk.StringVar()
        ttk.Entry(lf2, textvariable=self.var_2d_in, width=60).grid(row=0, column=1, padx=4, pady=3)
        ttk.Button(lf2, text="…", width=3,
                   command=lambda: self.var_2d_in.set(
                       filedialog.askopenfilename(filetypes=[("Occam dat", "*.dat")]))
                   ).grid(row=0, column=2, padx=3)
        self.var_2d_ip = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf2, text="包含IP反演 (ρ/m/τ/c 同时反演)",
                        variable=self.var_2d_ip).grid(row=0, column=3, padx=6)
        ttk.Label(lf2, text="★ LAMBDA_INIT:").grid(row=0, column=4, sticky="e", padx=4)
        self.var_2d_lam = tk.StringVar(value="1.0")
        ttk.Entry(lf2, textvariable=self.var_2d_lam, width=8).grid(row=0, column=5, padx=3)
        ttk.Button(lf2, text="▶ 2D 反演 (SWP24D)", style="Accent.TButton",
                   command=self._on_2d, width=22).grid(row=0, column=6, padx=8, pady=3)

        # 6.3 3D
        lf3 = ttk.LabelFrame(p, text="③ WLF 3D 层析 (TRANS24合并 + 各线2D→3D切片插值 · carmod3_kru)")
        lf3.pack(fill="x", padx=6, pady=4)
        ttk.Label(lf3, text="多选 Occam.dat (所有线):").grid(row=0, column=0, sticky="nw", padx=4, pady=3)
        self.txt_3d_dats = tk.Text(lf3, height=4, width=62, font=("Consolas", 10))
        self.txt_3d_dats.grid(row=0, column=1, padx=4, pady=3)
        ttk.Button(lf3, text="添加…", width=8, command=self._3d_add_dats).grid(row=0, column=2, padx=3, pady=3, sticky="n")
        ttk.Label(lf3, text="3D输出目录:").grid(row=1, column=0, sticky="e", padx=4, pady=3)
        self.var_3d_out = tk.StringVar()
        ttk.Entry(lf3, textvariable=self.var_3d_out, width=60).grid(row=1, column=1, padx=4, pady=3)
        ttk.Button(lf3, text="浏览…", width=8,
                   command=lambda: self.var_3d_out.set(filedialog.askdirectory())
                   ).grid(row=1, column=2, padx=3)
        ttk.Button(lf3, text="▶ 启动 3D 层析流水线", style="Accent.TButton",
                   command=self._on_3d, width=22).grid(row=0, column=3, padx=8, rowspan=2)

        # 6.4 输出面板 (桥接结果)
        lf4 = ttk.LabelFrame(p, text="桥接执行结果 / 错误 / BAT 输出")
        lf4.pack(fill="both", expand=True, padx=6, pady=4)
        self.txt_wlf = tk.Text(lf4, height=14, font=("Consolas", 10),
                               bg="#252526", fg="#e0e0e0", wrap="word")
        self.txt_wlf.pack(fill="both", expand=True, padx=4, pady=4)

    def _on_trans24(self):
        inp = self.var_t_in.get()
        if not Path(inp).exists():
            messagebox.showerror("错误", "请选 Occam.dat 文件"); return
        r = run_trans24_convert(inp, self.var_t_fmt.get(), log_fn=self.log)
        self._write_wlf_report("[TRANS24] " + json.dumps(r, ensure_ascii=False, indent=2))

    def _on_2d(self):
        inp = self.var_2d_in.get()
        if not Path(inp).exists():
            messagebox.showerror("错误", "请选 Occam.dat (线) 文件"); return
        r = run_2d_inv_wlf(inp, include_ip=self.var_2d_ip.get(),
                           lambda_init=float(self.var_2d_lam.get()), log_fn=self.log)
        self._write_wlf_report("[2D-SWP24D] " + json.dumps(r, ensure_ascii=False, indent=2))

    def _3d_add_dats(self):
        fs = filedialog.askopenfilenames(filetypes=[("Occam dat", "*.dat")])
        self.txt_3d_dats.insert("end", "\n".join(fs) + "\n")

    def _on_3d(self):
        raw = self.txt_3d_dats.get("1.0", "end").strip().splitlines()
        dats = [Path(x) for x in raw if x.strip() and Path(x.strip()).exists()]
        outd = self.var_3d_out.get() or filedialog.askdirectory(title="3D输出目录")
        if not dats or not outd:
            messagebox.showwarning("警告", "请选多个 Occam.dat + 输出目录"); return
        r = run_3d_inv_wlf(dats, Path(outd), log_fn=self.log)
        self._write_wlf_report("[3D-TOMO] " + json.dumps(r, ensure_ascii=False, indent=2))

    def _write_wlf_report(self, text):
        self.txt_wlf.insert("end", "-" * 72 + "\n" + text + "\n\n")
        self.txt_wlf.see("end")
        self.log(text.replace("\n", " | ")[:300])

    # =================================================================
    # TAB 7: Tx Current Processing
    # =================================================================
    def _build_tab_current(self):
        p = self.tab_cur
        ttk.Label(p, text="发射电流档处理 (CXXST01.DAT)\n提取: 平台幅值 | 关断时刻 | 纹波 | 与Excel I_tx 对照",
                  font=("Microsoft YaHei", 11, "bold"), foreground="#0a5cad").pack(pady=8)
        f1 = ttk.Frame(p); f1.pack(fill="x", padx=6)
        ttk.Label(f1, text="项目根:").pack(side="left", padx=2)
        self.var_cur_project = tk.StringVar(value=self.var_project.get())
        ttk.Entry(f1, textvariable=self.var_cur_project, width=60).pack(side="left", padx=2)
        ttk.Button(f1, text="浏览…", command=lambda: self.var_cur_project.set(
            filedialog.askdirectory(initialdir=self.var_cur_project.get()) or self.var_cur_project.get()
        )).pack(side="left", padx=2)
        f2 = ttk.Frame(p); f2.pack(fill="x", padx=6, pady=6)
        ttk.Label(f2, text="输出根:").pack(side="left", padx=2)
        self.var_cur_out = tk.StringVar(value=self.var_out.get())
        ttk.Entry(f2, textvariable=self.var_cur_out, width=60).pack(side="left", padx=2)
        ttk.Button(f2, text="浏览…", command=lambda: self.var_cur_out.set(
            filedialog.askdirectory(initialdir=self.var_cur_out.get()) or self.var_cur_out.get()
        )).pack(side="left", padx=2)
        bf = ttk.Frame(p); bf.pack(fill="x", padx=6, pady=8)
        ttk.Button(bf, text="▶ 处理全线电流档", command=self._on_run_current).pack(side="left", padx=4)
        ttk.Button(bf, text="打开输出目录", command=self._open_out).pack(side="left", padx=4)
        self.txt_cur = tk.Text(p, height=10, wrap="word", font=("Consolas", 10), bg="#1e1e1e", fg="#c0c0c0")
        self.txt_cur.pack(fill="both", expand=True, padx=6, pady=4)

    def _on_run_current(self):
        proj, out = self.var_cur_project.get(), self.var_cur_out.get()
        if not Path(proj).is_dir():
            messagebox.showerror("错误", f"项目根不存在: {proj}"); return
        self._start_worker(self._worker_current, proj, out)

    def _worker_current(self, project, out):
        log = lambda msg, level="I": (self.log(msg, level), self.txt_cur.insert("end", msg + "\n"), self.txt_cur.see("end"))
        run_all_current(project, out, log_fn=log)
        log("✓ 电流处理完成")

    # =================================================================
    # TAB 8: Advanced Tools (FEM + 静态校正 + 分辨率矩阵)
    # =================================================================
    def _build_tab_advanced(self):
        p = self.tab_adv
        # ---- FEM 频率域正演 ----
        lf_fem = ttk.LabelFrame(p, text="FEM 频率域正演")
        lf_fem.pack(fill="x", padx=6, pady=4)
        f = ttk.Frame(lf_fem); f.pack(fill="x", padx=4, pady=2)
        ttk.Label(f, text="层厚(m):").grid(row=0, column=0, sticky="w")
        self.var_fem_thick = tk.StringVar(value="50,100,inf")
        ttk.Entry(f, textvariable=self.var_fem_thick, width=20).grid(row=0, column=1, padx=2)
        ttk.Label(f, text="电阻率:").grid(row=0, column=2, sticky="w")
        self.var_fem_rho = tk.StringVar(value="100,500,100")
        ttk.Entry(f, textvariable=self.var_fem_rho, width=20).grid(row=0, column=3, padx=2)
        ttk.Label(f, text="频率(Hz):").grid(row=1, column=0, sticky="w")
        self.var_fem_freq = tk.StringVar(value="1,10000,20")
        ttk.Entry(f, textvariable=self.var_fem_freq, width=20).grid(row=1, column=1, padx=2)
        ttk.Label(f, text="面积(m²):").grid(row=1, column=2, sticky="w")
        self.var_fem_area = tk.StringVar(value="1e6")
        ttk.Entry(f, textvariable=self.var_fem_area, width=10).grid(row=1, column=3, padx=2)
        ttk.Button(f, text="▶ FEM 正演", command=self._on_fem_forward).grid(row=1, column=4, padx=4)

        # ---- 静态效应校正 ----
        lf_static = ttk.LabelFrame(p, text="静态效应/飞行点校正")
        lf_static.pack(fill="x", padx=6, pady=4)
        f2 = ttk.Frame(lf_static); f2.pack(fill="x", padx=4, pady=2)
        ttk.Label(f2, text="ρa断面文件:").pack(side="left")
        self.var_static_file = tk.StringVar()
        ttk.Entry(f2, textvariable=self.var_static_file, width=40).pack(side="left", padx=2)
        ttk.Button(f2, text="…", width=3,
                   command=lambda: self.var_static_file.set(
                       filedialog.askopenfilename(filetypes=[("NPY", "*.npy")]))).pack(side="left")
        ttk.Button(f2, text="▶ 校正", command=self._on_static_correct).pack(side="left", padx=4)

        # ---- 分辨率矩阵 ----
        lf_res = ttk.LabelFrame(p, text="深度分辨率矩阵")
        lf_res.pack(fill="x", padx=6, pady=4)
        f3 = ttk.Frame(lf_res); f3.pack(fill="x", padx=4, pady=2)
        ttk.Label(f3, text="反演结果(npy):").pack(side="left")
        self.var_res_inv = tk.StringVar()
        ttk.Entry(f3, textvariable=self.var_res_inv, width=35).pack(side="left", padx=2)
        ttk.Button(f3, text="…", width=3,
                   command=lambda: self.var_res_inv.set(
                       filedialog.askopenfilename(filetypes=[("NPY", "*.npy")]))).pack(side="left")
        ttk.Label(f3, text="观测数据:").pack(side="left")
        self.var_res_obs = tk.StringVar()
        ttk.Entry(f3, textvariable=self.var_res_obs, width=35).pack(side="left", padx=2)
        ttk.Button(f3, text="…", width=3,
                   command=lambda: self.var_res_obs.set(
                       filedialog.askopenfilename(filetypes=[("NPY", "*.npy")]))).pack(side="left")
        ttk.Button(f3, text="▶ 计算", command=self._on_compute_resolution).pack(side="left", padx=4)

        self.txt_adv = tk.Text(p, height=8, wrap="word", font=("Consolas", 9), bg="#1e1e1e", fg="#c0c0c0")
        self.txt_adv.pack(fill="both", expand=True, padx=6, pady=4)

    def _on_fem_forward(self):
        try:
            thick = np.array([float(x) for x in self.var_fem_thick.get().split(",")])
            rho = np.array([float(x) for x in self.var_fem_rho.get().split(",")])
            parts = [float(x) for x in self.var_fem_freq.get().split(",")]
            freq = np.geomspace(parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 20)
            area = float(self.var_fem_area.get())
            model = LayeredModel(thick, rho)
            self._start_worker(self._worker_fem, model, area, freq)
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))

    def _worker_fem(self, model, area, freq):
        log = lambda m: (self.log(m), self.txt_adv.insert("end", m + "\n"), self.txt_adv.see("end"))
        f, hz_mag, hz_phase = fem_forward_1d(model, area, 30.0, freq, log_fn=log)
        rhoa = fem_apparent_resistivity(hz_mag, freq, area, 30.0)
        log(f"  FEM 完成: {len(f)} 频率点, ρa ∈ [{rhoa.min():.1f}, {rhoa.max():.1f}] Ω·m")
        # 保存结果
        out = Path(self.var_out.get()) / "fem_forward.npz"
        np.savez(out, freq_hz=f, hz_mag=hz_mag, hz_phase=hz_phase, rhoa=rhoa)
        log(f"  结果 → {out}")

    def _on_static_correct(self):
        fpath = self.var_static_file.get()
        if not fpath or not Path(fpath).exists():
            messagebox.showerror("错误", "请选择 ρa 断面文件"); return
        self._start_worker(self._worker_static, fpath)

    def _worker_static(self, fpath):
        log = lambda m: (self.log(m), self.txt_adv.insert("end", m + "\n"), self.txt_adv.see("end"))
        rhoa = np.load(fpath)
        n_st = rhoa.shape[0]
        station_x = np.arange(n_st) * 50.0
        result = correct_profile(rhoa, station_x, log_fn=log)
        out = Path(self.var_out.get()) / "static_corrected.npz"
        np.savez(out, **result)
        log(f"  校正完成 → {out}")
        # 绘图
        out_png = Path(self.var_out.get()) / "static_correction.png"
        plot_correction_comparison(rhoa, result['rhoa_corrected'],
                                   result['outlier_mask'], station_x, out_png=str(out_png))
        log(f"  对比图 → {out_png}")

    def _on_compute_resolution(self):
        inv_file = self.var_res_inv.get()
        obs_file = self.var_res_obs.get()
        if not inv_file or not Path(inv_file).exists():
            messagebox.showerror("错误", "请选择反演结果文件"); return
        if not obs_file or not Path(obs_file).exists():
            messagebox.showerror("错误", "请选择观测数据文件"); return
        self._start_worker(self._worker_resolution, inv_file, obs_file)

    def _worker_resolution(self, inv_file, obs_file):
        log = lambda m: (self.log(m), self.txt_adv.insert("end", m + "\n"), self.txt_adv.see("end"))
        theta = np.load(inv_file)
        data = np.load(obs_file)
        t_g, rhoa_obs, err_obs = data['t'], data['rhoa'], data['err']
        z_c = np.geomspace(5.0, 1500.0, len(theta))
        from .forward import forward_rhoa
        res = compute_resolution_from_inversion(theta, t_g, rhoa_obs, err_obs, z_c,
                                                forward_rhoa, log_fn=log)
        out_png = Path(self.var_out.get()) / "resolution_matrix.png"
        plot_resolution(res, z_c, out_png=str(out_png))
        log(f"  分辨率矩阵 → {out_png}")

    # =================================================================
    # TAB 9: AI & Visualization (NN + AE + 3D Viz)
    # =================================================================
    def _build_tab_ai_viz(self):
        p = self.tab_ai
        # ---- NN 初始模型 ----
        lf_nn = ttk.LabelFrame(p, text="神经网络初始模型预测")
        lf_nn.pack(fill="x", padx=6, pady=4)
        f = ttk.Frame(lf_nn); f.pack(fill="x", padx=4, pady=2)
        ttk.Label(f, text="训练样本数:").grid(row=0, column=0, sticky="w")
        self.var_nn_samples = tk.StringVar(value="2000")
        ttk.Entry(f, textvariable=self.var_nn_samples, width=10).grid(row=0, column=1, padx=2)
        ttk.Label(f, text="Epochs:").grid(row=0, column=2, sticky="w")
        self.var_nn_epochs = tk.StringVar(value="50")
        ttk.Entry(f, textvariable=self.var_nn_epochs, width=8).grid(row=0, column=3, padx=2)
        ttk.Button(f, text="▶ 训练", command=self._on_nn_train).grid(row=0, column=4, padx=4)
        ttk.Button(f, text="▶ 预测", command=self._on_nn_predict).grid(row=0, column=5, padx=4)

        # ---- 去噪自编码器 ----
        lf_ae = ttk.LabelFrame(p, text="去噪自编码器")
        lf_ae.pack(fill="x", padx=6, pady=4)
        f2 = ttk.Frame(lf_ae); f2.pack(fill="x", padx=4, pady=2)
        ttk.Label(f2, text="样本数:").grid(row=0, column=0, sticky="w")
        self.var_ae_samples = tk.StringVar(value="2000")
        ttk.Entry(f2, textvariable=self.var_ae_samples, width=10).grid(row=0, column=1, padx=2)
        ttk.Label(f2, text="噪声水平:").grid(row=0, column=2, sticky="w")
        self.var_ae_noise = tk.StringVar(value="0.1")
        ttk.Entry(f2, textvariable=self.var_ae_noise, width=8).grid(row=0, column=3, padx=2)
        ttk.Button(f2, text="▶ 训练", command=self._on_ae_train).grid(row=0, column=4, padx=4)
        ttk.Button(f2, text="▶ 去噪", command=self._on_ae_denoise).grid(row=0, column=5, padx=4)

        # ---- 3D 可视化 ----
        lf_3d = ttk.LabelFrame(p, text="三维体可视化")
        lf_3d.pack(fill="x", padx=6, pady=4)
        f3 = ttk.Frame(lf_3d); f3.pack(fill="x", padx=4, pady=2)
        ttk.Label(f3, text="断面数据:").pack(side="left")
        self.var_3d_file = tk.StringVar()
        ttk.Entry(f3, textvariable=self.var_3d_file, width=40).pack(side="left", padx=2)
        ttk.Button(f3, text="…", width=3,
                   command=lambda: self.var_3d_file.set(
                       filedialog.askopenfilename(filetypes=[("NPY", "*.npy")]))).pack(side="left")
        ttk.Button(f3, text="▶ 生成 3D HTML", command=self._on_3d_viz).pack(side="left", padx=4)
        ttk.Button(f3, text="▶ 交互式断面", command=self._on_interactive_section).pack(side="left", padx=4)

        self.txt_ai = tk.Text(p, height=8, wrap="word", font=("Consolas", 9), bg="#1e1e1e", fg="#c0c0c0")
        self.txt_ai.pack(fill="both", expand=True, padx=6, pady=4)

        # 模型缓存
        self._nn_model = None
        self._ae_model = None

    def _on_nn_train(self):
        n = int(self.var_nn_samples.get())
        epochs = int(self.var_nn_epochs.get())
        self._start_worker(self._worker_nn_train, n, epochs)

    def _worker_nn_train(self, n_samples, n_epochs):
        log = lambda m: (self.log(m), self.txt_ai.insert("end", m + "\n"), self.txt_ai.see("end"))
        nn = NNInitialModel(n_gates=25, n_layers=25)
        log(f"  [NN] 生成 {n_samples} 样本...")
        history = nn.train_from_synthetic(n_samples=n_samples, n_epochs=n_epochs, log_fn=log)
        self._nn_model = nn
        save_path = Path(self.var_out.get()) / "nn_initial_model.npz"
        nn.save(save_path)
        log(f"  [NN] 训练完成, 模型 → {save_path}")

    def _on_nn_predict(self):
        if self._nn_model is None:
            # 尝试加载
            save_path = Path(self.var_out.get()) / "nn_initial_model.npz"
            if save_path.exists():
                self._nn_model = NNInitialModel(25, 25)
                self._nn_model.load(save_path)
            else:
                messagebox.showerror("错误", "请先训练或加载模型"); return
        self._start_worker(self._worker_nn_predict)

    def _worker_nn_predict(self):
        log = lambda m: (self.log(m), self.txt_ai.insert("end", m + "\n"), self.txt_ai.see("end"))
        # 从预处理结果加载 ρa 曲线
        out_dir = Path(self.var_out.get())
        rhoa_files = list(out_dir.glob("*_rhoa.npy"))
        if not rhoa_files:
            log("  [NN] 未找到 ρa 文件"); return
        rhoa = np.load(rhoa_files[0])
        rho_init = self._nn_model.predict_initial(rhoa)
        log(f"  [NN] 初始模型预测: ρ ∈ [{rho_init.min():.1f}, {rho_init.max():.1f}] Ω·m")
        out = out_dir / "nn_initial_model_pred.npy"
        np.save(out, rho_init)
        log(f"  [NN] 预测结果 → {out}")

    def _on_ae_train(self):
        n = int(self.var_ae_samples.get())
        noise = float(self.var_ae_noise.get())
        self._start_worker(self._worker_ae_train, n, noise)

    def _worker_ae_train(self, n_samples, noise_level):
        log = lambda m: (self.log(m), self.txt_ai.insert("end", m + "\n"), self.txt_ai.see("end"))
        ae = DenoisingAutoencoder(n_gates=25, n_latent=8)
        log(f"  [AE] 生成 {n_samples} 干净样本...")
        history = ae.train_from_synthetic(n_samples=n_samples, noise_level=noise_level, log_fn=log)
        self._ae_model = ae
        save_path = Path(self.var_out.get()) / "denoise_ae_model.npz"
        ae.save(save_path)
        log(f"  [AE] 训练完成, 模型 → {save_path}")

    def _on_ae_denoise(self):
        if self._ae_model is None:
            save_path = Path(self.var_out.get()) / "denoise_ae_model.npz"
            if save_path.exists():
                self._ae_model = DenoisingAutoencoder(25)
                self._ae_model.load(save_path)
            else:
                messagebox.showerror("错误", "请先训练或加载模型"); return
        self._start_worker(self._worker_ae_denoise)

    def _worker_ae_denoise(self):
        log = lambda m: (self.log(m), self.txt_ai.insert("end", m + "\n"), self.txt_ai.see("end"))
        out_dir = Path(self.var_out.get())
        decay_files = list(out_dir.glob("*_decay_curves.npz"))
        if not decay_files:
            log("  [AE] 未找到衰减曲线文件"); return
        data = np.load(decay_files[0])
        # 假设数据中有 'vt' 或 'dbzdt' 键
        key = 'vt' if 'vt' in data else 'dbzdt'
        noisy = data[key]
        clean = self._ae_model.denoise(noisy)
        out = out_dir / "denoised_curves.npz"
        np.savez(out, original=noisy, denoised=clean)
        log(f"  [AE] 去噪完成 → {out}")

    def _on_3d_viz(self):
        fpath = self.var_3d_file.get()
        if not fpath or not Path(fpath).exists():
            messagebox.showerror("错误", "请选择断面数据文件"); return
        self._start_worker(self._worker_3d_viz, fpath)

    def _worker_3d_viz(self, fpath):
        log = lambda m: (self.log(m), self.txt_ai.insert("end", m + "\n"), self.txt_ai.see("end"))
        data = np.load(fpath)
        # 假设数据是 (n_stations, n_gates)
        if data.ndim == 2:
            n_st, n_g = data.shape
            x = np.arange(n_st) * 50.0
            y = np.geomspace(5.0, 1500.0, n_g)
            z = np.array([0.0])
            vol = data[:, :, None]  # (n_st, n_g, 1)
            out_html = Path(self.var_out.get()) / "volume_3d.html"
            export_3d_volume_html(x, y, z, vol, str(out_html), log_fn=log)
        else:
            log("  [3D] 数据维度不支持")

    def _on_interactive_section(self):
        fpath = self.var_3d_file.get()
        if not fpath or not Path(fpath).exists():
            messagebox.showerror("错误", "请选择断面数据文件"); return
        self._start_worker(self._worker_interactive, fpath)

    def _worker_interactive(self, fpath):
        log = lambda m: (self.log(m), self.txt_ai.insert("end", m + "\n"), self.txt_ai.see("end"))
        data = np.load(fpath)
        n_st, n_g = data.shape
        x = np.arange(n_st) * 50.0
        y = np.geomspace(2e-6, 20e-3, n_g)
        out_html = Path(self.var_out.get()) / "interactive_section.html"
        export_section_interactive(x, y, data, str(out_html), log_fn=log)

    # =================================================================
    # TAB 10: Project Database (SQLite)
    # =================================================================
    def _build_tab_project(self):
        p = self.tab_proj
        # ---- 项目库操作 ----
        lf_db = ttk.LabelFrame(p, text="SQLite 项目库")
        lf_db.pack(fill="x", padx=6, pady=4)
        f = ttk.Frame(lf_db); f.pack(fill="x", padx=4, pady=2)
        ttk.Label(f, text="数据库:").grid(row=0, column=0, sticky="w")
        self.var_db_path = tk.StringVar(value=str(Path(self.var_out.get()) / "project.db"))
        ttk.Entry(f, textvariable=self.var_db_path, width=50).grid(row=0, column=1, padx=2)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self.var_db_path.set(
                       filedialog.asksaveasfilename(defaultextension=".db",
                                                    filetypes=[("SQLite", "*.db")]))).grid(row=0, column=2)
        ttk.Label(f, text="项目名:").grid(row=1, column=0, sticky="w")
        self.var_db_name = tk.StringVar(value="TFM_Project")
        ttk.Entry(f, textvariable=self.var_db_name, width=30).grid(row=1, column=1, padx=2)
        ttk.Button(f, text="▶ 创建/打开", command=self._on_db_open).grid(row=1, column=2, padx=4)
        ttk.Button(f, text="▶ 导入 Excel", command=self._on_db_import).grid(row=1, column=3, padx=4)
        ttk.Button(f, text="▶ 生成报告", command=self._on_db_report).grid(row=1, column=4, padx=4)

        # ---- 查询 ----
        lf_query = ttk.LabelFrame(p, text="数据查询")
        lf_query.pack(fill="x", padx=6, pady=4)
        f2 = ttk.Frame(lf_query); f2.pack(fill="x", padx=4, pady=2)
        ttk.Label(f2, text="测线号:").pack(side="left")
        self.var_query_line = tk.StringVar()
        ttk.Entry(f2, textvariable=self.var_query_line, width=8).pack(side="left", padx=2)
        ttk.Button(f2, text="查询测站", command=self._on_db_query).pack(side="left", padx=4)
        ttk.Button(f2, text="查询反演", command=self._on_db_query_inv).pack(side="left", padx=4)

        self.txt_proj = tk.Text(p, height=10, wrap="word", font=("Consolas", 9), bg="#1e1e1e", fg="#c0c0c0")
        self.txt_proj.pack(fill="both", expand=True, padx=6, pady=4)

        self._db: Optional[ProjectDB] = None

    def _on_db_open(self):
        db_path = self.var_db_path.get()
        name = self.var_db_name.get()
        try:
            self._db = ProjectDB(db_path)
            # 尝试创建项目（若已存在则忽略错误）
            try:
                self._db.create_project(name)
                self.log(f"[DB] 创建项目: {name}")
            except Exception:
                self.log(f"[DB] 打开项目: {name}")
            self.txt_proj.insert("end", f"数据库已连接: {db_path}\n")
            self.txt_proj.see("end")
        except Exception as exc:
            messagebox.showerror("错误", str(exc))

    def _on_db_import(self):
        if self._db is None:
            messagebox.showerror("错误", "请先创建/打开数据库"); return
        excel = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx;*.xls")])
        if not excel:
            return
        self._start_worker(self._worker_db_import, excel)

    def _worker_db_import(self, excel_path):
        log = lambda m: (self.log(m), self.txt_proj.insert("end", m + "\n"), self.txt_proj.see("end"))
        project = self._db.get_project(self.var_db_name.get())
        if not project:
            log("  [DB] 项目不存在"); return
        stats = import_from_excel(self._db, project['id'], excel_path, log_fn=log)
        log(f"  [DB] 导入完成: {stats}")

    def _on_db_report(self):
        if self._db is None:
            messagebox.showerror("错误", "请先创建/打开数据库"); return
        project = self._db.get_project(self.var_db_name.get())
        if not project:
            return
        out_html = Path(self.var_out.get()) / "project_report.html"
        self._db.export_summary_html(project['id'], str(out_html))
        self.log(f"[DB] 报告 → {out_html}")
        self.txt_proj.insert("end", f"报告 → {out_html}\n")
        self.txt_proj.see("end")

    def _on_db_query(self):
        if self._db is None:
            messagebox.showerror("错误", "请先创建/打开数据库"); return
        line_num = self.var_query_line.get()
        if not line_num:
            return
        project = self._db.get_project(self.var_db_name.get())
        if not project:
            return
        line = self._db.get_line(project['id'], int(line_num))
        if not line:
            self.txt_proj.insert("end", f"测线 {line_num} 不存在\n")
            return
        stations = self._db.list_stations(line['id'])
        self.txt_proj.insert("end", f"测线 {line_num}: {len(stations)} 站\n")
        for st in stations[:10]:
            self.txt_proj.insert("end", f"  {st['station_number']}: {st['code']} ({st['x']:.1f}, {st['y']:.1f})\n")
        if len(stations) > 10:
            self.txt_proj.insert("end", f"  ... 共 {len(stations)} 站\n")
        self.txt_proj.see("end")

    def _on_db_query_inv(self):
        if self._db is None:
            messagebox.showerror("错误", "请先创建/打开数据库"); return
        project = self._db.get_project(self.var_db_name.get())
        if not project:
            return
        invs = self._db.list_inversions(project['id'])
        self.txt_proj.insert("end", f"反演记录: {len(invs)} 条\n")
        for inv in invs[:5]:
            self.txt_proj.insert("end", f"  {inv['algorithm']}: RMSE={inv['rmse_mean']:.3f} @ {inv['created_at']}\n")
        self.txt_proj.see("end")


def main():
    root = tk.Tk()
    # 缺失的 simpledialog 导入
    import tkinter.simpledialog  # noqa: F401 (used in mdl_edit_cell)
    app = TFMMainApp(root)
    app.log("TFM GUI v2.0 启动 · WLF 全套算法集 · 100% OFFLINE 纯本地")
    app.log(f"  Python: {sys.executable}")
    app.log(f"  Tools : {_HERE}")
    root.mainloop()


if __name__ == "__main__":
    main()
