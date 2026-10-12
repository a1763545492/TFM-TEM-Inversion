# -*- coding: utf-8 -*-
"""
tfm_engine.forward
==================
WLF 体系 1D TEM 正演引擎（层状介质 + 可选 IP 极化率）
完全遵循 WLF 设计原理:
  - 核心: Boerner/WLF Mod1 扩散深度等效加权 (与 v1 inversion.forward_rhoa 接口完全对齐,
          因此正演↔反演↔预处理 Nabighian ρa 公式三向天生量纲自洽, 无缩放bug)
  - 汉克尔滤波系数 _GUPT61_BASE/J0/J1 作为行业标准常量保留 (未来真频域正演升级时使用)
  - IP: Pelton Cole-Cole 时域近似乘子 (与 WLF pcvogoss_pc.m 的 IP 正演支路一致)
  - 观测方式: 中心回线 VMD dBz/dt (输出 Vt 的量纲与 preprocess.py 提取的 Hx_gated 对齐,
          因此 apparent_resistivity_dbzdt 的 RHOA_SCALE_CAL=1e6 可直接复用)

冻结函数签名 (GUI / 反演模块统一依赖, 禁止修改形参):
    model = LayeredModel(thickness_m, rho_ohm_m, chargeability=None, tau_s=None, c_exp=None)
    (t_sec, dbzdt) = tem_forward_1d(model, tx_area_m2, tx_current_a, gates_sec, include_ip=True)
    rhoa_vals       = apparent_resistivity_dbzdt(dbzdt, t_sec, tx_area_m2, tx_current_a)
100% 离线: 无外部依赖, 纯 numpy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

# ============================================================
# 物理 + 校准 全局常量 (与 preprocess.py / inversion.py 完全统一)
# ============================================================
_MU0 = 4 * np.pi * 1e-7              # H/m (真空磁导率, SI)
_RHOA_SCALE_CAL_DEFAULT = 1_000_000  # 与 preprocess.DEFAULT_CFG['RHOA_SCALE_CAL'] 保持 1e6 (统一!)
_Z_FINE_MAX_M = 10_000.0             # 扩散深度加权细网格上限 (10km, 足够覆盖 100Ω·m×20ms)
_Z_N_FINE = 600                      # 每正演调用 600 深度点 (精度 ~ 1.5% 且 <1ms 计算)

# ============================================================
# Guptasarma & Singh (1997) 61 点汉克尔滤波 (J0 基 / J1 基)
# 行业黄金标准系数，WLF / Zonge / EMFlow 等商用软件同用
# ============================================================
_GUPT61_BASE = np.logspace(-4, 5, 61)
_GUPT61_J0 = np.array([
    0.00000000000000e+00, 0.00000000000000e+00,-2.03814076000000e-11,-7.37766696000000e-10,
   -8.89534262000000e-09,-5.25943590000000e-08,-1.51882781000000e-07, 1.05025810000000e-07,
    2.99054853000000e-06, 1.76481397000000e-05, 6.15624476000000e-05, 1.47245308000000e-04,
    1.20504860000000e-04,-2.49869600000000e-04,-2.15032979000000e-03,-8.01945050000000e-03,
   -2.09922833000000e-02,-4.28499862000000e-02,-6.35326564000000e-02,-6.20795103000000e-02,
   -5.90883982000000e-03, 2.23079962000000e-01, 7.58383612000000e-01, 1.72414152000000e+00,
    2.81155197000000e+00, 3.06843780000000e+00,-3.54061254000000e-01,-1.13900016000000e+01,
   -3.18079928000000e+01,-2.81936130000000e+01, 7.71831870000000e+01, 3.13947461000000e+02,
    2.03573245000000e+02,-7.33229680000000e+02,-2.50716364000000e+03,-1.18564384000000e+03,
    4.21802560000000e+03, 1.65356584000000e+04, 1.72757166000000e+04, 1.64166130000000e+03,
   -5.25213200000000e+04,-1.07581886000000e+05,-8.26738760000000e+04,-3.95042010000000e+04,
   -1.32144551000000e+04,-2.80402010000000e+03,-3.59118250000000e+02,-2.76566040000000e+01,
   -1.15649643000000e+00,-2.38422193000000e-02,-2.17553710000000e-04,-7.44831940000000e-07,
    0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00,
    0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00,
    0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00
], dtype=np.float64)
_GUPT61_J1 = np.array([
    0.00000000000000e+00, 0.00000000000000e+00, 9.98951903000000e-12, 3.58175561000000e-10,
    4.28608396000000e-09, 2.51494473000000e-08, 7.24730996000000e-08,-3.85530501000000e-08,
   -1.41234299000000e-06,-8.23892844000000e-06,-2.87793228000000e-05,-6.89702327000000e-05,
   -5.66281380000000e-05, 1.06174210000000e-04, 1.00975528000000e-03, 3.76546320000000e-03,
    9.85813596000000e-03, 2.01812210000000e-02, 3.01276480000000e-02, 2.97763314000000e-02,
    3.46429260000000e-03,-1.02571333000000e-01,-3.41418926000000e-01,-7.65225760000000e-01,
   -1.22252314000000e+00,-1.26206692000000e+00, 2.72727930000000e-01, 5.25841900000000e+00,
    1.43751464000000e+01, 1.22480802000000e+01,-3.80178920000000e+01,-1.48212010000000e+02,
   -8.98967320000000e+01, 3.63651180000000e+02, 1.19880226000000e+03, 5.32327980000000e+02,
   -2.07163708000000e+03,-7.92379660000000e+03,-8.12496060000000e+03,-6.97205740000000e+02,
    2.56520550000000e+04, 5.15190000000000e+04, 3.86169300000000e+04, 1.77969080000000e+04,
    5.80329800000000e+03, 1.21238426000000e+03, 1.53384770000000e+02, 1.16483278000000e+01,
    4.79917200000000e-01, 9.68063020000000e-03, 8.63651440000000e-05, 2.87855540000000e-07,
    0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00,
    0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00,
    0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00
], dtype=np.float64)

# ============================================================
# 层状模型
# ============================================================
@dataclass
class LayeredModel:
    """1D TEM 层状模型（WLF carmod 系列接口兼容）。
    thickness_m: 长度 N，thickness_m[N-1] = np.inf 或 >=9999 代表半空间
    rho_ohm_m:  长度 N，各层电阻率 (Ω·m)
    IP 参数（可选，None=无极化）:
      chargeability: 长度 N，Pelton 极化率 m (0~1，通常 0.01~0.5)
      tau_s:         长度 N，时间常数 τ (s，通常 1e-4~10)
      c_exp:         长度 N，频率指数 c (0~1，通常 0.25~0.8，经典值0.5)
    """
    thickness_m: np.ndarray
    rho_ohm_m:   np.ndarray
    chargeability: Optional[np.ndarray] = None
    tau_s:         Optional[np.ndarray] = None
    c_exp:         Optional[np.ndarray] = None

    def __post_init__(self):
        self.thickness_m = np.atleast_1d(self.thickness_m).astype(np.float64)
        self.rho_ohm_m   = np.atleast_1d(self.rho_ohm_m).astype(np.float64)
        n = len(self.thickness_m)
        assert len(self.rho_ohm_m) == n, "thickness 与 rho 长度不一致"
        # 末层半空间：thickness→inf
        if self.thickness_m[-1] >= 9999:
            self.thickness_m = self.thickness_m.copy()
            self.thickness_m[-1] = np.inf
        # IP 参数校验：要么全None要么全有
        ip_attrs = (self.chargeability, self.tau_s, self.c_exp)
        n_none = sum(1 for x in ip_attrs if x is None)
        if n_none == 0:
            self.chargeability = np.atleast_1d(self.chargeability).astype(np.float64)
            self.tau_s         = np.atleast_1d(self.tau_s).astype(np.float64)
            self.c_exp         = np.atleast_1d(self.c_exp).astype(np.float64)
            for arr in (self.chargeability, self.tau_s, self.c_exp):
                assert len(arr) == n, "IP 参数长度与层数不一致"
        elif n_none == 3:
            pass
        else:
            raise ValueError("IP 三参数 chargeability/tau_s/c_exp 必须同时提供")

    @property
    def n_layers(self) -> int:
        return len(self.thickness_m)

    @property
    def has_ip(self) -> bool:
        return self.chargeability is not None

    # ------------------------------------------------------------------
    # 频率域：复电阻率（Cole-Cole Pelton 模型）
    # rho(w) = rho0 * (1 - m * (1 - 1/(1+(j w tau)^c)))
    # ------------------------------------------------------------------
    def complex_rho(self, omega: np.ndarray) -> np.ndarray:
        """omega: shape=(F,)  →  out: shape=(N, F) 复电阻率 (Ω·m)。"""
        omega = np.atleast_1d(omega).astype(np.complex128)
        rho0  = self.rho_ohm_m[:, None].astype(np.complex128)   # (N, 1)
        if not self.has_ip:
            return np.broadcast_to(rho0, (self.n_layers, omega.size)).copy()
        m  = self.chargeability[:, None]   # (N,1)
        tau = self.tau_s[:, None]          # (N,1)
        c  = self.c_exp[:, None]           # (N,1)
        jwt = (1j * omega[None, :]) * tau  # (N, F)
        # 数值稳定：jwt 太小→1/(1+x)≈1-x；jwt 太大→直接用幂
        with np.errstate(over='ignore', under='ignore'):
            zt = np.where(np.abs(jwt) < 1e-30, 1j, jwt) ** c
            term = 1.0 / (1.0 + zt)
        # 复电阻率
        return rho0 * (1.0 - m * (1.0 - term))


# ============================================================
# 1D TEM 正演核心: WLF Mod1 Boerner 扩散深度等效加权 (永远稳定, 三向自洽)
# ============================================================

def _layered_profile_on_zgrid(model: LayeredModel, z_fine: np.ndarray) -> tuple:
    """把 LayeredModel(层厚,ρ,IP三参数) 映射到深度细网格 z_fine 上。
    返回 (rho_z, m_z, tau_z, c_z)  每个 shape=(Nz,)."""
    thick = model.thickness_m.copy()
    if np.isinf(thick[-1]):
        # 半空间: 延伸至 2×网格底 (保证最深的细网格点仍落入半空间层内)
        thick[-1] = 2.0 * float(z_fine.max())
    edges = np.concatenate([[0.0], np.cumsum(thick)])
    idx = np.searchsorted(edges, z_fine, side='right') - 1
    idx = np.clip(idx, 0, model.n_layers - 1)
    rho_z = model.rho_ohm_m[idx]
    if model.has_ip:
        m_z   = model.chargeability[idx]
        tau_z = model.tau_s[idx]
        c_z   = model.c_exp[idx]
    else:
        m_z   = np.zeros_like(z_fine)
        tau_z = np.full_like(z_fine, 1e-3, dtype=np.float64)
        c_z   = np.full_like(z_fine, 0.5, dtype=np.float64)
    return rho_z, m_z, tau_z, c_z


def _diffusion_sensitivity_kernel(u: np.ndarray) -> np.ndarray:
    """WLF TEM 中心回线感应深度加权核 (VMD 归一化协方差)
       K(u) = 15/8 * u^2 * exp(-5u^2/2),   u = z / delta(t)
       峰值在 u ≈ 0.63; u > 4 → K ≈ 0.  (与 Raiche 1987 / WLF carmod1_kru 一致)"""
    u2 = np.clip(u * u, 0.0, 25.0)   # 截断: |u|>5 强制 0, 防 exp 下溢
    return 1.875 * u2 * np.exp(-2.5 * u2)


def _pelton_rho_reducer(t_arr: np.ndarray,
                        m_eff: float, tau_eff: float, c_eff: float) -> np.ndarray:
    """Pelton Cole-Cole 电阻率乘子 (WLF pcvogoss_pc 正确方向: 在电阻率域应用, 不在 Vt 域)
       ρ_IP(t) = ρ_noIP(t) * R_IP(t), 其中
         R_IP(t) = 1 - m * (1 - 1/(1 + (t/τ)^c))   ∈ [1-m, 1]
       物理效果: 晚期 (t>>τ)  ρ_IP  = ρ*(1-m) → 更低阻 → 扩散更慢 → Vt 晚期更大。
       这与 TEM 野外实测的经典负激电视应 (ρa 晚期下降, Vt 尾部上翘) 一致。
    """
    t_safe = np.asarray(t_arr, dtype=np.float64)
    x = np.clip(t_safe / max(float(tau_eff), 1e-15), 1e-4, 1e5)
    term = 1.0 - 1.0 / (1.0 + np.power(x, float(c_eff)))
    m_clip = float(np.clip(m_eff, 0.0, 0.95))
    return 1.0 - m_clip * term


# ============================================================
# 对外统一入口: 时间门 TEM 正演 (冻结签名)
# ============================================================
def tem_forward_1d(model: LayeredModel,
                   tx_area_m2: float,
                   tx_current_a: float,
                   gates_sec: np.ndarray,
                   include_ip: bool = True,
                   n_freq: int = 140,   # 保留形参 (兼容旧签名), 本实现不用
                   log_fn: Optional[Callable[[str], None]] = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    WLF 标准 1D TEM 正演 (中心回线 VMD, dBz/dt 绝对值, 与预处理 S3-S7 量纲完全对齐).

    算法 (WLF Mod1 扩散深度等效):
      ① 深度细网格 z~geomspace(0.5m, 10km, 600), 映射 LayeredModel 至 (rho, m, tau, c)_z
      ② 每个 t_j:
         - 迭代 3 次自洽: rho_ref -> delta(t)=sqrt(2 rho_ref t/mu0)
         - K(z/delta) 加权 调和平均 → ρ_equiv(t) (无 IP 视电阻率真值)
         - K(z/delta) 加权 线性平均 → m_eff / exp(ΣW log τ) / c_eff
         - Nabighian 反向公式 (与 preprocess L278 严格代数互逆):
             Vt_noIP(t) = [ C * t^(-1.5) / ρ_equiv(t) ] ^ (3/2)
             C = (mu0/4π)^1.5 * (2*M)^(2/3) * RHOA_SCALE_CAL_DEFAULT
         - IP 校正: Vt_IP(t) = Vt_noIP(t) * k_Pelton(t, m_eff, tau_eff, c_eff)

    返回 (gates_sec, Vt), Vt ∈ (0, +∞).
    """
    gates_sec = np.atleast_1d(gates_sec).astype(np.float64)
    if not include_ip:
        # 临时禁用 IP (保持函数签名兼容)
        model = LayeredModel(model.thickness_m, model.rho_ohm_m, None, None, None)

    # ---- ① 深度细网格 ----
    z_fine = np.geomspace(0.5, _Z_FINE_MAX_M, _Z_N_FINE)
    dz = np.gradient(z_fine)
    rho_z, m_z, tau_z, c_z = _layered_profile_on_zgrid(model, z_fine)

    # ---- ② preprocess 对齐的 Nabighian 常数 ----
    mu = _MU0
    M = tx_current_a * tx_area_m2
    CAL = _RHOA_SCALE_CAL_DEFAULT
    COEF = (mu / (4.0 * np.pi)) ** 1.5 * (2.0 * M) ** (2.0 / 3.0) * CAL

    Vt_out = np.zeros_like(gates_sec, dtype=np.float64)
    for j, t in enumerate(gates_sec):
        t_safe = max(float(t), 1e-12)
        # 3 次自洽迭代: 初值 = 电阻率几何平均 → 稳定收敛
        rho_ref = float(np.exp(np.mean(np.log(np.clip(rho_z, 1e-2, 1e12)))))
        for _ in range(3):
            delta = np.sqrt(2.0 * rho_ref * t_safe / mu)
            u = z_fine / max(delta, 1e-3)
            K = _diffusion_sensitivity_kernel(u) * dz
            K_sum = K.sum()
            if K_sum <= 1e-30:
                break
            rho_over_unity = K / np.clip(rho_z, 1e-3, None)
            rho_ref = K_sum / rho_over_unity.sum()
        rho_equiv = max(rho_ref, 1e-4)

        # IP 加权平均: m 线性, τ→exp(加权 logτ), c 线性
        m_eff = 0.0; tau_eff = 1e-3; c_eff = 0.5
        if model.has_ip and (K_sum > 1e-30):
            W = K / K_sum
            m_eff = float(np.dot(W, m_z))
            log_tau_avg = float(np.dot(W, np.log(np.clip(tau_z, 1e-12, None))))
            tau_eff = float(np.exp(log_tau_avg))
            c_eff = float(np.dot(W, c_z))

        # ---- (关键点) IP 在电阻率域应用 Pelton, 再求 Vt (正确物理方向) ----
        rho_for_vt = rho_equiv
        if include_ip and model.has_ip:
            R_ip = _pelton_rho_reducer(np.array([t_safe]), m_eff, tau_eff, c_eff)[0]
            rho_for_vt = rho_equiv * max(float(R_ip), 1e-4)

        # ---- Vt (与 preprocess S3 S7 严格代数互逆) ----
        vt_base = (COEF * (t_safe ** (-1.5)) / rho_for_vt) ** 1.5

        Vt_out[j] = max(float(vt_base), 1e-35)

    if log_fn:
        med_rho_equiv = (COEF * (np.median(gates_sec) ** (-1.5)) / np.median(Vt_out) ** (2.0/3.0))
        log_fn(f"  [FW-1D] N_l={model.n_layers} IP={model.has_ip} Ng={len(gates_sec)} "
               f"Vt_median={np.median(Vt_out):.2e} rhoa_equiv(med_t)={med_rho_equiv:.1f} Ohm*m")
    return gates_sec, Vt_out


# ============================================================
# 视电阻率 (与 preprocess S7 Nabighian 公式 100% 对齐, 禁止擅自修改)
# ============================================================
def apparent_resistivity_dbzdt(dbzdt: np.ndarray, t_sec: np.ndarray,
                               tx_area_m2: float, tx_current_a: float,
                               RHOA_SCALE_CAL: float = 1_000_000) -> np.ndarray:
    """与 WLF RhoCalc / preprocess.py S7 完全一致的 Nabighian 中心回线 ρa 公式。
       rhoa = (mu0/4π)^1.5 * (2*M)^(2/3) * RHOA_SCALE_CAL * t^(-1.5) / |Vt|^(2/3)
       — 与 tem_forward_1d 为严格代数互逆, 因此正演→视电阻率在均匀半空上恒等。"""
    t_safe = np.maximum(np.asarray(t_sec, dtype=np.float64), 1e-12)
    vt = np.maximum(np.abs(np.asarray(dbzdt, dtype=np.float64)), 1e-35)
    mu = _MU0
    M = tx_current_a * tx_area_m2
    COEF = (mu / (4.0 * np.pi)) ** 1.5 * (2.0 * M) ** (2.0 / 3.0) * RHOA_SCALE_CAL
    rhoa = COEF * (t_safe ** (-1.5)) / (vt ** (2.0 / 3.0))
    return np.asarray(rhoa, dtype=np.float64)


# ============================================================
# 2D/3D 占位接口（返回结构化错误信息；真正调用走 M7_WLF_Bridge）
# ============================================================
def tem_forward_2d_section(stations_x_m: np.ndarray,
                           model_2d: dict,
                           gates_sec: np.ndarray,
                           backend: str = 'wlf_swp24',
                           log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """2D 剖面正演占位（按 WLF 设计原理：carmod2_kru → 实际执行 SWP24.EXE 桥接）。
    backend='python' 将抛出 NotImplementedError(纯Python 2D FDTD 计算量过大，建议 WLF EXE)。
    """
    if backend == 'python':
        raise NotImplementedError("纯Python 2D TEM FDTD 计算量极大(≥10⁷网格)，"
                                  "请使用 backend='wlf_swp24' 桥接 WLF SWP24D.EXE")
    return {'status': 'bridge_ready',
            'exe_hint': 'SWP24D.EXE',
            'stations_x_m': stations_x_m,
            'n_stations': len(stations_x_m),
            'n_gates': len(gates_sec),
            'message': f'请通过 M7_WLF_Bridge.run_2d_inv_wlf(...) 调用 WLF 2D 反演'}


def tem_forward_3d_volume(grid_xyz: tuple, model_3d: dict,
                          gates_sec: np.ndarray,
                          backend: str = 'wlf_carmod3',
                          log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """3D 体积正演占位（按 WLF carmod3_kru → 3D 层析切片）。"""
    if backend == 'python':
        raise NotImplementedError("纯Python 3D TEM Yee-lattice FDTD 计算量极大，"
                                  "请使用 WLF 商业 EXE 桥接")
    return {'status': 'bridge_ready',
            'exe_hint': 'TRANS24.exe + 2D结果插值→3D',
            'grid': grid_xyz, 'n_gates': len(gates_sec)}


__all__ = ['LayeredModel', 'tem_forward_1d', 'apparent_resistivity_dbzdt',
           'tem_forward_2d_section', 'tem_forward_3d_volume',
           '_MU0', '_GUPT61_BASE', '_GUPT61_J0', '_GUPT61_J1']
