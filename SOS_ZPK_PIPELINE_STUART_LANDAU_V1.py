"""
Stuart-Landau PPO Hybrid SOS/ZPK -- first hybrid_timesteps version.

Key semantic copied from the original HybridEnv:
    - one env.step(action) == one physical/solver step
    - PPO is trained with HYBRID_TIMESTEPS = 60000 physical steps
    - a new effective delta_theta is accepted only every DK_HOLD_STEPS
    - intermediate PPO actions are ignored and the last effective delta_theta is reused
    - theta_base is updated only at the end of the physical episode

Linear HybridEnv analogy:
    K_trial = K_global + dK
    u = -K_trial @ x

SOS/ZPK version:
    theta_trial = theta_base + delta_theta
    K_trial = DynamicControllerSOS(theta_trial)
    y = Csens @ xp
    u = K_trial.sample(y)
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import root

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    # Minimal fallback only for local smoke tests when gymnasium is not installed.
    # Stable-Baselines3 training still requires real gymnasium.
    class _FallbackEnv:
        metadata = {}

        def __init__(self, *args, **kwargs):
            self.np_random = np.random.default_rng()

        def reset(self, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)
            return None, {}

    class _FallbackBox:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
            self.shape = tuple(shape if shape is not None else self.low.shape)
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class _FallbackSpaces:
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    PPO = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ================================================================
# 0) Global defaults
# ================================================================

EPS = 1e-12
# Stuart-Landau parameters
OMEGA = 2.0
BETA = 0.5
MU_TARGET = 3

# Time and episode setup: same semantics as HybridEnv.
DT = 2e-2
TS = 2e-2
MAX_STEPS = 50
HYBRID_TIMESTEPS =40000
PPO_N_STEPS = 1024 #Bien con 512,1024,2048
BATCH_SIZE = 64

# Effective controller proposal hold duration.
DK_HOLD_STEPS = 50

# Hybrid update learning rates copied from HybridEnv.
LR_GOOD = 0.3
LR_BAD = 0.01

# Plant, sensing, saturation.
X0 = np.array([0.6, 0.1], dtype=float)
CSENS = np.array([[0, 1]], dtype=float)  # y = x2 by default
STATE_MAX = 2 * np.sqrt(MU_TARGET)
F_MAX = 15.0

# Costs and scoring.
Q_Y = 5
Q_X = 0
R_U = 5e-2
FAIL_PENALTY = 100.0
K_EVAL_STEPS = 500

SIM_ROLLOUT_K0_CSV = "stuart_landau_initial_closed_loop_K0_signals.csv"
SIM_ROLLOUT_KPPO_CSV = "stuart_landau_final_closed_loop_KPPO_signals.csv"
FINAL_CONTROLLER_KPPO = "stuart_landau_final_controller_KPPO.mat"
MODEL = "stuart_landau_PPO_hybrid"
# ================================================================
# 1) Utilities
# ================================================================


def as_1d(x: np.ndarray | List[float] | float, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def theta_to_string(theta: Optional[np.ndarray], precision: int = 8) -> str:
    if theta is None:
        return ""
    arr = np.asarray(theta, dtype=float).reshape(-1)
    return ";".join(f"{v:.{precision}g}" for v in arr)




def print_timestep_summary(
    max_steps: int = MAX_STEPS,
    hybrid_timesteps: int = HYBRID_TIMESTEPS,
    dk_hold_steps: int = DK_HOLD_STEPS,
) -> None:
    effective_episodes = hybrid_timesteps / max_steps
    proposals_per_episode = max_steps / dk_hold_steps
    proposals_total = hybrid_timesteps / dk_hold_steps
    print("=" * 72)
    print("Hybrid PPO/SOS-ZPK timestep semantics")
    print("=" * 72)
    print(f"MAX_STEPS                         = {max_steps}")
    print(f"HYBRID_TIMESTEPS                  = {hybrid_timesteps}")
    print(f"DK_HOLD_STEPS                     = {dk_hold_steps}")
    print(f"effective_episodes                = {effective_episodes:.3f}")
    print(f"effective_theta_proposals/episode = {proposals_per_episode:.3f}")
    print(f"effective_theta_proposals_total   = {proposals_total:.3f}")
    print("NOTE: PPO still sees total_timesteps = HYBRID_TIMESTEPS physical env steps.")
    print("      Distinct theta proposals are only accepted every DK_HOLD_STEPS.")
    print("=" * 72)


# ================================================================
# 2) Stuart-Landau plant
# ================================================================


class StuartLandauPlant:
    """
    Nonlinear Stuart-Landau plant.

    State:
        xp = [x1, x2]

    Dynamics:
        r2 = x1^2 + x2^2
        dx1 = mu*x1 - omega*x2 - r2*(x1 - beta*x2) + u
        dx2 = omega*x1 + mu*x2 - r2*(x2 + beta*x1)

    This class only knows the physical plant. It does not know PPO.
    """

    def __init__(
        self,
        dt: float = DT,
        Ts: float = TS,
        nu: int = 1,
        mu: float = MU_TARGET,
        omega: float = OMEGA,
        beta: float = BETA,
    ):
        self.nxp = 2
        self.nu = int(nu)
        if self.nu != 1:
            raise ValueError("This first version supports one real actuator input only.")
        self.dt = float(dt)
        self.Ts = float(Ts)
        self.mu = float(mu)
        self.omega = float(omega)
        self.beta = float(beta)
        if self.dt <= 0.0 or self.Ts <= 0.0:
            raise ValueError("dt and Ts must be > 0.")
        self.xp = np.zeros(self.nxp, dtype=float)

    def reset(self, xp0: Optional[np.ndarray] = None) -> np.ndarray:
        self.xp = np.array(X0 if xp0 is None else xp0, dtype=float).reshape(-1).copy()
        if self.xp.shape != (self.nxp,):
            raise ValueError(f"xp0 must have shape ({self.nxp},).")
        return self.xp.copy()

    def derivatives(self, xp: np.ndarray, u: np.ndarray) -> np.ndarray:
        xp = np.asarray(xp, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if xp.shape != (self.nxp,):
            raise ValueError(f"xp must have shape ({self.nxp},).")
        if u.shape != (self.nu,):
            raise ValueError(f"u must have shape ({self.nu},).")

        x1, x2 = xp
        force = float(u[0])
        r2 = x1 * x1 + x2 * x2
        dx1 = self.mu * x1 - self.omega * x2 - r2 * (x1 - self.beta * x2) + force
        dx2 = self.omega * x1 + self.mu * x2 - r2 * (x2 + self.beta * x1)
        return np.array([dx1, dx2], dtype=float)

    def step_crank_nicolson(self, u: np.ndarray, h: Optional[float] = None) -> np.ndarray:
        """One implicit trapezoidal / Crank-Nicolson step with constant ZOH input u."""
        h = self.dt if h is None else float(h)
        if h <= 0.0:
            raise ValueError("Integration step h must be > 0.")

        xp_now = self.xp.copy()
        f_now = self.derivatives(xp_now, u)

        def residual(xp_new: np.ndarray) -> np.ndarray:
            f_new = self.derivatives(xp_new, u)
            return xp_new - xp_now - 0.5 * h * (f_now + f_new)

        xp_guess = xp_now + h * f_now
        try:
            sol = root(residual, xp_guess, method="hybr", options={"maxfev": 60})
            if sol.success:
                self.xp = np.asarray(sol.x, dtype=float)
                return self.xp.copy()
        except Exception:
            pass

        # Fallback: explicit trapezoidal / Heun.
        f_pred = self.derivatives(xp_guess, u)
        self.xp = xp_now + 0.5 * h * (f_now + f_pred)
        return self.xp.copy()


# ================================================================
# 3) Dynamic SOS/ZPK controller
# ================================================================


class DynamicControllerSOS:
    """
    Discrete dynamic controller implemented as cascade of first/second-order sections.

    The structure pole_order / zero_order is fixed. PPO changes values of poles,
    zeros and global gain, but not the section topology.

    sample(y):
        1. computes u[k] from current controller states and y[k]
        2. updates internal controller states xK[k+1]
        3. stores u_hold for ZOH application to plant
    """

    def __init__(
        self,
        P_sections=None,
        Z_sections=None,
        pole_order=None,
        zero_order=None,
        Kg: float = 1.0,
        Ts: Optional[float] = None,
        tol: float = 1e-10,
        Csens: np.ndarray = CSENS,
        reset_states: bool = True,
    ):
        self.mode = "sos_rl"
        self.implementation = "sos_zpk_variable_order"
        self.ny = int(np.shape(Csens)[0])
        self.nu = 1
        self.Ts = TS if Ts is None else float(Ts)
        self.tol = float(tol)
        self.Kg = float(Kg)
        if self.ny != 1:
            raise ValueError("This first DynamicControllerSOS version supports SISO measurement only.")

        self.P_sections = None
        self.Z_sections = None
        self.pole_order = None
        self.zero_order = None
        self.local_sections: List[Dict[str, Any]] = []
        self.A_tot = None
        self.B_tot = None
        self.C_tot = None
        self.D_tot = None
        self.u_hold = np.zeros(self.nu, dtype=float)

        if P_sections is not None or Z_sections is not None:
            if P_sections is None or Z_sections is None:
                raise ValueError("Pass P_sections and Z_sections together.")
            if pole_order is None or zero_order is None:
                raise ValueError("Pass pole_order and zero_order.")
            self.set_sections(
                P_sections=P_sections,
                Z_sections=Z_sections,
                pole_order=pole_order,
                zero_order=zero_order,
                Kg=Kg,
                reset_states=reset_states,
            )

    def _is_nan(self, value) -> bool:
        return bool(np.isnan(np.real(value)) or np.isnan(np.imag(value)))

    def _real_scalar(self, value, name: str = "coef") -> float:
        if self._is_nan(value):
            raise ValueError(f"{name} is NaN, expected finite real value.")
        if abs(np.imag(value)) > self.tol:
            raise ValueError(f"{name} is not real within tolerance: {value}")
        return float(np.real(value))

    def _validate_pair(self, pair, name: str = "pair") -> None:
        x1, x2 = pair
        if self._is_nan(x1) or self._is_nan(x2):
            raise ValueError(f"{name} contains NaN, expected full pair: {pair}")
        if abs(np.imag(x1)) <= self.tol and abs(np.imag(x2)) <= self.tol:
            return
        if abs(x2 - np.conjugate(x1)) > self.tol:
            raise ValueError(f"{name} is not a valid conjugate pair: {pair}")

    def _normalize_sections_input(self, sections, name: str = "sections") -> np.ndarray:
        arr = np.asarray(sections)
        if arr.ndim == 1:
            if arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            else:
                arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D.")
        if arr.shape[1] == 1:
            nan_col = np.full((arr.shape[0], 1), np.nan, dtype=arr.dtype)
            arr = np.hstack([arr, nan_col])
        if arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape Lx2 or Lx1.")
        return arr.astype(complex)

    def _normalize_order_input(self, order, L: int, name: str = "order") -> np.ndarray:
        arr = np.asarray(order).reshape(-1).astype(int)
        if arr.size != L:
            raise ValueError(f"{name} must have length L={L}.")
        return arr

    def _validate_structure(self, P_sections, Z_sections, pole_order, zero_order) -> None:
        L = P_sections.shape[0]
        if Z_sections.shape[0] != L:
            raise ValueError("P_sections and Z_sections must have same number of sections.")
        for i in range(L):
            po = int(pole_order[i])
            zo = int(zero_order[i])
            if po not in (1, 2):
                raise ValueError(f"pole_order[{i}] must be 1 or 2.")
            if zo not in (0, 1, 2):
                raise ValueError(f"zero_order[{i}] must be 0, 1, or 2.")
            if zo > po:
                raise ValueError(f"Section {i}: zero_order={zo} > pole_order={po}.")

            p1, p2 = P_sections[i, :]
            z1, z2 = Z_sections[i, :]
            if po == 2:
                self._validate_pair((p1, p2), name=f"P_sections[{i}]")
            else:
                if self._is_nan(p1) or not self._is_nan(p2):
                    raise ValueError(f"Section {i}: pole_order=1 expects [p1, NaN].")
                self._real_scalar(p1, name=f"p1 section {i}")

            if zo == 2:
                self._validate_pair((z1, z2), name=f"Z_sections[{i}]")
            elif zo == 1:
                if self._is_nan(z1) or not self._is_nan(z2):
                    raise ValueError(f"Section {i}: zero_order=1 expects [z1, NaN].")
                self._real_scalar(z1, name=f"z1 section {i}")
            else:
                if not (self._is_nan(z1) and self._is_nan(z2)):
                    raise ValueError(f"Section {i}: zero_order=0 expects [NaN, NaN].")

    def _den_coeffs(self, poles_pair, pole_order: int):
        if int(pole_order) == 2:
            p1, p2 = poles_pair
            a1 = self._real_scalar(-(p1 + p2), name="a1")
            a2 = self._real_scalar(p1 * p2, name="a2")
            return a1, a2
        if int(pole_order) == 1:
            p1 = poles_pair[0]
            a1 = self._real_scalar(-p1, name="a1")
            return a1, None
        raise ValueError("pole_order must be 1 or 2.")

    def _num_coeffs(self, zeros_pair, pole_order: int, zero_order: int):
        pole_order = int(pole_order)
        zero_order = int(zero_order)
        z1, z2 = zeros_pair
        if pole_order == 2:
            if zero_order == 2:
                beta0, beta1, beta2 = 1.0, -(z1 + z2), z1 * z2
            elif zero_order == 1:
                beta0, beta1, beta2 = 0.0, 1.0, -z1
            elif zero_order == 0:
                beta0, beta1, beta2 = 0.0, 0.0, 1.0
            else:
                raise ValueError("Invalid zero_order for pole_order=2.")
            return (
                self._real_scalar(beta0, "beta0"),
                self._real_scalar(beta1, "beta1"),
                self._real_scalar(beta2, "beta2"),
            )
        if pole_order == 1:
            if zero_order == 1:
                beta0, beta1 = 1.0, -z1
            elif zero_order == 0:
                beta0, beta1 = 0.0, 1.0
            else:
                raise ValueError("For pole_order=1, zero_order must be 0 or 1.")
            return self._real_scalar(beta0, "beta0"), self._real_scalar(beta1, "beta1")
        raise ValueError("pole_order must be 1 or 2.")

    def _section_to_ss(self, poles_pair, zeros_pair, pole_order: int, zero_order: int) -> Dict[str, Any]:
        pole_order = int(pole_order)
        zero_order = int(zero_order)
        if pole_order == 2:
            a1, a2 = self._den_coeffs(poles_pair, 2)
            beta0, beta1, beta2 = self._num_coeffs(zeros_pair, 2, zero_order)
            A = np.array([[-a1, -a2], [1.0, 0.0]], dtype=float)
            B = np.array([[1.0], [0.0]], dtype=float)
            D = np.array([[beta0]], dtype=float)
            C = np.array([[beta1 - beta0 * a1, beta2 - beta0 * a2]], dtype=float)
            x = np.zeros((2, 1), dtype=float)
            return dict(
                pole_order=pole_order,
                zero_order=zero_order,
                poles=tuple(poles_pair),
                zeros=tuple(zeros_pair),
                A=A,
                B=B,
                C=C,
                D=D,
                x=x,
            )
        if pole_order == 1:
            a1, _ = self._den_coeffs(poles_pair, 1)
            beta0, beta1 = self._num_coeffs(zeros_pair, 1, zero_order)
            A = np.array([[-a1]], dtype=float)
            B = np.array([[1.0]], dtype=float)
            D = np.array([[beta0]], dtype=float)
            C = np.array([[beta1 - beta0 * a1]], dtype=float)
            x = np.zeros((1, 1), dtype=float)
            return dict(
                pole_order=pole_order,
                zero_order=zero_order,
                poles=tuple(poles_pair),
                zeros=tuple(zeros_pair),
                A=A,
                B=B,
                C=C,
                D=D,
                x=x,
            )
        raise ValueError("pole_order must be 1 or 2.")

    def set_sections(
        self,
        P_sections,
        Z_sections,
        pole_order,
        zero_order,
        Kg: Optional[float] = None,
        reset_states: bool = True,
    ) -> None:
        
        P_sections = self._normalize_sections_input(P_sections, "P_sections")
        Z_sections = self._normalize_sections_input(Z_sections, "Z_sections")
        L = P_sections.shape[0]
        pole_order = self._normalize_order_input(pole_order, L=L, name="pole_order")
        zero_order = self._normalize_order_input(zero_order, L=L, name="zero_order")
        self._validate_structure(P_sections, Z_sections, pole_order, zero_order)
        
        if Kg is not None:
            self.Kg = float(Kg)
            
        self.P_sections = P_sections.copy()
        self.Z_sections = Z_sections.copy()
        self.pole_order = pole_order.copy()
        self.zero_order = zero_order.copy()
        self.local_sections = []
        
        for i in range(L):
            sec = self._section_to_ss(
                poles_pair=self.P_sections[i, :],
                zeros_pair=self.Z_sections[i, :],
                pole_order=self.pole_order[i],
                zero_order=self.zero_order[i],
            )
            self.local_sections.append(sec)
        self.A_tot = self.B_tot = self.C_tot = self.D_tot = None
        if reset_states:
            self.reset()

    def update_sections(self, P_sections=None, Z_sections=None, Kg=None, reset_states: bool = False) -> None:
        if self.P_sections is None or self.Z_sections is None:
            raise RuntimeError("No sections initialized.")
        self.set_sections(
            P_sections=self.P_sections if P_sections is None else P_sections,
            Z_sections=self.Z_sections if Z_sections is None else Z_sections,
            pole_order=self.pole_order,
            zero_order=self.zero_order,
            Kg=self.Kg if Kg is None else Kg,
            reset_states=reset_states,
        )

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        nK = int(np.sum(self.pole_order)) if self.pole_order is not None else 0
        if x0 is None:
            for sec in self.local_sections:
                n = sec["A"].shape[0]
                sec["x"] = np.zeros((n, 1), dtype=float)
            self.u_hold = np.zeros(self.nu, dtype=float)
            return
        x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
        
        if x0.shape[0] != nK:
            raise ValueError(f"x0 must have dimension {nK} x 1.")
        idx = 0
        
        for sec in self.local_sections:
            n = sec["A"].shape[0]
            sec["x"] = x0[idx:idx + n, :].copy()
            idx += n
        self.u_hold = np.zeros(self.nu, dtype=float)

    def output(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape != (self.ny,):
            raise ValueError(f"y must have shape ({self.ny},).")
        v = float(y[0])
        for sec in self.local_sections:
            A, B, C, D, x = sec["A"], sec["B"], sec["C"], sec["D"], sec["x"]
            v_out = float((C @ x + D * v).squeeze())
            sec["x"] = A @ x + B * v
            v = v_out
        return np.array([self.Kg * v], dtype=float)

    def sample(self, y: np.ndarray) -> np.ndarray:
        self.u_hold = self.output(y)
        return self.u_hold.copy()

    def get_states(self) -> np.ndarray:
        if len(self.local_sections) == 0:
            return np.zeros((0, 1), dtype=float)
        return np.vstack([sec["x"] for sec in self.local_sections])

    def build_global_ss(self, apply_global_gain: bool = True):
        """Build global discrete SS of the cascade. Useful for export/debug."""
        L = len(self.local_sections)
        dims = [sec["A"].shape[0] for sec in self.local_sections]
        offsets = np.cumsum([0] + dims)
        n = int(offsets[-1])
        A_tot = np.zeros((n, n), dtype=float)
        B_tot = np.zeros((n, 1), dtype=float)
        C_tot = np.zeros((1, n), dtype=float)
        D_values = [float(sec["D"].squeeze()) for sec in self.local_sections]

        for i, sec_i in enumerate(self.local_sections):
            Ai, Bi = sec_i["A"], sec_i["B"]
            ii = slice(offsets[i], offsets[i + 1])
            A_tot[ii, ii] = Ai
            prod_before = 1.0
            for m in range(i):
                prod_before *= D_values[m]
            B_tot[ii, :] = Bi * prod_before
            for j in range(i):
                sec_j = self.local_sections[j]
                jj = slice(offsets[j], offsets[j + 1])
                prod_middle = 1.0
                for m in range(j + 1, i):
                    prod_middle *= D_values[m]
                A_tot[ii, jj] = Bi @ sec_j["C"] * prod_middle

        for j, sec_j in enumerate(self.local_sections):
            jj = slice(offsets[j], offsets[j + 1])
            prod_after = 1.0
            for m in range(j + 1, L):
                prod_after *= D_values[m]
            C_tot[:, jj] = sec_j["C"] * prod_after

        D_tot = np.array([[np.prod(D_values) if L > 0 else 1.0]], dtype=float)
        if apply_global_gain:
            C_tot = self.Kg * C_tot
            D_tot = self.Kg * D_tot
        self.A_tot, self.B_tot, self.C_tot, self.D_tot = A_tot, B_tot, C_tot, D_tot
        return A_tot, B_tot, C_tot, D_tot

    def summary(self) -> Dict[str, Any]:
        return {
            "implementation": self.implementation,
            "L": len(self.local_sections),
            "state_dimension": int(np.sum(self.pole_order)) if self.pole_order is not None else 0,
            "Kg": self.Kg,
            "Ts": self.Ts,
            "pole_order": None if self.pole_order is None else self.pole_order.tolist(),
            "zero_order": None if self.zero_order is None else self.zero_order.tolist(),
        }

    @classmethod
    def from_mat(
        cls,
        filename: str,
        p_key: str = "P_sections",
        z_key: str = "Z_sections",
        pole_order_key: str = "pole_order",
        zero_order_key: str = "zero_order",
        kg_key: str = "Kg",
        ts_key: str = "Ts",
        tol: float = 1e-10,
        Csens: np.ndarray = CSENS,
    ) -> "DynamicControllerSOS":
        
        data = loadmat(filename)
        P_sections = np.asarray(data[p_key])
        Z_sections = np.asarray(data[z_key])
        pole_order = np.asarray(data[pole_order_key]).reshape(-1)
        zero_order = np.asarray(data[zero_order_key]).reshape(-1)
        Kg = float(np.ravel(data[kg_key])[0]) if kg_key in data else 1.0
        Ts = float(np.ravel(data[ts_key])[0]) if ts_key in data else TS
        return cls(
            P_sections=P_sections,
            Z_sections=Z_sections,
            pole_order=pole_order,
            zero_order=zero_order,
            Kg=Kg,
            Ts=Ts,
            tol=tol,
            Csens=Csens,
            reset_states=True,
        )


# ================================================================
# 4) SOS/ZPK topology and action mapper
# ================================================================


def _has_nan_complex(x) -> bool:
    x = np.asarray(x)
    return bool(np.any(np.isnan(np.real(x))) or np.any(np.isnan(np.imag(x))))


def _is_real_scalar(x, tol: float = 1e-10) -> bool:
    if _has_nan_complex(x):
        return False
    return bool(abs(np.imag(x)) <= tol)


def _is_conjugate_pair(x1, x2, tol: float = 1e-10) -> bool:
    if _has_nan_complex(x1) or _has_nan_complex(x2):
        return False
    return bool(abs(x2 - np.conjugate(x1)) <= tol)


@dataclass
class ParamBlock:
    kind: str
    section: Optional[int]
    slots: Optional[List[int]]
    dim: int
    names: List[str]


class ZPKTopology:
    """Detect fixed ZPK topology and build a real parameter vector theta."""

    def __init__(
        self,
        P_sections,
        Z_sections,
        pole_order,
        zero_order,
        Kg: float,
        Ts: Optional[float] = None,
        tol: float = 1e-10,
    ):
        self.tol = float(tol)
        self.P_template = self._normalize_sections_input(P_sections, name="P_sections")
        self.Z_template = self._normalize_sections_input(Z_sections, name="Z_sections")
        self.L = int(self.P_template.shape[0])
        self.pole_order = np.asarray(pole_order, dtype=int).reshape(-1)
        self.zero_order = np.asarray(zero_order, dtype=int).reshape(-1)
        if self.pole_order.size != self.L or self.zero_order.size != self.L:
            raise ValueError("pole_order and zero_order must have length L.")
        self.Kg0 = float(Kg)
        self.Kg_sign = 1.0 if abs(self.Kg0) < EPS else float(np.sign(self.Kg0))
        self.Ts = TS if Ts is None else float(Ts)
        self.param_blocks: List[ParamBlock] = []
        self.detect_blocks()

    @staticmethod
    def _normalize_sections_input(sections, name: str = "sections") -> np.ndarray:
        arr = np.asarray(sections)
        if arr.ndim == 1:
            if arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            else:
                arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D.")
        if arr.shape[1] == 1:
            nan_col = np.full((arr.shape[0], 1), np.nan, dtype=arr.dtype)
            arr = np.hstack([arr, nan_col])
        if arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape Lx2 or Lx1.")
        return arr.astype(complex)

    @classmethod
    def from_controller(cls, controller: DynamicControllerSOS, tol: float = 1e-10) -> "ZPKTopology":
        if controller.P_sections is None or controller.Z_sections is None:
            raise ValueError("Controller must already contain P_sections and Z_sections.")
        return cls(
            P_sections=controller.P_sections,
            Z_sections=controller.Z_sections,
            pole_order=controller.pole_order,
            zero_order=controller.zero_order,
            Kg=controller.Kg,
            Ts=controller.Ts,
            tol=tol,
        )

    @property
    def action_dim(self) -> int:
        return int(sum(block.dim for block in self.param_blocks))

    def param_names(self) -> List[str]:
        names: List[str] = []
        for block in self.param_blocks:
            names.extend(block.names)
        return names

    def detect_blocks(self) -> None:
        self.param_blocks = []
        for l in range(self.L):
            po = int(self.pole_order[l])
            zo = int(self.zero_order[l])
            p1, p2 = self.P_template[l, :]
            z1, z2 = self.Z_template[l, :]

            if po == 1:
                if not _is_real_scalar(p1, self.tol):
                    raise ValueError(f"Section {l}: pole_order=1 requires one real pole.")
                self.param_blocks.append(ParamBlock("pole_real", l, [0], 1, [f"p{l}_real"]))
            elif po == 2:
                if _is_real_scalar(p1, self.tol) and _is_real_scalar(p2, self.tol):
                    self.param_blocks.append(ParamBlock("pole_real_pair", l, [0, 1], 2, [f"p{l}_1_real", f"p{l}_2_real"]))
                elif _is_conjugate_pair(p1, p2, self.tol):
                    self.param_blocks.append(ParamBlock("pole_complex_pair", l, [0, 1], 2, [f"rp{l}", f"thetap{l}"]))
                else:
                    raise ValueError(f"Section {l}: invalid pole pair {p1}, {p2}.")
            else:
                raise ValueError(f"Section {l}: pole_order must be 1 or 2.")

            if zo == 0:
                pass
            elif zo == 1:
                if not _is_real_scalar(z1, self.tol):
                    raise ValueError(f"Section {l}: zero_order=1 requires one real zero.")
                self.param_blocks.append(ParamBlock("zero_real", l, [0], 1, [f"z{l}_real"]))
            elif zo == 2:
                if _is_real_scalar(z1, self.tol) and _is_real_scalar(z2, self.tol):
                    self.param_blocks.append(ParamBlock("zero_real_pair", l, [0, 1], 2, [f"z{l}_1_real", f"z{l}_2_real"]))
                elif _is_conjugate_pair(z1, z2, self.tol):
                    self.param_blocks.append(ParamBlock("zero_complex_pair", l, [0, 1], 2, [f"rz{l}", f"thetaz{l}"]))
                else:
                    raise ValueError(f"Section {l}: invalid zero pair {z1}, {z2}.")
            else:
                raise ValueError(f"Section {l}: zero_order must be 0, 1, or 2.")

        self.param_blocks.append(ParamBlock("gain_log", None, None, 1, ["log_abs_Kg"]))

    def summary(self) -> Dict[str, Any]:
        return {
            "L": self.L,
            "action_dim": self.action_dim,
            "param_names": self.param_names(),
            "blocks": [block.__dict__ for block in self.param_blocks],
            "pole_order": self.pole_order.tolist(),
            "zero_order": self.zero_order.tolist(),
            "Kg0": self.Kg0,
            "Ts": self.Ts,
        }


class ZPKActionMapper:
    """Map PPO actions to projected structured ZPK perturbations."""

    def __init__(
        self,
        topology: ZPKTopology,
        scale_p_real: float = 0.01,
        scale_p_radius: float = 0.01,
        scale_p_angle: float = 0.01,
        scale_z_real: float = 0.01,
        scale_z_radius: float = 0.01,
        scale_z_angle: float = 0.02,
        scale_logKg: float = 0.2,
        r_p_min: float = 1e-5,
        r_p_max: float = 0.98,
        r_z_max: float = 2.0,
        logKg_min: float = -10.0,
        logKg_max: float = 10.0,    
    ):
        self.topology = topology
        self.scale_p_real = float(scale_p_real)
        self.scale_p_radius = float(scale_p_radius)
        self.scale_p_angle = float(scale_p_angle)
        self.scale_z_real = float(scale_z_real)
        self.scale_z_radius = float(scale_z_radius)
        self.scale_z_angle = float(scale_z_angle)
        self.scale_logKg = float(scale_logKg)
        self.r_p_min = float(r_p_min)
        self.r_p_max = float(r_p_max)
        self.r_z_max = float(r_z_max)
        self.logKg_min = float(logKg_min)
        self.logKg_max = float(logKg_max)
        if not (0.0 < self.r_p_min < self.r_p_max < 1.0):
            raise ValueError("Need 0 < r_p_min < r_p_max < 1 for discrete pole stability.")
        if self.r_z_max <= 0.0:
            raise ValueError("r_z_max must be > 0.")

    @property
    def action_dim(self) -> int:
        return self.topology.action_dim

    def zpk_to_theta(self, P_sections, Z_sections, Kg: float) -> np.ndarray:
        P_sections = self.topology._normalize_sections_input(P_sections, "P_sections")
        Z_sections = self.topology._normalize_sections_input(Z_sections, "Z_sections")
        theta: List[float] = []
        for block in self.topology.param_blocks:
            l = block.section
            if block.kind == "pole_real":
                theta.append(float(np.real(P_sections[l, 0])))
            elif block.kind == "pole_real_pair":
                theta.extend([float(np.real(P_sections[l, 0])), float(np.real(P_sections[l, 1]))])
            elif block.kind == "pole_complex_pair":
                p = P_sections[l, 0]
                theta.extend([float(np.abs(p)), float(np.angle(p))])
            elif block.kind == "zero_real":
                theta.append(float(np.real(Z_sections[l, 0])))
            elif block.kind == "zero_real_pair":
                theta.extend([float(np.real(Z_sections[l, 0])), float(np.real(Z_sections[l, 1]))])
            elif block.kind == "zero_complex_pair":
                z = Z_sections[l, 0]
                theta.extend([float(np.abs(z)), float(np.angle(z))])
            elif block.kind == "gain_log":
                theta.append(float(np.log(abs(float(Kg)) + EPS)))
            else:
                raise ValueError(f"Unknown block kind: {block.kind}")
        return self.project_theta(np.asarray(theta, dtype=np.float64))

    def theta_to_zpk(self, theta: np.ndarray):
        theta = self.project_theta(theta)
        P_new = self.topology.P_template.copy()
        Z_new = self.topology.Z_template.copy()
        Kg_new = self.topology.Kg0
        idx = 0
        for block in self.topology.param_blocks:
            l = block.section
            if block.kind == "pole_real":
                P_new[l, 0], P_new[l, 1] = theta[idx], np.nan
                idx += 1
            elif block.kind == "pole_real_pair":
                P_new[l, 0], P_new[l, 1] = theta[idx], theta[idx + 1]
                idx += 2
            elif block.kind == "pole_complex_pair":
                r, angle = theta[idx], theta[idx + 1]
                p = r * np.exp(1j * angle)
                P_new[l, 0], P_new[l, 1] = p, np.conjugate(p)
                idx += 2
            elif block.kind == "zero_real":
                Z_new[l, 0], Z_new[l, 1] = theta[idx], np.nan
                idx += 1
            elif block.kind == "zero_real_pair":
                Z_new[l, 0], Z_new[l, 1] = theta[idx], theta[idx + 1]
                idx += 2
            elif block.kind == "zero_complex_pair":
                r, angle = theta[idx], theta[idx + 1]
                z = r * np.exp(1j * angle)
                Z_new[l, 0], Z_new[l, 1] = z, np.conjugate(z)
                idx += 2
            elif block.kind == "gain_log":
                Kg_new = self.topology.Kg_sign * float(np.exp(theta[idx]))
                idx += 1
            else:
                raise ValueError(f"Unknown block kind: {block.kind}")
        return P_new, Z_new, float(Kg_new)

    def action_to_delta_theta(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {action.shape}.")
        action = np.clip(action, -1.0, 1.0)
        delta = np.zeros_like(action, dtype=np.float64)
        idx = 0
        for block in self.topology.param_blocks:
            if block.kind == "pole_real":
                delta[idx] = self.scale_p_real * action[idx]
                idx += 1
            elif block.kind == "pole_real_pair":
                delta[idx] = self.scale_p_real * action[idx]
                delta[idx + 1] = self.scale_p_real * action[idx + 1]
                idx += 2
            elif block.kind == "pole_complex_pair":
                delta[idx] = self.scale_p_radius * action[idx]
                delta[idx + 1] = self.scale_p_angle * action[idx + 1]
                idx += 2
            elif block.kind == "zero_real":
                delta[idx] = self.scale_z_real * action[idx]
                idx += 1
            elif block.kind == "zero_real_pair":
                delta[idx] = self.scale_z_real * action[idx]
                delta[idx + 1] = self.scale_z_real * action[idx + 1]
                idx += 2
            elif block.kind == "zero_complex_pair":
                delta[idx] = self.scale_z_radius * action[idx]
                delta[idx + 1] = self.scale_z_angle * action[idx + 1]
                idx += 2
            elif block.kind == "gain_log":
                delta[idx] = self.scale_logKg * action[idx]
                idx += 1
        return delta

    def project_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1).copy()
        if theta.shape != (self.action_dim,):
            raise ValueError(f"theta must have shape ({self.action_dim},), got {theta.shape}.")
        idx = 0
        for block in self.topology.param_blocks:
            if block.kind == "pole_real":
                theta[idx] = np.clip(theta[idx], -self.r_p_max, self.r_p_max)
                idx += 1
            elif block.kind == "pole_real_pair":
                theta[idx] = np.clip(theta[idx], -self.r_p_max, self.r_p_max)
                theta[idx + 1] = np.clip(theta[idx + 1], -self.r_p_max, self.r_p_max)
                idx += 2
            elif block.kind == "pole_complex_pair":
                theta[idx] = np.clip(theta[idx], self.r_p_min, self.r_p_max)
                theta[idx + 1] = wrap_angle(theta[idx + 1])
                idx += 2
            elif block.kind == "zero_real":
                theta[idx] = np.clip(theta[idx], -self.r_z_max, self.r_z_max)
                idx += 1
            elif block.kind == "zero_real_pair":
                theta[idx] = np.clip(theta[idx], -self.r_z_max, self.r_z_max)
                theta[idx + 1] = np.clip(theta[idx + 1], -self.r_z_max, self.r_z_max)
                idx += 2
            elif block.kind == "zero_complex_pair":
                theta[idx] = np.clip(theta[idx], 0.0, self.r_z_max)
                theta[idx + 1] = wrap_angle(theta[idx + 1])
                idx += 2
            elif block.kind == "gain_log":
                theta[idx] = np.clip(theta[idx], self.logKg_min, self.logKg_max)
                idx += 1
        return theta

    def normalize_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = self.project_theta(theta)
        out = np.zeros_like(theta, dtype=np.float64)
        idx = 0
        for block in self.topology.param_blocks:
            if block.kind == "pole_real":
                out[idx] = theta[idx] / max(self.r_p_max, EPS)
                idx += 1
            elif block.kind == "pole_real_pair":
                out[idx] = theta[idx] / max(self.r_p_max, EPS)
                out[idx + 1] = theta[idx + 1] / max(self.r_p_max, EPS)
                idx += 2
            elif block.kind == "pole_complex_pair":
                out[idx] = theta[idx] / max(self.r_p_max, EPS)
                out[idx + 1] = theta[idx + 1] / np.pi
                idx += 2
            elif block.kind == "zero_real":
                out[idx] = theta[idx] / max(self.r_z_max, EPS)
                idx += 1
            elif block.kind == "zero_real_pair":
                out[idx] = theta[idx] / max(self.r_z_max, EPS)
                out[idx + 1] = theta[idx + 1] / max(self.r_z_max, EPS)
                idx += 2
            elif block.kind == "zero_complex_pair":
                out[idx] = theta[idx] / max(self.r_z_max, EPS)
                out[idx + 1] = theta[idx + 1] / np.pi
                idx += 2
            elif block.kind == "gain_log":
                denom = max(abs(self.logKg_min), abs(self.logKg_max), 1.0)
                out[idx] = theta[idx] / denom
                idx += 1
        return np.clip(out, -10.0, 10.0)

    def build_candidate(self, theta_base: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        theta_base = self.project_theta(theta_base)
        delta_raw = self.action_to_delta_theta(action)
        theta_candidate = self.project_theta(theta_base + delta_raw)
        delta_theta = theta_candidate - theta_base
        P_candidate, Z_candidate, Kg_candidate = self.theta_to_zpk(theta_candidate)
        return {
            "theta_candidate": theta_candidate,
            "delta_theta": delta_theta,
            "P_sections": P_candidate,
            "Z_sections": Z_candidate,
            "Kg": Kg_candidate,
        }


# ================================================================
# 5) Closed-loop simulation and scoring helpers
# ================================================================


class ClosedLoopSimulatorSOS:
    """Glue class for plant + SOS controller in discrete ZOH closed-loop."""

    def __init__(
        self,
        plant: StuartLandauPlant,
        controller: DynamicControllerSOS,
        Csens: np.ndarray = CSENS,
        force_limit: float | np.ndarray = F_MAX,
        control_sign: float | np.ndarray = 1.0,
    ):
        self.plant = plant
        self.controller = controller
        self.nxp = self.plant.nxp
        self.Csens = np.asarray(Csens, dtype=float).copy()
        self.force_limit = np.asarray(force_limit, dtype=float).reshape(-1)
        self.control_sign = np.asarray(control_sign, dtype=float).reshape(-1)
        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.Csens.shape != (self.controller.ny, self.nxp):
            raise ValueError(f"Csens must have shape ({self.controller.ny}, {self.nxp}).")
        if self.controller.nu != self.plant.nu:
            raise ValueError("Controller output dimension must match plant input dimension.")
        if self.force_limit.size == 1:
            self.force_limit = np.full(self.plant.nu, float(self.force_limit.item()))
        if self.control_sign.size == 1:
            self.control_sign = np.full(self.plant.nu, float(self.control_sign.item()))

    def sensor_output(self, xp: Optional[np.ndarray] = None) -> np.ndarray:
        xp = self.plant.xp if xp is None else np.asarray(xp, dtype=float).reshape(-1)
        return np.asarray(self.Csens @ xp, dtype=float).reshape(self.controller.ny)

    def sat(self, u_raw: np.ndarray) -> np.ndarray:
        u_raw = np.asarray(u_raw, dtype=float).reshape(-1)
        return np.clip(u_raw, -self.force_limit, self.force_limit)

    def run(self, xp0=None, xk0=None, t_final: float = 10.0) -> Dict[str, np.ndarray]:
        self.plant.reset(xp0)
        self.controller.reset(xk0)
        t = 0.0
        log_t, log_xp, log_xk, log_y, log_u_raw, log_u_sat = [], [], [], [], [], []
        while t < t_final - 1e-15:
            y = self.sensor_output(self.plant.xp)
            u_raw = self.control_sign * self.controller.sample(y)
            u_sat = self.sat(u_raw)
            log_t.append(t)
            log_xp.append(self.plant.xp.copy())
            log_xk.append(self.controller.get_states().copy())
            log_y.append(y.copy())
            log_u_raw.append(u_raw.copy())
            log_u_sat.append(u_sat.copy())
            self.plant.step_crank_nicolson(u_sat, h=self.plant.dt)
            t += self.plant.dt
        return {
            "t": np.asarray(log_t, dtype=float),
            "xp": np.asarray(log_xp, dtype=float),
            "xk": np.asarray(log_xk, dtype=object),
            "y": np.asarray(log_y, dtype=float),
            "u_raw": np.asarray(log_u_raw, dtype=float),
            "u_sat": np.asarray(log_u_sat, dtype=float),
        }


# ================================================================
# 6) Hybrid PPO environment: exact HYBRID_TIMESTEPS semantics
# ================================================================


@dataclass
class HybridSOSConfig:
    dt: float = DT
    Ts: float = TS
    max_steps: int = MAX_STEPS
    hybrid_timesteps: int = HYBRID_TIMESTEPS
    ppo_n_steps: int = PPO_N_STEPS
    batch_size: int = BATCH_SIZE
    dk_hold_steps: int = DK_HOLD_STEPS
    lr_good: float = LR_GOOD
    lr_bad: float = LR_BAD
    alpha_update: float = 0.3
    allow_worse_base_update: bool = False
    beta_softmax: float =5.0 #LO CAMBIÉ, ANTES ESTABA EN 1
    k_eval_steps: int = K_EVAL_STEPS
    q_y: float = Q_Y
    q_x: float = Q_X
    r_u: float = R_U
    fail_penalty: float = FAIL_PENALTY
    state_max: float = STATE_MAX
    force_limit: float = F_MAX
    include_full_state_in_obs: bool = True
    random_ic: bool = False
    random_ic_scale: float = 0.05
    verbose_episode: bool = True


class StuartLandauHybridSOSZPKPPOEnv(gym.Env if gym is not None else object):
    """
    PPO Hybrid environment for SOS/ZPK controller parameters.

    Critical copied behavior from original HybridEnv:
        - one env.step(action_from_ppo) advances exactly one physical step
        - only every dk_hold_steps the new PPO action is accepted
        - in intermediate steps action_from_ppo is ignored
        - current_delta_theta is appended at every physical step, so it is repeated
          dk_hold_steps times in the buffer with different physical rewards
        - theta_base is updated only when the episode terminates/truncates
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: ZPKTopology,
        mapper: ZPKActionMapper,
        config: HybridSOSConfig = HybridSOSConfig(),
        plant_factory: Optional[Callable[[], StuartLandauPlant]] = None,
        Csens: np.ndarray = CSENS,
        xp0: np.ndarray = X0,
    ):
        super().__init__()
        self.topology = topology
        self.mapper = mapper
        self.cfg = config
        self.Csens = np.asarray(Csens, dtype=float).copy()
        self.xp0 = np.asarray(xp0, dtype=float).reshape(-1).copy()
        self.plant_factory = plant_factory or (lambda: StuartLandauPlant(dt=self.cfg.dt, Ts=self.cfg.Ts))

        if self.cfg.max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        if self.cfg.dk_hold_steps <= 0:
            raise ValueError("dk_hold_steps must be > 0.")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.topology.action_dim,),
            dtype=np.float32,
        )

        obs_dim = self.topology.action_dim
        obs_dim += 2 if self.cfg.include_full_state_in_obs else self.Csens.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.theta_base = self.mapper.zpk_to_theta(
            self.topology.P_template,
            self.topology.Z_template,
            self.topology.Kg0,
        )
        self.P_base, self.Z_base, self.Kg_base = self.mapper.theta_to_zpk(self.theta_base)

        # J_base is the reference cost used by the hybrid base-controller update.
        # theta_score is kept for backward-compatible logs, with higher = better.
        metrics_base = self.evaluate_theta(self.theta_base, n_steps=self.cfg.k_eval_steps)
        self.J_base = float(metrics_base["J"]) if metrics_base["valid"] else np.inf
        self.theta_score = -float(self.J_base) if np.isfinite(self.J_base) else -1e12
        self.alpha = 0.0

        self.episode_index = 0
        self.plant: StuartLandauPlant = self.plant_factory()
        self.current_controller: Optional[DynamicControllerSOS] = None
        self.current_delta_theta = np.zeros(self.topology.action_dim, dtype=np.float64)
        self.current_theta_trial = self.theta_base.copy()
        self.current_reward = 0.0
        self.current_cost = 0.0
        self.step_in_episode = 0
        self.global_physical_step = 0

        self.delta_accum: List[np.ndarray] = []
        self.reward_accum: List[float] = []
        self.cost_accum: List[float] = []
        self.logs: List[Dict[str, Any]] = []
        self.history: Dict[str, List[Any]] = {
            "theta_score": [self.theta_score],
            "J_base": [self.J_base],
            "J_probe": [],
            "accepted_update": [],
            "candidate_score": [],
            "lr": [],
            "alpha": [],
            "Kg_base": [self.Kg_base],
            "mean_reward": [],
            "best_reward": [],
            "delta_norm": [],
        }

    def _make_controller_from_theta(self, theta: np.ndarray, reset_states: bool = True) -> DynamicControllerSOS:
        P, Z, Kg = self.mapper.theta_to_zpk(theta)
        ctrl = DynamicControllerSOS(
            P_sections=P,
            Z_sections=Z,
            pole_order=self.topology.pole_order,
            zero_order=self.topology.zero_order,
            Kg=Kg,
            Ts=self.topology.Ts,
            Csens=self.Csens,
            reset_states=reset_states,
        )
        return ctrl

    def _get_measurement(self, xp: Optional[np.ndarray] = None) -> np.ndarray:
        xp = self.plant.xp if xp is None else np.asarray(xp, dtype=float).reshape(-1)
        return np.asarray(self.Csens @ xp, dtype=float).reshape(self.Csens.shape[0])

    def _get_obs(self) -> np.ndarray:
        theta_norm = self.mapper.normalize_theta(self.theta_base)
        if self.cfg.include_full_state_in_obs:
            plant_part = self.plant.xp.astype(np.float64)
        else:
            plant_part = self._get_measurement().astype(np.float64)
        obs = np.concatenate([plant_part, theta_norm])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant = self.plant_factory()
        xp0 = self.xp0.copy()
        if self.cfg.random_ic:
            xp0 = xp0 + self.np_random.normal(0.0, self.cfg.random_ic_scale, size=xp0.shape)
        self.plant.reset(xp0)

        self.current_controller = None
        self.current_delta_theta = np.zeros(self.topology.action_dim, dtype=np.float64)
        self.current_theta_trial = self.theta_base.copy()
        self.current_reward = 0.0
        self.current_cost = 0.0
        self.step_in_episode = 0
        self.delta_accum = []
        self.reward_accum = []
        self.cost_accum = []
        return self._get_obs(), {}


    def _instant_cost(self, xp: np.ndarray, y: np.ndarray, u_sat: np.ndarray) -> float:
        """
        Costo instantáneo por paso físico.
    
        No hay normalización por mu ni por fuerza máxima.
    
        Para un sensor escalar y:
            mean(y^2) = y^2
    
        Para varios sensores/estados:
            mean(...) solo promedia componentes del vector, no tiempo.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        xp = np.asarray(xp, dtype=float).reshape(-1)
        u_sat = np.asarray(u_sat, dtype=float).reshape(-1)
    
        return float(
            self.cfg.q_y * np.mean(np.square(y))
            + self.cfg.q_x * np.mean(np.square(xp))
            + self.cfg.r_u * np.mean(np.square(u_sat))
        )
    
    def _failure(self, xp: np.ndarray, y: np.ndarray, u_sat: np.ndarray) -> Tuple[bool, str]:
            if not (np.all(np.isfinite(xp)) and np.all(np.isfinite(y)) and np.all(np.isfinite(u_sat))):
                return True, "nan_or_inf"
            if float(np.max(np.abs(xp))) > self.cfg.state_max:
                return True, "state_limit"
            return False, "ok"

    def _accept_new_effective_action(self, action_from_ppo: np.ndarray) -> None:
        candidate = self.mapper.build_candidate(self.theta_base, action_from_ppo)
        self.current_delta_theta = candidate["delta_theta"].copy()
        self.current_theta_trial = candidate["theta_candidate"].copy()
        self.current_controller = DynamicControllerSOS(
            P_sections=candidate["P_sections"],
            Z_sections=candidate["Z_sections"],
            pole_order=self.topology.pole_order,
            zero_order=self.topology.zero_order,
            Kg=candidate["Kg"],
            Ts=self.topology.Ts,
            Csens=self.Csens,
            reset_states=True,
        )
        # Critical for SOS/ZPK: new poles/zeros imply a new controller state space.
        self.current_controller.reset(None)

    def step(self, action_from_ppo):
        # 1) Copy original HybridEnv: only accept PPO action every DK_HOLD_STEPS.
        if self.step_in_episode % self.cfg.dk_hold_steps == 0 or self.current_controller is None:
            self._accept_new_effective_action(action_from_ppo)
        assert self.current_controller is not None

        # 2) One physical step only.
        xp_before = self.plant.xp.copy()
        y = self._get_measurement(xp_before)
        u_raw = self.current_controller.sample(y)
        u_sat = np.clip(u_raw, -self.cfg.force_limit, self.cfg.force_limit)
        cost_k = self._instant_cost(xp_before, y, u_sat)
        
        # Reward PPO por step.
        # El reward acumulado del episodio queda:
        #   sum_k reward_k = - sum_k cost_k / max_steps
        # es decir, el negativo del costo medio del episodio de entrenamiento.
        reward = -float(cost_k) / float(self.cfg.max_steps)

        self.plant.step_crank_nicolson(u_sat, h=self.cfg.dt)
        xp_after = self.plant.xp.copy()
        failed, reason = self._failure(xp_after, y, u_sat)
        terminated = bool(failed)
        if terminated:
            reward -= self.cfg.fail_penalty

        self.current_reward = float(reward)
        self.current_cost = float(cost_k)

        # 3) Copy original HybridEnv: append repeated current_delta_theta at every physical step.
        self.delta_accum.append(self.current_delta_theta.copy())
        self.reward_accum.append(float(reward))
        self.cost_accum.append(float(cost_k))

        hold_block_id = self.step_in_episode // self.cfg.dk_hold_steps
        log_row = {
            "global_physical_step": self.global_physical_step,
            "episode": self.episode_index,
            "step_in_episode": self.step_in_episode,
            "hold_block_id": hold_block_id,
            "t": self.step_in_episode * self.cfg.dt,
            "x1": float(xp_before[0]),
            "x2": float(xp_before[1]),
            "y": float(y[0]),
            "u_raw": float(np.asarray(u_raw).reshape(-1)[0]),
            "u_sat": float(np.asarray(u_sat).reshape(-1)[0]),
            "cost_k": float(cost_k),
            "reward": float(reward),
            "theta_base": theta_to_string(self.theta_base),
            "theta_trial": theta_to_string(self.current_theta_trial),
            "delta_theta": theta_to_string(self.current_delta_theta),
            "accepted_update": "",
            "lr": "",
            "alpha": "",
            "theta_score": float(self.theta_score),
            "J_base": float(self.J_base),
            "J_probe": "",
            "reason": reason,
        }

        self.step_in_episode += 1
        self.global_physical_step += 1
        truncated = self.step_in_episode >= self.cfg.max_steps

        update_info: Dict[str, Any] = {}
        if terminated or truncated:
            update_info = self._update_theta_base()
            log_row["accepted_update"] = update_info.get("accepted_update", "")
            log_row["lr"] = update_info.get("lr", "")
            log_row["alpha"] = update_info.get("alpha", "")
            log_row["theta_score"] = update_info.get("theta_score", self.theta_score)
            log_row["J_base"] = update_info.get("J_base", self.J_base)
            log_row["J_probe"] = update_info.get("J_probe", "")
            self.episode_index += 1

        self.logs.append(log_row)

        info = {
            "cost_k": float(cost_k),
            "reward": float(reward),
            "y": y.copy(),
            "u_raw": np.asarray(u_raw).copy(),
            "u_sat": np.asarray(u_sat).copy(),
            "reason": reason,
            "new_effective_action_step": (self.step_in_episode - 1) % self.cfg.dk_hold_steps == 0,
            **update_info,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


    def evaluate_theta(self, theta: np.ndarray, n_steps: int = K_EVAL_STEPS) -> Dict[str, Any]:
        """
        Evalúa un theta SOS/ZPK en closed-loop.
    
        Convención final:
            J_eval = sum_k cost_k / n_steps
    
        Por tanto:
            entrenamiento: J_train = sum_k cost_k / max_steps
            evaluación:    J_eval  = sum_k cost_k / k_eval_steps
    
        Así la métrica de evaluación es comparable con el costo medio
        usado implícitamente por el reward total de entrenamiento.
        """
        try:
            theta = self.mapper.project_theta(theta)
    
            plant = self.plant_factory()
            plant.reset(self.xp0)
    
            controller = self._make_controller_from_theta(theta, reset_states=True)
    
            total_cost = 0.0
            ys: List[float] = []
            us: List[float] = []
            xs: List[float] = []
            reason = "ok"
    
            n_steps = int(n_steps)
            if n_steps <= 0:
                return {
                    "valid": False,
                    "J": np.inf,
                    "y_rms": np.inf,
                    "u_rms": np.inf,
                    "xmax": np.inf,
                    "reason": "invalid_n_steps",
                }
    
            for _ in range(n_steps):
                xp = plant.xp.copy()
                y = np.asarray(self.Csens @ xp, dtype=float).reshape(self.Csens.shape[0])
    
                u_raw = controller.sample(y)
                u_sat = np.clip(u_raw, -self.cfg.force_limit, self.cfg.force_limit)
    
                cost_k = self._instant_cost(xp, y, u_sat)
    
                if not np.isfinite(cost_k):
                    return {
                        "valid": False,
                        "J": np.inf,
                        "y_rms": rms(np.asarray(ys, dtype=float)),
                        "u_rms": rms(np.asarray(us, dtype=float)),
                        "xmax": float(np.max(xs)) if len(xs) else np.inf,
                        "reason": "nonfinite_cost",
                    }
    
                plant.step_crank_nicolson(u_sat, h=self.cfg.dt)
                xp_after = plant.xp.copy()
    
                failed, reason = self._failure(xp_after, y, u_sat)
    
                total_cost += float(cost_k)
                ys.append(float(np.linalg.norm(y)))
                us.append(float(np.linalg.norm(u_sat)))
                xs.append(float(np.max(np.abs(xp_after))))
    
                if failed:
                    return {
                        "valid": False,
                        "J": np.inf,
                        "y_rms": rms(np.asarray(ys, dtype=float)),
                        "u_rms": rms(np.asarray(us, dtype=float)),
                        "xmax": float(np.max(xs)) if len(xs) else np.inf,
                        "reason": reason,
                    }
    
            J = float(total_cost / float(n_steps))
    
            return {
                "valid": bool(np.isfinite(J)),
                "J": J,
                "y_rms": rms(np.asarray(ys, dtype=float)),
                "u_rms": rms(np.asarray(us, dtype=float)),
                "xmax": float(np.max(xs)) if len(xs) else 0.0,
                "reason": reason,
            }
    
        except Exception as exc:
            return {
                "valid": False,
                "J": np.inf,
                "y_rms": np.inf,
                "u_rms": np.inf,
                "xmax": np.inf,
                "reason": f"exception:{type(exc).__name__}",
            }

    def _update_theta_base(self) -> Dict[str, Any]:
        """
        End-of-episode hybrid update.
    
        Pipeline final:
            1) Se agrupan los rewards por bloques de DK_HOLD_STEPS.
            2) Cada bloque tiene:
                  - un único delta_theta efectivo
                  - un reward de bloque = suma de rewards del bloque
            3) Se calcula delta_mean con softmax sobre rewards de bloque.
            4) Se evalúa una sola vez:
                  theta_probe = theta_base + delta_mean
            5) Si J_probe mejora J_base:
                  alpha = LR_GOOD = 0.3
               Si no mejora:
                  alpha = LR_BAD = 0.01
            6) Si el probe es válido, siempre se actualiza:
                  theta_base <- theta_base + alpha * delta_mean
                  J_base    <- (1 - alpha) * J_base + alpha * J_probe
            7) Si el probe no es válido:
                  alpha = 0
                  theta_base queda igual
                  J_base queda igual
        """
    
        if len(self.delta_accum) == 0:
            return {
                "accepted_update": False,
                "alpha": 0.0,
                "lr": 0.0,
                "J_base": float(self.J_base),
                "J_probe": np.inf,
                "theta_score": float(self.theta_score),
                "candidate_score": -np.inf,
                "Kg_base": float(self.Kg_base),
                "mean_reward": 0.0,
                "best_reward": 0.0,
                "delta_norm": 0.0,
                "reason": "empty_episode",
            }
    
        # ============================================================
        # 0) Batch del episodio
        # ============================================================
        rewards = np.asarray(self.reward_accum, dtype=np.float64).reshape(-1)
        deltas = np.asarray(self.delta_accum, dtype=np.float64)
    
        if deltas.ndim == 1:
            deltas = deltas.reshape(1, -1)
    
        rewards_safe = np.where(np.isfinite(rewards), rewards, -1e12)
    
        mean_reward = float(np.mean(rewards_safe))
        best_reward = float(np.max(rewards_safe))
    
        # ============================================================
        # 1) Softmax ponderado por bloques del episodio
        # ============================================================
        H = int(self.cfg.dk_hold_steps)
        N = int(len(rewards_safe))
    
        block_rewards = []
        block_deltas = []
    
        for i0 in range(0, N, H):
            i1 = min(i0 + H, N)
    
            rewards_block = rewards_safe[i0:i1]
            deltas_block = deltas[i0:i1]
    
            if rewards_block.size == 0:
                continue
    
            # Reward total del bloque.
            # Como reward_k = -cost_k / max_steps,
            # R_block es la contribución del bloque al reward total del episodio.
            R_block = float(np.sum(rewards_block))
    
            # Delta único del bloque.
            # Durante DK_HOLD_STEPS se mantiene el mismo delta_theta efectivo.
            dtheta_block = np.asarray(deltas_block[0], dtype=np.float64).reshape(-1)
    
            block_rewards.append(R_block)
            block_deltas.append(dtheta_block)
    
        block_rewards = np.asarray(block_rewards, dtype=np.float64).reshape(-1)
        block_deltas = np.asarray(block_deltas, dtype=np.float64)
    
        block_rewards_safe = np.where(np.isfinite(block_rewards), block_rewards, -1e12)
    
        rmax = float(np.max(block_rewards_safe))
        weights = np.exp(self.cfg.beta_softmax * (block_rewards_safe - rmax))
        weights = weights / (np.sum(weights) + EPS)
    
        delta_mean = np.sum(weights[:, None] * block_deltas, axis=0)
        delta_mean = np.asarray(delta_mean, dtype=np.float64).reshape(-1)
    
        delta_norm = float(np.linalg.norm(delta_mean))
    
        # ============================================================
        # 2) Evaluación única: theta_base + delta_mean
        # ============================================================
        theta_probe = self.mapper.project_theta(self.theta_base + delta_mean)
    
        P_probe, Z_probe, Kg_probe = self.mapper.theta_to_zpk(theta_probe)
        _ = (P_probe, Z_probe, Kg_probe)  # Conversión explícita para debug.
    
        metrics_probe = self.evaluate_theta(
            theta_probe,
            n_steps=self.cfg.k_eval_steps,
        )
    
        accepted = False
        J_probe = np.inf
        reason = metrics_probe.get("reason", "unknown")
    
        # ============================================================
        # 3) GOOD si mejora, bad si no mejora.
        #    En ambos casos válidos se actualiza theta_base y J_base.
        # ============================================================
        if metrics_probe["valid"]:
            J_probe = float(metrics_probe["J"])
            J_base_old = float(self.J_base)
    
            if J_probe <= J_base_old:
                self.alpha = float(LR_GOOD)
                accepted = True
                reason = "accepted_good_improvement"
            else:
                self.alpha = float(LR_BAD)
                accepted = False
                reason = "bad_no_improvement"
    
            theta_new = self.mapper.project_theta(
                self.theta_base + self.alpha * delta_mean
            )
    
            P_new, Z_new, Kg_new = self.mapper.theta_to_zpk(theta_new)
    
            self.theta_base = theta_new.copy()
            self.P_base = P_new.copy()
            self.Z_base = Z_new.copy()
            self.Kg_base = float(Kg_new)
    
            self.J_base = (
                (1.0 - self.alpha) * J_base_old
                + self.alpha * J_probe
            )
    
            self.theta_score = -float(self.J_base)
    
        else:
            self.alpha = 0.0
            accepted = False
            reason = metrics_probe.get("reason", "invalid_probe")
    
        # ============================================================
        # 4) Historial
        # ============================================================
        self.history["theta_score"].append(float(self.theta_score))
        self.history["J_base"].append(float(self.J_base))
        self.history["J_probe"].append(float(J_probe))
        self.history["candidate_score"].append(
            float(-J_probe) if np.isfinite(J_probe) else -np.inf
        )
        self.history["accepted_update"].append(bool(accepted))
        self.history["lr"].append(float(self.alpha))
        self.history["alpha"].append(float(self.alpha))
        self.history["Kg_base"].append(float(self.Kg_base))
        self.history["mean_reward"].append(mean_reward)
        self.history["best_reward"].append(best_reward)
        self.history["delta_norm"].append(delta_norm)
    
        if self.cfg.verbose_episode:
            tag = "GOOD" if accepted else "bad"
            print(
                f"  [{tag:>4}] ep={self.episode_index:04d} "
                f"J_probe={J_probe: .4e} J_base={self.J_base: .4e} "
                f"alpha={self.alpha:.3g} Kg={self.Kg_base:.4g} "
                f"mean_r={mean_reward:.4e} best_r={best_reward:.4e} "
                f"|dtheta|={delta_norm:.4e} reason={reason}"
            )
    
        return {
            "accepted_update": bool(accepted),
            "alpha": float(self.alpha),
            "lr": float(self.alpha),
            "J_base": float(self.J_base),
            "J_probe": float(J_probe),
            "theta_score": float(self.theta_score),
            "candidate_score": float(-J_probe) if np.isfinite(J_probe) else -np.inf,
            "Kg_base": float(self.Kg_base),
            "mean_reward": mean_reward,
            "best_reward": best_reward,
            "delta_norm": delta_norm,
            "probe_valid": bool(metrics_probe["valid"]),
            "probe_y_rms": float(metrics_probe.get("y_rms", np.nan)),
            "probe_u_rms": float(metrics_probe.get("u_rms", np.nan)),
            "probe_xmax": float(metrics_probe.get("xmax", np.nan)),
            "reason": reason,
        }

    def score_theta(self, theta: np.ndarray, n_steps: int = K_EVAL_STEPS) -> float:
        """Backward-compatible score: higher is better."""
        metrics = self.evaluate_theta(theta, n_steps=n_steps)
        if metrics["valid"]:
            return -float(metrics["J"])
        return float(-1e12)

    def get_current_controller(self):
        return self.mapper.theta_to_zpk(self.theta_base)

    def evaluate_current_controller(self, t_final: float = 10.0) -> Dict[str, np.ndarray]:
        controller = self._make_controller_from_theta(self.theta_base, reset_states=True)
        plant = self.plant_factory()
        simulator = ClosedLoopSimulatorSOS(
            plant=plant,
            controller=controller,
            Csens=self.Csens,
            force_limit=self.cfg.force_limit,
        )
        return simulator.run(xp0=self.xp0, xk0=None, t_final=t_final)


# ================================================================
# 7) File IO and convenience builders
# ================================================================


def save_final_zpk_controller(
    filename: str,
    P_final,
    Z_final,
    Kg_final: float,
    theta_base: Optional[np.ndarray] = None,
    topology: Optional[ZPKTopology] = None,
) -> None:
    payload: Dict[str, Any] = {
        "P_sections_final": np.asarray(P_final, dtype=np.complex128),
        "Z_sections_final": np.asarray(Z_final, dtype=np.complex128),
        "Kg_final": np.array([[float(Kg_final)]], dtype=float),
    }
    if theta_base is not None:
        payload["theta_base_final"] = np.asarray(theta_base, dtype=float).reshape(1, -1)
    if topology is not None:
        payload["pole_order"] = np.asarray(topology.pole_order, dtype=int).reshape(1, -1)
        payload["zero_order"] = np.asarray(topology.zero_order, dtype=int).reshape(1, -1)
        payload["Ts"] = np.array([[float(topology.Ts)]], dtype=float)
    savemat(filename, payload)


def save_results_csv(results: Dict[str, np.ndarray], savepath: str) -> None:
    t = np.asarray(results["t"], dtype=float)
    xp = np.asarray(results["xp"], dtype=float)
    y = np.asarray(results.get("y", np.full((len(t), 1), np.nan)), dtype=float)
    u_raw = np.asarray(results["u_raw"], dtype=float)
    u_sat = np.asarray(results["u_sat"], dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if u_raw.ndim == 1:
        u_raw = u_raw[:, None]
    if u_sat.ndim == 1:
        u_sat = u_sat[:, None]
    data = np.column_stack([t[:, None], xp, y, u_raw, u_sat])
    header = ["t", "x1", "x2"]
    header += [f"y_{j}" for j in range(y.shape[1])]
    header += [f"u_raw_{j}" for j in range(u_raw.shape[1])]
    header += [f"u_sat_{j}" for j in range(u_sat.shape[1])]
    np.savetxt(savepath, data, delimiter=",", header=",".join(header), comments="")


def make_env_from_mat(
    mat_filename: str,
    config: HybridSOSConfig = HybridSOSConfig(),
    Csens: np.ndarray = CSENS,
    xp0: np.ndarray = X0,
) -> StuartLandauHybridSOSZPKPPOEnv:
    ctrl0 = DynamicControllerSOS.from_mat(mat_filename, Csens=Csens)
    topology = ZPKTopology.from_controller(ctrl0)
    mapper = ZPKActionMapper(topology)
    env = StuartLandauHybridSOSZPKPPOEnv(
        topology=topology,
        mapper=mapper,
        config=config,
        plant_factory=lambda: StuartLandauPlant(dt=config.dt, Ts=config.Ts),
        Csens=Csens,
        xp0=xp0,
    )
    print("Topology:", topology.summary())
    return env


# ================================================================
# 8) Debug, training, final evaluation
# ================================================================


def run_debug_rollout(env: StuartLandauHybridSOSZPKPPOEnv, n_steps: Optional[int] = None) -> None:
    n_steps = env.cfg.max_steps if n_steps is None else int(n_steps)
    obs, _ = env.reset()
    print("Debug rollout")
    print("  obs shape:", obs.shape)
    print("  action shape:", env.action_space.shape)
    for k in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if k < 5 or terminated or truncated:
            print(
                f"  k={k:04d} reward={reward: .4e} "
                f"new_action={info['new_effective_action_step']} "
                f"term={terminated} trunc={truncated} reason={info.get('reason')}"
            )
        if terminated or truncated:
            break


def train_hybrid_ppo(env: StuartLandauHybridSOSZPKPPOEnv, total_timesteps: int = HYBRID_TIMESTEPS):
    if PPO is None:
        raise ImportError("stable-baselines3 is required for PPO training.")
    device = "cuda"
    if torch is None or not torch.cuda.is_available():
        device = "cpu"
    print("Training PPO Hybrid SOS/ZPK")
    print("  device:", device)
    print("  total_timesteps:", total_timesteps)
    print("  n_steps:", env.cfg.ppo_n_steps)
    print("  batch_size:", env.cfg.batch_size)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=env.cfg.ppo_n_steps,
        batch_size=env.cfg.batch_size,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def run_evaluate_final(env: StuartLandauHybridSOSZPKPPOEnv, t_final: float = K_EVAL_STEPS*DT ) -> Dict[str, np.ndarray]:
    results = env.evaluate_current_controller(t_final=t_final)
    print("Final controller evaluation")
    print("  samples:", len(results["t"]))
    print("  y_rms:", rms(results["y"]))
    print("  u_rms:", rms(results["u_sat"]))
    print("  x_max:", float(np.max(np.abs(results["xp"]))))
    return results


# ================================================================
# 9) Main
# ================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Stuart-Landau Hybrid PPO SOS/ZPK, HYBRID_TIMESTEPS semantics.")
    parser.add_argument("--mat-file", default="controller_K0_SOS_ZPK.mat", help="Initial MATLAB SOS/ZPK controller file.")
    parser.add_argument("--mode", choices=["debug", "train"], default="train")
    parser.add_argument("--total-timesteps", type=int, default=HYBRID_TIMESTEPS)
    parser.add_argument("--dk-hold-steps", type=int, default=DK_HOLD_STEPS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--no-full-state-obs", action="store_true")
    parser.add_argument("--create-demo-if-missing", action="store_true")
    parser.add_argument("--eval-final-time", type=float, default=10.0)
    args = parser.parse_args()

    cfg = HybridSOSConfig(
        max_steps=args.max_steps,
        hybrid_timesteps=args.total_timesteps,
        dk_hold_steps=args.dk_hold_steps,
        include_full_state_in_obs=not args.no_full_state_obs,
    )
    print_timestep_summary(cfg.max_steps, cfg.hybrid_timesteps, cfg.dk_hold_steps)
    env = make_env_from_mat(args.mat_file, config=cfg, Csens=CSENS, xp0=X0)

    if args.mode == "debug":
        run_debug_rollout(env, n_steps=cfg.max_steps)
        
    elif args.mode == "train":
        initial_results = run_evaluate_final(env, t_final=args.eval_final_time)
        save_results_csv(initial_results,SIM_ROLLOUT_K0_CSV)
        print("Saved initial closed-loop K0 CSV:", SIM_ROLLOUT_K0_CSV)
        model = train_hybrid_ppo(env, total_timesteps=args.total_timesteps)
        model_path = os.path.join(MODEL, "ppo_hybrid_sos_zpk_stuart_landau.zip")
        model.save(model_path)

    # Always export whatever theta_base is current after debug/train.
    final_results = run_evaluate_final(env, t_final=args.eval_final_time)
    save_results_csv(final_results, SIM_ROLLOUT_KPPO_CSV)
    print("Saved final closed-loop KPPO CSV:", SIM_ROLLOUT_KPPO_CSV)


    P_final, Z_final, Kg_final = env.get_current_controller()
    mat_path = FINAL_CONTROLLER_KPPO
    save_final_zpk_controller(
        mat_path,
        P_final=P_final,
        Z_final=Z_final,
        Kg_final=Kg_final,
        theta_base=env.theta_base,
        topology=env.topology,
    )
    print("Saved final controller MAT:", mat_path)


if __name__ == "__main__":
    main()

