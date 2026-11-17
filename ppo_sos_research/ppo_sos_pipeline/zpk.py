from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

EPS = 1e-12
from .controller_sos import ControllerSOS
from .utils import wrap_angle


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
        self.Ts = 1.0 if Ts is None else float(Ts)
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
    def from_controller(cls, controller: ControllerSOS, tol: float = 1e-10) -> "ZPKTopology":
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
