from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.io import loadmat

EPS = 1e-12
TS = 0.05


class ControllerSOS:
    """Discrete-time SOS/ZPK controller implemented as cascade sections.

    This is the final clean version of DynamicControllerSOS.

    It loads the same ZPK MATLAB files:
        P_sections, Z_sections, pole_order, zero_order, Kg, Ts

    Online interface:
        u = controller.step(y, dt=None)

    The controller is already discrete. The optional dt is accepted only for
    compatibility, and must match Ts; no resampling is performed.

    Online implementation remains the original cascade filter:
        v_i[k] = C_i x_i[k] + D_i v_{i-1}[k]
        x_i[k+1] = A_i x_i[k] + B_i v_{i-1}[k]
        u[k] = Kg v_L[k]
    """

    def __init__(
        self,
        P_sections=None,
        Z_sections=None,
        pole_order=None,
        zero_order=None,
        Kg=None,
        Ts=None,
        dt: Optional[float] = None,
        x0: Optional[np.ndarray] = None,
        reset_states: bool = True,
    ):
        self.ny = 1
        self.nu = 1
        self.tol = float(1e-10)

        self.Ts = float(Ts if Ts is not None else (dt if dt is not None else TS))
        self.dt = self.Ts

        self.P_sections = None
        self.Z_sections = None
        self.pole_order = None
        self.zero_order = None
        self.Kg = None
        self.local_sections: List[Dict[str, Any]] = []

        # Kept for export/debug only. Online step uses the cascade above.
        self.A_tot = None
        self.B_tot = None
        self.C_tot = None
        self.D_tot = None

        self.x0 = None if x0 is None else np.asarray(x0, dtype=float).reshape(-1, 1)

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
                reset_states=False,
            )
            if reset_states:
                self.reset(self.x0)

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
            return dict(pole_order=pole_order, zero_order=zero_order, poles=tuple(poles_pair), zeros=tuple(zeros_pair), A=A, B=B, C=C, D=D, x=np.zeros((2, 1), dtype=float))
        if pole_order == 1:
            a1, _ = self._den_coeffs(poles_pair, 1)
            beta0, beta1 = self._num_coeffs(zeros_pair, 1, zero_order)
            A = np.array([[-a1]], dtype=float)
            B = np.array([[1.0]], dtype=float)
            D = np.array([[beta0]], dtype=float)
            C = np.array([[beta1 - beta0 * a1]], dtype=float)
            return dict(pole_order=pole_order, zero_order=zero_order, poles=tuple(poles_pair), zeros=tuple(zeros_pair), A=A, B=B, C=C, D=D, x=np.zeros((1, 1), dtype=float))
        raise ValueError("pole_order must be 1 or 2.")

    def set_sections(self, P_sections, Z_sections, pole_order, zero_order, Kg: float, reset_states: bool = True) -> None:
        P_sections = self._normalize_sections_input(P_sections, "P_sections")
        Z_sections = self._normalize_sections_input(Z_sections, "Z_sections")
        L = P_sections.shape[0]
        pole_order = self._normalize_order_input(pole_order, L=L, name="pole_order")
        zero_order = self._normalize_order_input(zero_order, L=L, name="zero_order")
        self._validate_structure(P_sections, Z_sections, pole_order, zero_order)

        self.Kg = float(Kg)
        self.P_sections = P_sections.copy()
        self.Z_sections = Z_sections.copy()
        self.pole_order = pole_order.copy()
        self.zero_order = zero_order.copy()
        self.local_sections = []

        for i in range(L):
            self.local_sections.append(
                self._section_to_ss(
                    poles_pair=self.P_sections[i, :],
                    zeros_pair=self.Z_sections[i, :],
                    pole_order=self.pole_order[i],
                    zero_order=self.zero_order[i],
                )
            )

        self.A_tot = self.B_tot = self.C_tot = self.D_tot = None
        if reset_states:
            self.reset(self.x0)

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

    def _state_dim(self) -> int:
        return int(np.sum(self.pole_order)) if self.pole_order is not None else 0

    def reset(self, x0: Optional[np.ndarray] = None) -> None:
        nK = self._state_dim()
        if x0 is None:
            x0 = self.x0
        if x0 is None:
            x0 = np.zeros((nK, 1), dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
        if x0.shape[0] != nK:
            raise ValueError(f"x0 must have dimension {nK} x 1.")

        idx = 0
        for sec in self.local_sections:
            n = sec["A"].shape[0]
            sec["x"] = x0[idx:idx + n, :].copy()
            idx += n

    def get_states(self) -> np.ndarray:
        if len(self.local_sections) == 0:
            return np.zeros((0, 1), dtype=float)
        return np.vstack([sec["x"] for sec in self.local_sections])

    def output(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape != (self.ny,):
            raise ValueError(f"y must have shape ({self.ny},), got {y.shape}.")

        v = float(y[0])
        for sec in self.local_sections:
            A, B, C, D, x = sec["A"], sec["B"], sec["C"], sec["D"], sec["x"]
            v_out = float((C @ x + D * v).squeeze())
            sec["x"] = A @ x + B * v
            v = v_out
        return np.array([self.Kg * v], dtype=float)

    def step(self, y: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        if dt is not None:
            self.validate_dt(dt)
        return self.output(y)

    def validate_dt(self, dt: float) -> None:
        if not np.isfinite(self.Ts) or self.Ts <= 0 or not np.isclose(
            self.Ts, float(dt), rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"Discrete controller Ts={self.Ts} incompatible with dt={dt}; no conversion.")

    def build_global_ss(self, apply_global_gain: bool = True):
        """Build global discrete state-space realization of the cascade.

        Kept for export/debug. The online step uses the cascade directly.
        """
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
            "L": len(self.local_sections),
            "state_dimension": self._state_dim(),
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
        x0: Optional[np.ndarray] = None,
        expected_dt: Optional[float] = None,
    ) -> "ControllerSOS":
        data = loadmat(filename)
        P_sections = np.asarray(data[p_key])
        Z_sections = np.asarray(data[z_key])
        pole_order = np.asarray(data[pole_order_key]).reshape(-1)
        zero_order = np.asarray(data[zero_order_key]).reshape(-1)
        Kg = float(np.ravel(data[kg_key])[0])
        if ts_key not in data or np.asarray(data[ts_key]).size != 1:
            raise ValueError(f"{filename}: missing or nonscalar {ts_key}; specify the discrete sample time in the MAT file.")
        Ts = float(np.ravel(data[ts_key])[0])
        if not np.isfinite(Ts) or Ts <= 0:
            raise ValueError(f"{filename}: Ts must be a positive discrete sample time.")
        controller = cls(
            P_sections=P_sections,
            Z_sections=Z_sections,
            pole_order=pole_order,
            zero_order=zero_order,
            Kg=Kg,
            Ts=Ts,
            x0=x0,
            reset_states=True,
        )
        if expected_dt is not None:
            controller.validate_dt(expected_dt)
        return controller


# Backward-compatible name if old code imports DynamicControllerSOS.
DynamicControllerSOS = ControllerSOS
