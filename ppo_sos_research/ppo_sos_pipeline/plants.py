from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from scipy.optimize import root



class StuartLandauPlant:
    """Nonlinear Stuart-Landau plant with internal measurement.

    Final public interface expected by the RL environment:
        plant.reset()
        plant.measure() -> y[k]
        plant.step(u)   -> advances one physical step

    Csens and xp0 are plant attributes. The environment does not know them.
    """

    def __init__(
        self,
        dt: float = 0.05,
        Ts: float = 0.05,
        Csens: Optional[np.ndarray] = None,
        xp0: Optional[Sequence[float]] = (0.6, 0.1),
        nu: int = 1,
        mu: float = 2.4,
        omega: float = 2.0,
        beta: float = 0.5,
    ):
        self.dt = float(dt)
        self.Ts = float(Ts)
        if self.dt <= 0.0 or self.Ts <= 0.0:
            raise ValueError("dt and Ts must be > 0.")

        self.Csens = np.asarray([[0.0, 1.0]] if Csens is None else Csens, dtype=float).copy()
        self.state_names = ["x1", "x2"]
        if self.Csens.ndim != 2:
            raise ValueError("Csens must be a 2D array.")

        # Generalized from Csens shape.
        self.ny = int(self.Csens.shape[0])
        self.nxp = int(self.Csens.shape[1])
        if self.nxp != 2:
            raise ValueError("Stuart-Landau dynamics require Csens.shape[1] == 2.")

        self.nu = int(nu)
        if self.nu != 1:
            raise ValueError("This StuartLandauPlant version supports one actuator input only.")

        self.mu = float(mu)
        self.omega = float(omega)
        self.beta = float(beta)

        if xp0 is None:
            self.xp0 = np.zeros(self.nxp, dtype=float)
        else:
            self.xp0 = np.asarray(xp0, dtype=float).reshape(-1)
        if self.xp0.shape != (self.nxp,):
            raise ValueError(f"xp0 must have shape ({self.nxp},).")

        self.xp = self.xp0.copy()

    def reset(self) -> np.ndarray:
        self.xp = self.xp0.copy()
        return self.xp.copy()

    def state(self) -> np.ndarray:
        """Optional debug/export state. Not used by the RL cost or observation."""
        return self.xp.copy()

    def measure(self) -> np.ndarray:
        return np.asarray(self.Csens @ self.xp, dtype=float).reshape(self.ny)

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
        """One implicit trapezoidal / Crank-Nicolson step with constant input u."""
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

    def step(self, u: np.ndarray) -> np.ndarray:
        return self.step_crank_nicolson(u, h=self.dt)
