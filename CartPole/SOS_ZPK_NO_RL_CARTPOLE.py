import os
from typing import Optional, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import root

# ================================================================
# 0) Physical parameters and simulation setup
# ================================================================
# Plant constants (cart-pole nonlinear model)
M = 1.0
m = 0.1
l = 0.5
g = 9.81
F_MAX = 15.0  # actuator saturation

# Default time values requested
DEFAULT_DT = 1e-3
DEFAULT_TS = 1e-3

# We keep Ts FIXED. The total simulation time is set with the number of
# controller samples we want to simulate.
N_CTRL_STEPS = 5000
T_FINAL = N_CTRL_STEPS * DEFAULT_TS

# Initial condition around theta = 0 (upright)
X0 = np.array([0.0, 0.0, np.deg2rad(10.0), 0.0], dtype=float)

# Default sensing matrix: full-state measurement
#Csens = np.eye(len(X0), dtype=float)
Csens = np.array([[0, 0, 1, 0]], dtype=float)

BUILD_DIR = 'build_cartpole_minimal'
os.makedirs(BUILD_DIR, exist_ok=True)


# ================================================================
# 3) Linearized cart-pole model around theta = 0 (upright)
#    Useful only for diagnostics.
# ================================================================
def linearized_cartpole_matrices(Cp: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plant linearized around theta = 0 (upright).

    State order:
        xp = [x, xdot, theta, thetadot]^T
    """
    Ap = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -(m * g) / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, ((M + m) * g) / (l * M), 0.0],
    ], dtype=float)
    Bp = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (l * M)]], dtype=float)
    Cp = np.eye(Ap.shape[0], dtype=float) if Cp is None else np.array(Cp, dtype=float, copy=True)
    return Ap, Bp, Cp


# ================================================================
# 4) Plant class
#    This class ONLY knows the cart-pole dynamics.
# ================================================================
class CartPolePlant:
    """
    Nonlinear cart-pole plant.

    Responsibility of this class:
      - store the physical state xp = [x, xdot, theta, thetadot]
      - compute nonlinear derivatives
      - advance the plant one step with Crank-Nicolson

    Notes on MIMO:
      - the class already exposes the plant input as a vector u with shape (nu,)
      - the current cart-pole physics still uses a single real force input
      - therefore nu must remain 1 unless the physical model itself is extended
    """

    def __init__(
        self,
        dt: float = DEFAULT_DT,
        Ts: float = DEFAULT_TS,
        nu: int = 1,
    ):
        self.nxp = 4
        self.nu = int(nu)
        if self.nu != 1:
            raise ValueError('Current cart-pole physics only supports one real actuator input, so nu must be 1.')

        self.dt = DEFAULT_DT if dt is None else float(dt)
        self.Ts = DEFAULT_TS if Ts is None else float(Ts)
        if self.dt <= 0.0:
            raise ValueError('dt must be > 0')
        if self.Ts <= 0.0:
            raise ValueError('Ts must be > 0')

        self.xp = np.zeros(self.nxp, dtype=float)

    def reset(self, xp0: Optional[np.ndarray] = None) -> np.ndarray:
        self.xp = np.array(X0 if xp0 is None else xp0, dtype=float).reshape(-1).copy()
        if self.xp.shape != (self.nxp,):
            raise ValueError(f'xp0 must have shape ({self.nxp},)')
        return self.xp.copy()

    def derivatives(self, xp: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Nonlinear cart-pole dynamics for theta = 0 upright.

        Returns:
            xpdot = [xdot, xddot, thetadot, thetaddot]
        """
        xp = np.asarray(xp, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)

        if xp.shape != (self.nxp,):
            raise ValueError(f'xp must have shape ({self.nxp},)')
        if u.shape != (self.nu,):
            raise ValueError(f'u must have shape ({self.nu},)')

        force = u[0] #INPUT SISO, DEPENDERÁ DE CADA PLANTA. 
        x, xdot, theta, thetadot = xp
        st, ct = np.sin(theta), np.cos(theta)
        denom = M + m - m * ct**2

        thetaddot = ((M + m) * g * st - ct * (force + m * l * thetadot**2 * st)) / (l * denom)
        xddot = (force + m * l * (thetadot**2 * st - thetaddot * ct)) / (M + m)

        return np.array([xdot, xddot, thetadot, thetaddot], dtype=float)

    def step_crank_nicolson(self, u: np.ndarray, h: Optional[float] = None) -> np.ndarray:
        """
        One implicit trapezoidal / Crank-Nicolson step for the plant alone.

        h can be smaller than self.dt. This is useful in discrete control mode
        when we want to hit the sampling instants exactly.
        """
        h = self.dt if h is None else float(h)
        if h <= 0.0:
            raise ValueError('Integration step h must be > 0')

        xp_now = self.xp.copy() #Para cada dt, trae el estado xp
        f_now = self.derivatives(xp_now, u) #Vector de estados. Cálcula derivadas

        def residual(xp_new: np.ndarray) -> np.ndarray:
            f_new = self.derivatives(xp_new, u) #Cálculo vector de estados para predicción
            return xp_new - xp_now - 0.5 * h * (f_now + f_new)

        xp_guess = xp_now + h * f_now #Predicción simple, tipo Euler explícito

        try:
            sol = root(residual, xp_guess, method='hybr', options={'maxfev': 60})
            if sol.success:
                self.xp = np.asarray(sol.x, dtype=float)
                return self.xp.copy()
        except Exception:
            pass

        # fallback: Heun / predictor-corrector
        f_pred = self.derivatives(xp_guess, u) #Usa Euler explícito si solver falla
        self.xp = xp_now + 0.5 * h * (f_now + f_pred) #Aproximación trapezoidal del estado actual/próximo
        return self.xp.copy() #Actualiza y devuelve valor de la planta para el instante t+dt


# ================================================================
# 5) CONTROLLER FOR SOS RL class
#    JUST CONTROLLER DYNAMICS BASED ON SOS FOR ZPK IMPLEMENTATION
# ================================================================  
class DynamicControllerSOS:
    """
    Controlador discreto en cascada con secciones de orden 1 o 2.

    Estructura fija por sección:
        pole_order[l] ∈ {1, 2}
        zero_order[l] ∈ {0, 1, 2}
        con zero_order[l] <= pole_order[l]

    MATLAB debe entregar:
        P_sections : matriz L x 2, con NaN en slots vacíos
        Z_sections : matriz L x 2, con NaN en slots vacíos
        pole_order : vector L
        zero_order : vector L
        Kg         : ganancia global
        Ts         : tiempo de muestreo

    Importante:
        - La estructura pole_order / zero_order queda fija.
        - El RL puede modificar valores de polos/ceros después,
          pero no debería cambiar pole_order / zero_order.
        - Las matrices A,B,C,D se reconstruyen solo al cargar o actualizar ZPK.
        - Durante sample(), solo se propagan estados.
    """

    def __init__(
        self,
        P_sections=None,
        Z_sections=None,
        pole_order=None,
        zero_order=None,
        Kg=1.0,
        Ts=None,
        tol=1e-10,
        Csens=Csens,
        reset_states=True,
    ):
        self.mode = 'sos_rl'
        self.implementation = 'sos_zpk_variable_order'

        self.ny = np.shape(Csens)[0]
        self.nu = 1
        self.Ts = Ts
        self.tol = float(tol)
        self.Kg = float(Kg)

        if self.ny != 1:
            raise ValueError("DynamicControllerSOS currently supports SISO measurement only.")

        self.P_sections = None
        self.Z_sections = None
        self.pole_order = None
        self.zero_order = None

        self.local_sections = []

        self.A_tot = None
        self.B_tot = None
        self.C_tot = None
        self.D_tot = None

        self.u_hold = np.zeros(self.nu, dtype=float)

        if P_sections is not None or Z_sections is not None:
            if P_sections is None or Z_sections is None:
                raise ValueError("Debes pasar P_sections y Z_sections a la vez.")
            if pole_order is None or zero_order is None:
                raise ValueError("Debes pasar pole_order y zero_order.")

            self.set_sections(
                P_sections=P_sections,
                Z_sections=Z_sections,
                pole_order=pole_order,
                zero_order=zero_order,
                Kg=Kg,
                reset_states=reset_states,
            )

    # =========================================================
    # UTILIDADES
    # =========================================================
    def _is_nan(self, value):
        return bool(np.isnan(value))

    def _is_finite_number(self, value):
        return not self._is_nan(value)

    def _real_scalar(self, value, name="coef"):
        if self._is_nan(value):
            raise ValueError(f"{name} es NaN, pero se esperaba un valor finito.")

        if abs(np.imag(value)) > self.tol:
            raise ValueError(f"{name} no es real dentro de tolerancia: {value}")

        return float(np.real(value))

    def _validate_pair(self, pair, name="pair"):
        """
        Acepta:
            - dos reales
            - par complejo conjugado
        No acepta NaN.
        """
        x1, x2 = pair

        if self._is_nan(x1) or self._is_nan(x2):
            raise ValueError(f"{name} contiene NaN, pero se esperaba par completo: {pair}")

        if np.isreal(x1) and np.isreal(x2):
            return

        if abs(x2 - np.conjugate(x1)) > self.tol:
            raise ValueError(f"{name} no es un par conjugado válido: {pair}")

    def _normalize_sections_input(self, sections, name="sections"):
        """
        Devuelve matriz L x 2.

        Acepta:
            - L x 2
            - L x 1, que se rellena con NaN en segunda columna
            - vector plano de longitud par
        """
        arr = np.asarray(sections)

        if arr.ndim == 1:
            if arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            else:
                arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser 2D.")

        if arr.shape[1] == 1:
            nan_col = np.full((arr.shape[0], 1), np.nan, dtype=arr.dtype)
            arr = np.hstack([arr, nan_col])

        if arr.shape[1] != 2:
            raise ValueError(f"{name} debe tener forma Lx2 o Lx1.")

        return arr.astype(complex)

    def _normalize_order_input(self, order, L, name="order"):
        arr = np.asarray(order).reshape(-1)

        if arr.size != L:
            raise ValueError(f"{name} debe tener longitud L={L}.")

        arr = arr.astype(int)

        return arr

    # =========================================================
    # VALIDACIÓN DE ESTRUCTURA FIJA
    # =========================================================
    def _validate_structure(self, P_sections, Z_sections, pole_order, zero_order):
        L = P_sections.shape[0]

        if Z_sections.shape[0] != L:
            raise ValueError("P_sections y Z_sections deben tener el mismo número de secciones.")

        if pole_order.size != L or zero_order.size != L:
            raise ValueError("pole_order y zero_order deben tener longitud L.")

        for i in range(L):
            po = int(pole_order[i])
            zo = int(zero_order[i])

            if po not in (1, 2):
                raise ValueError(f"pole_order[{i}] debe ser 1 o 2.")

            if zo not in (0, 1, 2):
                raise ValueError(f"zero_order[{i}] debe ser 0, 1 o 2.")

            if zo > po:
                raise ValueError(
                    f"Sección {i}: zero_order={zo} > pole_order={po}. "
                    "Cada sección debe ser propia localmente."
                )

            p1, p2 = P_sections[i, :]
            z1, z2 = Z_sections[i, :]

            if po == 2:
                self._validate_pair((p1, p2), name=f"P_sections[{i}]")

            elif po == 1:
                if self._is_nan(p1):
                    raise ValueError(f"Sección {i}: pole_order=1 requiere p1 finito.")
                if not self._is_nan(p2):
                    raise ValueError(f"Sección {i}: pole_order=1 espera p2=NaN.")
                self._real_scalar(p1, name=f"p1 sección {i}")

            if zo == 2:
                self._validate_pair((z1, z2), name=f"Z_sections[{i}]")

            elif zo == 1:
                if self._is_nan(z1):
                    raise ValueError(f"Sección {i}: zero_order=1 requiere z1 finito.")
                if not self._is_nan(z2):
                    raise ValueError(f"Sección {i}: zero_order=1 espera z2=NaN.")
                self._real_scalar(z1, name=f"z1 sección {i}")

            elif zo == 0:
                if not (self._is_nan(z1) and self._is_nan(z2)):
                    raise ValueError(f"Sección {i}: zero_order=0 espera [NaN, NaN].")

    # =========================================================
    # COEFICIENTES POR SECCIÓN
    # =========================================================
    def _den_coeffs(self, poles_pair, pole_order):
        """
        Denominador normalizado.

        pole_order = 2:
            den = 1 + a1 z^-1 + a2 z^-2

        pole_order = 1:
            den = 1 + a1 z^-1
        """
        pole_order = int(pole_order)

        if pole_order == 2:
            p1, p2 = poles_pair
            a1 = self._real_scalar(-(p1 + p2), name="a1")
            a2 = self._real_scalar(p1 * p2, name="a2")
            return a1, a2

        if pole_order == 1:
            p1 = poles_pair[0]
            a1 = self._real_scalar(-p1, name="a1")
            return a1, None

        raise ValueError("pole_order debe ser 1 o 2.")

    def _num_coeffs(self, zeros_pair, pole_order, zero_order):
        """
        Numerador normalizado según estructura fija.

        Para pole_order = 2:
            zero_order = 2:
                beta = [1, -(z1+z2), z1*z2]
            zero_order = 1:
                beta = [0, 1, -z1]
            zero_order = 0:
                beta = [0, 0, 1]

        Para pole_order = 1:
            zero_order = 1:
                beta = [1, -z1]
            zero_order = 0:
                beta = [0, 1]
        """
        pole_order = int(pole_order)
        zero_order = int(zero_order)

        z1, z2 = zeros_pair

        if pole_order == 2:
            if zero_order == 2:
                beta0 = 1.0
                beta1 = -(z1 + z2)
                beta2 = z1 * z2

            elif zero_order == 1:
                beta0 = 0.0
                beta1 = 1.0
                beta2 = -z1

            elif zero_order == 0:
                beta0 = 0.0
                beta1 = 0.0
                beta2 = 1.0

            else:
                raise ValueError("zero_order inválido para pole_order=2.")

            beta0 = self._real_scalar(beta0, name="beta0")
            beta1 = self._real_scalar(beta1, name="beta1")
            beta2 = self._real_scalar(beta2, name="beta2")

            return beta0, beta1, beta2

        if pole_order == 1:
            if zero_order == 1:
                beta0 = 1.0
                beta1 = -z1

            elif zero_order == 0:
                beta0 = 0.0
                beta1 = 1.0

            else:
                raise ValueError("Para pole_order=1, zero_order debe ser 0 o 1.")

            beta0 = self._real_scalar(beta0, name="beta0")
            beta1 = self._real_scalar(beta1, name="beta1")

            return beta0, beta1

        raise ValueError("pole_order debe ser 1 o 2.")

    # =========================================================
    # SECCIÓN -> ESPACIO DE ESTADOS
    # =========================================================
    def _section_to_ss(self, poles_pair, zeros_pair, pole_order, zero_order):
        pole_order = int(pole_order)
        zero_order = int(zero_order)

        if zero_order > pole_order:
            raise ValueError("Cada sección debe cumplir zero_order <= pole_order.")

        if pole_order == 2:
            a1, a2 = self._den_coeffs(poles_pair, pole_order=2)
            beta0, beta1, beta2 = self._num_coeffs(
                zeros_pair=zeros_pair,
                pole_order=2,
                zero_order=zero_order,
            )

            A = np.array([
                [-a1, -a2],
                [1.0, 0.0]
            ], dtype=float)

            B = np.array([
                [1.0],
                [0.0]
            ], dtype=float)

            D = np.array([[beta0]], dtype=float)

            C = np.array([
                [beta1 - beta0 * a1, beta2 - beta0 * a2]
            ], dtype=float)

            x = np.zeros((2, 1), dtype=float)

            return {
                "pole_order": pole_order,
                "zero_order": zero_order,
                "poles": tuple(poles_pair),
                "zeros": tuple(zeros_pair),
                "a1": a1,
                "a2": a2,
                "beta0": beta0,
                "beta1": beta1,
                "beta2": beta2,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "x": x,
            }

        if pole_order == 1:
            a1, _ = self._den_coeffs(poles_pair, pole_order=1)
            beta0, beta1 = self._num_coeffs(
                zeros_pair=zeros_pair,
                pole_order=1,
                zero_order=zero_order,
            )

            A = np.array([[-a1]], dtype=float)
            B = np.array([[1.0]], dtype=float)
            D = np.array([[beta0]], dtype=float)
            C = np.array([[beta1 - beta0 * a1]], dtype=float)
            x = np.zeros((1, 1), dtype=float)

            return {
                "pole_order": pole_order,
                "zero_order": zero_order,
                "poles": tuple(poles_pair),
                "zeros": tuple(zeros_pair),
                "a1": a1,
                "a2": None,
                "beta0": beta0,
                "beta1": beta1,
                "beta2": None,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "x": x,
            }

        raise ValueError("pole_order debe ser 1 o 2.")

    # =========================================================
    # SET / UPDATE
    # =========================================================
    def set_sections(
        self,
        P_sections,
        Z_sections,
        pole_order,
        zero_order,
        Kg=None,
        reset_states=True,
    ):
        """
        Reconstruye las matrices locales A,B,C,D.
        Se llama al cargar desde MATLAB o cuando tú actualices ZPK.
        No se llama en cada sample().
        """
        P_sections = self._normalize_sections_input(P_sections, name="P_sections")
        Z_sections = self._normalize_sections_input(Z_sections, name="Z_sections")

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

        self.A_tot = None
        self.B_tot = None
        self.C_tot = None
        self.D_tot = None

        if reset_states:
            self.reset()

    def update_sections(
        self,
        P_sections=None,
        Z_sections=None,
        Kg=None,
        reset_states=False,
    ):
        """
        Para RL.

        Actualiza polos/ceros/Kg, pero conserva pole_order y zero_order.
        Es decir, la estructura de cada sección queda fija.
        """
        if self.P_sections is None or self.Z_sections is None:
            raise RuntimeError("No hay secciones inicializadas.")

        new_P = self.P_sections if P_sections is None else P_sections
        new_Z = self.Z_sections if Z_sections is None else Z_sections
        new_Kg = self.Kg if Kg is None else Kg

        self.set_sections(
            P_sections=new_P,
            Z_sections=new_Z,
            pole_order=self.pole_order,
            zero_order=self.zero_order,
            Kg=new_Kg,
            reset_states=reset_states,
        )

    # =========================================================
    # RESET / SAMPLE / OUTPUT
    # =========================================================
    def reset(self, x0=None):
        """
        Resetea estados internos.

        Dimensión total:
            nK = sum(pole_order)
        """
        nK = int(np.sum(self.pole_order)) if self.pole_order is not None else 0

        if x0 is None:
            for sec in self.local_sections:
                n = sec["A"].shape[0]
                sec["x"] = np.zeros((n, 1), dtype=float)

            self.u_hold = np.zeros(self.nu, dtype=float)
            return

        x0 = np.asarray(x0, dtype=float).reshape(-1, 1)

        if x0.shape[0] != nK:
            raise ValueError(f"x0 debe tener dimensión {nK} x 1.")

        idx = 0
        for sec in self.local_sections:
            n = sec["A"].shape[0]
            sec["x"] = x0[idx:idx+n, :].copy()
            idx += n

        self.u_hold = np.zeros(self.nu, dtype=float)

    def output(self, y):
        """
        Ejecuta un paso discreto de la cascada:

            y[k] -> H1 -> H2 -> ... -> HL -> Kg -> u[k]

        Aquí NO se reconstruyen matrices.
        Solo se usan A,B,C,D ya construidas.
        """
        y = np.asarray(y, dtype=float).reshape(-1)

        if y.shape != (self.ny,):
            raise ValueError(f"y must have shape ({self.ny},)")

        v = float(y[0])

        for sec in self.local_sections:
            A = sec["A"]
            B = sec["B"]
            C = sec["C"]
            D = sec["D"]
            x = sec["x"]

            v_out = float((C @ x + D * v).squeeze())
            x_next = A @ x + B * v

            sec["x"] = x_next
            v = v_out

        return np.array([self.Kg * v], dtype=float)

    def sample(self, y):
        """
        Un update del controlador en un instante de muestreo.

        Orden correcto:
            1. calcula u[k] con xK[k], y[k]
            2. actualiza xK[k+1]
            3. guarda u_hold para el ZOH
        """
        if self.mode != 'sos_rl':
            raise RuntimeError("sample() is only for discrete controllers.")

        y = np.asarray(y, dtype=float).reshape(-1)

        if y.shape != (self.ny,):
            raise ValueError(f"y must have shape ({self.ny},)")

        self.u_hold = self.output(y)
        return self.u_hold.copy()

    # =========================================================
    # GLOBAL SS OPCIONAL
    # =========================================================
    def build_global_ss(self, apply_global_gain=True):
        """
        Construye A_tot, B_tot, C_tot, D_tot para la cascada completa.

        Válido para secciones de orden 1 o 2, con D_l arbitrario.
        """
        L = len(self.local_sections)

        dims = [sec["A"].shape[0] for sec in self.local_sections]
        offsets = np.cumsum([0] + dims)
        n = offsets[-1]

        A_tot = np.zeros((n, n), dtype=float)
        B_tot = np.zeros((n, 1), dtype=float)
        C_tot = np.zeros((1, n), dtype=float)

        D_values = [float(sec["D"].squeeze()) for sec in self.local_sections]

        for i, sec_i in enumerate(self.local_sections):
            Ai = sec_i["A"]
            Bi = sec_i["B"]

            ii = slice(offsets[i], offsets[i+1])

            A_tot[ii, ii] = Ai

            prod_before = 1.0
            for m in range(i):
                prod_before *= D_values[m]

            B_tot[ii, :] = Bi * prod_before

            for j in range(i):
                sec_j = self.local_sections[j]
                Cj = sec_j["C"]
                jj = slice(offsets[j], offsets[j+1])

                prod_middle = 1.0
                for m in range(j + 1, i):
                    prod_middle *= D_values[m]

                A_tot[ii, jj] = Bi @ Cj * prod_middle

        for j, sec_j in enumerate(self.local_sections):
            Cj = sec_j["C"]
            jj = slice(offsets[j], offsets[j+1])

            prod_after = 1.0
            for m in range(j + 1, L):
                prod_after *= D_values[m]

            C_tot[:, jj] = Cj * prod_after

        D_tot = np.array([[np.prod(D_values) if L > 0 else 1.0]], dtype=float)

        if apply_global_gain:
            C_tot = self.Kg * C_tot
            D_tot = self.Kg * D_tot

        self.A_tot = A_tot
        self.B_tot = B_tot
        self.C_tot = C_tot
        self.D_tot = D_tot

        return A_tot, B_tot, C_tot, D_tot

    # =========================================================
    # INSPECCIÓN
    # =========================================================
    def get_states(self):
        if len(self.local_sections) == 0:
            return np.zeros((0, 1), dtype=float)

        return np.vstack([sec["x"] for sec in self.local_sections])

    def get_coeffs(self):
        out = []

        for i, sec in enumerate(self.local_sections):
            out.append({
                "index": i,
                "pole_order": sec["pole_order"],
                "zero_order": sec["zero_order"],
                "poles": sec["poles"],
                "zeros": sec["zeros"],
                "a1": sec["a1"],
                "a2": sec["a2"],
                "beta0": sec["beta0"],
                "beta1": sec["beta1"],
                "beta2": sec["beta2"],
                "D": float(sec["D"].squeeze()),
            })

        return out

    def summary(self):
        return {
            "implementation": self.implementation,
            "L": len(self.local_sections),
            "state_dimension": int(np.sum(self.pole_order)) if self.pole_order is not None else 0,
            "Kg": self.Kg,
            "Ts": self.Ts,
            "pole_order": None if self.pole_order is None else self.pole_order.tolist(),
            "zero_order": None if self.zero_order is None else self.zero_order.tolist(),
            "sections": self.get_coeffs(),
        }

    # =========================================================
    # CARGA DESDE MATLAB
    # =========================================================
    @classmethod
    def from_mat(
        cls,
        filename,
        p_key="P_sections",
        z_key="Z_sections",
        pole_order_key="pole_order",
        zero_order_key="zero_order",
        kg_key="Kg",
        ts_key="Ts",
        tol=1e-10,
        Csens=Csens,
    ):
        data = loadmat(filename)

        P_sections = np.asarray(data[p_key])
        Z_sections = np.asarray(data[z_key])

        pole_order = np.asarray(data[pole_order_key]).reshape(-1)
        zero_order = np.asarray(data[zero_order_key]).reshape(-1)

        Kg = float(np.ravel(data[kg_key])[0]) if kg_key in data else 1.0
        Ts = float(np.ravel(data[ts_key])[0]) if ts_key in data else None

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
# 6) Simulator class for SOS RL
#    This class coordinates plant + SOS controller.
# ================================================================
class ClosedLoopSimulatorSOS:
    """
    This class glues the pieces together.

    Why a class?
    - The plant class only knows physics.
    - The SOS controller class only knows cascade filter equations.
    - The simulator class knows how they interact in time.
    """

    def __init__(
        self,
        plant: CartPolePlant,
        controller: DynamicControllerSOS,
        Csens: Optional[np.ndarray] = None,
        force_limit: float | np.ndarray = F_MAX,
        control_sign: float | np.ndarray = 1.0,
    ):
        self.plant = plant
        self.controller = controller
        self.nxp = self.plant.nxp
        self.Csens = np.eye(self.nxp, dtype=float) if Csens is None else np.array(Csens, dtype=float, copy=True)
        self.force_limit = np.array(force_limit, dtype=float).reshape(-1)
        self.control_sign = np.array(control_sign, dtype=float).reshape(-1)
        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.Csens.ndim != 2:
            raise ValueError('Csens must be 2D')
            
        if self.Csens.shape != (self.controller.ny, self.nxp):
            raise ValueError(
                f'Csens must have shape ({self.controller.ny}, {self.nxp}) to map plant state to controller input'
            )
            
        if self.controller.nu != self.plant.nu:
            raise ValueError(
                f'Controller output dimension nu={self.controller.nu} must match plant input dimension nu={self.plant.nu}'
            )

        if self.force_limit.size == 1:
            self.force_limit = np.full(self.plant.nu, self.force_limit.item(), dtype=float)
            
        if self.force_limit.shape != (self.plant.nu,):
            raise ValueError(f'force_limit must be scalar or have shape ({self.plant.nu},)')
            
        if np.any(self.force_limit <= 0.0):
            raise ValueError('Every entry of force_limit must be > 0')

        if self.control_sign.size == 1:
            self.control_sign = np.full(self.plant.nu, self.control_sign.item(), dtype=float)
            
        if self.control_sign.shape != (self.plant.nu,):
            raise ValueError(f'control_sign must be scalar or have shape ({self.plant.nu},)')

    def sensor_output(self, xp: Optional[np.ndarray] = None) -> np.ndarray:
        xp = self.plant.xp if xp is None else np.asarray(xp, dtype=float).reshape(-1)
        
        if xp.shape != (self.nxp,):
            raise ValueError(f'xp must have shape ({self.nxp},)')
            
        y = self.Csens @ xp
        
        return np.asarray(y, dtype=float).reshape(self.controller.ny)

    def sat(self, u_raw: np.ndarray) -> np.ndarray:
        u_raw = np.asarray(u_raw, dtype=float).reshape(-1)
        
        if u_raw.shape != (self.plant.nu,):
            raise ValueError(f'u_raw must have shape ({self.plant.nu},)')
            
        return np.clip(u_raw, -self.force_limit, self.force_limit)

    def run(self, xp0=None, xk0=None, t_final=T_FINAL):
        self.plant.reset(xp0)
        self.controller.reset(xk0)
    
        t = 0.0
        next_sample = 0.0
        Tu = []
    
        log_t = []
        log_xp = []
        log_xk = []
        log_u_raw = []
        log_u_sat = []
        log_solver_mode = []
    
        # Primer sample en t = 0
        yk = self.sensor_output(self.plant.xp)
        self.controller.sample(yk)
        next_sample = self.controller.Ts
        Tu.append(t)
    
        while t < t_final - 1e-15:
    
            # Fuerza constante durante el intervalo actual
            u_raw = self.control_sign * self.controller.u_hold
            u_sat = self.sat(u_raw)
    
            # Log al inicio del intervalo
            log_t.append(t)
            log_xp.append(self.plant.xp.copy())
            log_xk.append(self.controller.get_states().copy())
            log_u_raw.append(u_raw.copy())
            log_u_sat.append(u_sat.copy())
            log_solver_mode.append('plant-only-cn')
    
            # Integramos hasta la próxima muestra
            h = min(self.plant.dt, next_sample - t, t_final - t)
    
            if h <= 1e-15:
                yk = self.sensor_output(self.plant.xp)
                self.controller.sample(yk)
                Tu.append(next_sample)
                next_sample += self.controller.Ts
                continue
    
            self.plant.step_crank_nicolson(u_sat, h=h)
            t += h
    
            # Al llegar al instante de muestreo, actualizamos u[k+1]
            if t >= next_sample - 1e-12:
                yk = self.sensor_output(self.plant.xp)
                self.controller.sample(yk)
                Tu.append(next_sample)
                next_sample += self.controller.Ts
    
        return {
            't': np.asarray(log_t, dtype=float),
            'xp': np.asarray(log_xp, dtype=float),
            'xk': np.asarray(log_xk, dtype=float),
            'u_raw': np.asarray(log_u_raw, dtype=float),
            'u_sat': np.asarray(log_u_sat, dtype=float),
            'solver_mode': np.asarray(log_solver_mode, dtype=object),
            'Tu': np.asarray(Tu, dtype=float),
        }

# ================================================================
# 7) Plot helpers
# ================================================================
def plot_results(results: Dict[str, np.ndarray], title: str, savepath: str) -> None:
    t = results['t']
    xp = results['xp']
    u_raw = results['u_raw']
    u_sat = results['u_sat']

    if u_raw.ndim == 1:
        u_raw = u_raw[:, None]
    if u_sat.ndim == 1:
        u_sat = u_sat[:, None]

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    for j in range(u_raw.shape[1]):
        axes[0].plot(t, u_raw[:, j], label=f'u_raw[{j}]', lw=1.4)
        axes[0].plot(t, u_sat[:, j], label=f'u_sat[{j}]', lw=1.4)

    axes[0].axhline(F_MAX, ls='--', color='k', alpha=0.4)
    axes[0].axhline(-F_MAX, ls='--', color='k', alpha=0.4)
    axes[0].set_ylabel('Input')
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    labels = ['x [m]', 'xdot [m/s]', 'theta [rad]', 'thetadot [rad/s]']
    for i in range(4):
        axes[i + 1].plot(t, xp[:, i], lw=1.3)
        axes[i + 1].set_ylabel(labels[i])
        axes[i + 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_results_csv(results: Dict[str, np.ndarray], savepath: str) -> None:
    t = results['t']
    xp = results['xp']
    u_raw = results['u_raw']
    u_sat = results['u_sat']

    if u_raw.ndim == 1:
        u_raw = u_raw[:, None]
    if u_sat.ndim == 1:
        u_sat = u_sat[:, None]

    data_cols = [t[:, None], xp, u_raw, u_sat]
    data = np.column_stack(data_cols)

    header = ['t', 'x', 'xdot', 'theta', 'thetadot']
    header += [f'u_raw_{j}' for j in range(u_raw.shape[1])]
    header += [f'u_sat_{j}' for j in range(u_sat.shape[1])]

    np.savetxt(savepath, data, delimiter=',', header=','.join(header), comments='')


# ================================================================
# 9) Main: minimal test only
# ================================================================
def main() -> None:
    print('=' * 70)
    print('Minimal cart-pole test: continuous controller and discrete controller')
    print('=' * 70)
    print(f'dt      = {DEFAULT_DT:.6f} s  (plant CN integration step)')
    print(f'Ts      = {DEFAULT_TS:.6f} s  (controller sample time for discrete mode)')
    print(f'T_final = {T_FINAL:.6f} s = N_ctrl_steps * Ts')
    print(f'N_ctrl_steps = {N_CTRL_STEPS}')
    
    # ---------------- discrete controller SOS RL ----------------
    
    #P_sections, Z_sections, Kg, Ts = load_controller_matrices('controller.mat')
    
    ctrl_sos = DynamicControllerSOS.from_mat("controller_sos.mat")
    
    
    plant_sos = CartPolePlant(dt=DEFAULT_DT, Ts=DEFAULT_TS)
    
    sim_sos = ClosedLoopSimulatorSOS(
        plant=plant_sos,
        controller=ctrl_sos,
        Csens=Csens,
        force_limit=F_MAX,
    )
    
    res_sos = sim_sos.run(xp0=X0,xk0=None,t_final=T_FINAL,)
    
    BUILD_CSV = r"C:\Users\Personal\OneDrive\Documents\ONERA - Flow Stabilization\MATLAB\Cartpole"
    save_results_csv(res_sos, os.path.join(BUILD_CSV, 'discrete_controller_SOS_signals.csv'))

    

if __name__ == '__main__':
    main()