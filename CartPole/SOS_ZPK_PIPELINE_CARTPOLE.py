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
BUILD_CSV = r"C:\Users\Personal\OneDrive\Documents\ONERA - Flow Stabilization\MATLAB\Cartpole"
os.makedirs(BUILD_DIR, exist_ok=True)




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
        - Durante sample(), solo se propagan estados. Sample() está separado del avance temporal PPO
        - Mientras que para RL un step será un rollout completo donde propone un \delta zpk
        -sample() actualiza el controlador - closed-loop simulation por rollout (manteniendo lógica original)
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
        
        #Dimensions I/O for controller (Csens + U: Measurable states + Real Inputs)
        self.ny = np.shape(Csens)[0]
        self.nu = 1
        self.Ts = Ts
        self.tol = float(tol)
        self.Kg = float(Kg) #Assumes Kg = 1; Change for None in 2nd iteration

        if self.ny != 1: #2nd iteration generalises for np.shape(Csens)[0]. Based on measurable states!!
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

        self.u_hold = np.zeros(self.nu, dtype=float) #Descripción MIMO entrada

        if P_sections is not None or Z_sections is not None: #Condición None INDICA IMPORTAR ZPK-SOS MATLAB
            if P_sections is None or Z_sections is None:
                raise ValueError("Debes pasar P_sections y Z_sections a la vez.")
            if pole_order is None or zero_order is None:
                raise ValueError("Debes pasar pole_order y zero_order.")
                
            """ Updates if zpk + order specified. Otherwise, information from MATLAB"""    
            self.set_sections(
                P_sections=P_sections,
                Z_sections=Z_sections,
                pole_order=pole_order,
                zero_order=zero_order,
                Kg=Kg,
                reset_states=reset_states,
            )

    # =========================================================
    # ZPK NUMERICAL VALIDATION: FORMAT & TOLERANCES
    # =========================================================
    
    def _is_nan(self, value):
        return bool(np.isnan(value)) #(True,False where Nan: Important to identify real SOS orders)

    def _is_finite_number(self, value):
        return not self._is_nan(value) #General number. Identifies existence ZPK and REAL VALUE

    def _real_scalar(self, value, name="coef"):
        if self._is_nan(value):
            raise ValueError(f"{name} es NaN, pero se esperaba un valor finito.") #TF coefficients must be real! 

        if abs(np.imag(value)) > self.tol:
            raise ValueError(f"{name} no es real dentro de tolerancia: {value}") #Tolerance - complex pair

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

        if abs(x2 - np.conjugate(x1)) > self.tol: #Guarantees numerical stability for complex pair
            raise ValueError(f"{name} no es un par conjugado válido: {pair}")
            
    # =========================================================
    # ZPK SECTIONS & ORDER CONSTRUCTION + VALIDATION
    # =========================================================        

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

        return arr.astype(complex) #Returns general P & Z_sections with complex array format (General form)!

    def _normalize_order_input(self, order, L, name="order"):
        arr = np.asarray(order).reshape(-1)

        if arr.size != L:
            raise ValueError(f"{name} debe tener longitud L={L}.")

        arr = arr.astype(int)

        return arr #Returns int array to establish section orders in {0,1,2} (General form)!

    # =========================================================
    # KEEP INITIAL K0 STRUCTURE FIXED + VALIDATION 
    # =========================================================
    
    def _validate_structure(self, P_sections, Z_sections, pole_order, zero_order):
        L = P_sections.shape[0] #Number SOS sections described MATLAB
        
        #IMPORTANT: FOR VALIDATION WE MUST APPLY NAN JUST IN SECOND COLUMN. 
        #COMING FROM MATLAB, WE CAN HAVE JUST (p1, nan) or (z1, nan) TO KEEP VALIDATION!

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
                
                
            #VALIDATION FOR LOCAL ORDER. 
            if zo > po:
                raise ValueError(
                    f"Sección {i}: zero_order={zo} > pole_order={po}. "
                    "Cada sección debe ser propia localmente."
                )
                
            #ZPK EXTRACTION TO VALIDATE FORMAT
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
    # TF COEFICIENTS CONSTRUCTED FROM ZPK INFORMATION
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
        
        #TF COEFICIENTS CONSTRUCTED FROM ZPK INFORMATION

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
# 7) RL-ZPK PARAMETRIZATION LAYER
#    PPO does not act on u[k]. It proposes structured perturbations
#    of the SOS/ZPK controller parameters.
# ================================================================

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # gymnasium is optional until RL training is requested
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
except Exception:  # stable-baselines3 is optional until train_sos_ppo() is called
    PPO = None

from dataclasses import dataclass
from typing import Callable, List


EPS = 1e-12


def _has_nan_complex(x) -> bool:
    x = np.asarray(x)
    return bool(np.any(np.isnan(np.real(x))) or np.any(np.isnan(np.imag(x))))


def _is_finite_complex(x) -> bool:
    x = np.asarray(x)
    return bool(np.all(np.isfinite(np.real(x))) and np.all(np.isfinite(np.imag(x))))


def _is_real_scalar(x, tol: float = 1e-10) -> bool:
    if _has_nan_complex(x):
        return False
    return bool(abs(np.imag(x)) <= tol)


def _is_conjugate_pair(x1, x2, tol: float = 1e-10) -> bool:
    if _has_nan_complex(x1) or _has_nan_complex(x2):
        return False
    return bool(abs(x2 - np.conjugate(x1)) <= tol)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


@dataclass
class ParamBlock:
    """
    One structured real block in the RL parametrization.

    Rules:
      - one real pole/zero               -> dim = 1
      - two independent real poles/zeros -> dim = 2
      - one complex-conjugate pair       -> dim = 2, represented by (radius, angle)
      - global gain                      -> dim = 1, represented by log(abs(Kg))
    """
    kind: str
    section: Optional[int]
    slots: Optional[List[int]]
    dim: int
    names: List[str]


class ZPKTopology:
    """
    Detects and stores the fixed ZPK topology of the SOS controller.

    This class does not simulate anything. It only converts the initial
    controller structure into a list of ParamBlock objects used by RL.
    """

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
        self.Ts = Ts
        self.param_blocks: List[ParamBlock] = []
        self.detect_blocks()

    @classmethod
    def from_controller(cls, controller: DynamicControllerSOS, tol: float = 1e-10):
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

    @staticmethod
    def _normalize_sections_input(sections, name="sections"):
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

    @property
    def action_dim(self) -> int:
        return int(sum(block.dim for block in self.param_blocks))

    def param_names(self) -> List[str]:
        names = []
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

            # ---------------- poles ----------------
            if po == 1:
                if not _is_real_scalar(p1, self.tol):
                    raise ValueError(f"Section {l}: pole_order=1 requires one real pole.")
                self.param_blocks.append(
                    ParamBlock(
                        kind="pole_real",
                        section=l,
                        slots=[0],
                        dim=1,
                        names=[f"p{l}_real"],
                    )
                )
            elif po == 2:
                if _is_real_scalar(p1, self.tol) and _is_real_scalar(p2, self.tol):
                    self.param_blocks.append(
                        ParamBlock(
                            kind="pole_real_pair",
                            section=l,
                            slots=[0, 1],
                            dim=2,
                            names=[f"p{l}_1_real", f"p{l}_2_real"],
                        )
                    )
                elif _is_conjugate_pair(p1, p2, self.tol):
                    self.param_blocks.append(
                        ParamBlock(
                            kind="pole_complex_pair",
                            section=l,
                            slots=[0, 1],
                            dim=2,
                            names=[f"rp{l}", f"thetap{l}"],
                        )
                    )
                else:
                    raise ValueError(f"Section {l}: invalid pole pair {p1}, {p2}.")
            else:
                raise ValueError(f"Section {l}: pole_order must be 1 or 2.")

            # ---------------- zeros ----------------
            if zo == 0:
                pass
            elif zo == 1:
                if not _is_real_scalar(z1, self.tol):
                    raise ValueError(f"Section {l}: zero_order=1 requires one real zero.")
                self.param_blocks.append(
                    ParamBlock(
                        kind="zero_real",
                        section=l,
                        slots=[0],
                        dim=1,
                        names=[f"z{l}_real"],
                    )
                )
            elif zo == 2:
                if _is_real_scalar(z1, self.tol) and _is_real_scalar(z2, self.tol):
                    self.param_blocks.append(
                        ParamBlock(
                            kind="zero_real_pair",
                            section=l,
                            slots=[0, 1],
                            dim=2,
                            names=[f"z{l}_1_real", f"z{l}_2_real"],
                        )
                    )
                elif _is_conjugate_pair(z1, z2, self.tol):
                    self.param_blocks.append(
                        ParamBlock(
                            kind="zero_complex_pair",
                            section=l,
                            slots=[0, 1],
                            dim=2,
                            names=[f"rz{l}", f"thetaz{l}"],
                        )
                    )
                else:
                    raise ValueError(f"Section {l}: invalid zero pair {z1}, {z2}.")
            else:
                raise ValueError(f"Section {l}: zero_order must be 0, 1, or 2.")

        self.param_blocks.append(
            ParamBlock(
                kind="gain_log",
                section=None,
                slots=None,
                dim=1,
                names=["log_abs_Kg"],
            )
        )

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
    """
    Maps stable-baselines3 actions into structured ZPK perturbations.

    PPO action a in [-1, 1]^n is converted to delta_theta, then projected
    and reconstructed as P_sections, Z_sections, Kg.
    """

    def __init__(
        self,
        topology: ZPKTopology,
        scale_p_real: float = 0.005,
        scale_p_radius: float = 0.005,
        scale_p_angle: float = 0.01,
        scale_z_real: float = 0.01,
        scale_z_radius: float = 0.01,
        scale_z_angle: float = 0.02,
        scale_logKg: float = 0.03,
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
        P_sections = self.topology._normalize_sections_input(P_sections, name="P_sections")
        Z_sections = self.topology._normalize_sections_input(Z_sections, name="Z_sections")
        theta = []

        for block in self.topology.param_blocks:
            l = block.section

            if block.kind == "pole_real":
                theta.append(float(np.real(P_sections[l, 0])))

            elif block.kind == "pole_real_pair":
                theta.extend([
                    float(np.real(P_sections[l, 0])),
                    float(np.real(P_sections[l, 1])),
                ])

            elif block.kind == "pole_complex_pair":
                p = P_sections[l, 0]
                theta.extend([float(np.abs(p)), float(np.angle(p))])

            elif block.kind == "zero_real":
                theta.append(float(np.real(Z_sections[l, 0])))

            elif block.kind == "zero_real_pair":
                theta.extend([
                    float(np.real(Z_sections[l, 0])),
                    float(np.real(Z_sections[l, 1])),
                ])

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
                P_new[l, 0] = theta[idx]
                P_new[l, 1] = np.nan
                idx += 1

            elif block.kind == "pole_real_pair":
                P_new[l, 0] = theta[idx]
                P_new[l, 1] = theta[idx + 1]
                idx += 2

            elif block.kind == "pole_complex_pair":
                r = theta[idx]
                angle = theta[idx + 1]
                p = r * np.exp(1j * angle)
                P_new[l, 0] = p
                P_new[l, 1] = np.conjugate(p)
                idx += 2

            elif block.kind == "zero_real":
                Z_new[l, 0] = theta[idx]
                Z_new[l, 1] = np.nan
                idx += 1

            elif block.kind == "zero_real_pair":
                Z_new[l, 0] = theta[idx]
                Z_new[l, 1] = theta[idx + 1]
                idx += 2

            elif block.kind == "zero_complex_pair":
                r = theta[idx]
                angle = theta[idx + 1]
                z = r * np.exp(1j * angle)
                Z_new[l, 0] = z
                Z_new[l, 1] = np.conjugate(z)
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
        """Light normalization for PPO observations."""
        theta = self.project_theta(theta)
        out = np.zeros_like(theta, dtype=np.float64)
        idx = 0
        for block in self.topology.param_blocks:
            if block.kind in ("pole_real",):
                out[idx] = theta[idx] / max(self.r_p_max, EPS)
                idx += 1
            elif block.kind in ("pole_real_pair",):
                out[idx] = theta[idx] / max(self.r_p_max, EPS)
                out[idx + 1] = theta[idx + 1] / max(self.r_p_max, EPS)
                idx += 2
            elif block.kind in ("pole_complex_pair",):
                out[idx] = theta[idx] / max(self.r_p_max, EPS)
                out[idx + 1] = theta[idx + 1] / np.pi
                idx += 2
            elif block.kind in ("zero_real",):
                out[idx] = theta[idx] / max(self.r_z_max, EPS)
                idx += 1
            elif block.kind in ("zero_real_pair",):
                out[idx] = theta[idx] / max(self.r_z_max, EPS)
                out[idx + 1] = theta[idx + 1] / max(self.r_z_max, EPS)
                idx += 2
            elif block.kind in ("zero_complex_pair",):
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
        delta_theta_raw = self.action_to_delta_theta(action)
        theta_candidate = self.project_theta(theta_base + delta_theta_raw)
        delta_theta = theta_candidate - theta_base
        P_candidate, Z_candidate, Kg_candidate = self.theta_to_zpk(theta_candidate)
        return {
            "theta_candidate": theta_candidate,
            "delta_theta": delta_theta,
            "P_sections": P_candidate,
            "Z_sections": Z_candidate,
            "Kg": Kg_candidate,
        }


class ControllerEvaluatorSOS:
    """
    Evaluates one candidate SOS/ZPK controller with a full closed-loop rollout.

    One evaluate(...) call is what one RL env.step(action) uses internally.
    If H_eval = 5 s and Ts = 1e-3, the controller is sampled 5000 times.
    """

    def __init__(
        self,
        topology: ZPKTopology,
        plant_factory: Optional[Callable[[], CartPolePlant]] = None,
        Csens_eval: Optional[np.ndarray] = None,
        xp0: Optional[np.ndarray] = None,
        xk0: Optional[np.ndarray] = None,
        H_eval: float = 1.0,
        lambda_u: float = 1e-3,
        force_limit: float | np.ndarray = F_MAX,
        control_sign: float | np.ndarray = 1.0,
        y_limit: Optional[float] = None,
        x_limit: Optional[float] = None,
    ):
        self.topology = topology
        self.plant_factory = plant_factory or (lambda: CartPolePlant(dt=DEFAULT_DT, Ts=DEFAULT_TS))
        self.Csens = np.array(Csens if Csens_eval is None else Csens_eval, dtype=float, copy=True)
        self.xp0 = np.array(X0 if xp0 is None else xp0, dtype=float).reshape(-1).copy()
        self.xk0 = xk0
        self.H_eval = float(H_eval)
        self.lambda_u = float(lambda_u)
        self.force_limit = force_limit
        self.control_sign = control_sign
        self.y_limit = y_limit
        self.x_limit = x_limit

        if self.H_eval <= 0.0:
            raise ValueError("H_eval must be > 0.")

    def make_controller(self, P_sections, Z_sections, Kg) -> DynamicControllerSOS:
        return DynamicControllerSOS(
            P_sections=P_sections,
            Z_sections=Z_sections,
            pole_order=self.topology.pole_order,
            zero_order=self.topology.zero_order,
            Kg=Kg,
            Ts=self.topology.Ts if self.topology.Ts is not None else DEFAULT_TS,
            Csens=self.Csens,
            reset_states=True,
        )

    def evaluate(self, P_sections, Z_sections, Kg) -> Dict[str, Any]:
        try:
            plant = self.plant_factory()
            controller = self.make_controller(P_sections, Z_sections, Kg)
            simulator = ClosedLoopSimulatorSOS(
                plant=plant,
                controller=controller,
                Csens=self.Csens,
                force_limit=self.force_limit,
                control_sign=self.control_sign,
            )
            results = simulator.run(xp0=self.xp0, xk0=self.xk0, t_final=self.H_eval)
        except Exception as exc:
            return {
                "valid": False,
                "J": float("inf"),
                "reason": f"simulation_failed: {type(exc).__name__}: {exc}",
            }

        try:
            xp = np.asarray(results["xp"], dtype=float)
            u_sat = np.asarray(results["u_sat"], dtype=float)
            if u_sat.ndim == 1:
                u_sat = u_sat[:, None]
            y = (self.Csens @ xp.T).T

            if xp.size == 0 or y.size == 0 or u_sat.size == 0:
                return {"valid": False, "J": float("inf"), "reason": "empty_rollout"}

            if not (np.all(np.isfinite(xp)) and np.all(np.isfinite(y)) and np.all(np.isfinite(u_sat))):
                return {"valid": False, "J": float("inf"), "reason": "nan_or_inf"}

            y_max = float(np.max(np.abs(y)))
            x_max = float(np.max(np.abs(xp)))
            u_max = float(np.max(np.abs(u_sat)))

            if self.y_limit is not None and y_max > self.y_limit:
                return {"valid": False, "J": float("inf"), "reason": "y_limit_exceeded", "y_max": y_max}
            if self.x_limit is not None and x_max > self.x_limit:
                return {"valid": False, "J": float("inf"), "reason": "x_limit_exceeded", "x_max": x_max}

            force_limit = np.asarray(self.force_limit, dtype=float).reshape(-1)
            if force_limit.size == 1:
                force_scale = float(force_limit.item())
            else:
                force_scale = np.maximum(force_limit.reshape(1, -1), EPS)

            u_norm = u_sat / force_scale
            J_x = float(np.mean(np.square(xp[:,0])))
            J_y = float(np.mean(np.square(y)))
            J_u = float(np.mean(np.square(u_norm)))
            J = J_x + J_y + self.lambda_u * J_u

            return {
                "valid": True,
                "J": float(J),
                "J_y": J_y,
                "J_u": J_u,
                "y_rms": rms(y),
                "u_rms": rms(u_sat),
                "y_max": y_max,
                "u_max": u_max,
                "x_max": x_max,
                "reason": "ok",
            }
        except Exception as exc:
            return {
                "valid": False,
                "J": float("inf"),
                "reason": f"metric_failed: {type(exc).__name__}: {exc}",
            }


class SOSZPKPPOEnv(gym.Env if gym is not None else object):
    """
    Gymnasium environment for parameter-level PPO.

    One env.step(action) = one full closed-loop evaluation of one candidate K.
    episode_len = number of candidates tested before updating theta_base.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: ZPKTopology,
        mapper: ZPKActionMapper,
        evaluator: ControllerEvaluatorSOS,
        episode_len: int = 4,
        alpha: float = 0.1,
        beta: float = 5.0,
        reward_fail: float = -10.0,
        allow_worse_base_update: bool = False,
    ):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to use SOSZPKPPOEnv.")
        super().__init__()
        self.topology = topology
        self.mapper = mapper
        self.evaluator = evaluator
        self.episode_len = int(episode_len)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.reward_fail = float(reward_fail)
        self.allow_worse_base_update = bool(allow_worse_base_update)

        if self.episode_len <= 0:
            raise ValueError("episode_len must be >= 1.")
        if self.mapper.action_dim != self.topology.action_dim:
            raise ValueError("mapper.action_dim and topology.action_dim do not match.")

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.topology.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.topology.action_dim + 3,),
            dtype=np.float32,
        )

        self.theta_base = self.mapper.zpk_to_theta(
            self.topology.P_template,
            self.topology.Z_template,
            self.topology.Kg0,
        )
        self.P_base, self.Z_base, self.Kg_base = self.mapper.theta_to_zpk(self.theta_base)
        base_metrics = self.evaluator.evaluate(self.P_base, self.Z_base, self.Kg_base)
        if not base_metrics["valid"]:
            raise RuntimeError(f"Initial K_base is invalid: {base_metrics.get('reason')}")
        self.J_base = float(base_metrics["J"])

        self.step_in_episode = 0
        self.delta_buffer: List[np.ndarray] = []
        self.reward_buffer: List[float] = []
        self.J_buffer: List[float] = []
        self.last_reward = 0.0
        self.last_J = self.J_base
        self.last_valid = 1.0

        self.history = {
            "J_base": [self.J_base],
            "mean_reward": [],
            "best_reward": [],
            "Kg_base": [self.Kg_base],
            "accepted_update": [],
        }

    def _get_obs(self) -> np.ndarray:
        theta_norm = self.mapper.normalize_theta(self.theta_base)
        J_norm = self.last_J / (abs(self.J_base) + EPS)
        obs = np.concatenate([
            theta_norm,
            np.asarray([
                float(np.clip(self.last_reward, -1e6, 1e6)),
                float(np.clip(J_norm, -1e6, 1e6)),
                float(self.step_in_episode / max(self.episode_len, 1)),
            ], dtype=np.float64),
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_in_episode = 0
        self.delta_buffer = []
        self.reward_buffer = []
        self.J_buffer = []
        self.last_reward = 0.0
        self.last_J = self.J_base
        self.last_valid = 1.0
        return self._get_obs(), {}

    def step(self, action):
        candidate = self.mapper.build_candidate(self.theta_base, action)
        metrics = self.evaluator.evaluate(
            P_sections=candidate["P_sections"],
            Z_sections=candidate["Z_sections"],
            Kg=candidate["Kg"],
        )

        if metrics["valid"]:
            J_candidate = float(metrics["J"])
            reward = float(self.J_base - J_candidate)
        else:
            J_candidate = float("inf")
            reward = self.reward_fail

        self.delta_buffer.append(candidate["delta_theta"].copy())
        self.reward_buffer.append(float(reward))
        self.J_buffer.append(float(J_candidate))
        self.last_reward = float(reward)
        self.last_J = float(J_candidate if np.isfinite(J_candidate) else 10.0 * (abs(self.J_base) + 1.0))
        self.last_valid = float(metrics["valid"])
        self.step_in_episode += 1

        terminated = False
        truncated = self.step_in_episode >= self.episode_len
        accepted_update = None
        if truncated:
            accepted_update = self._end_episode_update()

        info = {
            "J_candidate": J_candidate,
            "J_base": self.J_base,
            "reward": reward,
            "valid": bool(metrics["valid"]),
            "reason": metrics.get("reason", "ok"),
            "accepted_update": accepted_update,
        }
        for key in ("J_y", "J_u", "y_rms", "u_rms", "y_max", "u_max"):
            if key in metrics:
                info[key] = metrics[key]

        return self._get_obs(), float(reward), terminated, truncated, info

    def _end_episode_update(self) -> bool:
        if len(self.delta_buffer) == 0:
            return False

        rewards = np.asarray(self.reward_buffer, dtype=np.float64)
        deltas = np.vstack(self.delta_buffer)
        rewards_safe = np.where(np.isfinite(rewards), rewards, self.reward_fail)

        rmax = float(np.max(rewards_safe))
        weights = np.exp(self.beta * (rewards_safe - rmax))
        weights = weights / (np.sum(weights) + EPS)
        delta_mean = np.sum(weights[:, None] * deltas, axis=0)

        theta_candidate_base = self.mapper.project_theta(self.theta_base + self.alpha * delta_mean)
        P_new, Z_new, Kg_new = self.mapper.theta_to_zpk(theta_candidate_base)
        metrics_new = self.evaluator.evaluate(P_new, Z_new, Kg_new)

        accepted = False
        if metrics_new["valid"]:
            J_new = float(metrics_new["J"])
            if self.allow_worse_base_update or J_new <= self.J_base:
                self.theta_base = theta_candidate_base
                self.P_base, self.Z_base, self.Kg_base = P_new, Z_new, Kg_new
                self.J_base = J_new
                accepted = True

        self.history["J_base"].append(float(self.J_base))
        self.history["mean_reward"].append(float(np.mean(rewards_safe)))
        self.history["best_reward"].append(float(np.max(rewards_safe)))
        self.history["Kg_base"].append(float(self.Kg_base))
        self.history["accepted_update"].append(bool(accepted))
        return bool(accepted)

    def get_current_controller(self):
        return self.mapper.theta_to_zpk(self.theta_base)


# ================================================================
# 8) Convenience functions: environment creation and PPO training
# ================================================================

def make_sos_rl_env_from_mat(
    mat_filename: str = "controller_sos.mat",
    H_eval: float = 1.0,
    episode_len: int = 4,
    lambda_u: float = 1e-3,
    alpha: float = 0.1,
    beta: float = 5.0,
    reward_fail: float = -10.0,
    Csens_eval: Optional[np.ndarray] = None,
    xp0: Optional[np.ndarray] = None,
) -> SOSZPKPPOEnv:
    """Create the full RL environment from a MATLAB SOS/ZPK controller file."""
    ctrl0 = DynamicControllerSOS.from_mat(mat_filename, Csens=Csens if Csens_eval is None else Csens_eval)
    topology = ZPKTopology.from_controller(ctrl0)
    mapper = ZPKActionMapper(topology)

    evaluator = ControllerEvaluatorSOS(
        topology=topology,
        plant_factory=lambda: CartPolePlant(dt=DEFAULT_DT, Ts=DEFAULT_TS),
        Csens_eval=Csens if Csens_eval is None else Csens_eval,
        xp0=X0 if xp0 is None else xp0,
        H_eval=H_eval,
        lambda_u=lambda_u,
        force_limit=F_MAX,
    )

    env = SOSZPKPPOEnv(
        topology=topology,
        mapper=mapper,
        evaluator=evaluator,
        episode_len=episode_len,
        alpha=alpha,
        beta=beta,
        reward_fail=reward_fail,
        allow_worse_base_update=False,
    )
    return env


def train_sos_ppo(
    mat_filename: str = "controller_sos.mat",
    total_timesteps: int = 256,
    H_eval: float = 1.0,
    episode_len: int = 4,
    ppo_n_steps: int = 16,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    verbose: int = 1,
):
    """
    Minimal PPO training entry point.

    For early debugging, keep total_timesteps, H_eval, and episode_len small.
    Increase H_eval to 5.0 only after validating the pipeline.
    """
    if PPO is None:
        raise ImportError("stable-baselines3 is required to train PPO.")

    env = make_sos_rl_env_from_mat(
        mat_filename=mat_filename,
        H_eval=H_eval,
        episode_len=episode_len,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=ppo_n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, env


def random_search_sos_env(env: SOSZPKPPOEnv, n_episodes: int = 5, seed: int = 0) -> Dict[str, Any]:
    """
    Lightweight baseline/debugger: random actions using exactly the same env pipeline.
    Useful before running PPO.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    return env.history


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
# 10) Main: choose minimal simulation, random RL pipeline test, or PPO training
# ================================================================
def main() -> None:
    print('=' * 70)
    print('Cart-pole SOS/ZPK pipeline with optional RL layer')
    print('=' * 70)
    print(f'dt      = {DEFAULT_DT:.6f} s  (plant CN integration step)')
    print(f'Ts      = {DEFAULT_TS:.6f} s  (controller sample time for discrete mode)')
    print(f'T_final = {T_FINAL:.6f} s = N_ctrl_steps * Ts')
    print(f'N_ctrl_steps = {N_CTRL_STEPS}')

    mat_file = "controller_sos.mat"
    
    

    # ------------------------------------------------------------------
    # Toggle only one of these at a time.
    # ------------------------------------------------------------------
    RUN_MINIMAL_CLOSED_LOOP = False
    RUN_RANDOM_RL_PIPELINE_TEST = False
    RUN_PPO_TRAINING = True

    if not os.path.exists(mat_file):
        print(f"MAT file not found: {mat_file}")
        print("Place controller_sos.mat next to this script, or change mat_file in main().")
        return

    if RUN_MINIMAL_CLOSED_LOOP:
        ctrl_sos = DynamicControllerSOS.from_mat(mat_file)
        plant_sos = CartPolePlant(dt=DEFAULT_DT, Ts=DEFAULT_TS)
        sim_sos = ClosedLoopSimulatorSOS(
            plant=plant_sos,
            controller=ctrl_sos,
            Csens=Csens,
            force_limit=F_MAX,
        )
        res_sos = sim_sos.run(xp0=X0, xk0=None, t_final=T_FINAL)
        csv_path = os.path.join(BUILD_CSV, 'discrete_controller_SOS_signals.csv')
        save_results_csv(res_sos, csv_path)
        print(f"Saved closed-loop CSV to: {csv_path}")
        

    if RUN_RANDOM_RL_PIPELINE_TEST:
        env = make_sos_rl_env_from_mat(
            mat_filename=mat_file,
            H_eval=1.0,
            episode_len=4,
            lambda_u=1e-3,
            alpha=0.1,
            beta=5.0,
        )
        print('Topology summary:')
        print(env.topology.summary())
        #history = random_search_sos_env(env, n_episodes=3, seed=1)
        #print('Random RL pipeline history:')
        #print(history)
        P_final, Z_final, Kg_final = env.get_current_controller()
        print('Final Kg after random pipeline:', Kg_final)
        print('Final P_sec after random pipeline:', P_final)
        print('Final Z_sec after random pipeline:', Z_final)

    if RUN_PPO_TRAINING:
        model, env = train_sos_ppo(
            mat_filename=mat_file,
            total_timesteps=128,
            H_eval=1.0,
            episode_len=1, #4
            ppo_n_steps=16,
            batch_size=16,
            verbose=1,
        )
        P_final, Z_final, Kg_final = env.get_current_controller()
        print('PPO training done.')
        print('Final Kg:', Kg_final)
        print('Final P:', P_final)
        print('Final Z:', Z_final)
        
        


if __name__ == '__main__':
    main()
#%%
