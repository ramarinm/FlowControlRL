# PPO SOS/ZPK: pipeline de investigación

Modificación de la implementación adjunta, con `controller_lti`, `model_free` y
`residual_rl`. No se ha sustituido PPO ni el algoritmo de guía. El código sigue
siendo SISO; cambiar el número de estados de la planta no cambia el ambiente.

## Mathematical Formulation and Research Objectives

The following describes the research workflow. Initial controller synthesis and
ZOH discretization are performed externally; the Python pipeline consumes a
compatible discrete controller. Parameter continuation is performed manually by
loading the controller obtained at the preceding parameter value.

A nonlinear plant is considered, with measurements depending only on its state:

$$
\dot{x}(t)=f(x(t),u(t),\mu), \qquad y(t)=g(x(t))
$$

$$
x\in\mathbb{R}^{n_x}, \qquad
u\in\mathbb{R}^{n_u}, \qquad
y\in\mathbb{R}^{n_y}
$$

The continuous LTI controller receives the measured output directly:

$$
\dot{x}_c=A_cx_c+B_cy, \qquad
u=C_cx_c+D_cy
$$

The continuous closed loop can be expressed using an augmented state:

$$
\frac{d}{dt}
\begin{bmatrix}x\\x_c\end{bmatrix}
=
\begin{bmatrix}
f(x,C_cx_c+D_cg(x),\mu)\\
A_cx_c+B_cg(x)
\end{bmatrix}
$$

### Sampled-Data Implementation

This work adopts a **continuous plant coupled to a discrete LTI controller through
a zero-order hold (ZOH)**, consistent with the sampled interaction commonly used
in reinforcement learning:

$$
\begin{aligned}
t_k&=kT_s, \qquad y_k=g(x(t_k)),\\
u_k&=C_dx_c[k]+D_dy_k,\\
x_c[k+1]&=A_dx_c[k]+B_dy_k,\\
u(t)&=u_k,\qquad t\in[t_k,t_{k+1}),\\
\dot{x}(t)&=f(x(t),u_k,\mu).
\end{aligned}
$$

The numerical solver integrates the plant between controller updates. Rollouts
are used for PPO training, and the resulting LTI controller is validated with
fixed coefficients.

### Methodology

1. Synthesize an initial controller from the linearized model in the design region:

   $$
   \mu\leq\mu_c
   $$

2. Discretize the controller using ZOH.
3. Obtain an equivalent second-order sections (SOS) realization, organized by
   channel for MIMO controllers. The current implementation is SISO.
4. Couple the discrete controller to the nonlinear plant and validate the
   closed-loop response.

### Code Objectives

- Synthesize the initial LTI controller and obtain its discrete SOS realization:

  $$
  K_0(\mu_i)\longrightarrow K_{0,d}(\mu_i)
  \longrightarrow \mathrm{SOS}_0(\mu_i)
  $$

- Iteratively construct the controller for the next parameter value through
  RL–PPO optimization in Python, initialized from the previous controller:

  $$
  K^\star(\mu_i)
  \xrightarrow[\mu=\mu_{i+1}]{\mathrm{RL-PPO}}
  K^\star(\mu_{i+1})
  $$

### Parameter Continuation and Validation

The continuation sequence is defined as:

$$
\mu_0=\mu_{\mathrm{base}}, \qquad
\mu_{i+1}=\mu_i+\Delta\mu, \qquad
\mu_N=\mu_{\mathrm{target}},
\qquad i=0,\ldots,N-1.
$$

## Auditoría y decisiones

| Responsabilidad | Código original y ubicación actual |
|---|---|
| Dinámica, medición, integrador | `plants.py`: `StuartLandauPlant`, `Csens @ xp`, Crank–Nicolson con `scipy.optimize.root`; fallback Heun, ambos avanzan un `dt` |
| Estado interno y ejecución de K | `controller_sos.py`: `ControllerSOS`, cascada de secciones discretas |
| Parametrización, límites y normalización | `zpk.py`: `ZPKTopology`, `ZPKActionMapper`, conservados |
| Interacción PPO y guía | `env.py`: `HybridSOSZPKPPOEnv`, `_update_theta_base`, conservando sus fórmulas |
| PPO | `training.py`: `stable_baselines3.PPO('MlpPolicy', ...)` |
| Simulación y costo compartidos | `simulation.py`: `physical_step`, `instant_cost`, `run_rollout`; se conserva `ClosedLoopSimulatorSOS` |
| Exportación y métricas | `io_utils.py`; nuevo `policy_io.py` exclusivamente para persistencia e inferencia |
| Parámetros y lanzamiento | `main.py`; `config.py` queda como importación compatible de `HybridSOSConfig`, sin configuración global activa |

El archivo adjunto incluye **solamente `Kinit_mu1.0.mat`**. El `main` original
buscaba `stuart_landau_final_controller_KPPO_mu2.0.mat`, que no estaba en el ZIP.
Se corrige el archivo predeterminado; se conservan `mu_target=2.4` y los demás
valores físicos originales. Los ejemplos siguientes fijan explícitamente 1.0.
No falta ningún archivo imprescindible para ejecutar la dinámica Stuart–Landau
suministrada. No se adjuntaron la dinámica cartpole, los scripts MATLAB de diseño
ni evidencia del valor real de `mu_design`. No se han reconstruido ni inventado.

**La selección final original es el último `theta_base`**, actualizado al final
de los episodios completos. No es la última acción individual de PPO, el mejor
probe ni una nueva predicción determinista de PPO. Se conserva esa regla. Si el
presupuesto termina dentro de un episodio, no se fuerza una guía adicional.

Se mantiene el reinicio a cero de los estados internos de K cada vez que se
acepta una nueva propuesta LTI. También se conservan la proyección inicial,
los límites existentes de polos/ceros/ganancia y la parametrización real/radio/
ángulo/log-ganancia. No se añaden restricciones de estabilidad.

Correcciones explícitas: `from_mat` exige `Ts` discreto positivo y compatible con
`dt`; ya no puede adoptar silenciosamente un valor por defecto. Se detecta una
medición no finita inmediatamente después del avance físico y se devuelve un
reward finito con la penalización existente. Los episodios fallidos tienen
`valid=false`, costos no comparables (`null` en JSON), motivo y longitud; nunca
se presentan como mejora por ser cortos. Las trayectorias válidas conservan el
orden y las fórmulas originales.

## Pipeline analítico

`x_k = plant.xp` es el estado físico; `y_k = Csens @ x_k` es la medición.
Los estados `sec['x']` de la cascada son estados de K, distintos de la planta y
de la observación de PPO. En todos los modos:

```text
y_k       = h(x_k)
u_sat,k   = clip(u_total,k, -force_limit, force_limit)
ell_k     = q_y * mean(y_k**2) + r_u * mean(u_cost,k**2)
r_k       = -ell_k / max_steps       (menos fail_penalty si falla)
x_(k+1)   = F_dt(x_k, u_sat,k; mu_target)
J_total   = sum_k ell_k              (suma discreta, sin multiplicar por dt)
J_mean    = J_total / N
```

El costo se calcula con **la medición anterior al avance**, igual que antes.
`q` es penalización de medición, no energía del estado ni `PKE²`. Para SISO:
`q = q_y*y²`. El retorno de un episodio completo válido es `-J_total/max_steps`;
se conserva esa normalización, aunque conceptualmente se busque minimizar J.
No hay VecNormalize ni estadísticas aprendidas de normalización.

**controller_lti.** `o_k = [y_k, normalize(theta_base)]`. Con el MAT suministrado,
`theta=[rp0,thetap0,z0_real,log_abs_Kg]`: 4 acciones y 5 observaciones.
La acción PPO se recorta a `[-1,1]^4` y se convierte en incrementos con las escalas
existentes. Cada `dk_hold_steps` se acepta una propuesta
`theta_trial = project(theta_base + delta_raw)`; el incremento efectivo es
`delta = theta_trial-theta_base`. K permanece fijo durante ese bloque, con sus
estados evolucionando. `u_total=K(theta_trial)[y]`, `u_cost=u_sat`. El signo de
realimentación está incluido en `Kg`; no se añade otro signo negativo.

Al finalizar cada episodio:

```text
R_b         = suma de rewards del bloque b (incluye bloque final parcial)
w_b         = softmax(beta_softmax * R_b)
delta_mean  = sum_b w_b * delta_b
theta_probe = project(theta_base + delta_mean)
J_probe     = J_mean de UN rollout fijo de k_eval_steps con theta_probe
alpha       = lr_good si J_probe < J_base; lr_bad en otro caso válido
theta_update= project(theta_base + alpha * delta_mean)
J_base      = (1-alpha)*J_base + alpha*J_probe
```

Si el probe falla, se conserva la base. **`J_base` es una referencia heurística**
a partir de su primera evaluación; no es el costo medido de `theta_update`.
No se evalúa `theta_update` en la guía. `metrics['guidance']` conserva pesos,
bloques, `theta_before`, `theta_probe`, su costo, `theta_update`, alpha y referencias.
La referencia inicial interna de `k_eval_steps` y el rollout inicial de
`eval_final_time` ya existían: ambos se mantienen y se contabilizan aparte.
Con los defaults `dk_hold_steps=max_steps=50` hay una propuesta por episodio;
para varias propuestas se debe reducir explícitamente `--dk-hold-steps`.

**model_free.** `a_k=pi(o_k)`, `u_rl=action_scale*clip(a_k,-1,1)`,
`u_total=u_rl`, `u_cost=u_sat`. No requiere MAT, topología ni guía.

**residual_rl.** `u_base=K_base[y]`, `u_rl=action_scale*clip(a_k,-1,1)`,
`u_total=u_base+u_rl`, **`u_cost=u_rl` antes de saturar la suma**.
K_base se carga sin pasar por la proyección de ZPK y sus coeficientes permanecen
fijos. Por ese canal de esfuerzo diferente, su objetivo no es numéricamente
idéntico al de las otras modalidades.

Para ambos modos directos:

```text
o_k = [y_k, y_(k-1), ..., y_(k-nf), u_sat_(k-1), ..., u_sat_(k-nf)]
dim(o) = ny + nf*(ny+nu) = 1+2*nf en este proyecto; dim(a) = nu = 1
```

Buffers inicialmente cero y reiniciados en `reset`, unidades físicas, más reciente
primero. Se insertan `y_k,u_sat,k` después de aplicar la acción; no se introduce
`u_k` antes de calcularla ni se repite `y_k`. `nf=0` deja solo la medición actual.
No se añaden estados físicos ni internos de K a las observaciones. Este historial
no garantiza observabilidad completa de cualquier POMDP.

## Ejecución y cuatro ejemplos

Python 3.12 fue el entorno de verificación. Desde la carpeta que contiene el
paquete `ppo_sos_pipeline/`:

```bash
python -m venv .venv
source .venv/bin/activate
# Opcional para usar la distribución CPU verificada:
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Los argumentos de `main.py` están ordenados por modalidad, física, tiempo,
controlador, PPO/costo, semilla/memoria y salidas. Se pueden editar sus defaults.

```bash
# 1. Primer entrenamiento LTI, utilizando el controlador realmente adjunto.
python -m ppo_sos_pipeline.main --mode controller_lti \
  --mu-target 1.0 --mat-file ppo_sos_pipeline/Kinit_mu1.0.mat \
  --episode-duration 2.5 --eval-final-time 10 --total-timesteps 70000 \
  --output-dir runs/lti_mu1.0

# 2. Continuación MANUAL con otro mu_target; PPO nuevo, K inicial anterior.
python -m ppo_sos_pipeline.main --mode controller_lti \
  --mu-target 1.05 --mat-file runs/lti_mu1.0/Kfinal_mu1.0.mat \
  --total-timesteps 70000 --output-dir runs/lti_mu1.05

# 3. Acción directa; no se carga ningún controlador.
python -m ppo_sos_pipeline.main --mode model_free --mu-target 1.0 \
  --nf 4 --action-scale 15 --total-timesteps 70000 --output-dir runs/mf_mu1.0

# 4. Corrección residual con controlador fijo.
python -m ppo_sos_pipeline.main --mode residual_rl --mu-target 1.0 \
  --mat-file runs/lti_mu1.0/Kfinal_mu1.0.mat --nf 4 --action-scale 2 \
  --total-timesteps 70000 --output-dir runs/residual_mu1.0
```

`--x0`, `--omega`, `--beta`, `--csens`, `--dt`, `--force-limit`, `--q-y`, `--r-u`,
`--dk-hold-steps` y `--k-eval-steps` permiten cambiar lo indicado por sus nombres.
`--max-steps` es alternativa a `--episode-duration`. Una duración no múltiplo de
dt se redondea hacia arriba y se registra la duración efectiva. No se cambia Ts
al modificar dt: un MAT incompatible produce un error claro.

`--mu-design` es solo metadato conocido, nunca se utiliza para linealizar ni
identificar en `mu_target`; por defecto es desconocido. Se separan el `mu_target`
del experimento, `mu_design` y el texto `mu` del nombre de archivo. `mu_tag`
usa la representación decimal que conserva el valor float, incluida `1.0`.

Para cambiar la dinámica, sustituir la función local `plant_factory` en `main` y
sus parámetros físicos. El contrato es `reset()`, `measure()`, `step(u)`, `ny`,
`nu`, `dt`; `state()` y `state_names` son opcionales para exportar. Su `step`
debe avanzar un dt aunque use subpasos internos. No hay implementación cartpole
en la entrega original que pueda reutilizarse.

La semilla predeterminada es **0, constante**: no se incrementa durante el
entrenamiento. Los generadores avanzan normalmente; no se repite el mismo ruido
reinicializándolos en cada step. SB3 completa bloques `n_steps`: con presupuesto
70000 y `n_steps=1024` puede usar **70656** pasos. Se guardan ambas cantidades y
se muestran los timesteps reales. Para inspección sin entrenamiento:
`python -m ppo_sos_pipeline.main --run debug --mode controller_lti`.

## Guardado, inferencia y transferencia

Cada ejecución escribe en pwd o `--output-dir`. Usar carpetas distintas para
conservar experimentos. Los outputs son:

- LTI: `sim_kinit_mu1.0.csv`, `Kfinal_mu1.0.mat`, `sim_kfinal_mu1.0.csv`.
- Directo: `sim_mf_mu1.0.csv`. Residual: `sim_residual_mu1.0.csv`.
- Todos: `ppo_{modo}_mu1.0.zip`, `actor_{modo}_mu1.0.mat`,
  `experiment_{modo}_mu1.0.json`, `metrics_{modo}_mu1.0.json` y dos figuras PNG.
- Cuando hay controlador: copia exacta `controller_source_{modo}_mu1.0.mat`.

Todos los CSV: `t,x1,x2,y,u_sat`; residual añade `u_rl`. Los estados usan los
nombres de la dinámica, o `x1,x2,...`. Cada fila contiene `x_k,y_k` al inicio del
intervalo `[t_k,t_k+dt)`, y el control realmente aplicado en ese intervalo.
No existe una fila terminal con una acción inventada. El rollout final LTI carga
y evalúa **el mismo MAT recién exportado**. Se mantienen claves ZPK originales,
órdenes, ganancia y Ts; se añaden SOS y matrices SS regeneradas coherentes.

El ZIP nativo SB3 contiene actor, crítico, arquitectura, activaciones, optimizador,
hiperparámetros y contrato de inferencia. No se duplican en un archivo `.pt`.
El JSON guarda configuración efectiva, versiones, hashes de código/controlador,
canal de costo y presupuestos de simulación. El estado de la guía se conserva en
el atributo `pipeline_guide_state` del ZIP para análisis; su `J_reference` sigue
siendo heurístico.

Para inferencia, construir un ambiente compatible y usar:

```python
from ppo_sos_pipeline.policy_io import load_policy
model = load_policy("runs/mf_mu1.0/ppo_model_free_mu1.0.zip", env)
obs, _ = env.reset(seed=env.cfg.seed)
action, _ = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)
# En LTI, para despliegue usar ControllerSOS.from_mat(Kfinal), no este actor PPO.
```

Para inicializar otro entrenamiento, añadir `--policy-path <ppo_...zip>` al
comando correspondiente, manteniendo modo, nf, canales, dt, límites/escalado,
arquitectura y parámetros PPO compatibles. Se permite cambiar explícitamente
mu_target/x0/costos y el controlador inicial con topología compatible. La recarga
rechaza incompatibilidades; no adapta pesos. Conserva el estado del optimizador,
pero comienza un experimento nuevo con estado físico/historial reiniciado y,
en LTI, referencia evaluada en el controlador indicado. No promete reanudar
bit a bit un rollout interrumpido ni reproducibilidad exacta entre dispositivos.

El actor MAT exporta las tres capas densas (`W[out,in]`, `b[out,1]`), Tanh/Tanh/
lineal, tipo float32, clipping, escalado y contrato JSON de observaciones. El actor
determinista es la media gaussiana **recortada**, sin squash Tanh de salida.
`matlab/ppo_actor_predict.m` implementa esa inferencia; `check_actor.m` compara
con observaciones y acciones de referencia incluidas. La saturación física final
y K_base son externos a la NN residual. No se ha implementado un modelo Simulink.

## Métricas y verificación

Las métricas están en `env.metrics` y JSON, sin CSV de entrenamiento ni trayectorias
de todos los rollouts. `J_probe` conserva el significado original **J/N**;
`J_total` es la suma discreta. `theta_update` no hereda un costo medido.

`cost_timesteps_*.png` muestra costo instantáneo, media por episodio y probes en
el timestep que activó la guía. Solo añade una referencia inicial si su horizonte
coincide con una serie representada. `cost_episodes_*.png` separa media y total
en dos paneles, identificando entrenamiento y K ponderado fijo. Los horizontes N
se indican: medias de horizontes distintos pueden diferir por transitorios y los
totales con N diferentes no permiten clasificar controladores directamente.
K puede cambiar dentro del episodio LTI; su costo no representa un único K fijo.
No se usa la suma acumulada de todos los costos ni se promete descenso monótono.
El episodio parcial final queda registrado aparte, sin aparentar uno completo.

`simulation_budget` separa entrenamiento, referencia inicial, rollout inicial,
guías, rollout final y otras evaluaciones. Las guías no incrementan los pasos de
PPO. `valid`, `N` y `planned_steps` permiten detectar fallos y horizontes distintos.

```bash
python -m pytest tests -q
```

Ver resultados realmente ejecutados y sus límites en `VERIFICATION.md`. No se
han realizado certificaciones de estabilidad, RoA, Safe-RL o MBRL.
