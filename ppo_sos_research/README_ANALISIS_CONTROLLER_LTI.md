# Análisis matemático y crítico de `controller_lti`

Comparación de `ppo_sos_pipeline` y `ppo_pipeline_test`, con las mejoras y configuraciones discutidas progresivamente.

## 1. Alcance y procedencia

Este README reúne el análisis presentado en la conversación sobre los dos ZIP adjuntos, exclusivamente para la modalidad `controller_lti`. Conserva sus hallazgos, ecuaciones, resultados de diagnóstico, recomendaciones y límites, y añade las configuraciones de ejecución discutidas después.

La comparación corresponde a estas copias de los archivos:

- `ppo_sos_pipeline.zip`: versión original con configuración global y ambiente LTI.
- `ppo_pipeline_test.zip`: versión posterior con ambiente compartido, guardado nativo de PPO, métricas y exportación de actor.

Los valores descritos son los de esos ZIP. Una edición posterior del código o un argumento de terminal puede producir otra configuración efectiva. La presencia de un archivo en el repositorio tampoco significa que estuviera incluido en los ZIP analizados.

**Conclusión de la comparación:** el núcleo matemático de la optimización LTI se conserva. Las diferencias más importantes están en la configuración efectiva, la instrumentación y el tratamiento de errores.

Tres problemas deben revisarse antes de aumentar el presupuesto:

1. El `main` nuevo usa episodios de 50 pasos por defecto, aunque la clase declare 1000.
2. Con esos 50 pasos y `dk_hold_steps=100`, hay una sola propuesta por episodio: el softmax asigna peso 1.
3. PPO recibe acciones en cada paso, pero la mayoría se ignoran. Además, `J_probe` no entra directamente en el reward de PPO.

Este documento es un análisis del código y de los experimentos indicados. Las recomendaciones no significan que se hayan implementado o validado como mejoras de convergencia.

## 2. Misma implementación y mismo experimento son comparaciones diferentes

### 2.1. Configuración efectiva al ejecutar cada `main`

Sin argumentos adicionales, los valores son:

| Cantidad | `ppo_sos_pipeline` | `ppo_pipeline_test` |
|---|---:|---:|
| `mu_target` | 2.4 | 2.0 |
| `x0` | `[0.6, 0.1]` | `[1.0, 0.5]` |
| `dt` | 0.05 | 0.05 |
| `max_steps` efectivo | 50 | **50** |
| `dk_hold_steps` | 50 | **100** |
| Propuestas por episodio completo | 1 | **1** |
| `k_eval_steps` | 100 | 200 |
| Evaluación inicial/final | 10 s | 10 s |
| `ppo_n_steps` | 1024 | 2048 |
| `batch_size` | 64 | 64 |
| `lr_good`, `lr_bad` | 0.3, 0.01 | 0.3, 0.01 |
| `q_y`, `r_u` | 6, 0.005 | 1, 0.01 |
| Saturación | ±15 | ±5 |
| Semilla PPO explícita | No | 0 |
| Controlador inicial solicitado | `stuart_landau_final_controller_KPPO_mu2.0.mat` | `Kfinal_mu1.0.mat` |

El archivo solicitado por defecto en el ZIP original no está incluido. Para ejecutar esa versión hay que indicar un MAT existente. El nuevo ZIP incluye y carga por defecto un `Kfinal` anterior.

Por tanto, ejecutar las dos versiones con sus defaults no compara únicamente una refactorización: también cambia la planta del experimento, la condición inicial, el controlador, el costo y el límite de fuerza.

### 2.2. Por qué editar `HybridSOSConfig` no siempre cambia el entrenamiento

En el `main.py` nuevo, alrededor de la línea 78 del archivo analizado, aparece:

```python
max_steps = (
    steps_for_duration(args.episode_duration, args.dt)
    if args.episode_duration is not None
    else (50 if args.max_steps is None else args.max_steps)
)
```

Después, `main` construye `HybridSOSConfig` pasando explícitamente los valores de `args`. Los valores por defecto de la clase solo se utilizan cuando no se proporciona el campo al construirla.

En consecuencia:

```text
Clase: max_steps=1000, hold=300, evaluación=1000,
       lr_good=0.5, lr_bad=0.001

Main:  max_steps=50,   hold=100, evaluación=200,
       lr_good=0.3, lr_bad=0.01
```

Los parámetros que se modifican habitualmente deben ajustarse en los `default=` de `parse_args()` o mediante los argumentos de terminal. La referencia para comprobar una ejecución es el campo `runtime` de `experiment_controller_lti_mu{mu_tag}.json`.

### 2.3. El objetivo numérico también cambió

La relación entre penalización de esfuerzo y medición pasó de:

$$
\left(\frac{r_u}{q_y}\right)_{\mathrm{original}}
=\frac{0.005}{6}
$$

a:

$$
\left(\frac{r_u}{q_y}\right)_{\mathrm{nuevo}}
=\frac{0.01}{1}.
$$

El esfuerzo pesa doce veces más respecto a la medición. Una reducción numérica de costo entre ambos experimentos no demuestra, por sí sola, una mejora del algoritmo.

### 2.4. Comprobación numérica del núcleo LTI

Se ejecutaron ambas versiones con el mismo MAT, planta, parámetros y 63 acciones suministradas, repartidas en tres episodios con tres propuestas por episodio.

La diferencia máxima observada fue 0 en:

- Estados de la planta.
- Observaciones.
- Rewards.
- `theta_trial`.
- `theta_base`.
- `J_base`.
- `J_probe`.

La coincidencia corresponde a esas trayectorias válidas en el entorno utilizado. No implica que las rutas de fallo sean idénticas ni que los defaults representen el mismo experimento.

## 3. Planta, medición e integración temporal

### 3.1. Dinámica no lineal

La planta implementada en `plants.py` es:

$$
r^2=x_1^2+x_2^2,
$$

$$
\dot x_1=\mu x_1-\omega x_2-r^2(x_1-\beta x_2)+u,
$$

$$
\dot x_2=\omega x_1+\mu x_2-r^2(x_2+\beta x_1).
$$

La medición predeterminada es:

$$
y_k=C_{\mathrm{sens}}x_k=x_{2,k},
\qquad C_{\mathrm{sens}}=[0\quad1].
$$

No se incorpora una identificación ni una nueva linealización en `mu_target`. El controlador inicial se aporta mediante el archivo MATLAB.

### 3.2. Convención temporal

El flujo físico mínimo es:

$$
y_k=h(x_k),
$$

$$
u_{\mathrm{sat},k}
=\operatorname{clip}(u_{\mathrm{raw},k},-F_{\max},F_{\max}),
$$

$$
x_{k+1}=F_{dt}(x_k,u_{\mathrm{sat},k};\mu_{\mathrm{target}}).
$$

Una llamada válida a `env.step` implica una llamada a `plant.step` y un avance físico de `dt`. Las iteraciones del solucionador implícito no son pasos adicionales de RL.

El integrador realiza Crank–Nicolson:

$$
x_{k+1}=x_k+\frac{dt}{2}
\left[f(x_k,u_{\mathrm{sat},k})+
f(x_{k+1},u_{\mathrm{sat},k})\right].
$$

Si falla el solucionador, utiliza Heun. En ambos casos, la entrada permanece constante durante ese avance.

La dinámica y el integrador son esencialmente los mismos en ambos proyectos. El fallback a Heun es silencioso: conviene registrar cuántas veces ocurre y el residuo del solucionador. Cerca de una trayectoria difícil, un cambio de costo puede tener una componente numérica. Esa instrumentación puede añadirse sin cambiar el algoritmo de aprendizaje.

### 3.3. Cuatro objetos que deben distinguirse

| Objeto | Significado |
|---|---|
| $x_k$ | Estado físico de la planta |
| $\xi_k$ | Estados internos de las secciones del controlador |
| $o_k$ | Observación de PPO |
| $a_k$ | Propuesta normalizada de modificación de parámetros |

PPO no recibe directamente ni $x_k$ ni $\xi_k$. Los estados físicos pueden utilizarse para exportación y métricas sin convertirse en observaciones.

## 4. Controlador SOS/ZPK y significado de una acción PPO

### 4.1. Parametrización conservada

Los dos controladores adjuntos tienen una sección con dos polos conjugados y un cero real. La parametrización es:

$$
\theta=
\begin{bmatrix}
r_p & \varphi_p & z_0 & \log|K_g|
\end{bmatrix}^{\mathsf T}.
$$

Los polos y la ganancia se reconstruyen mediante:

$$
p_{1,2}=r_p e^{\pm j\varphi_p},
\qquad
K_g=\operatorname{sign}(K_{g,0})e^{\theta_4}.
$$

El signo de la ganancia queda fijado por el controlador cargado.

La propuesta efectiva se construye como:

$$
\theta_b=\Pi\left(
\theta_{\mathrm{base}}+
D\,\operatorname{clip}(a_b,-1,1)
\right),
$$

con las escalas actuales:

$$
D=\operatorname{diag}(0.01,\;0.01,\;0.01,\;0.2).
$$

El incremento almacenado es el que resulta después de la proyección:

$$
\delta_b=\theta_b-\theta_{\mathrm{base}}.
$$

Las restricciones existentes son:

- Radio del par de polos entre $10^{-5}$ y $0.98$.
- Cero real entre $-2$ y $2$.
- Logaritmo de ganancia entre $-10$ y $10$.
- Ángulos envueltos según la convención de `wrap_angle`.

Son restricciones sobre el controlador. No garantizan estabilidad del lazo cerrado no lineal.

`zpk.py` conserva el mismo algoritmo de parametrización, mapeo, normalización y proyección; la diferencia de fuente relevante en ese archivo es la eliminación de su dependencia del `EPS` global de `config.py`.

### 4.2. Ejecución del controlador

La cascada calcula:

$$
v_{\ell,k}=C_\ell\xi_{\ell,k}+D_\ell v_{\ell-1,k},
$$

$$
\xi_{\ell,k+1}=A_\ell\xi_{\ell,k}+B_\ell v_{\ell-1,k},
$$

$$
u_{\mathrm{raw},k}=K_gv_{L,k},
\qquad v_{0,k}=y_k.
$$

No hay un signo negativo externo adicional: el signo de realimentación está incluido en el controlador.

Dos consecuencias del código son importantes:

1. Cada propuesta parte de `theta_base`. No se acumula sobre la propuesta anterior del mismo episodio.
2. Al aceptar una propuesta nueva, los estados internos de K se reinician a cero, mientras la planta continúa desde su estado actual.

Un bloque posterior incluye tanto otros coeficientes como un reinicio del controlador y un estado inicial de planta heredado del bloque anterior.

Ese reinicio pertenece al mecanismo original. Sustituirlo por conservación de estados o por una transferencia sin salto cambiaría las condiciones del experimento y requeriría una comparación explícita.

### 4.3. Tiempo de muestreo

El nuevo `ControllerSOS.from_mat` exige un `Ts` discreto positivo y valida su compatibilidad con `dt`. El original podía adoptar un valor predeterminado si faltaba `Ts`, y `step(..., dt=...)` almacenaba el valor sin validar que el controlador discreto correspondiera a ese tiempo.

La condición del pipeline actual es:

$$
T_s\simeq dt
$$

dentro de la tolerancia numérica implementada. No se hace una conversión o un remuestreo silencioso.

### 4.4. `action_scale` no es la escala de exploración LTI

`action_scale=5` corresponde a las modalidades de acción física. En `controller_lti`, las perturbaciones las determina `ZPKActionMapper`.

Para este controlador, los campos relevantes son `scale_p_radius`, `scale_p_angle`, `scale_z_real` y `scale_logKg`. Cambiar `action_scale` no reduce las perturbaciones ZPK.

## 5. Sensibilidad de las coordenadas y cero cercano a uno

Los archivos MATLAB analizados contienen:

| Propiedad | `Kinit_mu1.0.mat` | `Kfinal_mu1.0.mat` |
|---|---:|---:|
| Radio de polos | 0.689839 | 0.539949 |
| Ángulo del primer polo | −0.098672 | −0.061087 |
| Cero real | 1.080003 | **0.992024** |
| Ganancia $K_g$ | −0.948947 | −10.317804 |
| $T_s$ | 0.05 | 0.05 |

Para esta sección:

$$
K(z)=K_g\frac{z-z_0}
{z^2-2r_p\cos(\varphi_p)z+r_p^2}.
$$

La ganancia en continua es:

$$
K(1)=K_g\frac{1-z_0}
{1-2r_p\cos(\varphi_p)+r_p^2}.
$$

En el `Kfinal` cargado:

$$
1-z_0\approx0.007976,
\qquad |\Delta z_0|\leq0.01.
$$

Una propuesta puede cruzar $z_0=1$ y cambiar el signo de $K(1)$ aunque el signo de $K_g$ permanezca fijo.

Fuera del cruce:

$$
\frac{\partial\log|K(1)|}{\partial z_0}
=-\frac{1}{1-z_0}
\approx-125.4.
$$

Por eso una escala de `0.01` aparentemente pequeña puede ser grande en términos funcionales.

La escala de log-ganancia tampoco es diminuta:

$$
\Delta\log|K_g|\in[-0.2,0.2]
\Longrightarrow
\frac{|K_g^{\mathrm{trial}}|}{|K_g^{\mathrm{base}}|}
\in[0.819,\;1.221].
$$

Una propuesta permite aproximadamente −18 % a +22 % de cambio de ganancia.

La mejora propuesta es ajustar las escalas por sensibilidad de cada coordenada, conservando la parametrización. No conviene tratar todos los `0.01` como perturbaciones funcionalmente equivalentes. También hay que vigilar las diferencias angulares al cruzar ±π, porque una diferencia numérica envuelta puede no representar el desplazamiento angular corto que se pretendía.

## 6. Observación de PPO y observabilidad parcial

En LTI:

$$
o_k=\left[
y_k,\;\operatorname{normalize}(\theta_{\mathrm{base}})
\right].
$$

Con el controlador adjunto:

$$
\dim(o_k)=5,
\qquad \dim(a_k)=4.
$$

La observación no incluye:

- El controlador `theta_trial` que está actuando.
- Sus estados internos.
- El tiempo restante del bloque.
- La fase del episodio.
- La referencia heurística `J_base`.

El problema observado por la MLP no es plenamente markoviano. Dos observaciones iguales pueden corresponder a distintos estados internos, propuestas activas y momentos del episodio, con evoluciones futuras diferentes.

Es una limitación heredada, no una modificación introducida en el ZIP nuevo. Añadir memoria, el controlador activo o la fase del bloque podría ayudar, pero cambiaría el espacio de observación LTI y la compatibilidad de las políticas guardadas.

`nf` debe permanecer en cero en `controller_lti`; el código rechaza su uso para ampliar las observaciones de esta modalidad.

## 7. Costo, reward, normalización y terminación

### 7.1. Definiciones conservadas

El costo SISO es:

$$
\ell_k=q_y y_k^2+r_u u_{\mathrm{sat},k}^2,
$$

$$
u_{\mathrm{sat},k}
=\operatorname{clip}(u_{\mathrm{raw},k},-F_{\max},F_{\max}).
$$

La implementación general usa `mean(y**2)` y `mean(u_cost**2)`; en el SISO analizado son exactamente los cuadrados escalares anteriores.

Se utiliza la medición $y_k$ anterior al avance de la planta. Se penaliza la entrada realmente aplicada después de saturar.

El reward del ambiente es:

$$
r_k=-\frac{\ell_k}{N}
-P_{\mathrm{fail}}\,\mathbf{1}_{\mathrm{fallo}},
\qquad N=\texttt{max\_steps}.
$$

Para un episodio completo válido:

$$
J_{\mathrm{total}}=\sum_{k=0}^{N-1}\ell_k,
\qquad
J_{\mathrm{mean}}=\frac{J_{\mathrm{total}}}{N},
$$

$$
\sum_{k=0}^{N-1}r_k=-J_{\mathrm{mean}}.
$$

No se multiplica por `dt`. No hay VecNormalize ni estadísticas aprendidas de normalización en este pipeline. La normalización de `theta_base` forma parte de la observación; la normalización de ventajas es una operación interna diferente de PPO.

### 7.2. Medición frente a energía

El término de estado del costo actual es una penalización de medición:

$$
q_k=q_y y_k^2.
$$

No es la energía completa de la planta. Si se definiera otra cantidad:

$$
\mathrm{PKE}=\sum_i x_i^2,
$$

penalizar `PKE` sería distinto de penalizar `PKE²`. No se ha introducido ninguna de esas sustituciones.

### 7.3. Fallos y significado de `valid`

En el nuevo pipeline se detecta una medición no finita inmediatamente después de avanzar la planta, se evita devolver una observación no finita a PPO y se utiliza la penalización de fallo existente. También se comprueban las señales/costo antes de integrar.

El original comprobaba en el entrenamiento las señales anteriores al avance, lo que podía retrasar la detección del fallo y propagar valores no finitos. Las rutas válidas mantienen la secuencia y las fórmulas, pero estas rutas de fallo no son idénticas.

Las métricas nuevas conservan longitud, validez y presupuesto. Un episodio fallido no debe presentarse como mejora por haber acumulado menos pasos; sus costos inválidos se representan con `valid=false` y `null` en JSON.

**Una trayectoria válida no necesariamente es una trayectoria estabilizada.** `valid` comprueba esencialmente finitud numérica, no convergencia al equilibrio.

### 7.4. Comprobación de la diferencia entre bajo costo y estabilización

Se evaluó el `Kfinal_mu1.0.mat` adjunto con:

```text
mu_target = 2
x0 = [1.0, 0.5]
dt = 0.05
force_limit = 5
q_y = 1
r_u = 0.01
duración = 50 s
```

Se obtuvo:

$$
J_{\mathrm{mean}}\approx0.193111.
$$

Durante los últimos 10 segundos:

$$
\sqrt{\operatorname{mean}\|x_k\|^2}\approx0.8959.
$$

La norma del estado varió aproximadamente entre 0.453 y 1.204 en esa ventana. Fue una evaluación numéricamente válida, sin aproximación al origen en esa ventana. Ese costo no demuestra estabilización.

No se extrapola este resultado al experimento posterior en `mu_target=1.0` o `1.5`, ni a otro controlador inicial.

## 8. Guía offline: secuencia exacta y análisis matemático

### 8.1. Propuestas y retención

Sea:

$$
H=\texttt{dk\_hold\_steps}.
$$

La propuesta se acepta cuando:

$$
k\bmod H=0,
$$

o cuando todavía no existe un controlador activo. En los pasos intermedios se sigue utilizando el controlador aceptado y se ignora la nueva acción de PPO.

En un episodio de $N$ pasos, el número de propuestas aceptadas es:

$$
B=\left\lceil\frac{N}{H}\right\rceil.
$$

Este conteo incluye el bloque final parcial. El contador se reinicia en cada episodio.

### 8.2. Ponderación por bloques

Para cada bloque $b$:

$$
\delta_b=\theta_b-\theta_{\mathrm{base}},
\qquad
R_b=\sum_{k\in b}r_k.
$$

La ponderación es:

$$
w_b=\frac{\exp(\beta_{\mathrm{softmax}}R_b)}
{\sum_j\exp(\beta_{\mathrm{softmax}}R_j)},
$$

$$
\bar\delta=\sum_b w_b\delta_b.
$$

El código resta el máximo de los rewards de bloque dentro de la exponencial para mejorar la estabilidad numérica; no cambia los pesos matemáticos.

Aquí $\beta_{\mathrm{softmax}}$ es `beta_softmax`. No es el parámetro físico `beta` de Stuart–Landau.

### 8.3. Un único rollout del controlador ponderado

Se construye:

$$
\theta_{\mathrm{probe}}
=\Pi(\theta_{\mathrm{base}}+\bar\delta).
$$

Se evalúa exactamente una vez, con planta y estados internos de K reiniciados:

$$
J_{\mathrm{probe}}=
\frac{1}{N_e}\sum_{k=0}^{N_e-1}
\ell_k(\theta_{\mathrm{probe}}),
\qquad N_e=\texttt{k\_eval\_steps}.
$$

`J_probe` conserva el significado histórico de costo medio. El costo total de ese rollout es $N_eJ_{\mathrm{probe}}$ cuando se completa válidamente.

### 8.4. Actualización posterior

Después del rollout:

$$
\alpha=
\begin{cases}
\texttt{lr\_good}, & J_{\mathrm{probe}}<J_{\mathrm{base}},\\
\texttt{lr\_bad}, & \text{probe válido en otro caso},\\
0, & \text{probe inválido}.
\end{cases}
$$

Se calcula:

$$
\theta_{\mathrm{update}}=
\Pi(\theta_{\mathrm{base}}+\alpha\bar\delta).
$$

Si el probe es válido:

$$
\theta_{\mathrm{base}}\leftarrow\theta_{\mathrm{update}},
$$

$$
J_{\mathrm{base}}\leftarrow
(1-\alpha)J_{\mathrm{base}}+\alpha J_{\mathrm{probe}}.
$$

Si es inválido, se conserva la base y su referencia. **No hay un segundo rollout para medir `theta_update`.**

Los campos `alpha_update` y `allow_worse_base_update` no intervienen en esta actualización. Cambiarlos no modifica el método.

### 8.5. Con una propuesta, el softmax no selecciona

Si $B=1$:

$$
w_1=1.
$$

El resultado no depende de `beta_softmax`. Ocurre con:

- Defaults originales: $N=50$, $H=50$.
- Defaults del `main` nuevo: $N=50$, $H=100$.

La cadencia se comprobó ejecutando un episodio con los defaults nuevos: se aceptó una acción y el vector de pesos fue `[1.0]`.

### 8.6. Sesgo a favor de bloques cortos

Con la configuración declarada en tu clase:

$$
N=1000,\qquad H=300,
$$

las longitudes de bloque son:

$$
300,\quad300,\quad300,\quad100.
$$

Supongamos el mismo costo instantáneo constante $\ell_k=1$ y ningún fallo:

$$
R_b=(-0.3,-0.3,-0.3,-0.1).
$$

Con $\beta_{\mathrm{softmax}}=5$:

$$
w_{\mathrm{último}}=
\frac{e^{-0.5}}{3e^{-1.5}+e^{-0.5}}
\approx0.47537.
$$

Cada bloque largo recibe aproximadamente 0.175. El bloque de 100 pasos obtiene casi la mitad del peso sin tener mejor costo por paso.

La corrección compatible con la fórmula actual es elegir $N$ múltiplo de $H$:

$$
N\bmod H=0.
$$

Cambiar la puntuación a costo medio por bloque también eliminaría ese sesgo de duración, pero modificaría el criterio de ponderación y debería tratarse como una variante del algoritmo.

### 8.7. Los bloques no son comparaciones independientes de controladores

Los bloques sucesivos parten de estados diferentes. Lo que se compara es del tipo:

$$
J_b=J(\theta_b;\,x_{b,\mathrm{inicio}},\,\xi=0).
$$

El estado $x_{b,\mathrm{inicio}}$ depende de los bloques anteriores. Un bloque puede recibir buen reward porque comienza cerca del equilibrio o en una fase favorable de la oscilación.

Usar bloques suficientemente informativos puede reducir la dependencia del transitorio heredado, pero no convierte estas puntuaciones en evaluaciones independientes desde el mismo estado. La evaluación offline del controlador ponderado sí reinicia la planta y K.

### 8.8. `J_base` es una referencia heurística

Inicialmente procede de una evaluación real de la base. Después pasa a ser una combinación de referencias y costos de probes.

En general:

$$
J(\theta+\alpha\bar\delta)
\neq
(1-\alpha)J(\theta)+\alpha J(\theta+\bar\delta).
$$

Por tanto:

- `J_probe` pertenece a `theta_probe`.
- `theta_update` no tiene un costo medido por esa guía.
- `GOOD` significa mejor que la referencia heurística, no necesariamente mejor que la base actual medida.
- Un `BAD` válido sigue modificando la base cuando `lr_bad > 0`.
- Aumentar `lr_good` de 0.3 a 0.5 aumenta la distancia recorrida; no aporta evidencia adicional sobre la calidad de esa dirección.

No debe presentarse `J_base` como validación offline de `theta_update`. El nuevo historial conserva explícitamente `theta_before`, `theta_probe`, `theta_update`, pesos, rewards de bloques, alpha y referencias para poder estudiar esta distinción.

### 8.9. Referencia inicial infinita

Hay un caso de fallo heredado. Si la referencia inicial es infinita, para $0<\alpha<1$ y un probe finito:

$$
(1-\alpha)\infty+\alpha J_{\mathrm{probe}}=\infty.
$$

La referencia no se recupera mediante esa fórmula aunque aparezcan probes válidos. Conviene tratar explícitamente la referencia no finita, conservando la distinción entre el costo medido del probe y la actualización posterior. Ese ajuste no estaba implementado en los ZIP analizados.

## 9. PPO: objetivo, acciones ignoradas y crédito temporal

### 9.1. Arquitectura e hiperparámetros

Las dos versiones utilizan `MlpPolicy`. En SB3 2.9.2 se verificaron los defaults:

- Actor: dos capas ocultas de 64 unidades.
- Crítico: dos capas ocultas de 64 unidades.
- Activaciones ocultas Tanh y salida lineal del actor.
- Extractor de características `FlattenExtractor`.
- Distribución de acciones gaussiana diagonal, con clipping para respetar el espacio.
- `learning_rate=3e-4` y `ent_coef=0.01`, explícitos en el proyecto.
- `gamma=0.99`.
- `gae_lambda=0.95`.
- `clip_range=0.2`.
- `n_epochs=10`.
- `normalize_advantage=True`.

Los ZIP no fijaban una versión original de SB3 que permitiera garantizar retrospectivamente todos sus defaults. Los valores anteriores se verificaron con la versión usada para la revisión.

### 9.2. Objetivo de PPO y relación con la guía

PPO optimiza el surrogate:

$$
L_{\mathrm{clip}}=
\mathbb E\left[
\min\left(
\rho_k\hat A_k,
\operatorname{clip}(\rho_k,1-\epsilon,1+\epsilon)\hat A_k
\right)
\right],
$$

donde:

$$
\rho_k=
\frac{\pi_\phi(a_k\mid o_k)}
{\pi_{\phi_{\mathrm{old}}}(a_k\mid o_k)}.
$$

El entrenamiento también incluye pérdida del crítico y bonificación de entropía.

**PPO no recibe `-J_probe` como reward.** El costo offline decide el paso heurístico de la base. Su efecto sobre PPO es indirecto, mediante los episodios y observaciones posteriores.

Conviven tres criterios diferentes:

1. Costos físicos de entrenamiento, con K posiblemente variable dentro del episodio.
2. Retornos descontados y ventajas que utiliza PPO.
3. Costo medio offline de un único K fijo.

El retorno bruto de un episodio completo válido es `-J_total/N`, pero el objetivo de aprendizaje de PPO utiliza descuento y targets de valor; no debe identificarse sin matices con la minimización directa del `J_probe` de la guía.

### 9.3. Acciones que no tienen efecto sobre el controlador

El ambiente aplica:

$$
a_k\longrightarrow
\begin{cases}
\text{nueva propuesta}, & k\bmod H=0,\\
\text{sin efecto}, & \text{resto del bloque}.
\end{cases}
$$

PPO almacena acciones y calcula pérdidas en todos esos pasos.

Con los defaults nuevos, aproximadamente el 2 % de las acciones modifica el controlador. Con $N=1000$, $H=300$, la proporción ronda el 0.4 %.

Los pasos restantes aportan información al crítico, pero sus acciones no tienen efecto causal directo sobre la trayectoria. En condiciones ideales, la contribución esperada de esas acciones al gradiente del retorno sería nula. En muestras finitas añaden ruido y siguen participando en la regularización de entropía.

Aumentar `total_timesteps` no aumenta las decisiones efectivas en la misma proporción que en un PPO donde cada acción modifica la planta o el controlador.

### 9.4. Descuento y GAE

Las ventajas se construyen con residuos TD:

$$
\hat A_k=\sum_{\ell\geq0}
(\gamma\lambda)^\ell\delta^{TD}_{k+\ell}.
$$

Con los defaults:

$$
\gamma\lambda=0.99\times0.95=0.9405.
$$

La constante de decaimiento en pasos es:

$$
\tau_{\mathrm{GAE,pasos}}
=-\frac{1}{\log(0.9405)}
\approx16.3.
$$

Para `dt=0.05`:

$$
\tau_{\mathrm{GAE}}\approx0.815\;\mathrm{s}.
$$

En un bloque de 300 pasos:

$$
(\gamma\lambda)^{300}\approx10^{-8}.
$$

La escala del descuento puro es:

$$
\tau_\gamma=-\frac{dt}{\log\gamma}
\approx4.975\;\mathrm{s},
$$

y:

$$
\gamma^{300}\approx0.04904.
$$

Esto no significa que el crítico no pueda representar efectos largos, pero sí que la evidencia TD distante llega muy atenuada a la acción que inició el bloque. Mantener propuestas durante 15 segundos y utilizar esa escala GAE merece revisión explícita.

### 9.5. Final de episodio por timeout

El ambiente devuelve `truncated=True` cuando alcanza `max_steps`. SB3 trata ese timeout mediante bootstrap con el valor de la observación terminal.

El reward bruto del ambiente conserva la fórmula indicada, pero los targets internos del crítico incluyen ese tratamiento del timeout. Si el episodio se interpreta como un problema estrictamente finito, debe revisarse la convención. Cambiarla alteraría el aprendizaje y no debe hacerse silenciosamente.

### 9.6. Exploración y clipping inicial

Con la política gaussiana inicial de desviación estándar aproximadamente uno y media cercana a cero, para una componente:

$$
P(|a|>1)\approx31.7\%.
$$

Para cuatro componentes independientes con esa distribución, la probabilidad de que al menos una se recorte ronda el 78 %.

Es el comportamiento esperado de esa configuración inicial de SB3, no una medición universal de todas las etapas del entrenamiento. Implica que la exploración inicial no se concentra únicamente en perturbaciones pequeñas dentro del intervalo.

### 9.7. Semilla fija y determinismo

El entrenamiento muestrea acciones de una política estocástica. La semilla fija, por defecto cero, busca reproducir la secuencia aleatoria al repetir el experimento bajo las mismas condiciones.

La semilla no cambia durante el entrenamiento, pero el generador avanza: no produce siempre la misma acción y no se reinicia el ruido en cada step.

En LTI:

- Durante entrenamiento, las propuestas aceptadas son estocásticas.
- La guía evalúa un controlador ponderado fijo, sin muestrear acciones de PPO.
- La evaluación final ejecuta el Kfinal exportado, sin utilizar el actor PPO en línea.

Repetir un rollout del mismo controlador, con la misma planta, condición inicial, saturación y horizonte, produce el mismo costo en este simulador determinista. Repetir un entrenamiento completo con la misma semilla también puede reproducir el resultado.

Si el costo parece constante entre guías sucesivas de una única ejecución, hay que comprobar `theta_probe`, `theta_update`, los pesos, los cambios por coordenada y la precisión de impresión. No se deduce un fallo solo de la igualdad de costos entre evaluaciones repetidas del mismo K.

No se garantiza reproducibilidad exacta entre dispositivos, versiones de bibliotecas o CPU/GPU.

### 9.8. `ep_len_mean` no mide mejora del controlador

`ep_len_mean` es la longitud media, en steps, de los episodios recientes registrados por SB3. Si todos terminan al alcanzar `max_steps`, permanece cerca de ese valor aunque el costo cambie mucho.

- `max_steps` determina la duración máxima del episodio.
- `ppo_n_steps` determina la cantidad de transiciones recolectadas antes de optimizar la política.
- `dk_hold_steps` determina la cadencia de propuestas aceptadas.
- `k_eval_steps` determina la duración de los probes offline.

Cambiar los tres últimos no obliga a cambiar `ep_len_mean`. Si se espera 1000 y se observa 50, primero hay que comprobar la configuración efectiva de `main` y el JSON del experimento.

## 10. Escalas temporales y presupuesto de simulación

### 10.1. Definiciones

Se utilizan:

$$
N=\texttt{max\_steps},\quad
H=\texttt{dk\_hold\_steps},\quad
L=\texttt{ppo\_n\_steps},\quad
N_e=\texttt{k\_eval\_steps}.
$$

Las escalas físicas son:

$$
T_{\mathrm{episodio}}=Ndt,
\quad T_{\mathrm{hold}}=Hdt,
\quad T_{\mathrm{probe}}=N_e dt,
\quad T_{\mathrm{recolección}}=Ldt.
$$

El último valor es una suma de tiempo físico simulado a través de episodios. No representa necesariamente una trayectoria continua ni el tiempo de reloj del ordenador.

La retención efectiva termina cuando termina el episodio. Por eso un hold nominal de 100 pasos no llega a completarse dentro de un episodio de 50 pasos.

### 10.2. Comparación de escalas

| Escala | Original: defaults | Nuevo: defaults de `main` | Valores de tu clase, si se activan explícitamente |
|---|---:|---:|---:|
| Paso físico/RL | 0.05 s | 0.05 s | 0.05 s |
| Episodio $Ndt$ | 2.5 s | **2.5 s** | 50 s |
| Retención nominal $Hdt$ | 2.5 s | 5 s | 15 s |
| Retención realmente alcanzada | 2.5 s | **2.5 s: termina el episodio** | 15, 15, 15 y 5 s |
| Propuestas por episodio | 1 | 1 | 4 |
| Evaluación de guía $N_e dt$ | 5 s | 10 s | 50 s |
| Evaluación inicial/final | 10 s | 10 s | 10 s, salvo argumento explícito |
| Recolección PPO $Ldt$ | 51.2 s | 102.4 s | 102.4 s |
| Decaimiento del descuento $-dt/\log\gamma$ | ≈4.975 s | ≈4.975 s | ≈4.975 s |
| Decaimiento GAE | ≈0.815 s | ≈0.815 s | ≈0.815 s |

### 10.3. Redondeo de SB3 y conteo de propuestas

Para un entorno y un entrenamiento nuevo, SB3 completa bloques de $L$ transiciones. Sin parada anticipada del entrenamiento:

$$
T=L\left\lceil\frac{T_{\mathrm{solicitado}}}{L}\right\rceil.
$$

Sin terminaciones anticipadas de episodios:

$$
E=\left\lfloor\frac{T}{N}\right\rfloor,
\qquad R=T-EN.
$$

$E$ es el número de episodios completos y $R$ el número de pasos del episodio parcial final. El número de propuestas aceptadas, incluyendo ese parcial, es:

$$
P=E\left\lceil\frac{N}{H}\right\rceil+
\begin{cases}
\left\lceil R/H\right\rceil,&R>0,\\
0,&R=0.
\end{cases}
$$

No debe utilizarse simplemente $T/H$ cuando los episodios reinician la cadencia o el hold supera la duración del episodio.

El número de ciclos de recolección/optimización PPO es $T/L$. La guía se realiza al final de los episodios completos, no al final de cada bloque de recolección PPO.

### 10.4. Presupuesto para 70000 pasos solicitados

Estas cuentas se derivan del código; no son resultados de entrenamientos completos ejecutados durante la revisión. Suponen episodios y evaluaciones válidos que completan su horizonte.

| Cantidad | Original | Nuevo `main` | Clase activada explícitamente |
|---|---:|---:|---:|
| Pasos reales PPO | 70656 | 71680 | 71680 |
| Ciclos de recolección/optimización PPO | 69 | 35 | 35 |
| Episodios completos / guías | 1413 | 1433 | 71 |
| Pasos del episodio final parcial | 6 | 30 | 680 |
| Propuestas aceptadas, incluido el parcial | 1414 | 1434 | 287 |
| Transiciones de guía | 141300 | 286600 | 71000 |
| Transiciones totales, incluidas evaluaciones | 212456 | **358880** | 144080 |

La cuenta total incluye:

1. Pasos de entrenamiento PPO.
2. Referencia inicial interna de `k_eval_steps`.
3. Rollout inicial solicitado para CSV.
4. Un probe por guía.
5. Rollout final solicitado para CSV.

Si las evaluaciones inicial y final tienen $N_f$ pasos cada una:

$$
T_{\mathrm{sim,total}}=T+(E+1)N_e+2N_f.
$$

Aquí $N_f=200$ para 10 s con `dt=0.05`. Las transiciones offline no incrementan `model.num_timesteps`.

Con los defaults nuevos, aproximadamente el 80 % de las transiciones simuladas pertenece a evaluaciones. El tiempo físico simulado total es 17944 s, sumado sobre múltiples trayectorias y reinicios, no tiempo de ejecución del ordenador.

Con 10 épocas y minibatches de 64, el número aproximado de pasos Adam es:

$$
\frac{T}{L}\times10\times\frac{L}{64}.
$$

Es 11040 para la versión original y 11200 para la nueva, con esos defaults y sin interrupciones de optimización. Duplicar `n_steps` reduce la frecuencia de actualización de la política, pero no reduce a la mitad los pasos totales del optimizador.

### 10.5. Escalas de la planta y de los polos de K

Cerca del origen, el período angular nominal sin control es:

$$
\frac{2\pi}{\omega}\approx3.14\;\mathrm{s}
\qquad(\omega=2).
$$

Para el ciclo límite sin control, las ecuaciones dan:

$$
\dot\varphi=\omega-\beta\mu.
$$

En $\mu=2$, $\omega=2$, $\beta=0.5$, el período es aproximadamente 6.28 s.

Para el decaimiento asociado al radio $r_p$ de los polos del controlador:

$$
\tau_K=-\frac{dt}{\log r_p}.
$$

Se obtiene aproximadamente:

- `Kinit_mu1.0.mat`: 0.135 s.
- `Kfinal_mu1.0.mat`: 0.081 s.
- Radio máximo permitido de 0.98: 2.475 s.

Son referencias del sistema conocido, no una nueva linealización en `mu_target`. Los transitorios del lazo cerrado pueden ser más lentos.

Un episodio de 2.5 s puede evaluar solo una parte del transitorio y ni siquiera un ciclo completo de la oscilación de referencia.

## 11. Pipeline actual y responsabilidades por archivo

### 11.1. Recorrido de ejecución

```text
main
  → builders: MAT → ControllerSOS → ZPKTopology → ZPKActionMapper → env
  → referencia inicial interna de k_eval_steps
  → rollout inicial y CSV

  → PPO recolecta pasos:
      observación → acción
      → aceptar o ignorar según la cadencia
      → K → saturación → costo → avance físico
      → al terminar episodio:
          softmax → un probe offline → actualización heurística → reset

  → al completar ppo_n_steps:
      optimización de actor/crítico

  → fin del presupuesto:
      guardar PPO
      exportar K(theta_base)
      cargar ese MAT y evaluarlo
      guardar CSV, métricas y figuras
```

La referencia inicial interna y el rollout inicial para CSV ya eran dos operaciones distintas en el código original. Se mantienen y deben contabilizarse separadamente.

### 11.2. Comparación y mejora por archivo

| Archivo | Papel en `controller_lti` y diferencia relevante | Punto de mejora |
|---|---|---|
| `__init__.py` | Importa las clases públicas. El nuevo carga Torch primero, como corrección del orden de DLL en Windows. | Esa carga también vuelve obligatoria la importación de Torch para utilizar el paquete. |
| `config.py` | Original: constantes globales, dataclass y mensajes al importar. Nuevo: alias de `HybridSOSConfig` definido en `env.py`. | Evitar que los defaults de la clase y de `main` parezcan dos configuraciones activas equivalentes. |
| `main.py` | Define el experimento, crea la planta, evalúa inicialmente, entrena, guarda, evalúa el MAT exportado y genera métricas. | Mostrar siempre la configuración efectiva; eliminar la sorpresa del `50` implícito. |
| `builders.py` | Carga MAT → valida Ts → construye topología → mapper → ambiente. Ahora admite una fábrica de planta. | Permitir configurar las escalas del mapper desde `main`. |
| `plants.py` | Dinámica no lineal, medición, condición inicial e integrador. Ecuaciones conservadas. | Registrar fallos del solucionador/fallbacks y estudiar sensibilidad numérica sin cambiar el período de K. |
| `controller_sos.py` | Mantiene estados de K y ejecuta la cascada. Ahora valida `Ts=dt`. | Analizar el efecto de reiniciar K en cada propuesta; conservarlo mientras se compare el método actual. |
| `zpk.py` | Parametrización, escalas, proyección, normalización y reconstrucción de K. Algoritmo conservado. | Escalas sensibles al punto actual; cuidado con el cero cerca de 1 y con diferencias angulares al cruzar ±π. |
| `env.py` | Observación, retención de propuestas, interacción física, reward, cierre de episodio y guía. | Acciones ignoradas, observabilidad parcial, bloques desiguales y referencia heurística. |
| `simulation.py` | Original: simulador offline separado. Nuevo: costo, saturación y avance compartidos. | Separar validez numérica de éxito de control; instrumentar saturación y métricas de cola. |
| `training.py` | Construye PPO y ejecuta `learn`. Nuevo: semilla fija y transferencia desde ZIP. | Hacer visibles `gamma`, `gae_lambda` y la relación entre decisiones efectivas y pasos recolectados. |
| `policy_io.py` | Nuevo: guardado nativo, contrato de compatibilidad y actor MATLAB. | La arquitectura 64–64 está validada explícitamente; cambiarla exige actualizar el contrato/exportación. |
| `io_utils.py` | CSV, MATLAB, JSON y figuras. El nuevo guarda métricas y reconstrucciones SOS/SS coherentes. | No comparar como equivalentes costos de horizontes distintos; añadir indicadores de convergencia de la cola. |
| `utils.py` | RMS, ángulos, etiquetas y resumen temporal. | El resumen heredado usa `N/H`; debe contrastarse con el conteo real con `ceil` y reinicios por episodio. |
| Archivos `.mat` | Determinan el controlador de partida. El nuevo incluye y carga un `Kfinal` anterior. | Separar procedencia, `mu_target` y `mu_design`; el nombre no demuestra el punto de diseño. |

### 11.3. Selección final

Kfinal es la última base resultante de las guías completadas:

$$
K_{\mathrm{final}}=K(\theta_{\mathrm{base,última}}).
$$

No se elige el mejor probe histórico ni la última propuesta individual de PPO. Tampoco se genera una nueva propuesta determinista para seleccionar Kfinal.

Si termina el presupuesto dentro de un episodio, no se fuerza una guía adicional para el parcial.

La última actualización de pesos PPO sucede después de recolectar el último bloque. Esos pesos recién actualizados pueden no haber propuesto ningún controlador antes de exportar Kfinal. Por tanto, el actor guardado y el Kfinal exportado tienen funciones distintas: el primero es el optimizador entrenado; el segundo es el controlador de despliegue.

### 11.4. Exportación, métricas y reproducción

El pipeline nuevo conserva accesibles los datos de las guías sin exportar un CSV por rollout. Se guardan las métricas por timestep/episodio y los parámetros del controlador ponderado realmente evaluado.

Las curvas distinguen costo instantáneo, costo medio por episodio y evaluaciones offline. El costo de entrenamiento puede corresponder a un controlador que cambia dentro del episodio; no equivale necesariamente al costo de un único K fijo.

Se distinguen:

$$
J_{\mathrm{total}}=\sum_k\ell_k,
\qquad J_{\mathrm{mean}}=J_{\mathrm{total}}/N.
$$

No se utiliza como indicador de mejora la suma acumulada de todos los costos desde el inicio. Con costos no negativos crecería aunque el entrenamiento mejorase.

Las medias sobre horizontes diferentes pueden cambiar por transitorios. Los totales con distintos números de pasos tampoco permiten clasificar directamente controladores como si el objetivo fuera el mismo.

Los CSV inicial y final utilizan:

```text
t, estados de la planta, y, u_sat
```

Cada fila contiene $x_k,y_k$ al inicio de $[t_k,t_k+dt)$ y la entrada aplicada durante ese intervalo. No se inventa una acción final para completar una fila terminal.

En el nuevo `main`, el rollout final carga y evalúa exactamente el MAT recién exportado. El formato conserva ZPK, órdenes, ganancia y Ts; las representaciones SOS/SS adicionales se regeneran a partir del mismo controlador.

El ZIP nativo de SB3 permite guardar pesos, arquitectura, optimizador e información de entrenamiento. La carga valida espacios, transformaciones y arquitectura. El estado de la guía se conserva para análisis, pero iniciar otro entrenamiento desde el ZIP reinicia el experimento físico: no es una reanudación exacta de un rollout interrumpido.

El actor MATLAB incluye pesos, sesgos, activaciones, orientación de matrices, clipping y metadatos de inferencia. En LTI su salida representa acciones de parámetros, no la entrada física del controlador en línea.

### 11.5. `mu_design`, `mu_target` y continuación manual

Se mantienen separados:

- `mu_design`: punto conocido donde se diseñó inicialmente el controlador; puede ser desconocido.
- `mu_target`: parámetro de la planta del experimento.
- La etiqueta `mu` del nombre del archivo cargado.

`Kinit_mu1.0.mat` no prueba que el controlador se haya diseñado linealizando en `mu=1.0`. No se añade identificación ni linealización en el valor objetivo.

La continuación es manual: se carga un Kfinal previo para entrenar en el siguiente `mu_target`. No se automatiza una trayectoria de continuación numérica.

La función `mu_tag` conserva suficiente precisión de la representación float y mantiene nombres como `mu1.0`. Si MATLAB usa `sprintf('%.1f', mu_target)`, coincide para valores con un decimal, pero no distingue valores como 1.05; al utilizar esos valores hay que adaptar el formato para que coincida con Python.

En la función MATLAB discutida, el primer punto debe cargar `Kinit_mu{mu_target}.mat`; los siguientes pueden cargar `Kfinal_mu{mu_anterior}.mat`. La cadena literal `Kinit_mustart.mat` solo cargaría un archivo que realmente tenga ese nombre.

## 12. Cómo aumentar la probabilidad de propuestas útiles con el mecanismo actual

### 12.1. Definir qué se considera éxito

Manteniendo el costo actual, conviene distinguir:

$$
\text{trayectoria numéricamente válida},
$$

$$
J_{\mathrm{probe}}<J_{\mathrm{referencia}},
$$

$$
\operatorname{RMS}_{\mathrm{cola}}\|x\|<\varepsilon.
$$

El último criterio puede calcularse con los estados exportados sin añadirlos a la observación de PPO. El umbral, la ventana final y cualquier límite físico deben fijarse según el objetivo del experimento.

No debe equipararse `valid=True`, reducción de la medición y convergencia al equilibrio.

### 12.2. Varios bloques iguales y horizontes comparables

La propuesta temporal de diagnóstico fue:

```text
max_steps       = 1024
dk_hold_steps   = 128
k_eval_steps    = 1024
ppo_n_steps     = 2048
eval_final_time = 51.2
dt              = 0.05
```

Produce ocho propuestas por episodio, bloques de 6.4 s, dos episodios por recolección PPO y evaluaciones de igual duración.

No se presentó como una configuración óptima. Mejora la coherencia temporal y evita el bloque final corto, pero mantiene pocas decisiones efectivas respecto a los pasos físicos.

### 12.3. Ajustar primero la perturbación del cero con una prueba controlada

Se ejecutó un diagnóstico local con 32 vectores de acción comunes, gaussianos recortados, alrededor del `Kfinal_mu1.0.mat` adjunto. Se mantuvieron iguales las otras escalas, se reinició cada evaluación desde el mismo `x0` y se utilizó `mu_target=2`.

| Evaluación de 50 s | Escala del cero 0.01 | Escala del cero 0.002 |
|---|---:|---:|
| Propuestas válidas | 32/32 | 32/32 |
| Propuestas con menor costo que la base | 7/32 | 11/32 |
| Costo mediano | 0.203071 | 0.195233 |
| Mejor costo | 0.166250 | 0.180820 |
| Peor costo | 0.224840 | 0.229342 |

La base dio:

$$
J_{\mathrm{mean}}\approx0.193111.
$$

La escala menor produjo más mejoras en esa muestra y una mediana más próxima a la base. No mejoró el peor caso y perdió la mejora extrema más favorable.

Es evidencia local para experimentar, no una estimación fiable de la probabilidad de éxito de la política entrenada. Se evaluaron propuestas para diagnóstico; esas evaluaciones adicionales no se incorporaron a la guía del entrenamiento.

Con la escala reducida, cuatro de las 32 propuestas cambiaron de clasificación «mejora/no mejora» al pasar de una evaluación de 10 s a otra de 50 s. El horizonte puede cambiar la decisión de la guía.

Para probar la reducción discutida, el argumento del constructor de `ZPKActionMapper` en `zpk.py` es:

```python
scale_z_real: float = 0.002,
```

El ZIP analizado no expone ese campo como argumento de terminal. Cambiar `--action-scale` no realiza esta modificación. Conviene probar primero la corrección temporal y después la escala, guardando cada experimento en una carpeta distinta.

La evidencia local anterior corresponde a `mu=2`. Debe comprobarse de nuevo su efecto al utilizar `mu=1`, `mu=1.5` o un controlador diferente.

### 12.4. Calibrar `beta_softmax` después de fijar los bloques

Con $B$ bloques iguales, cada uno de longitud $H=N/B$, y ningún fallo:

$$
R_b=-\frac{H}{N}\bar\ell_b=-\frac{\bar\ell_b}{B},
$$

$$
w_b\propto
\exp\left(-\frac{\beta_{\mathrm{softmax}}}{B}\bar\ell_b\right).
$$

Con ocho bloques y `beta_softmax=5`, la escala efectiva sobre el costo medio del bloque es $5/8$.

Conviene observar los pesos y el número efectivo de propuestas:

$$
N_{\mathrm{efectivo}}=\frac{1}{\sum_b w_b^2}.
$$

- Si todos los pesos son casi iguales, se promedian perturbaciones que pueden cancelarse.
- Si un peso domina siempre, puede estar seleccionando ruido, transitorios o una fase favorable de la planta.

Subir β indiscriminadamente amplifica también esos sesgos. Mantener inicialmente β=5 al corregir los tiempos permite estudiar el efecto temporal sin cambiar simultáneamente la selectividad.

### 12.5. Separar el aprendizaje de la red y la actualización de K

`learning_rate=3e-4` actualiza los pesos de PPO. `lr_good` y `lr_bad` actualizan los parámetros del controlador mediante la guía.

Conviene ajustar una familia cada vez. La recomendación inicial fue mantener `lr_good=0.3` en lugar de 0.5: un paso mayor hace más agresivo el desplazamiento sin aportar nueva evidencia sobre su calidad.

Reducir `lr_bad` limita la deriva por probes válidos clasificados como malos, pero también reduce exploración y no corrige el sesgo de la referencia `J_base`. En los comandos posteriores se conservó `lr_bad=0.001`, tal como propusiste.

### 12.6. Instrumentar las decisiones útiles

Antes de aumentar `total_timesteps`, y sin añadir rollouts a la guía, conviene analizar:

- Número de propuestas efectivamente aceptadas.
- Cambios por coordenada de theta, normalizados por sus escalas.
- Pesos softmax y cancelación entre propuestas.
- Fracción de tiempo saturado.
- Cambio real de `theta_base`.
- Métricas de estado y medición al final de las evaluaciones existentes.

`delta_norm` mezcla radios, ángulos, ceros y log-ganancia. Puede estar dominada por la escala de ganancia y ocultar cambios pequeños pero críticos en el cero.

### 12.7. Revisar después el horizonte PPO y la exploración

Hacer explícitos `gamma` y `gae_lambda` permite relacionarlos con los tiempos de respuesta observados. Ajustarlos conserva PPO y la guía, pero cambia el balance entre sesgo, varianza y horizonte del aprendizaje.

En cambio, las siguientes opciones sí requieren una modificación más sustancial:

- Entrenar al actor solamente en instantes de propuesta.
- Convertir cada bloque en una decisión temporal.
- Ampliar la observación con memoria, fase o controlador activo.
- Cambiar la puntuación softmax a costo medio por bloque.
- Cambiar el reinicio de estados internos de K.
- Cambiar el criterio de selección final.

Deben tratarse como variantes identificadas, con una comparación explícita, no como ajustes silenciosos del mismo método.

### 12.8. Por qué una perturbación pequeña no garantiza mejora

Localmente:

$$
J(\theta+\delta)-J(\theta)
\approx\nabla J(\theta)^{\mathsf T}\delta.
$$

Si la perturbación es simétrica y centrada, el gradiente no es nulo y la aproximación local es válida, aproximadamente la mitad de las direcciones son descendentes.

Reducir la escala controla la distancia y el riesgo de una propuesta, pero no crea una dirección favorable. Cerca de restricciones, singularidades funcionales, saturación o cambios de régimen, esa aproximación necesita especial cuidado.

Aumentar la probabilidad de mejora requiere que la política o la ponderación identifiquen direcciones útiles para el objetivo que realmente importa. El mecanismo actual no garantiza mejora en cada rollout ni descenso monótono del costo.

## 13. Configuraciones discutidas progresivamente

Esta sección conserva la evolución de los parámetros. No todas las configuraciones representan el mismo experimento ni fueron ejecutadas en la revisión.

### 13.1. Configuración inicial indicada en la conversación

```text
mu_target       = 1.0
dt              = 0.05
total_timesteps = 70000
max_steps       = 1000
ppo_n_steps     = 2048
batch_size      = 64
dk_hold_steps   = 300
lr_good         = 0.5
lr_bad          = 0.001
beta_softmax    = 5
k_eval_steps    = 1000
q_y             = 1
r_u             = 0.01
force_limit     = 5
seed            = 0
```

La limitación temporal identificada fue el reparto de bloques 300, 300, 300 y 100. La recomendación fue pasar a bloques iguales y reducir la agresividad de `lr_good` para la prueba inicial.

### 13.2. Propuesta de corrección temporal

Desde la carpeta padre de `ppo_pipeline_test`, en PowerShell:

```powershell
python -m ppo_pipeline_test.main `
  --mode controller_lti `
  --mu-target 1.0 `
  --dt 0.05 `
  --total-timesteps 70000 `
  --max-steps 1024 `
  --ppo-n-steps 2048 `
  --batch-size 64 `
  --dk-hold-steps 128 `
  --lr-good 0.3 `
  --lr-bad 0.001 `
  --beta-softmax 5.0 `
  --k-eval-steps 1024 `
  --eval-final-time 51.2 `
  --q-y 1.0 `
  --r-u 0.01 `
  --force-limit 5.0 `
  --seed 0 `
  --output-dir resultados_corregidos
```

Con el `main` del ZIP analizado, este comando carga por defecto `Kfinal_mu1.0.mat`. Para otro controlador debe indicarse `--mat-file`.

| Consecuencia | Valor |
|---|---:|
| Episodio | 51.2 s |
| Bloque de propuesta | 6.4 s |
| Propuestas por episodio | 8 |
| Episodios por recolección PPO | 2 |
| Probe, evaluación inicial y evaluación final | 51.2 s cada uno |
| Pasos reales PPO para 70000 solicitados | 71680 |
| Episodios completos y guías, sin fallos | 70 |

La reducción de `scale_z_real` a 0.002 es una prueba separada y no está incluida en ese comando.

### 13.3. Variante posterior con bloques de 32 pasos

Después se plantearon `mu_target=1.5`, presupuesto solicitado 80000, `max_steps=1024`, `dk_hold_steps=32`, `k_eval_steps=1024` y evaluación final de 10 s.

Eso conserva bloques iguales, pero cambia:

- El bloque a 1.6 s.
- Las propuestas por episodio a 32.
- El horizonte inicial/final a 10 s, distinto de los probes de 51.2 s.

Con 32 bloques, `beta_softmax=5` implica una escala efectiva de $5/32$ sobre el costo medio de bloque. La selección es diferente de la configuración con ocho bloques; conservar el mismo β no conserva la misma selectividad por unidad de costo medio.

También apareció `--force-limit .0`: ese valor equivale a cero y es rechazado por el ambiente. Se corrigió a `--force-limit 5.0`.

### 13.4. Configuración posterior de 128 pasos y menor penalización de esfuerzo

En el JSON posterior se indicaron:

```text
max_steps       = 128
dk_hold_steps   = 32
total_timesteps = 80000
ppo_n_steps     = 2048
k_eval_steps    = 1024
r_u             = 0.001
eval_final_time = 10
```

Estos valores son distintos del comando previo de `max_steps=1024` y `r_u=0.01`. Para reproducir el JSON hay que cambiar explícitamente ambos argumentos.

| Cantidad | Valor |
|---|---:|
| Duración del episodio | **6.4 s** |
| Duración de cada propuesta | **1.6 s** |
| Propuestas por episodio | **4**, de igual longitud |
| Episodios por recolección PPO | **16** |
| Duración de cada probe offline | **51.2 s** |
| Duración de evaluación inicial/final | **10 s** |

El menor `r_u` reduce diez veces la penalización del esfuerzo respecto al comando anterior, manteniendo `q_y=1`. Es un cambio del objetivo, además del cambio temporal.

`Ts=0.05` se valida al cargar el controlador. `nf=0` conserva el espacio de observación LTI. `action_scale=5` no modifica las perturbaciones ZPK. `alpha_update` y `allow_worse_base_update` siguen inactivos en la actualización y no tienen argumentos CLI.

### 13.5. Comando para la configuración de 128 pasos

Este bloque utiliza la ubicación de trabajo indicada en la conversación. La ruta del controlador se solicita para evitar depender del nombre predeterminado de `main`.

```powershell
cd "C:\Users\rmarinme\Documents\Flow Stabilization\PYTHON\RL\Codes\Stuart Landau\PPO_Transfer_Function"

$controllerPath = (Read-Host "Ruta completa del controlador MATLAB").Trim('"')

if (-not (Test-Path -LiteralPath $controllerPath -PathType Leaf)) {
    throw "No existe el archivo: $controllerPath"
}

python -m ppo_pipeline_test.main `
  --mode controller_lti `
  --mat-file "$controllerPath" `
  --mu-target 1.5 `
  --dt 0.05 `
  --total-timesteps 80000 `
  --max-steps 128 `
  --ppo-n-steps 2048 `
  --batch-size 64 `
  --dk-hold-steps 32 `
  --lr-good 0.3 `
  --lr-bad 0.001 `
  --beta-softmax 5.0 `
  --k-eval-steps 1024 `
  --eval-final-time 10 `
  --q-y 1.0 `
  --r-u 0.001 `
  --fail-penalty 100 `
  --force-limit 5.0 `
  --seed 0 `
  --nf 0 `
  --action-scale 5.0 `
  --output-dir resultados
```

Para igualar la evaluación inicial y final al horizonte de los probes, sustituir `--eval-final-time 10` por `--eval-final-time 51.2`.

Los imports relativos requieren ejecutar el paquete mediante `python -m ...` desde su carpeta padre. `Path(__file__).with_name(...)` busca el MAT junto a `main.py`; `--mat-file` permite indicar otra ruta explícita. El valor de `mu_target` no selecciona por sí solo un archivo inicial ni automatiza la continuación.

### 13.6. Comparación temporal de las configuraciones posteriores

Las siguientes cuentas son derivaciones aritméticas de los parámetros, no resultados nuevos de entrenamiento. Suponen un entorno, ausencia de fallos y evaluaciones completas.

| Cantidad | Propuesta temporal | Variante de 32 pasos | Configuración del JSON posterior |
|---|---:|---:|---:|
| `mu_target` discutido | 1.0 | 1.5 | 1.5 |
| Presupuesto solicitado | 70000 | 80000 | 80000 |
| `max_steps` | 1024 | 1024 | 128 |
| `dk_hold_steps` | 128 | 32 | 32 |
| `ppo_n_steps` | 2048 | 2048 | 2048 |
| `k_eval_steps` | 1024 | 1024 | 1024 |
| Evaluación inicial/final | 51.2 s | 10 s | 10 s |
| `r_u` | 0.01 | 0.01 | 0.001 |
| Propuestas por episodio | 8 | 32 | 4 |
| Episodios por recolección PPO | 2 | 2 | 16 |
| Pasos PPO reales | 71680 | 81920 | 81920 |
| Ciclos de recolección/optimización PPO | 35 | 40 | 40 |
| Episodios completos / guías | 70 | 80 | 640 |
| Propuestas aceptadas totales | 560 | 2560 | 2560 |
| Transiciones de guía | 71680 | 81920 | 655360 |
| Transiciones totales con referencia e inicial/final | 146432 | 165264 | 738704 |

Aunque las dos últimas configuraciones aceptan 2560 propuestas, no son equivalentes: cambian el número de guías, la frecuencia de actualización de la base, el estado inicial de cada propuesta, el número de resets y el costo de simulación.

Con `max_steps=128`, el probe es ocho veces más largo que un episodio de entrenamiento. La comparación de sus costos requiere indicar esa diferencia incluso si se representan medias.

## 14. Prioridades de mejora y alcance de cada cambio

| Prioridad | Acción | Relación con el mecanismo actual |
|---|---|---|
| 1 | Comprobar los valores efectivos de `main` y el JSON. | Diagnóstico/configuración; evita entrenar otro experimento del que se pretendía. |
| 2 | Elegir bloques completos de igual longitud y horizontes documentados. | Ajuste de parámetros; conserva las fórmulas. |
| 3 | Examinar la sensibilidad del cero y de la ganancia. | Ajuste explícito de escalas; conserva la parametrización. |
| 4 | Registrar saturación, cambios por coordenada, pesos y métricas de cola. | Instrumentación con las trayectorias ya solicitadas; no exige otro probe por guía. |
| 5 | Calibrar β y los pasos heurísticos una vez fijados los tiempos. | Ajuste de parámetros del mecanismo actual. |
| 6 | Corregir la referencia no finita y hacer visibles los hiperparámetros temporales de PPO. | Corrección explícita y control experimental; debe verificarse. |
| 7 | Estudiar acciones ignoradas, observación y atribución temporal. | Cambios más sustanciales si se decide alterar los contratos actuales. |

No se propone añadir certificación de estabilidad, región de atracción, Safe-RL, identificación ni MBRL. La convergencia observada en simulación no constituye una garantía de estabilidad.

## 15. Comprobaciones realizadas y límites de la evidencia

Durante el análisis comparativo descrito se realizaron:

- Inspección de las diferencias de código de los dos ZIP.
- Lectura de parámetros de los archivos MATLAB.
- Comparación numérica de 63 pasos con condiciones y acciones iguales.
- Comprobación de una sola propuesta y peso `[1.0]` con los defaults nuevos.
- Simulaciones de controladores fijos y análisis de la cola de la trayectoria descrita.
- Diagnóstico local de 32 perturbaciones comunes para cada escala de cero indicada.
- Comparación de esas propuestas a horizontes de 10 y 50 s.
- Cálculos de escalas temporales y presupuestos a partir del código.

La coincidencia del núcleo LTI se obtuvo en las trayectorias válidas probadas. Los diagnósticos de exploración fueron pruebas locales de candidatos, no campañas de entrenamiento PPO.

Durante esa revisión no se ejecutó un entrenamiento nuevo completo de 70000 pasos ni una campaña estadística de convergencia. Las configuraciones posteriores de 80000 pasos son ejemplos y cuentas de presupuesto; este documento no les atribuye resultados de rendimiento.

Los resultados locales destacados corresponden a `mu_target=2`, `x0=[1.0,0.5]`, el Kfinal adjunto, `q_y=1`, `r_u=0.01` y límite de fuerza 5. No se transfieren automáticamente a `mu=1.0`, `mu=1.5`, otro MAT o el costo posterior con `r_u=0.001`.

La versión original no especificaba las versiones de las dependencias. Se verificaron los defaults PPO expuestos usando SB3 2.9.2 en el entorno de revisión. No se garantiza igualdad de resultados entre versiones, plataformas o dispositivos.

Este README conserva la distinción entre comportamiento implementado, comprobaciones ejecutadas, cálculos derivados y propuestas de mejora. No presenta una mejora por rollout, un descenso monótono del costo o una garantía de estabilidad como resultados demostrados.
