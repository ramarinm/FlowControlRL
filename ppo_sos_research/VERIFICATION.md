# Verificaciones ejecutadas

Se ejecutó el código, no solamente una revisión estática. Entorno Linux, Python
3.12.14, CPU; NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.15.0+cpu,
stable-baselines3 2.9.2, Gymnasium 1.4.0, Matplotlib 3.10.8, pytest 9.2.0.
El ZIP original no especificaba versiones de dependencias.

## Suite automatizada

Resultado final: **19 pruebas pasadas, 0 fallos, 17.61 s**. El único warning
provino del plugin `anyio` preinstalado en el entorno y de una deprecación interna
de pytest, no de la simulación. Logs exactos: `verification/pytest.txt` y
`verification/pytest.xml`.

Cobertura ejecutada:

- Regresión frente al **código original ejecutado**: 21 acciones fijas, tres
  episodios de 7 pasos, hold=3, bloque final parcial, 11 pasos por guía y límite
  físico 0.2. Observaciones, estados físicos, rewards, theta_trial, theta_base,
  J_base y J_probe coinciden dentro de `rtol=atol=1e-12`.
  `tests/fixtures/original_lti_trace.*` conserva resultados y hashes de procedencia.
- Conteo de llamadas: un solo rollout por guía, pesos softmax por bloques,
  theta_probe, theta_update y referencia heurística. No se evalúa theta_update.
- Una llamada a `plant.step` y avance físico dt por acción válida, en los tres
  modos. Saturación compartida y canales de esfuerzo correctos. El caso residual
  usa una base sintética identificada como fixture de prueba para provocar
  saturación y distinguir u_rl de u_sat y de u_sat-u_base.
- nf=0 y nf=2 en ambos modos directos: dimensiones, orden temporal, valores y
  reinicio. Coeficientes de K_base fijos y estado interno reiniciado.
- Rechazo de Ts ausente o incompatible; CSV con datos al inicio del intervalo,
  entrada aplicada y número exacto de filas; exportación/recarga ZPK y equivalencia
  cascada/SOS/SS. El rollout de Kfinal coincide con el controlador exportado.
- Detección inmediata de fallo tras el integrador, reward finito con penalización
  y métricas que no consideran una trayectoria fallida una mejora por duración.
- Entrenamientos PPO cortos en los tres modos, guardado, recarga, igualdad exacta
  de acciones deterministas y del estado del optimizador; inicialización de otro
  entrenamiento desde el ZIP. Rechazo de arquitectura, límites o nf incompatibles.
- Inferencia NumPy leyendo el MAT contra SB3, incluyendo prueba deliberada de
  clipping con sesgo de salida alto. Tres ejecuciones completas de main con
  CSV, figuras, MAT, ZIP, metadatos y presupuesto. Presupuesto solicitado 17 y
  usado 32 verifica explícitamente el redondeo de SB3 a bloques de rollout.
- Dos entrenamientos repetidos CPU con la misma semilla produjeron la misma
  acción determinista en la observación probada; mu_tag distingue floats vecinos.

## Ejecuciones con tamaños PPO originales

También se completaron tres experimentos de **1024 timesteps reales** cada uno,
con `n_steps=1024`, `batch_size=64`, arquitectura MLP original y demás parámetros
PPO originales. `mu_target=1.0`, semilla 0, dt=0.05, episodio de 50 steps,
evaluación final de 10 s. En los modos directos nf=2 y action_scale=15.
LTI/residual cargaron el `Kinit_mu1.0.mat` adjunto. Todas las evaluaciones finales
completaron sus 200 transiciones con `valid=true`.

| Modo | Entrenamiento | Referencia inicial | CSV inicial | Guía offline | Evaluación final |
|---|---:|---:|---:|---:|---:|
| controller_lti | 1024 | 100 | 200 | 2000 (20 rollouts) | 200 |
| model_free | 1024 | 0 | 0 | 0 | 200 |
| residual_rl | 1024 | 0 | 0 | 0 | 200 |

Errores máximos absolutos, comparando el actor MAT leído en NumPy con SB3 sobre
17 observaciones representativas por experimento:

| Modo | Acción PPO recortada | Corrección física u_rl |
|---|---:|---:|
| controller_lti | 2.24e-8 | No corresponde: acción de parámetros |
| model_free | 1.12e-8 | 1.79e-7 |
| residual_rl | 1.49e-8 | 2.38e-7 |

Los resultados completos de estas ejecuciones se incluyen en el archivo separado
`ppo_sos_validation_runs.zip`: son pruebas de funcionamiento, no controladores
recomendados ni demostraciones de convergencia. Las curvas se inspeccionaron.
El archivo de código no incorpora trayectorias de entrenamiento ni dependencias
binarias. Solo guarda las trayectorias inicial/final solicitadas y las métricas.

## Pendiente y límites reales

- MATLAB y Octave no estaban disponibles. No se ejecutaron los `.m`, MATLAB ni
  Simulink. La prueba realizada es lectura del MAT e inferencia independiente en
  Python. Se entrega `matlab/check_actor.m` para verificar también en MATLAB.
- No se ejecutó el presupuesto completo de 70000 pasos ni se evaluó convergencia
  estadística, múltiples semillas, GPU o reproducibilidad entre dispositivos.
- No se ejecutó cartpole: esa implementación no fue adjuntada. Tampoco se verificó
  el diseño inicial MATLAB, su linealización o el valor de mu_design, ausentes.
- La carga nativa permite inferencia y transferencia con optimizador; no captura
  el estado exacto de un rollout interrumpido ni todos los generadores externos.
- El costo residual penaliza otro canal; sus valores no se comparan como el mismo
  objetivo numérico de LTI/model_free. No hay certificación de estabilidad.
