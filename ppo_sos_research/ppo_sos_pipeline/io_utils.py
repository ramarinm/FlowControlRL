from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.io import savemat

from .controller_sos import ControllerSOS


def write_json(filename, data):
    """Standards-compliant JSON: failed/nonfinite metrics are null plus valid=false."""
    def clean(obj):
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, np.generic):
            return clean(obj.item())
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, Path):
            return str(obj)
        return obj
    Path(filename).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_final_zpk_controller(filename, P_final, Z_final, Kg_final, theta_base=None, topology=None,
                              *, mu_target=None, mu_design=None):
    """Preserve original keys/dimensions; add regenerated SOS/SS, not stale input data."""
    payload = {
        'P_sections_final': np.asarray(P_final, dtype=np.complex128),
        'Z_sections_final': np.asarray(Z_final, dtype=np.complex128),
        'Kg_final': np.array([[float(Kg_final)]]),
        'P_sections': np.asarray(P_final, dtype=np.complex128),
        'Z_sections': np.asarray(Z_final, dtype=np.complex128),
        'Kg': np.array([[float(Kg_final)]]),
    }
    if theta_base is not None:
        payload['theta_base_final'] = np.asarray(theta_base, dtype=float).reshape(1, -1)
    if topology is not None:
        payload.update(pole_order=np.asarray(topology.pole_order, dtype=int).reshape(1, -1),
                       zero_order=np.asarray(topology.zero_order, dtype=int).reshape(1, -1),
                       Ts=np.array([[float(topology.Ts)]]))
        controller = ControllerSOS(P_final, Z_final, topology.pole_order, topology.zero_order, Kg_final, topology.Ts)
        A, B, C, D = controller.build_global_ss()
        sos = []
        for sec in controller.local_sections:
            po, zo = sec['pole_order'], sec['zero_order']
            numerator = list(controller._num_coeffs(sec['zeros'], po, zo))
            a1, a2 = controller._den_coeffs(sec['poles'], po)
            if po == 1:
                numerator.append(0.0)
                a2 = 0.0
            sos.append(numerator + [1.0, a1, a2])
        payload.update(K0_sos_raw=np.asarray(sos), K0_sos_norm=np.asarray(sos),
                       Kg_raw=np.array([[float(Kg_final)]]), Ad_sos=A, Bd_sos=B, Cd_sos=C, Dd_sos=D)
    if mu_target is not None:
        payload['mu_target'] = np.array([[mu_target]])
    if mu_design is not None:
        payload['mu_design'] = np.array([[mu_design]])
    savemat(filename, payload)


def save_results_csv(results, savepath):
    """Row k = t_k, x_k, y_k, input held on [t_k,t_k+dt); no terminal row."""
    t = np.asarray(results['t'], dtype=float).reshape(-1)
    def block(key):
        data = np.asarray(results[key], dtype=float)
        if data.ndim == 1:
            data = data.reshape(len(t), -1)
        if data.ndim != 2 or data.shape[0] != len(t):
            raise ValueError(f'Invalid CSV block {key}: {data.shape}.')
        return data
    x, y, u_sat = block('x'), block('y'), block('u_sat')
    names = results.get('state_names')
    if names is None:
        names = [f'x{i+1}' for i in range(x.shape[1])]
    if len(names) != x.shape[1]:
        raise ValueError('state_names length must match plant state dimension.')
    header = ['t'] + list(names)
    blocks = [t[:, None], x]
    for name, data in [('y', y), ('u_sat', u_sat)]:
        header.extend([name] if data.shape[1] == 1 else [f'{name}{i+1}' for i in range(data.shape[1])])
        blocks.append(data)
    if 'u_rl' in results:
        u_rl = block('u_rl')
        header.extend(['u_rl'] if u_rl.shape[1] == 1 else [f'u_rl{i+1}' for i in range(u_rl.shape[1])])
        blocks.append(u_rl)
    if len(set(header)) != len(header) or any(',' in n or '\n' in n for n in header):
        raise ValueError('CSV column names must be unique and contain no commas/newlines.')
    np.savetxt(savepath, np.column_stack(blocks), delimiter=',', header=','.join(header), comments='')


def plot_training_metrics(env, directory, tag):
    """Means and totals are separate axes; failed runs are marked, never ranked as low costs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    steps, episodes, guidance = (env.metrics[k] for k in ('steps', 'episodes', 'guidance'))
    episode_label = 'Entrenamiento (K variable)' if env.mode == 'controller_lti' else 'Entrenamiento'
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    if steps:
        ax.plot([s['timestep'] for s in steps], [s['cost_k'] if s['valid'] else np.nan for s in steps],
                color='0.75', lw=0.6, label='Costo instantáneo ell_k')
    if episodes:
        ax.plot([e['timestep'] for e in episodes], [e['J_mean'] for e in episodes],
                'o-', ms=3, label=episode_label + ' — J/N por episodio')
    if guidance:
        ax.scatter([g['timestep'] for g in guidance], [g['J_mean'] for g in guidance], marker='D', s=24,
                   label=f"Offline K ponderado — J_probe = J/N (N={env.cfg.k_eval_steps})")
    # A mean over a different horizon is not made a comparable initial baseline.
    for i, reference in enumerate(env.metrics['evaluations']):
        if reference['role'] not in ('initial_reference', 'initial') or not reference['valid']:
            continue
        if reference['N'] == env.cfg.max_steps or (guidance and reference['N'] == env.cfg.k_eval_steps):
            ax.axhline(reference['J_mean'], color='tab:green', ls='--', alpha=0.7,
                       label=f"K inicial — J/N (N={reference['N']}, {reference['role']})")
    for e in episodes:
        if not e['valid']:
            ax.axvline(e['timestep'], color='red', ls=':', alpha=.5)
    ax.set(xlabel='Timesteps de PPO (excluye evaluaciones)', ylabel='Costo instantáneo / costo medio',
           title=(f'{env.mode}: entrenamiento y evaluaciones de K fijo' if env.mode == 'controller_lti'
                  else f'{env.mode}: entrenamiento de la política'))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.2)
    fig.savefig(directory / f'cost_timesteps_{tag}.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout='constrained')
    for ax, key, label in zip(axes, ('J_mean', 'J_total'), ('Costo medio J/N', 'Costo total J = sum ell_k')):
        if episodes:
            ax.plot([e['episode'] for e in episodes], [e[key] for e in episodes], 'o-', ms=3,
                    label=f'{episode_label} (N máximo={env.cfg.max_steps})')
        if guidance:
            ax.scatter([g['episode'] for g in guidance], [g[key] for g in guidance], marker='D', s=24,
                       label=f'Offline K ponderado — {"J_probe=J/N" if key == "J_mean" else "J_total"} (N={env.cfg.k_eval_steps})')
        for e in episodes:
            if not e['valid']:
                ax.axvline(e['episode'], color='red', ls=':')
                ax.text(e['episode'], .95, 'fallo', transform=ax.get_xaxis_transform(), rotation=90, va='top', color='red')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title(f'{env.mode}: costo por episodio; los horizontes N se indican en la leyenda')
    axes[1].set_xlabel('Episodio completado de entrenamiento')
    fig.savefig(directory / f'cost_episodes_{tag}.png', dpi=160)
    plt.close(fig)
