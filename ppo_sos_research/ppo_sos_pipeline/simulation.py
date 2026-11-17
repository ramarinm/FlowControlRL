from __future__ import annotations

from typing import Callable
import numpy as np

from .controller_sos import ControllerSOS


def instant_cost(y, u_cost, q_y: float, r_u: float) -> float:
    """Original measurement cost (no energy squaring, no dt factor)."""
    return float(q_y * np.mean(np.square(y)) + r_u * np.mean(np.square(u_cost)))


def saturate(u_total, force_limit):
    return np.clip(np.asarray(u_total, dtype=float).reshape(-1), -np.asarray(force_limit), np.asarray(force_limit))


def physical_step(plant, y, u_total, *, force_limit, q_y, r_u, u_cost=None):
    """Signals at t_k; u_sat acts on [t_k,t_k+dt). Exactly one plant.step.

    Failed/nonfinite input is rejected before integration. A nonfinite result is
    detected immediately after integration, without feeding NaNs back to PPO.
    State is only logged, never inserted into the policy observation.
    """
    y = np.asarray(y, dtype=float).reshape(plant.ny)
    x = np.asarray(plant.state(), dtype=float).reshape(-1).copy() if hasattr(plant, 'state') else np.zeros(0)
    u_sat = saturate(u_total, force_limit)
    u_cost = u_sat if u_cost is None else np.asarray(u_cost, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        cost = instant_cost(y, u_cost, q_y, r_u)
    valid = bool(np.all(np.isfinite(y)) and np.all(np.isfinite(u_total)) and np.isfinite(cost))
    reason = 'ok' if valid else 'nonfinite_signal_or_cost'
    transitions = 0
    if valid:
        plant.step(u_sat)
        transitions = 1
        valid = bool(np.all(np.isfinite(plant.measure())))
        if not valid:
            reason = 'nonfinite_y_after_step'
    return dict(x=x, y=y.copy(), u_raw=np.asarray(u_total).copy(), u_sat=u_sat.copy(),
                u_cost=u_cost.copy(), cost_k=cost, valid=valid, reason=reason, transitions=transitions)


def steps_for_duration(t_final: float, dt: float) -> int:
    if not np.isfinite(t_final) or t_final <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Duration and dt must be finite and positive.')
    # Same left-endpoint convention as the original while t < t_final loop.
    return int(np.ceil(t_final / dt - 1e-12))


def run_rollout(plant, control: Callable, n_steps: int, *, force_limit=15.0,
                q_y=6.0, r_u=5e-3, residual=False, capture=True, after_step=None):
    """Shared offline loop; controller/plant resets are the caller's responsibility.

    control(y) returns (u_total, u_rl). No final invented action or terminal row.
    J retains the legacy mean-cost meaning; J_total and J_mean are explicit.
    """
    if n_steps <= 0:
        raise ValueError('n_steps must be positive.')
    logs = {key: [] for key in ('t', 'x', 'y', 'u_raw', 'u_sat', 'u_rl')}
    total = sum_y2 = sum_u2 = 0.0
    transitions = 0
    valid, reason = True, 'ok'
    observed = 0
    for k in range(n_steps):
        y = np.asarray(plant.measure(), dtype=float).reshape(plant.ny)
        u_total, u_rl = control(y)
        tr = physical_step(plant, y, u_total, force_limit=force_limit, q_y=q_y, r_u=r_u,
                           u_cost=u_rl if residual else None)
        transitions += tr['transitions']
        observed += 1
        if capture:
            logs['t'].append(k * plant.dt)
            for key in ('x', 'y', 'u_raw', 'u_sat'):
                logs[key].append(tr[key])
            logs['u_rl'].append(np.asarray(u_rl, dtype=float).reshape(plant.nu))
        if after_step is not None:
            after_step(y, tr['u_sat'])
        total += tr['cost_k']
        sum_y2 += float(np.mean(np.square(y)))
        sum_u2 += float(np.mean(np.square(tr['u_sat'])))
        if not tr['valid']:
            valid, reason = False, tr['reason']
            break
    J_total = float(total) if valid else np.inf
    result = dict(valid=valid, reason=reason, J=J_total / observed, J_mean=J_total / observed,
                  J_total=J_total, N=observed, transitions=transitions, planned_steps=n_steps,
                  y_rms=float(np.sqrt(sum_y2 / observed)), u_rms=float(np.sqrt(sum_u2 / observed)))
    if capture:
        result.update({key: np.asarray(values, dtype=float) for key, values in logs.items()})
        result['state_names'] = getattr(plant, 'state_names', None)
        result['dt'] = float(plant.dt)
        if not residual:
            result.pop('u_rl')
    return result


class ClosedLoopSimulatorSOS:
    """Existing public simulator, now using the shared cost and physical step."""
    def __init__(self, plant, controller: ControllerSOS, force_limit=15.0,
                 control_sign=1.0, q_y=6.0, r_u=5e-3):
        self.plant, self.controller = plant, controller
        self.force_limit, self.control_sign = force_limit, np.asarray(control_sign)
        self.q_y, self.r_u = q_y, r_u
        if controller.ny != plant.ny or controller.nu != plant.nu:
            raise ValueError('Controller and plant dimensions must match.')
        controller.validate_dt(plant.dt)

    def sat(self, u_raw):
        return saturate(u_raw, self.force_limit)

    def run(self, t_final=10.0, *, n_steps=None, capture=True):
        self.plant.reset()
        self.controller.reset()
        def control(y):
            return self.control_sign * self.controller.step(y, dt=self.plant.dt), np.zeros(self.plant.nu)
        return run_rollout(self.plant, control,
                           steps_for_duration(t_final, self.plant.dt) if n_steps is None else int(n_steps),
                           force_limit=self.force_limit, q_y=self.q_y, r_u=self.r_u, capture=capture)
