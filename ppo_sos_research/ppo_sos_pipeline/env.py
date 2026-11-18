from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    class _FallbackEnv:
        metadata = {}

        def __init__(self, *args, **kwargs):
            self.np_random = np.random.default_rng()

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
            return None, {}

    class _FallbackBox:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
            self.shape = tuple(shape if shape is not None else self.low.shape)
            self.dtype = dtype
            self.seed(0)

        def seed(self, seed):
            self.np_random = np.random.default_rng(seed)

        def sample(self):
            return self.np_random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class _FallbackSpaces:
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

from dataclasses import dataclass
from copy import deepcopy
from .controller_sos import ControllerSOS
from .simulation import ClosedLoopSimulatorSOS, instant_cost, physical_step, run_rollout, steps_for_duration
from .zpk import ZPKActionMapper, ZPKTopology


@dataclass
class HybridSOSConfig:
    """Runtime contract. Experiment choices are explicit in main, no global config."""
    dt: float = 0.05
    Ts: float = 0.05
    max_steps: int = 50
    hybrid_timesteps: int = 70000
    ppo_n_steps: int = 1024
    batch_size: int = 64
    dk_hold_steps: int = 50
    lr_good: float = 0.3
    lr_bad: float = 0.01
    beta_softmax: float = 5.0
    k_eval_steps: int = 100
    q_y: float = 6.0
    r_u: float = 5e-3
    fail_penalty: float = 100.0
    force_limit: float = 15.0
    verbose_episode: bool = True
    mode: str = 'controller_lti'
    seed: int = 0
    nf: int = 0
    action_scale: float = 15.0
    # Legacy inert fields: retained for callers; the original update never used them.
    alpha_update: float = 0.3
    allow_worse_base_update: bool = False


class HybridSOSZPKPPOEnv(gym.Env if gym is not None else object):
    """Shared measurement-only environment; original LTI algorithm plus direct/residual control.

    Semantics:
        - one env.step(action) = one physical/solver step
        - controller_lti: delta_theta accepted every dk_hold_steps; other actions ignored
        - controller_lti: theta_base updated only at episode end
        - direct/residual: every action produces a physical u_rl; no guide

    Required plant interface:
        plant.reset()
        plant.measure() -> ndarray shape (ny,)
        plant.step(u)
        plant.ny, plant.nu, plant.dt
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: Optional[ZPKTopology],
        mapper: Optional[ZPKActionMapper],
        plant_factory: Callable[[], Any],
        config: Optional[HybridSOSConfig] = None,
        base_controller: Optional[ControllerSOS] = None,
    ):
        super().__init__()
        self.topology = topology
        self.mapper = mapper
        self.plant_factory = plant_factory
        self.cfg = config if config is not None else HybridSOSConfig()
        self.mode = self.cfg.mode
        if self.mode not in ('controller_lti', 'model_free', 'residual_rl'):
            raise ValueError(f'Unknown mode: {self.mode}')
        if not isinstance(self.cfg.nf, int) or self.cfg.nf < 0:
            raise ValueError('nf must be a nonnegative integer.')
        if self.mode == 'controller_lti' and self.cfg.nf != 0:
            raise ValueError('nf is only defined for model_free and residual_rl.')
        if not np.isfinite(self.cfg.dt) or self.cfg.dt <= 0:
            raise ValueError('dt must be finite and positive.')
        if self.cfg.k_eval_steps <= 0:
            raise ValueError('k_eval_steps must be positive.')
        for name in ('force_limit', 'action_scale'):
            if not np.isfinite(getattr(self.cfg, name)) or getattr(self.cfg, name) <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        for name in ('q_y', 'r_u', 'fail_penalty'):
            if not np.isfinite(getattr(self.cfg, name)) or getattr(self.cfg, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        self.base_controller = deepcopy(base_controller)
        self.metrics = {'steps': [], 'episodes': [], 'guidance': [], 'evaluations': []}
        self.budget = {'training': 0, 'initial_reference': 0, 'initial': 0, 'guidance': 0, 'final': 0, 'other': 0}
        self._episode_costs = []
        self._episode_rewards = []
        self.observation_samples = []

        if self.cfg.max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        if self.cfg.dk_hold_steps <= 0:
            raise ValueError("dk_hold_steps must be > 0.")

        probe_plant = self.plant_factory()
        self.ny = int(probe_plant.ny)
        self.nu = int(probe_plant.nu)
        if self.ny != 1 or self.nu != 1:
            raise ValueError("This ControllerSOS/RL version currently assumes SISO ny=1, nu=1.")

        if not np.isclose(probe_plant.dt, self.cfg.dt, rtol=1e-10, atol=1e-12):
            raise ValueError('plant.dt must equal config.dt.')
        if self.mode == 'controller_lti':
            if topology is None or mapper is None:
                raise ValueError('controller_lti requires a topology and mapper.')
            if not np.isclose(topology.Ts, self.cfg.dt, rtol=1e-10, atol=1e-12):
                raise ValueError('Controller Ts must equal dt.')
            if not np.isclose(self.cfg.Ts, self.cfg.dt, rtol=1e-10, atol=1e-12):
                raise ValueError('config.Ts must equal dt.')
            action_dim = self.topology.action_dim
            self.obs_dim = self.ny + action_dim
        else:
            action_dim = self.nu
            self.obs_dim = self.ny + self.cfg.nf * (self.ny + self.nu)
            if self.mode == 'residual_rl':
                if self.base_controller is None:
                    raise ValueError('residual_rl requires a fixed base_controller.')
                self.base_controller.validate_dt(self.cfg.dt)
                if (self.base_controller.ny, self.base_controller.nu) != (self.ny, self.nu):
                    raise ValueError('Base controller dimensions do not match plant.')
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(self.cfg.seed)
        self._reset_memory()

        if self.mode == 'controller_lti':
            self.theta_base = self.mapper.zpk_to_theta(
                self.topology.P_template, self.topology.Z_template, self.topology.Kg0)
            self.P_base, self.Z_base, self.Kg_base = self.mapper.theta_to_zpk(self.theta_base)
            metrics_base = self.evaluate_theta(self.theta_base, n_steps=self.cfg.k_eval_steps, role='initial_reference')
            self.J_base = float(metrics_base['J']) if metrics_base['valid'] else np.inf
            self.theta_score = -float(self.J_base) if np.isfinite(self.J_base) else -1e12
        else:
            self.theta_base = np.zeros(0)
            self.J_base, self.theta_score, self.Kg_base = np.nan, np.nan, np.nan
        self.alpha = 0.0

        self.episode_index = 0
        self.plant = self.plant_factory()
        self.current_controller: Optional[ControllerSOS] = None
        self.current_delta_theta = np.zeros(self.theta_base.size, dtype=np.float64)
        self.current_theta_trial = self.theta_base.copy()
        self.current_reward = 0.0
        self.current_cost = 0.0
        self.step_in_episode = 0
        self.global_physical_step = 0

        self.delta_accum: List[np.ndarray] = []
        self.reward_accum: List[float] = []
        self.history: Dict[str, List[Any]] = {
            "theta_score": [self.theta_score],
            "J_base": [self.J_base],
            "J_probe": [],
            "accepted_update": [],
            "candidate_score": [],
            "lr": [],
            "alpha": [],
            "Kg_base": [self.Kg_base],
            "mean_reward": [],
            "best_reward": [],
            "delta_norm": [],
        }

    def _make_controller_from_theta(self, theta: np.ndarray, reset_states: bool = True) -> ControllerSOS:
        P, Z, Kg = self.mapper.theta_to_zpk(theta)
        return ControllerSOS(
            P_sections=P,
            Z_sections=Z,
            pole_order=self.topology.pole_order,
            zero_order=self.topology.zero_order,
            Kg=Kg,
            Ts=self.topology.Ts,
            x0=None,
            reset_states=reset_states,
        )

    def _get_measurement(self) -> np.ndarray:
        return np.asarray(self.plant.measure(), dtype=float).reshape(self.ny)

    def _get_obs(self) -> np.ndarray:
        y = self._get_measurement()
        if self.mode == 'controller_lti':
            obs = np.concatenate([y.reshape(-1), self.mapper.normalize_theta(self.theta_base)])
        else:
            # Newest first, all measurement lags followed by all applied-input lags.
            obs = np.concatenate([y, self.y_history.ravel(), self.u_history.ravel()])
        return obs.astype(np.float32)

    def _reset_memory(self):
        self.y_history = np.zeros((self.cfg.nf, self.ny))
        self.u_history = np.zeros((self.cfg.nf, self.nu))

    def _push_history(self, y, u_sat):
        if self.mode != 'controller_lti' and self.cfg.nf:
            self.y_history[1:] = self.y_history[:-1].copy()
            self.u_history[1:] = self.u_history[:-1].copy()
            self.y_history[0], self.u_history[0] = y, u_sat

    def reset(self, seed=None, options=None):
        # Seed is fixed for this experiment; never increment it per episode.
        if seed is not None and seed != self.cfg.seed:
            raise ValueError('reset seed must match the experiment seed.')
        super().reset(seed=self.cfg.seed if seed is not None else None)
        self._reset_memory()
        self._episode_costs, self._episode_rewards = [], []
        self.plant = self.plant_factory()
        self.plant.reset()

        self.current_controller = deepcopy(self.base_controller) if self.mode == 'residual_rl' else None
        if self.current_controller is not None:
            self.current_controller.reset()
        self.current_delta_theta = np.zeros(self.theta_base.size, dtype=np.float64)
        self.current_theta_trial = self.theta_base.copy()
        self.current_reward = 0.0
        self.current_cost = 0.0
        self.step_in_episode = 0
        self.delta_accum = []
        self.reward_accum = []
        return self._get_obs(), {}

    def _instant_cost(self, y: np.ndarray, u_sat: np.ndarray) -> float:
        return instant_cost(y, u_sat, self.cfg.q_y, self.cfg.r_u)

    def _failure(self, y: np.ndarray, u_sat: np.ndarray) -> Tuple[bool, str]:
        
        y = np.asarray(y, dtype=float).reshape(-1)
        u_sat = np.asarray(u_sat, dtype=float).reshape(-1)
        
        if not np.all(np.isfinite(y)):
            return True, "nonfinite_y"
        
        if not np.all(np.isfinite(u_sat)):
            return True, "nonfinite_u"
        
        return False, "ok"

    def _accept_new_effective_action(self, action_from_ppo: np.ndarray) -> None:
        candidate = self.mapper.build_candidate(self.theta_base, action_from_ppo)
        self.current_delta_theta = candidate["delta_theta"].copy()
        self.current_theta_trial = candidate["theta_candidate"].copy()
        self.current_controller = ControllerSOS(
            P_sections=candidate["P_sections"],
            Z_sections=candidate["Z_sections"],
            pole_order=self.topology.pole_order,
            zero_order=self.topology.zero_order,
            Kg=candidate["Kg"],
            Ts=self.topology.Ts,
            x0=None,
            reset_states=True,
        )

    def _control(self, action, y):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError('Action must be finite and match action_space.')
        new_action = True
        if self.mode == 'controller_lti':
            new_action = self.step_in_episode % self.cfg.dk_hold_steps == 0 or self.current_controller is None
            if new_action:
                self._accept_new_effective_action(action)
            u_base = self.current_controller.step(y, dt=self.cfg.dt)
            u_rl = np.zeros(self.nu)
        else:
            u_base = (self.current_controller.step(y, dt=self.cfg.dt)
                      if self.mode == 'residual_rl' else np.zeros(self.nu))
            u_rl = self.cfg.action_scale * np.clip(action, -1.0, 1.0)
        # The original loop has no extra minus sign: its feedback sign lives in Kg.
        return u_base, u_rl, bool(new_action)

    def step(self, action_from_ppo):
        if len(self.observation_samples) < 256:
            self.observation_samples.append(self._get_obs().copy())
        y = self._get_measurement()
        u_base, u_rl, new_action = self._control(action_from_ppo, y)
        tr = physical_step(self.plant, y, u_base + u_rl, force_limit=self.cfg.force_limit,
                           q_y=self.cfg.q_y, r_u=self.cfg.r_u,
                           u_cost=u_rl if self.mode == 'residual_rl' else None)
        cost_k = tr['cost_k']
        # For valid trajectories, exactly the original reward normalization.
        reward = -float(cost_k) / self.cfg.max_steps if np.isfinite(cost_k) else 0.0
        terminated = not tr['valid']
        if terminated:
            reward -= self.cfg.fail_penalty
        self.current_reward, self.current_cost = float(reward), float(cost_k)
        if self.mode == 'controller_lti':
            self.delta_accum.append(self.current_delta_theta.copy())
            self.reward_accum.append(float(reward))
        self._push_history(y, tr['u_sat'])
        self.step_in_episode += 1
        self.global_physical_step += 1
        self.budget['training'] += tr['transitions']
        self._episode_costs.append(float(cost_k))
        self._episode_rewards.append(float(reward))
        self.metrics['steps'].append(dict(timestep=self.global_physical_step, cost_k=float(cost_k), reward=float(reward), valid=tr['valid']))
        truncated = self.step_in_episode >= self.cfg.max_steps
        update_info = {}
        if terminated or truncated:
            self.metrics['episodes'].append(self.episode_metrics(valid=not terminated, complete=True))
            if self.mode == 'controller_lti':
                update_info = self._update_theta_base()
                update_info['guide_reason'] = update_info['reason']
            self.episode_index += 1
        obs = self._get_obs()
        if terminated:
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        info = {**update_info, **tr, 'reward': float(reward), 'u_base': u_base, 'u_rl': u_rl,
                'new_effective_action_step': new_action}
        return obs, float(reward), terminated, truncated, info

    def episode_metrics(self, valid=True, complete=True):
        N = len(self._episode_costs)
        total = float(np.sum(self._episode_costs)) if valid else np.inf
        return dict(episode=self.episode_index + 1, timestep=self.global_physical_step, N=N,
                    J_total=total, J_mean=total / max(N, 1), valid=valid, complete=complete,
                    reward_total=float(np.sum(self._episode_rewards)), planned_steps=self.cfg.max_steps)

    def _record_evaluation(self, result, role):
        self.budget[role] += int(result['transitions'])
        keys = ('valid', 'reason', 'J_total', 'J_mean', 'N', 'transitions', 'planned_steps')
        self.metrics['evaluations'].append(dict(role=role, **{k: result[k] for k in keys}))
        return result

    def evaluate_theta(self, theta, n_steps=None, role='other', capture=False):
        simulator = ClosedLoopSimulatorSOS(self.plant_factory(), self._make_controller_from_theta(theta),
                                            self.cfg.force_limit, q_y=self.cfg.q_y, r_u=self.cfg.r_u)
        result = simulator.run(n_steps=self.cfg.k_eval_steps if n_steps is None else n_steps, capture=capture)
        return self._record_evaluation(result, role)

    def evaluate_policy(self, model, t_final=10.0, role='final'):
        if self.mode == 'controller_lti':
            raise ValueError('LTI deployment evaluates theta_base, not the PPO actor.')
        # Fresh environment for identical observation/history semantics; no training or guide.
        evaluation = HybridSOSZPKPPOEnv(None, None, self.plant_factory, self.cfg, self.base_controller)
        evaluation.reset(seed=self.cfg.seed)
        def control(y):
            action, _ = model.predict(evaluation._get_obs(), deterministic=True)
            u_base, u_rl, _ = evaluation._control(action, y)
            return u_base + u_rl, u_rl
        result = run_rollout(evaluation.plant, control, steps_for_duration(t_final, self.cfg.dt),
                             force_limit=self.cfg.force_limit, q_y=self.cfg.q_y, r_u=self.cfg.r_u,
                             residual=self.mode == 'residual_rl', after_step=evaluation._push_history)
        return self._record_evaluation(result, role)

    def _score_theta(self, theta: np.ndarray, n_steps: Optional[int] = None) -> float:
        metrics = self.evaluate_theta(theta, n_steps=n_steps)
        if metrics["valid"]:
            return -float(metrics["J"])
        return float(-1e12)

    def _update_theta_base(self) -> Dict[str, Any]:
        """
        End-of-episode hybrid update.
    
        Final semantics:
            1) During the physical episode, PPO proposes actions.
            2) Effective delta_theta is held during dk_hold_steps.
            3) At episode end, block rewards are used to compute delta_mean.
            4) One additional probe rollout evaluates:
    
                   theta_probe = theta_base + delta_mean
    
            5) If the probe is valid:
                  - if J_probe < J_base: alpha = lr_good = 0.3
                  - else:                alpha = lr_bad  = 0.01
    
               Then always update:
    
                   theta_base <- theta_base + alpha * delta_mean
                   J_base     <- (1 - alpha) * J_base + alpha * J_probe
    
            6) If the probe is invalid:
                  theta_base and J_base remain unchanged.
        """
        if len(self.delta_accum) == 0:
            return {
                "updated": False,
                "improved": False,
                "reason": "empty_accum",
                "J_probe": np.inf,
                "J_base": float(self.J_base),
                "alpha": 0.0,
                "lr": 0.0,
                "delta_norm": 0.0,
            }
    
        # ============================================================
        # 1) Episode data
        # ============================================================
        deltas = np.asarray(self.delta_accum, dtype=np.float64)
        rewards = np.asarray(self.reward_accum, dtype=np.float64).reshape(-1)
    
        if deltas.ndim == 1:
            deltas = deltas.reshape(1, -1)
    
        rewards_safe = np.where(np.isfinite(rewards), rewards, -1e12)
    
        n = len(rewards_safe)
        block_size = int(self.cfg.dk_hold_steps)
        n_blocks = int(np.ceil(n / block_size))
    
        block_deltas = []
        block_rewards = []
    
        for b in range(n_blocks):
            i0 = b * block_size
            i1 = min((b + 1) * block_size, n)
    
            if i0 >= i1:
                continue
    
            # One effective delta_theta per hold block.
            block_deltas.append(np.asarray(deltas[i0], dtype=np.float64).reshape(-1))
    
            # Total reward over this block.
            block_rewards.append(float(np.sum(rewards_safe[i0:i1])))
    
        block_deltas = np.asarray(block_deltas, dtype=np.float64)
        block_rewards = np.asarray(block_rewards, dtype=np.float64).reshape(-1)
    
        if block_deltas.size == 0:
            return {
                "updated": False,
                "improved": False,
                "reason": "empty_blocks",
                "J_probe": np.inf,
                "J_base": float(self.J_base),
                "alpha": 0.0,
                "lr": 0.0,
                "delta_norm": 0.0,
            }
    
        # ============================================================
        # 2) Softmax over block rewards -> delta_mean
        # ============================================================
        beta = float(self.cfg.beta_softmax)
    
        block_rewards_safe = np.where(np.isfinite(block_rewards), block_rewards, -1e12)
        rmax = float(np.max(block_rewards_safe))
    
        weights = np.exp(beta * (block_rewards_safe - rmax))
        weights = weights / max(float(np.sum(weights)), 1e-12)
    
        delta_mean = np.sum(block_deltas * weights[:, None], axis=0)
        delta_mean = np.asarray(delta_mean, dtype=np.float64).reshape(-1)
    
        delta_norm = float(np.linalg.norm(delta_mean))
    
        # ============================================================
        # 3) One probe rollout only:
        #       theta_probe = theta_base + delta_mean
        # ============================================================
        theta_before = self.theta_base.copy()
        theta_probe = self.mapper.project_theta(self.theta_base + delta_mean)
    
        metrics_probe = self.evaluate_theta(
            theta_probe,
            n_steps=self.cfg.k_eval_steps,
            role='guidance',
        )
    
        J_probe = float(metrics_probe["J"]) if metrics_probe["valid"] else np.inf
        J_base_old = float(self.J_base)
    
        updated = False
        improved = False
        alpha = 0.0
        lr = 0.0
        reason = metrics_probe.get("reason", "unknown")
    
        # ============================================================
        # 4) If valid: always update, with alpha depending on improvement
        # ============================================================
        if metrics_probe["valid"]:
            improved = bool(np.isfinite(J_probe) and J_probe < J_base_old)
    
            if improved:
                alpha = float(self.cfg.lr_good)   # normally 0.3
                reason = "valid_improved"
            else:
                alpha = float(self.cfg.lr_bad)    # normally 0.01
                reason = "valid_not_improved"
    
            lr = alpha
    
            theta_new = self.mapper.project_theta(
                self.theta_base + alpha * delta_mean
            )
    
            P_new, Z_new, Kg_new = self.mapper.theta_to_zpk(theta_new)
    
            self.theta_base = theta_new.copy()
            self.P_base = P_new.copy()
            self.Z_base = Z_new.copy()
            self.Kg_base = float(Kg_new)
    
            # IMPORTANT:
            # No second evaluation here.
            # J_base is a heuristic reference, NOT the measured cost of theta_new.
            self.J_base = (
                (1.0 - alpha) * J_base_old
                + alpha * J_probe
            )
    
            self.theta_score = -float(self.J_base)
            self.alpha = float(alpha)
            updated = True
    
        else:
            # Invalid candidate: keep base unchanged.
            alpha = 0.0
            lr = 0.0
            updated = False
            improved = False
            self.alpha = 0.0
            reason = metrics_probe.get("reason", "invalid_probe")
    
        # ============================================================
        # 5) Logging/history
        # ============================================================
        mean_reward = float(np.mean(rewards_safe)) if rewards_safe.size else 0.0
        best_reward = float(np.max(block_rewards_safe)) if block_rewards_safe.size else 0.0
    
        self.metrics['guidance'].append(dict(
            timestep=self.global_physical_step, episode=self.episode_index + 1,
            theta_before=theta_before.tolist(), theta_probe=theta_probe.tolist(),
            theta_update=self.theta_base.tolist(), J_probe=J_probe,
            J_total=metrics_probe['J_total'], J_mean=metrics_probe['J_mean'],
            N=metrics_probe['N'], planned_steps=self.cfg.k_eval_steps, valid=metrics_probe['valid'],
            weights=weights.tolist(), block_deltas=block_deltas.tolist(), block_rewards=block_rewards.tolist(),
            alpha=alpha, reason=reason, J_reference_old=J_base_old, J_reference_new=float(self.J_base)))
        self.history["theta_score"].append(float(self.theta_score))
        self.history["J_base"].append(float(self.J_base))
        self.history["J_probe"].append(float(J_probe))
        self.history["accepted_update"].append(bool(updated))
        self.history["candidate_score"].append(
            float(-J_probe) if np.isfinite(J_probe) else -1e12
        )
        self.history["lr"].append(float(lr))
        self.history["alpha"].append(float(alpha))
        self.history["Kg_base"].append(float(self.Kg_base))
        self.history["mean_reward"].append(mean_reward)
        self.history["best_reward"].append(best_reward)
        self.history["delta_norm"].append(delta_norm)
    
        if self.cfg.verbose_episode:
            tag = "GOOD" if improved else ("BAD" if updated else "INVALID")
            print(
                f"episode={self.episode_index} [{tag}] "
                f"J_probe={J_probe:.6e} "
                f"J_base_old={J_base_old:.6e} "
                f"J_base_new={self.J_base:.6e} "
                f"updated={updated} improved={improved} "
                f"alpha={alpha:.3g} |delta|={delta_norm:.3e} "
                f"reason={reason}"
            )
    
        return {
            "updated": bool(updated),
            "accepted_update": bool(updated),
            "improved": bool(improved),
            "J_probe": float(J_probe),
            "J_base": float(self.J_base),
            "J_base_old": float(J_base_old),
            "alpha": float(alpha),
            "lr": float(lr),
            "delta_norm": float(delta_norm),
            "reason": reason,
        }

    def get_current_controller(self):
        return self.mapper.theta_to_zpk(self.theta_base)

    def evaluate_current_controller(self, t_final=10.0, role='other'):
        return self.evaluate_theta(self.theta_base, steps_for_duration(t_final, self.cfg.dt), role=role, capture=True)


# Backward-compatible public names; no wrapper environment changes the spaces.
StuartLandauHybridSOSZPKPPOEnv = HybridSOSZPKPPOEnv
