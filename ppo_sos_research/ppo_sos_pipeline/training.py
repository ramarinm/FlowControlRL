from __future__ import annotations

import numpy as np
from .policy_io import load_policy
from .utils import rms


def run_debug_rollout(env, n_steps=None):
    n_steps = env.cfg.max_steps if n_steps is None else int(n_steps)
    obs, _ = env.reset(seed=env.cfg.seed)
    print('Debug rollout:', env.mode, 'observation', obs.shape, 'action', env.action_space.shape)
    for k in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if k < 5 or terminated or truncated:
            print(f'k={k} reward={reward:.6g} new_action={info["new_effective_action_step"]} reason={info["reason"]}')
        if terminated or truncated:
            break


def train_hybrid_ppo(env, total_timesteps=70000, *, policy_path=None, learning_rate=3e-4,
                     ent_coef=0.01, device='auto', verbose=1):
    """Same SB3 PPO and defaults in all modes. ZIP loading retains optimizer state.

    A loaded policy initializes a NEW training run on a fresh environment, not a
    bit-exact resume of an interrupted rollout. SB3 collects full n_steps batches.
    """
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    if total_timesteps <= 0:
        raise ValueError('total_timesteps must be positive.')
    if env.cfg.ppo_n_steps <= 1 or env.cfg.batch_size <= 1 or env.cfg.batch_size > env.cfg.ppo_n_steps:
        raise ValueError('Need 1 < batch_size <= ppo_n_steps for the single environment.')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # One constant seed per run. Random generators advance normally; no reseeding
    # each episode or each PPO update (which would repeat the same action noise).
    set_random_seed(env.cfg.seed, using_cuda=str(device).startswith('cuda'))
    if policy_path is None:
        model = PPO('MlpPolicy', env, n_steps=env.cfg.ppo_n_steps, batch_size=env.cfg.batch_size,
                    learning_rate=learning_rate, ent_coef=ent_coef, verbose=verbose, device=device,
                    seed=env.cfg.seed)
    else:
        model = load_policy(policy_path, env, device=device)
        # These values determine rollout buffer and optimizer schedules. Reject
        # implicit overrides; native checkpoint settings remain authoritative.
        expected = (env.cfg.ppo_n_steps, env.cfg.batch_size, learning_rate, ent_coef)
        actual = (model.n_steps, model.batch_size, model.learning_rate, model.ent_coef)
        if actual != expected:
            raise ValueError(f'Loaded PPO hyperparameters {actual} differ from requested {expected}.')
        model.seed = env.cfg.seed
        model.set_random_seed(env.cfg.seed)
        model.verbose = verbose
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True)
    if 0 < env.step_in_episode < env.cfg.max_steps and env._episode_costs:
        env.metrics['partial_episode'] = env.episode_metrics(complete=False)
    return model


def evaluate_closed_loop(env, label, t_final, role='other'):
    results = env.evaluate_current_controller(t_final=t_final, role=role)
    print(f'{label}: N={results["N"]}, J_total={results["J_total"]:.8g}, J_mean={results["J_mean"]:.8g}, valid={results["valid"]}')
    return results


def ppo_parameters(model):
    """Effective settings including the originally implicit SB3 defaults."""
    keys = ('n_steps', 'batch_size', 'learning_rate', 'ent_coef', 'n_epochs', 'gamma', 'gae_lambda',
            'vf_coef', 'max_grad_norm', 'normalize_advantage', 'target_kl', 'use_sde', 'seed')
    result = {k: getattr(model, k) for k in keys}
    result.update(clip_range=float(model.clip_range(1.0)), clip_range_vf=None if model.clip_range_vf is None else float(model.clip_range_vf(1.0)),
                  device=str(model.device), optimizer=type(model.policy.optimizer).__name__,
                  optimizer_defaults={k: v for k, v in model.policy.optimizer.defaults.items() if isinstance(v, (str, float, int, bool, tuple)) or v is None},
                  policy_class=type(model.policy).__name__, feature_extractor=type(model.policy.features_extractor).__name__,
                  net_arch=model.policy.net_arch, activation=type(model.policy.activation_fn()).__name__)
    return result
