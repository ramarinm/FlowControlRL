"""Native SB3 persistence and deterministic actor interchange (not a second PPO)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat


def inference_contract(env):
    plant = env.plant_factory()
    contract = dict(version=1, mode=env.mode, dt=env.cfg.dt, nf=env.cfg.nf,
                    observation_shape=list(env.observation_space.shape),
                    action_shape=list(env.action_space.shape), observation_dtype='float32',
                    observation_normalization='none; theta_base normalized only in controller_lti',
                    action_low=env.action_space.low.tolist(), action_high=env.action_space.high.tolist(),
                    force_limit=env.cfg.force_limit,
                    action_scale=None if env.mode == 'controller_lti' else env.cfg.action_scale,
                    observation_order='[y_k, normalize(theta_base)]' if env.mode == 'controller_lti' else
                                      '[y_k, y_(k-1), ..., y_(k-nf), u_sat_(k-1), ..., u_sat_(k-nf)]',
                    history_initialization='zeros; reset every episode; newest first; physical units',
                    control_sign='no external minus sign; base sign in Kg',
                    feature_extractor='FlattenExtractor', actor_hidden=[64, 64], critic_hidden=[64, 64],
                    activation='Tanh', output_activation='linear', squash_output=False,
                    deterministic_action='clip(actor_mean, action_low, action_high)',
                    ny=env.ny, nu=env.nu)
    if hasattr(plant, 'Csens'):
        contract['measurement_matrix'] = np.asarray(plant.Csens).tolist()
    if env.mode == 'controller_lti':
        contract['theta_names'] = env.topology.param_names()
        contract['theta_blocks'] = [b.__dict__ for b in env.topology.param_blocks]
        contract['gain_sign'] = env.topology.Kg_sign
        contract['mapper'] = {k: v for k, v in vars(env.mapper).items() if k != 'topology'}
        contract['dk_hold_steps'] = env.cfg.dk_hold_steps
    return contract


def _actor_layers(model):
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import FlattenExtractor
    policy = model.policy
    if not isinstance(policy.features_extractor, FlattenExtractor) or policy.squash_output or model.get_vec_normalize_env() is not None:
        raise ValueError('Unsupported extractor, normalization wrapper, or squashing; export cannot silently omit preprocessing.')
    if policy.use_sde:
        raise ValueError('This pipeline expects the original diagonal Gaussian PPO policy.')
    modules = list(policy.mlp_extractor.policy_net) + [policy.action_net]
    if [type(m) for m in modules] != [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]:
        raise ValueError('Actor architecture must match the original 64/64 Tanh MlpPolicy.')
    if [modules[0].out_features, modules[2].out_features] != [64, 64]:
        raise ValueError('Incompatible hidden-layer widths.')
    critic = list(policy.mlp_extractor.value_net)
    if [type(m) for m in critic] != [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh] or [critic[0].out_features, critic[2].out_features] != [64, 64]:
        raise ValueError('Incompatible critic architecture.')
    return [modules[0], modules[2], modules[4]]


def load_policy(filename, env, device='cpu'):
    """Load for inference or transfer training. No resizing/remapping of weights.

    Physical mu/x0/cost may change intentionally in main. Observation channels,
    dt, nf, action semantics/scales, topology and architecture must remain compatible.
    """
    from stable_baselines3 import PPO
    model = PPO.load(str(filename), env=env, device=device)
    expected = inference_contract(env)
    actual = getattr(model, 'pipeline_contract', None)
    if actual != expected:
        keys = sorted(k for k in set(expected) | set(actual or {}) if expected.get(k) != (actual or {}).get(k))
        raise ValueError(f'Policy/preprocessing mismatch: {keys}. No automatic weight adaptation.')
    _actor_layers(model)
    return model


def save_policy(model, env, filename):
    _actor_layers(model)
    model.pipeline_contract = inference_contract(env)
    # SB3 ZIP already stores actor/critic weights, optimizer state, policy kwargs,
    # spaces, counters and hyperparameters. Do not duplicate them in another .pt.
    model.save(str(filename))


def actor_from_mat(filename, observations):
    """Independent NumPy replay of the actual MAT payload, batch as rows."""
    data = loadmat(filename)
    z = np.asarray(observations, dtype=np.float32)
    if z.ndim == 1:
        z = z[None, :]
    n_layers = int(data['n_layers'].item())
    if z.shape[1] != data['W1'].shape[1]:
        raise ValueError('Observation dimension mismatch.')
    for i in range(1, n_layers + 1):
        z = z @ data[f'W{i}'].astype(np.float32).T + data[f'b{i}'].reshape(1, -1).astype(np.float32)
        if i < n_layers:
            z = np.tanh(z)
    action = np.clip(z, data['action_low'].reshape(1, -1), data['action_high'].reshape(1, -1))
    scale = float(data['physical_action_scale'].item())
    return action, action * scale


def export_actor(model, env, filename, experiment_metadata=None):
    """Export mean network + clipping + history/scaling metadata and check numerically."""
    layers = _actor_layers(model)
    contract = inference_contract(env)
    payload = dict(n_layers=np.array([[len(layers)]]),
                   action_low=env.action_space.low.reshape(-1, 1), action_high=env.action_space.high.reshape(-1, 1),
                   physical_action_scale=np.array([[1.0 if env.mode == 'controller_lti' else env.cfg.action_scale]]),
                   activations=np.asarray(['tanh', 'tanh', 'linear'], dtype=object),
                   matrix_orientation='W has shape [out,in]; column inference W*z+b',
                   computation_dtype='single / float32',
                   contract_json=json.dumps(contract, ensure_ascii=False),
                   experiment_json=json.dumps(experiment_metadata or {}, ensure_ascii=False),
                   physical_output_meaning='not a physical input; ZPK delta action' if env.mode == 'controller_lti' else 'u_rl before final saturation')
    for i, layer in enumerate(layers, start=1):
        payload[f'W{i}'] = layer.weight.detach().cpu().numpy()
        payload[f'b{i}'] = layer.bias.detach().cpu().numpy().reshape(-1, 1)
    samples = np.asarray(env.observation_samples, dtype=np.float32)
    if samples.size:
        indices = np.linspace(0, len(samples) - 1, min(16, len(samples)), dtype=int)
        observations = samples[indices]
    else:
        observations = env._get_obs()[None, :]
    observations = np.vstack([observations, np.zeros((1, env.obs_dim), dtype=np.float32)])
    expected, _ = model.predict(observations, deterministic=True)
    payload['validation_observations_rows'] = observations
    payload['validation_actions_rows'] = expected
    savemat(filename, payload)
    actual, physical = actor_from_mat(filename, observations)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    scale = 1.0 if env.mode == 'controller_lti' else env.cfg.action_scale
    np.testing.assert_allclose(physical, expected * scale, rtol=2e-5, atol=2e-6 * max(1, scale))
    return dict(checked_observations=len(observations), max_abs_action_error=float(np.max(np.abs(actual - expected))),
                max_abs_physical_error=float(np.max(np.abs(physical - expected * scale))),
                method='SB3 deterministic predict vs independent NumPy reading saved MAT', matlab_executed=False)
