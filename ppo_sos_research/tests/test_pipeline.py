from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.io import loadmat, savemat
from scipy.signal import sosfilt

from ppo_sos_pipeline.builders import make_env, make_stuart_landau_env_from_mat
from ppo_sos_pipeline.env import HybridSOSConfig, HybridSOSZPKPPOEnv
from ppo_sos_pipeline.controller_sos import ControllerSOS
from ppo_sos_pipeline.io_utils import save_results_csv, save_final_zpk_controller
from ppo_sos_pipeline.simulation import ClosedLoopSimulatorSOS
from ppo_sos_pipeline.policy_io import save_policy, load_policy, export_actor, actor_from_mat
from ppo_sos_pipeline.training import train_hybrid_ppo
from ppo_sos_pipeline.main import main
from ppo_sos_pipeline.utils import mu_tag

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / 'ppo_sos_pipeline' / 'Kinit_mu1.0.mat'


def make(mode='controller_lti', **kwargs):
    cfg = HybridSOSConfig(mode=mode, max_steps=7, dk_hold_steps=3, k_eval_steps=11, verbose_episode=False, **kwargs)
    return make_stuart_landau_env_from_mat(MAT, cfg)


def test_original_lti_regression():
    """Fixture was produced by executing the actual unmodified uploaded sources."""
    original = np.load(Path(__file__).parent / 'fixtures' / 'original_lti_trace.npz')
    env = make(force_limit=0.2)
    env.reset()
    for k, action in enumerate(original['actions']):
        obs, reward, term, trunc, _ = env.step(action)
        for name, actual in [('obs', obs), ('reward', reward), ('x', env.plant.state()),
                             ('theta_base', env.theta_base), ('theta_trial', env.current_theta_trial), ('J_base', env.J_base)]:
            np.testing.assert_allclose(actual, original[name][k], rtol=1e-12, atol=1e-12, err_msg=f'{name}, step {k}')
        if term or trunc:
            env.reset()
    np.testing.assert_allclose(env.history['J_probe'], original['J_probe'], rtol=1e-12, atol=1e-12)


def test_one_guide_rollout_and_exact_softmax_update(monkeypatch):
    env = make()
    original_evaluate = env.evaluate_theta
    evaluated = []
    def spy(theta, *args, **kwargs):
        evaluated.append(np.array(theta))
        return original_evaluate(theta, *args, **kwargs)
    monkeypatch.setattr(env, 'evaluate_theta', spy)
    env.reset()
    for k in range(14):
        _, _, term, trunc, _ = env.step(np.full(4, (-1)**k * .7))
        if term or trunc:
            env.reset()
    assert len(evaluated) == len(env.metrics['guidance']) == 2
    assert env.budget['training'] == 14
    assert env.budget['guidance'] == 22
    assert env.budget['initial_reference'] == 11
    for checked, guide in zip(evaluated, env.metrics['guidance']):
        np.testing.assert_array_equal(checked, guide['theta_probe'])
        w = np.exp(env.cfg.beta_softmax * (np.array(guide['block_rewards']) - max(guide['block_rewards'])))
        w /= w.sum()
        np.testing.assert_allclose(w, guide['weights'])
        delta = np.sum(np.array(guide['block_deltas']) * w[:, None], axis=0)
        np.testing.assert_allclose(guide['theta_probe'], env.mapper.project_theta(np.array(guide['theta_before']) + delta))
        np.testing.assert_allclose(guide['theta_update'], env.mapper.project_theta(np.array(guide['theta_before']) + guide['alpha'] * delta))
        assert guide['J_probe'] == guide['J_mean']
        assert guide['J_total'] == pytest.approx(guide['J_mean'] * guide['N'])
        assert guide['J_reference_new'] == pytest.approx((1-guide['alpha'])*guide['J_reference_old']+guide['alpha']*guide['J_probe'])


class CountingPlant:
    ny = nu = 1
    dt = .05
    state_names = ['position']
    def __init__(self):
        self.reset()
    def reset(self):
        self.x = np.array([1.])
        self.steps = 0
        self.time = 0.
    def state(self):
        return self.x.copy()
    def measure(self):
        return self.x.copy()
    def step(self, u):
        self.steps += 1
        self.time += self.dt
        self.x += self.dt * u


def static_controller(gain=3.):
    # Synthetic test controller: identical pole and zero -> static gain.
    return ControllerSOS([[.5, np.nan]], [[.5, np.nan]], [1], [1], gain, .05)


@pytest.mark.parametrize('mode', ['controller_lti', 'model_free', 'residual_rl'])
def test_one_dt_and_cost_channels(mode):
    cfg = HybridSOSConfig(mode=mode, max_steps=4, k_eval_steps=3, force_limit=.5, action_scale=2., verbose_episode=False)
    if mode == 'controller_lti':
        env = make_env(CountingPlant, cfg, MAT)
    else:
        env = HybridSOSZPKPPOEnv(None, None, CountingPlant, cfg, static_controller() if mode == 'residual_rl' else None)
    env.reset(seed=0)
    action = np.full(env.action_space.shape, .7)
    _, reward, term, trunc, info = env.step(action)
    assert env.plant.steps == 1
    assert env.plant.time == pytest.approx(cfg.dt)
    np.testing.assert_array_equal(info['x'], [1.])
    np.testing.assert_allclose(env.plant.state(), info['x'] + cfg.dt * info['u_sat'])
    assert np.max(np.abs(info['u_sat'])) <= cfg.force_limit
    penalized = info['u_rl'] if mode == 'residual_rl' else info['u_sat']
    cost = cfg.q_y + cfg.r_u * np.mean(penalized**2)
    assert info['cost_k'] == pytest.approx(cost)
    assert reward == pytest.approx(-cost / cfg.max_steps)
    if mode == 'residual_rl':
        np.testing.assert_allclose(info['u_base'], [3.])
        np.testing.assert_allclose(info['u_rl'], [1.4])
        np.testing.assert_allclose(info['u_sat'], [.5])
        assert info['cost_k'] != pytest.approx(cfg.q_y + cfg.r_u * (info['u_sat'][0]-3.)**2)
    assert not term and not trunc


@pytest.mark.parametrize('mode', ['model_free', 'residual_rl'])
@pytest.mark.parametrize('nf', [0, 2])
def test_memory_order_reset_and_fixed_base(mode, nf):
    cfg = HybridSOSConfig(mode=mode, nf=nf, max_steps=4, action_scale=2., force_limit=.5, verbose_episode=False)
    base = static_controller()
    before = base.P_sections.copy(), base.Z_sections.copy(), base.Kg
    env = HybridSOSZPKPPOEnv(None, None, CountingPlant, cfg, base if mode == 'residual_rl' else None)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1+2*nf,)
    np.testing.assert_array_equal(obs, [1.] + [0.]*(2*nf))
    obs, _, _, _, info1 = env.step([.7])
    if nf:
        np.testing.assert_allclose(obs, [env.plant.measure()[0], 1., 0., .5, 0.])
    y1 = env.plant.measure()[0]
    obs, _, _, _, info2 = env.step([-.5])
    if nf:
        np.testing.assert_allclose(obs, [env.plant.measure()[0], y1, 1., info2['u_sat'][0], .5])
    for _ in range(2):
        env.step([0.])
    assert not env.metrics['guidance']
    assert env.budget['guidance'] == 0
    for actual, expected in zip((base.P_sections, base.Z_sections, base.Kg), before):
        np.testing.assert_array_equal(actual, expected)
    if mode == 'residual_rl':
        np.testing.assert_array_equal(env.current_controller.P_sections, before[0])
        assert env.current_controller.Kg == before[2]
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, [1.] + [0.]*(2*nf))
    if mode == 'residual_rl':
        np.testing.assert_array_equal(env.current_controller.get_states(), [[0.]])


def test_mat_sampling_validation(tmp_path):
    ControllerSOS.from_mat(MAT, expected_dt=.05)
    with pytest.raises(ValueError, match='incompatible'):
        ControllerSOS.from_mat(MAT, expected_dt=.01)
    data = {k: v for k, v in loadmat(MAT).items() if not k.startswith('__') and k != 'Ts'}
    missing = tmp_path / 'missing_ts.mat'
    savemat(missing, data)
    with pytest.raises(ValueError, match='missing'):
        ControllerSOS.from_mat(missing)
    with pytest.raises(ValueError, match='incompatible'):
        static_controller().step([1.], dt=.1)


def test_mat_sos_ss_csv_and_exact_exported_controller(tmp_path):
    env = make()
    env.reset()
    for _ in range(7):
        env.step(np.full(4, .5))
    P, Z, Kg = env.get_current_controller()
    path = tmp_path / 'final.mat'
    save_final_zpk_controller(path, P, Z, Kg, env.theta_base, env.topology)
    data = loadmat(path)
    exported = ControllerSOS.from_mat(path, expected_dt=.05)
    np.testing.assert_array_equal(exported.P_sections, P)
    np.testing.assert_array_equal(exported.Z_sections, Z)
    assert exported.Kg == Kg
    y = np.linspace(-1, 1, 25)
    cascade = np.array([exported.step([v])[0] for v in y])
    np.testing.assert_allclose(cascade, sosfilt(data['K0_sos_norm'], y)*Kg, atol=1e-12)
    state = np.zeros((data['Ad_sos'].shape[0], 1))
    ss_values = []
    for v in y:
        ss_values.append(float((data['Cd_sos']@state + data['Dd_sos']*v).item()))
        state = data['Ad_sos']@state + data['Bd_sos']*v
    np.testing.assert_allclose(cascade, ss_values, atol=1e-12)
    result = ClosedLoopSimulatorSOS(env.plant_factory(), exported, env.cfg.force_limit).run(.5)
    expected = env.evaluate_current_controller(.5)
    for key in ('x', 'y', 'u_sat'):
        np.testing.assert_allclose(result[key], expected[key], rtol=0, atol=0)
    csv = tmp_path / 'rollout.csv'
    save_results_csv(result, csv)
    assert csv.read_text().splitlines()[0] == 't,x1,x2,y,u_sat'
    rows = np.loadtxt(csv, delimiter=',', skiprows=1)
    assert rows.shape == (10, 5)
    np.testing.assert_allclose(rows[:, 0], np.arange(10)*.05)
    plant = env.plant_factory()
    plant.reset()
    for row in rows:
        np.testing.assert_allclose(row[1:3], plant.state(), rtol=0, atol=0)
        np.testing.assert_allclose(row[3], plant.measure()[0], rtol=0, atol=0)
        plant.step([row[4]])


def test_failure_not_reported_as_improvement():
    class FailedPlant(CountingPlant):
        def step(self, u):
            super().step(u)
            self.x[:] = np.nan
    env = make_env(FailedPlant, HybridSOSConfig(mode='model_free', max_steps=4), None)
    env.reset()
    obs, reward, term, _, info = env.step([0.])
    assert term and np.all(np.isfinite(obs))
    assert reward == pytest.approx(-6/4 -100)
    assert not env.metrics['episodes'][0]['valid']
    assert np.isinf(env.metrics['episodes'][0]['J_total'])
    assert env.metrics['episodes'][0]['N'] == 1
    assert env.budget['training'] == 1


@pytest.mark.parametrize('mode', ['controller_lti', 'model_free', 'residual_rl'])
def test_ppo_native_actor_transfer_and_compatibility(tmp_path, mode):
    env = make(mode, nf=0 if mode == 'controller_lti' else 2, ppo_n_steps=16, batch_size=8)
    model = train_hybrid_ppo(env, 32, device='cpu', verbose=0)
    filename = tmp_path / 'policy.zip'
    save_policy(model, env, filename)
    restored = load_policy(filename, env)
    observations = np.asarray(env.observation_samples)
    np.testing.assert_array_equal(model.predict(observations, deterministic=True)[0], restored.predict(observations, deterministic=True)[0])
    original_opt, loaded_opt = model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict()
    assert original_opt['param_groups'] == loaded_opt['param_groups']
    assert original_opt['state'].keys() == loaded_opt['state'].keys()
    for key in original_opt['state']:
        for name in original_opt['state'][key]:
            np.testing.assert_array_equal(original_opt['state'][key][name].cpu(), loaded_opt['state'][key][name].cpu())
    actor = tmp_path / 'actor.mat'
    validation = export_actor(model, env, actor)
    assert validation['max_abs_action_error'] < 2e-6
    # Exercise actual output clipping independently of how small smoke-training means are.
    import torch
    with torch.no_grad():
        model.policy.action_net.bias.fill_(3.)
    export_actor(model, env, actor)
    np.testing.assert_array_equal(actor_from_mat(actor, observations[:3])[0], np.ones((3, env.action_space.shape[0])))
    model.policy.mlp_extractor.policy_net[1] = torch.nn.ReLU()
    with pytest.raises(ValueError, match='architecture'):
        save_policy(model, env, tmp_path / 'incompatible.zip')
    fresh = make(mode, nf=env.cfg.nf, ppo_n_steps=16, batch_size=8)
    continued = train_hybrid_ppo(fresh, 16, policy_path=filename, device='cpu', verbose=0)
    assert continued.num_timesteps == 16
    mismatched = make(mode, nf=env.cfg.nf, ppo_n_steps=16, batch_size=8, force_limit=1.)
    with pytest.raises(ValueError, match='mismatch'):
        load_policy(filename, mismatched)
    if mode != 'controller_lti':
        mismatched = make(mode, nf=0, ppo_n_steps=16, batch_size=8)
        with pytest.raises(ValueError):
            load_policy(filename, mismatched)


@pytest.mark.parametrize('mode,prefix', [('controller_lti','kfinal'), ('model_free','mf'), ('residual_rl','residual')])
def test_main_end_to_end(tmp_path, mode, prefix):
    result = main(['--mode', mode, '--mu-target', '1.0', '--total-timesteps', '17', '--ppo-n-steps', '16',
                   '--batch-size', '8', '--max-steps', '7', '--dk-hold-steps', '3', '--k-eval-steps', '11',
                   '--eval-final-time', '.5', '--nf', '0' if mode=='controller_lti' else '2',
                   '--output-dir', str(tmp_path), '--device', 'cpu', '--quiet'])
    env, model = result
    metadata = json.loads((tmp_path / f'experiment_{mode}_mu1.0.json').read_text())
    assert metadata['actual_ppo_timesteps'] == model.num_timesteps == 32
    assert metadata['simulation_budget']['training'] == 32
    assert metadata['simulation_budget']['final'] == 10
    assert (tmp_path / f'cost_timesteps_{mode}_mu1.0.png').is_file()
    assert (tmp_path / f'cost_episodes_{mode}_mu1.0.png').is_file()
    csv = tmp_path / f'sim_{prefix}_mu1.0.csv'
    assert csv.read_text().splitlines()[0] == 't,x1,x2,y,u_sat' + (',u_rl' if mode == 'residual_rl' else '')
    rows = np.loadtxt(csv, delimiter=',', skiprows=1)
    assert len(rows) == 10
    if mode == 'controller_lti':
        assert metadata['simulation_budget']['guidance'] == 4*11
        ctrl = ControllerSOS.from_mat(tmp_path / 'Kfinal_mu1.0.mat')
        np.testing.assert_array_equal(ctrl.P_sections, env.get_current_controller()[0])
        assert metadata['mu_design'] is None
        assert metadata['controller']['filename_mu_label'] == '1.0'
    else:
        assert metadata['simulation_budget']['guidance'] == 0


def test_fixed_seed_repeated_training_and_tags():
    models = []
    for _ in range(2):
        env = make('model_free', nf=1, ppo_n_steps=8, batch_size=4)
        models.append(train_hybrid_ppo(env, 16, device='cpu', verbose=0))
    obs = np.array([[1., 0., 0.]], dtype=np.float32)
    np.testing.assert_array_equal(models[0].predict(obs, deterministic=True)[0], models[1].predict(obs, deterministic=True)[0])
    assert mu_tag(1.) == '1.0'
    assert mu_tag(1.01) != mu_tag(1.001)
    assert float(mu_tag(np.nextafter(1., 2.))) == np.nextafter(1., 2.)
