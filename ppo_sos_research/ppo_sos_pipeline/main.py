from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import platform
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from scipy.io import loadmat

from .builders import make_env
from .controller_sos import ControllerSOS
from .env import HybridSOSConfig
from .io_utils import save_final_zpk_controller, save_results_csv, write_json, plot_training_metrics
from .plants import StuartLandauPlant
from .policy_io import save_policy, load_policy, export_actor
from .simulation import ClosedLoopSimulatorSOS, steps_for_duration
from .training import evaluate_closed_loop, run_debug_rollout, train_hybrid_ppo, ppo_parameters
from .utils import mu_tag


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='PPO research pipeline: LTI optimization, direct or residual control.')
    # 1. Modality and physical dynamics. The supplied archive contains Stuart-Landau only.
    parser.add_argument('--mode', choices=['controller_lti', 'model_free', 'residual_rl'], default='controller_lti')
    parser.add_argument('--run', choices=['train', 'debug'], default='train')
    parser.add_argument('--mu-target', type=float, default=2.4)
    parser.add_argument('--mu-design', type=float, default=None, help='Known design point only; never inferred from filename.')
    parser.add_argument('--omega', type=float, default=2.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--x0', nargs=2, type=float, default=[0.6, 0.1])
    parser.add_argument('--csens', nargs=2, type=float, default=[0.0, 1.0])
    # 2. Time: one environment step = one dt, even with an implicit integrator.
    parser.add_argument('--dt', type=float, default=0.05)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--episode-duration', type=float, default=None, help='Seconds; rounded up to a whole dt.')
    duration.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--eval-final-time', type=float, default=10.0, help='Initial and final rollout duration (seconds).')
    # 3. Initial/fixed controller. Default resolves the file actually included in the ZIP.
    parser.add_argument('--mat-file', type=Path, default=Path(__file__).with_name('Kinit_mu1.0.mat'))
    # 4. PPO, guidance and shared cost. Defaults preserve the uploaded implementation.
    parser.add_argument('--total-timesteps', type=int, default=70000)
    parser.add_argument('--ppo-n-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--dk-hold-steps', type=int, default=50)
    parser.add_argument('--k-eval-steps', type=int, default=100)
    parser.add_argument('--lr-good', type=float, default=0.3)
    parser.add_argument('--lr-bad', type=float, default=0.01)
    parser.add_argument('--beta-softmax', type=float, default=5.0)
    parser.add_argument('--q-y', type=float, default=6.0)
    parser.add_argument('--r-u', type=float, default=5e-3)
    parser.add_argument('--fail-penalty', type=float, default=100.0)
    parser.add_argument('--force-limit', type=float, default=15.0)
    # 5. Fixed seed, memory, physical action scaling and optional transfer learning.
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nf', type=int, default=0)
    parser.add_argument('--action-scale', type=float, default=15.0)
    parser.add_argument('--policy-path', type=Path, default=None, help='Native PPO ZIP to initialize a new training run.')
    parser.add_argument('--device', default='auto')
    # 6. Outputs. Use a distinct folder per experiment to retain previous runs.
    parser.add_argument('--output-dir', type=Path, default=Path.cwd())
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    args = parse_args(argv)
    max_steps = (steps_for_duration(args.episode_duration, args.dt) if args.episode_duration is not None
                 else (50 if args.max_steps is None else args.max_steps))
    steps_for_duration(args.eval_final_time, args.dt)  # Validate before training.
    config = HybridSOSConfig(
        mode=args.mode, dt=args.dt, Ts=args.dt, max_steps=max_steps,
        hybrid_timesteps=args.total_timesteps, ppo_n_steps=args.ppo_n_steps, batch_size=args.batch_size,
        dk_hold_steps=args.dk_hold_steps, k_eval_steps=args.k_eval_steps,
        lr_good=args.lr_good, lr_bad=args.lr_bad, beta_softmax=args.beta_softmax,
        q_y=args.q_y, r_u=args.r_u, fail_penalty=args.fail_penalty, force_limit=args.force_limit,
        seed=args.seed, nf=args.nf, action_scale=args.action_scale, verbose_episode=not args.quiet)

    # Replace this factory to use another dynamics with reset/measure/step, ny/nu/dt.
    # state() and state_names are optional, used only for CSV export. No target linearization.
    physical_parameters = dict(mu=args.mu_target, omega=args.omega, beta=args.beta,
                               xp0=args.x0, Csens=[args.csens])
    def plant_factory():
        return StuartLandauPlant(dt=args.dt, Ts=args.dt, **physical_parameters)

    controller_path = args.mat_file if args.mode != 'model_free' else None
    if controller_path is not None and not controller_path.is_file():
        raise FileNotFoundError(f'Required controller file missing: {controller_path}')
    env = make_env(plant_factory, config, controller_path)
    if args.run == 'debug':
        run_debug_rollout(env)
        return env

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tag = f'{args.mode}_mu{mu_tag(args.mu_target)}'
    controller_metadata = None
    mu_design = args.mu_design
    if controller_path is not None:
        data = loadmat(controller_path)
        file_design = float(np.asarray(data['mu_design']).item()) if 'mu_design' in data else None
        if mu_design is not None and file_design is not None and mu_design != file_design:
            raise ValueError('Explicit mu_design conflicts with controller metadata.')
        mu_design = file_design if mu_design is None else mu_design
        match = re.search(r'mu([-+0-9.eE]+)\.mat$', controller_path.name)
        source_copy = output / f'controller_source_{tag}.mat'
        if controller_path.resolve() != source_copy:
            shutil.copy2(controller_path, source_copy)
        controller_metadata = dict(path=str(controller_path.resolve()), sha256=sha256(controller_path),
                                   stored_copy=source_copy.name, filename_mu_label=match.group(1) if match else None,
                                   mu_design=mu_design, Ts=float(data['Ts'].item()))
    effective = dict(schema_version=1, mode=args.mode, mu_target=args.mu_target, mu_design=mu_design,
                     plant='StuartLandauPlant', physical_parameters=physical_parameters, runtime=asdict(config),
                     episode_duration_effective=max_steps * args.dt, evaluation_duration_requested=args.eval_final_time,
                     evaluation_steps=steps_for_duration(args.eval_final_time, args.dt),
                     controller=controller_metadata, requested_ppo_timesteps=args.total_timesteps,
                     transfer_source=None if args.policy_path is None else dict(path=str(args.policy_path.resolve()), sha256=sha256(args.policy_path)),
                     cost_definition='q_y*mean(y_k**2)+r_u*mean(u_cost_k**2); discrete sum, no dt factor',
                     u_cost='u_rl (scaled, before final saturation)' if args.mode == 'residual_rl' else 'u_sat',
                     reward_definition='-ell_k/max_steps; subtract fail_penalty on failure',
                     final_controller_selection='last theta_base after completed guide; not last PPO trial' if args.mode == 'controller_lti' else None,
                     versions={name: version(name) for name in ('numpy', 'scipy', 'torch', 'stable-baselines3', 'gymnasium', 'matplotlib')},
                     python=sys.version, platform=platform.platform(),
                     source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output / f'experiment_{tag}.json', effective)

    if args.mode == 'controller_lti':
        initial = evaluate_closed_loop(env, 'Initial K', args.eval_final_time, role='initial')
        save_results_csv(initial, output / f'sim_kinit_mu{mu_tag(args.mu_target)}.csv')

    model = train_hybrid_ppo(env, args.total_timesteps, policy_path=args.policy_path,
                             learning_rate=args.learning_rate, ent_coef=args.ent_coef,
                             device=args.device, verbose=0 if args.quiet else 1)
    effective['ppo'] = ppo_parameters(model)
    effective['actual_ppo_timesteps'] = model.num_timesteps
    # Save guide state for analysis/recovery; transfer intentionally starts a new
    # experiment at the supplied controller and re-evaluates its initial reference.
    if args.mode == 'controller_lti':
        model.pipeline_guide_state = dict(theta_base=env.theta_base.copy(), J_reference=env.J_base,
                                         completed_episodes=env.episode_index)
    native_file = output / f'ppo_{tag}.zip'
    save_policy(model, env, native_file)
    restored = load_policy(native_file, env, device=str(model.device))
    representative = np.asarray(env.observation_samples)
    original_action, _ = model.predict(representative, deterministic=True)
    restored_action, _ = restored.predict(representative, deterministic=True)
    np.testing.assert_allclose(restored_action, original_action, rtol=0, atol=0)
    effective['native_reload_verified'] = True
    effective['actor_validation'] = export_actor(model, env, output / f'actor_{tag}.mat', effective)

    if args.mode == 'controller_lti':
        P, Z, Kg = env.get_current_controller()
        final_mat = output / f'Kfinal_mu{mu_tag(args.mu_target)}.mat'
        save_final_zpk_controller(final_mat, P, Z, Kg, env.theta_base, env.topology,
                                  mu_target=args.mu_target, mu_design=mu_design)
        # Evaluate exactly the file just exported (one final rollout, no update probe).
        exported = ControllerSOS.from_mat(final_mat, expected_dt=args.dt)
        final = ClosedLoopSimulatorSOS(plant_factory(), exported, config.force_limit,
                                       q_y=config.q_y, r_u=config.r_u).run(args.eval_final_time)
        env._record_evaluation(final, 'final')
        csv_name = f'sim_kfinal_mu{mu_tag(args.mu_target)}.csv'
    else:
        final = env.evaluate_policy(restored, args.eval_final_time)
        prefix = 'mf' if args.mode == 'model_free' else 'residual'
        csv_name = f'sim_{prefix}_mu{mu_tag(args.mu_target)}.csv'
    save_results_csv(final, output / csv_name)
    effective['simulation_budget'] = env.budget
    effective['final_evaluation'] = env.metrics['evaluations'][-1]
    write_json(output / f'experiment_{tag}.json', effective)
    write_json(output / f'metrics_{tag}.json', dict(**env.metrics, simulation_budget=env.budget,
                                                 J_base_meaning='heuristic reference only', legacy_guide_history=env.history if args.mode == 'controller_lti' else None))
    plot_training_metrics(env, output, tag)
    print(f'{args.mode}: {model.num_timesteps} PPO steps; final J/N={final["J_mean"]:.8g}, valid={final["valid"]}')
    print(f'Outputs: {output}')
    return env, model


if __name__ == '__main__':
    main()
