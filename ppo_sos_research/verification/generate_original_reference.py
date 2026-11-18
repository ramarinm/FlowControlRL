"""Regenerate regression fixture by executing the user's ORIGINAL ZIP.

Usage, from the revised project root:
    python verification/generate_original_reference.py /path/to/original/ppo_sos_pipeline.zip

Run in a fresh Python process so imports cannot resolve to the revised package.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import argparse
import hashlib
import json
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('original_zip', type=Path)
args = parser.parse_args()
output = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'
with TemporaryDirectory() as tmp:
    root = Path(tmp)
    with ZipFile(args.original_zip) as archive:
        for entry in archive.infolist():
            if not (root / entry.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall(root)
    source = root / 'ppo_sos_pipeline'
    sys.path.insert(0, str(root))
    from ppo_sos_pipeline.builders import make_stuart_landau_env_from_mat
    from ppo_sos_pipeline.config import HybridSOSConfig
    cfg = HybridSOSConfig(max_steps=7, dk_hold_steps=3, k_eval_steps=11,
                          force_limit=0.2, verbose_episode=False)
    env = make_stuart_landau_env_from_mat(str(source / 'Kinit_mu1.0.mat'), cfg)
    actions = np.random.default_rng(2026).uniform(-1, 1, (21, env.action_space.shape[0])).astype(np.float32)
    env.reset()
    data = {k: [] for k in ('obs', 'reward', 'x', 'theta_base', 'theta_trial', 'J_base')}
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        values = dict(obs=obs, reward=reward, x=env.plant.state(), theta_base=env.theta_base.copy(),
                      theta_trial=env.current_theta_trial.copy(), J_base=env.J_base)
        for key, value in values.items():
            data[key].append(value)
        if terminated or truncated:
            env.reset()
    np.savez(output / 'original_lti_trace.npz', actions=actions, **data, J_probe=env.history['J_probe'])
    metadata = dict(source='Unmodified ppo_sos_pipeline.zip uploaded by user', cfg=vars(cfg),
                    original_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
                    generation='21 deterministic supplied actions, three episodes of seven steps, hold=3, offline N=11; original nonlinear Stuart-Landau default mu=2.4')
    (output / 'original_lti_trace.json').write_text(json.dumps(metadata, indent=2))
