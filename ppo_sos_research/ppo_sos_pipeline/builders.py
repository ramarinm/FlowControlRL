from __future__ import annotations

from .env import HybridSOSConfig, HybridSOSZPKPPOEnv
from .controller_sos import ControllerSOS
from .plants import StuartLandauPlant
from .zpk import ZPKActionMapper, ZPKTopology


def make_env(plant_factory, config=None, mat_filename=None):
    """The only builder: inject a plant factory and optionally a discrete controller."""
    config = HybridSOSConfig() if config is None else config
    controller = None
    topology = mapper = None
    if config.mode != 'model_free':
        if mat_filename is None:
            raise ValueError(f'{config.mode} requires a MATLAB controller file.')
        controller = ControllerSOS.from_mat(mat_filename, expected_dt=config.dt)
        if config.mode == 'controller_lti':
            topology = ZPKTopology.from_controller(controller)
            mapper = ZPKActionMapper(topology)
    return HybridSOSZPKPPOEnv(topology, mapper, plant_factory, config,
                             base_controller=controller if config.mode == 'residual_rl' else None)


def make_stuart_landau_env_from_mat(mat_filename, config=None, **plant_parameters):
    """Compatibility entry point; physical parameters can now be explicit."""
    config = HybridSOSConfig() if config is None else config
    return make_env(lambda: StuartLandauPlant(dt=config.dt, Ts=config.dt, **plant_parameters),
                    config, mat_filename)
