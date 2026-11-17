from __future__ import annotations

from typing import List, Optional
import numpy as np




def as_1d(x: np.ndarray | List[float] | float, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def theta_to_string(theta: Optional[np.ndarray], precision: int = 8) -> str:
    if theta is None:
        return ""
    arr = np.asarray(theta, dtype=float).reshape(-1)
    return ";".join(f"{v:.{precision}g}" for v in arr)


def print_timestep_summary(
    max_steps: int = 50,
    hybrid_timesteps: int = 70000,
    dk_hold_steps: int = 50,
) -> None:
    effective_episodes = hybrid_timesteps / max_steps
    proposals_per_episode = max_steps / dk_hold_steps
    proposals_total = hybrid_timesteps / dk_hold_steps
    print("=" * 72)
    print("Hybrid PPO/SOS-ZPK timestep semantics")
    print("=" * 72)
    print(f"MAX_STEPS                         = {max_steps}")
    print(f"HYBRID_TIMESTEPS                  = {hybrid_timesteps}")
    print(f"DK_HOLD_STEPS                     = {dk_hold_steps}")
    print(f"effective_episodes                = {effective_episodes:.3f}")
    print(f"effective_theta_proposals/episode = {proposals_per_episode:.3f}")
    print(f"effective_theta_proposals_total   = {proposals_total:.3f}")
    print("NOTE: PPO still sees total_timesteps = HYBRID_TIMESTEPS physical env steps.")
    print("      Distinct theta proposals are only accepted every DK_HOLD_STEPS.")
    print("=" * 72)


def mu_tag(mu_target: float) -> str:
    """Shortest round-trip decimal, retaining .0 for integer-valued floats."""
    if not np.isfinite(mu_target):
        raise ValueError("mu_target must be finite.")
    return str(float(mu_target))
