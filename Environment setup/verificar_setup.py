"""Verifica las versiones acordadas y un entrenamiento corto de PPO en CPU."""

import sys
from importlib import import_module
from importlib.metadata import version

EXPECTED = {
    "numpy": "1.26.4",
    "scipy": "1.14.1",
    "pandas": "2.2.3",
    "meshio": "5.3.5",
    "control": "0.10.1",
    "torch": "2.9.1",
    "gymnasium": "1.0.0",
    "stable-baselines3": "2.9.2",
}


def main():
    print("Python:", sys.version.split()[0])
    print("Ejecutable:", sys.executable)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Selecciona el interprete o kernel de ENV_RL con Python 3.10.")

    for package, expected in EXPECTED.items():
        module = "stable_baselines3" if package == "stable-baselines3" else package
        import_module(module)
        installed = version(package)
        print(f"{package}: {installed}")
        if installed != expected:
            raise RuntimeError(
                f"{package}: se esperaba {expected}, pero se encontro {installed}."
            )

    import matplotlib
    import numpy as np
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO

    print("matplotlib:", matplotlib.__version__)
    print("CUDA de esta compilacion:", torch.version.cuda)
    if torch.version.cuda is not None:
        raise RuntimeError("Este setup requiere la compilacion de PyTorch para CPU.")

    # Comprobar tambien la interoperabilidad de NumPy y PyTorch.
    total = torch.from_numpy(np.array([1, 2, 3], dtype=np.float32)).sum().item()
    if total != 6.0:
        raise RuntimeError("Fallo la prueba de interoperabilidad NumPy/PyTorch.")

    env = gym.make("CartPole-v1")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            seed=42,
            verbose=0,
        )
        model.learn(total_timesteps=128)
    finally:
        env.close()

    print("PRUEBA COMPLETADA: versiones y entrenamiento en CPU correctos.")


if __name__ == "__main__":
    main()
