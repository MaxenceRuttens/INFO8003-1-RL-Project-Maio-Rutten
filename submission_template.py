"""
Submission template for the dogfight tournament.

Rename this file to groupXX.py (e.g., group01.py) and implement load_model().
Your model must expose a .predict(obs, deterministic=True) -> (action, info) interface.

  - obs:    np.ndarray of shape (37,)   — dogfight observation
  - action: np.ndarray of shape (4,)    — [roll, pitch, yaw, throttle] in [-1, 1]

You may use any RL library or custom implementation. The only requirement is
that load_model() returns an object with the predict() signature above.

Example with SB3:
    from stable_baselines3 import PPO
    def load_model(path=None):
        return PPO.load(path or "my_model.zip")

Example with a custom policy:
    import numpy as np
    class MyPolicy:
        def __init__(self, weights_path):
            self.weights = np.load(weights_path)
        def predict(self, obs, deterministic=True):
            action = ...  # your inference code
            return action, {}
    def load_model(path=None):
        return MyPolicy(path or "my_weights.npy")
"""

import numpy as np


def load_model(path=None):
    """Load and return a trained model for the dogfight tournament.

    Args:
        path: Optional path to model weights / checkpoint.

    Returns:
        An object with .predict(obs, deterministic=True) -> (action, info).
        obs: np.ndarray(37,), action: np.ndarray(4,).
    """
    raise NotImplementedError(
        "Replace this with your model loading code. "
        "See the docstring above for examples."
    )
