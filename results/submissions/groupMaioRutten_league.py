"""Tournament submission for groupXX (league).

Loads a Stable-Baselines3 PPO checkpoint trained via the three-phase self-play
curriculum described in Section 7 of the report. The weights file
(groupXX_league.zip) is bundled in the same directory as this script.
"""

import os
from stable_baselines3 import PPO


def load_model(path=None):
    """Return a trained model with .predict(obs, deterministic=True)."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "groupXX_league.zip")
    return PPO.load(path)
