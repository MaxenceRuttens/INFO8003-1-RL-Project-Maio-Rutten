"""Custom Gymnasium wrappers for PyFlyt environments."""

import gymnasium
import numpy as np
from gymnasium import spaces


class FlattenWaypointEnv(gymnasium.ObservationWrapper):
    """Flattens the Dict observation of PyFlyt Waypoints envs into a single Box.

    The Waypoints env returns:
      - 'attitude': Box(21,) - drone state
      - 'target_deltas': (N, 3) waypoint deltas — N can decrease as waypoints are reached

    This wrapper pads/truncates target_deltas to a fixed number of waypoints
    and concatenates everything into a single flat vector.
    """

    def __init__(self, env, max_waypoints=4):
        super().__init__(env)
        self.max_waypoints = max_waypoints

        # Determine attitude dim from the observation space
        self.attitude_dim = env.observation_space["attitude"].shape[0]
        total_dim = self.attitude_dim + self.max_waypoints * 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float64
        )

    def observation(self, obs):
        attitude = obs["attitude"]
        targets = obs["target_deltas"]  # shape (N, 3), N may vary

        # Pad or truncate to max_waypoints
        padded = np.zeros((self.max_waypoints, 3), dtype=np.float64)
        n = min(len(targets), self.max_waypoints)
        padded[:n] = targets[:n]

        return np.concatenate([attitude, padded.flatten()])


class HERWaypointEnv(gymnasium.ObservationWrapper):
    """Goal-conditioned wrapper for PyFlyt Waypoints, compatible with HER.

    The wrapped observation follows the standard Gymnasium goal-conditioned
    convention:
      - observation: flat state used by the policy
      - achieved_goal: current drone position in world coordinates
      - desired_goal: current target waypoint in world coordinates

    For the policy input, we keep the same flattened observation as in
    FlattenWaypointEnv so that the main change is HER itself, not a totally new
    state representation.
    """

    def __init__(self, env, max_waypoints=4):
        super().__init__(env)
        self.max_waypoints = max_waypoints
        self.attitude_dim = env.observation_space["attitude"].shape[0]
        self.goal_dim = 3
        self.goal_reach_distance = float(env.unwrapped.waypoints.goal_reach_distance)

        flat_dim = self.attitude_dim + self.max_waypoints * 3
        flat_box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float64
        )
        goal_box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.goal_dim,), dtype=np.float64
        )

        self.observation_space = spaces.Dict(
            {
                "observation": flat_box,
                "achieved_goal": goal_box,
                "desired_goal": goal_box,
            }
        )

    def _pad_targets(self, targets):
        padded = np.zeros((self.max_waypoints, 3), dtype=np.float64)
        n = min(len(targets), self.max_waypoints)
        padded[:n] = targets[:n]
        return padded

    def _flat_observation(self, obs):
        attitude = np.asarray(obs["attitude"], dtype=np.float64)
        padded_targets = self._pad_targets(obs["target_deltas"])
        return np.concatenate([attitude, padded_targets.flatten()])

    def _drone_position(self):
        position, *_ = self.unwrapped.compute_attitude()
        return np.asarray(position, dtype=np.float64)

    def _current_target_position(self):
        targets = np.asarray(self.unwrapped.waypoints.targets, dtype=np.float64)
        return targets[0].copy()

    def observation(self, obs):
        return {
            "observation": self._flat_observation(obs),
            "achieved_goal": self._drone_position(),
            "desired_goal": self._current_target_position(),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal = np.asarray(desired_goal, dtype=np.float64)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > self.goal_reach_distance).astype(np.float32)
