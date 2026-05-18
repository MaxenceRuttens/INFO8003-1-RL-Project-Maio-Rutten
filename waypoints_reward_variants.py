"""Grouped Waypoints wrappers and reward variants for the final SAC notebook."""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces


class FlattenWaypointEnv(gymnasium.ObservationWrapper):
    """Flatten PyFlyt Waypoints dict observations into one fixed-size Box."""

    def __init__(self, env, max_waypoints: int = 4):
        super().__init__(env)
        self.max_waypoints = max_waypoints
        self.attitude_dim = env.observation_space['attitude'].shape[0]
        total_dim = self.attitude_dim + self.max_waypoints * 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float64,
        )

    def observation(self, obs):
        attitude = np.asarray(obs['attitude'], dtype=np.float64)
        targets = np.asarray(obs['target_deltas'], dtype=np.float64)

        padded = np.zeros((self.max_waypoints, 3), dtype=np.float64)
        n = min(len(targets), self.max_waypoints)
        padded[:n] = targets[:n]
        return np.concatenate([attitude, padded.flatten()])


REWARD_VARIANTS = {
    'distance_dense': {
        'description': 'Signed progress + strong smooth proximity shaping.',
        'time_penalty': 0.02,
        'yaw_scale': 0.01,
        'waypoint_bonus': 30.0,
        'crash_penalty': 25.0,
    },
    'relative_progress': {
        'description': 'Normalized signed progress only, no proximity helper.',
        'time_penalty': 0.1,
        'yaw_scale': 0.01,
        'waypoint_bonus': 100.0,
        'crash_penalty': 100.0,
        'progress_scale': 120.0,
        'distance_eps': 1e-6,
    },
    'signed_progress': {
        'description': 'Baseline reward with raw signed progress.',
        'time_penalty': 0.1,
        'yaw_scale': 0.01,
        'waypoint_bonus': 100.0,
        'crash_penalty': 100.0,
        'progress_scale': 3.0,
        'distance_scale': 0.1,
        'distance_eps': 1e-9,
    },
    'soft_signed': {
        'description': 'Signed progress with a deadband and milder regress penalty.',
        'time_penalty': 0.1,
        'yaw_scale': 0.01,
        'waypoint_bonus': 100.0,
        'crash_penalty': 100.0,
        'progress_scale': 3.0,
        'regress_scale': 1.0,
        'progress_deadband': 0.02,
        'distance_scale': 0.1,
        'distance_eps': 1e-9,
    },
}


def list_reward_variants() -> list[str]:
    """Return the supported reward-variant names."""

    return sorted(REWARD_VARIANTS)


class RewardVariantWaypointEnv(gymnasium.Wrapper):
    """Replace the default PyFlyt Waypoints reward with one selected variant."""

    def __init__(self, env, variant: str):
        super().__init__(env)
        if variant not in REWARD_VARIANTS:
            raise ValueError(f'Unknown reward variant: {variant!r}')

        self.variant = variant
        self.variant_cfg = REWARD_VARIANTS[variant]
        self._prev_targets_reached = 0
        self._segment_start_distance = 1.0

    def _current_distance(self) -> float:
        distance = float(self.unwrapped.waypoints.distance_to_next_target)
        if not np.isfinite(distance):
            return 0.0
        return distance

    def _current_progress(self) -> float:
        progress = float(self.unwrapped.waypoints.progress_to_next_target)
        if not np.isfinite(progress):
            return 0.0
        return progress

    def _current_yaw_rate(self) -> float:
        yaw_rate = abs(float(self.unwrapped.env.state(0)[0][2]))
        if not np.isfinite(yaw_rate):
            return 0.0
        return yaw_rate

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_targets_reached = int(info.get('num_targets_reached', 0))
        self._segment_start_distance = max(self._current_distance(), 1.0)
        return obs, info

    def _dense_distance_reward(self, progress: float, distance: float, yaw_rate: float):
        progress_term = 3.0 * float(np.clip(progress, -3.0, 3.0))
        distance_term = 3.0 * (1.0 - float(np.tanh(distance / 25.0)))
        yaw_term = -self.variant_cfg['yaw_scale'] * yaw_rate**2
        reward = -self.variant_cfg['time_penalty'] + progress_term + distance_term + yaw_term
        return reward, progress_term, distance_term, yaw_term

    def _relative_progress_reward(self, progress: float, yaw_rate: float):
        denom = max(self._segment_start_distance, self.variant_cfg['distance_eps'])
        progress_term = self.variant_cfg['progress_scale'] * progress / denom
        distance_term = 0.0
        yaw_term = -self.variant_cfg['yaw_scale'] * yaw_rate**2
        reward = -self.variant_cfg['time_penalty'] + progress_term + yaw_term
        return reward, progress_term, distance_term, yaw_term

    def _signed_progress_reward(self, progress: float, distance: float, yaw_rate: float):
        progress_term = self.variant_cfg['progress_scale'] * progress
        distance_term = self.variant_cfg['distance_scale'] / max(distance, self.variant_cfg['distance_eps'])
        yaw_term = -self.variant_cfg['yaw_scale'] * yaw_rate**2
        reward = -self.variant_cfg['time_penalty'] + progress_term + distance_term + yaw_term
        return reward, progress_term, distance_term, yaw_term

    def _soft_signed_reward(self, progress: float, distance: float, yaw_rate: float):
        if progress > self.variant_cfg['progress_deadband']:
            progress_term = self.variant_cfg['progress_scale'] * progress
        elif progress < -self.variant_cfg['progress_deadband']:
            progress_term = self.variant_cfg['regress_scale'] * progress
        else:
            progress_term = 0.0
        distance_term = self.variant_cfg['distance_scale'] / max(distance, self.variant_cfg['distance_eps'])
        yaw_term = -self.variant_cfg['yaw_scale'] * yaw_rate**2
        reward = -self.variant_cfg['time_penalty'] + progress_term + distance_term + yaw_term
        return reward, progress_term, distance_term, yaw_term

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        progress = self._current_progress()
        distance = self._current_distance()
        yaw_rate = self._current_yaw_rate()

        current_targets_reached = int(info.get('num_targets_reached', self._prev_targets_reached))
        reached_waypoint = current_targets_reached > self._prev_targets_reached
        crashed = bool(
            info.get('collision', False)
            or info.get('out_of_bounds', False)
            or info.get('crashed', False)
        )

        if self.variant == 'distance_dense':
            reward, progress_term, distance_term, yaw_term = self._dense_distance_reward(progress, distance, yaw_rate)
        elif self.variant == 'relative_progress':
            reward, progress_term, distance_term, yaw_term = self._relative_progress_reward(progress, yaw_rate)
        elif self.variant == 'signed_progress':
            reward, progress_term, distance_term, yaw_term = self._signed_progress_reward(progress, distance, yaw_rate)
        elif self.variant == 'soft_signed':
            reward, progress_term, distance_term, yaw_term = self._soft_signed_reward(progress, distance, yaw_rate)
        else:
            raise ValueError(f'Unknown reward variant: {self.variant!r}')

        if reached_waypoint:
            reward += self.variant_cfg['waypoint_bonus']
        if crashed:
            reward -= self.variant_cfg['crash_penalty']

        info['reward_variant'] = self.variant
        info['reward_time_term'] = float(-self.variant_cfg['time_penalty'])
        info['reward_progress_term'] = float(progress_term)
        info['reward_distance_term'] = float(distance_term)
        info['reward_yaw_term'] = float(yaw_term)
        info['reward_waypoint_bonus'] = float(self.variant_cfg['waypoint_bonus'] if reached_waypoint else 0.0)
        info['reward_crash_penalty'] = float(-self.variant_cfg['crash_penalty'] if crashed else 0.0)
        info['reward_shaped_total'] = float(reward)

        self._prev_targets_reached = current_targets_reached
        if reached_waypoint and np.isfinite(distance):
            self._segment_start_distance = max(distance, 1.0)

        return obs, float(reward), terminated, truncated, info
