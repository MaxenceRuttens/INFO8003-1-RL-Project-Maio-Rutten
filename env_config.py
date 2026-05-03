"""Centralized environment configuration for PyFlyt environments.

Provides env_kwargs that override restrictive PyFlyt defaults for Waypoints.
Default PyFlyt Waypoints uses goal_reach_distance=0.2m, flight_dome_size=5.0m,
max_duration_seconds=10s — far too restrictive for standard RL algorithms.
Community-tested values: goal_reach_distance=4.0, flight_dome_size=150.0,
max_duration_seconds=120.0.
"""

WAYPOINT_ENV_KWARGS = dict(
    goal_reach_distance=4.0,
    flight_dome_size=150.0,
    max_duration_seconds=120.0,
    num_targets=4,
)


def get_env_kwargs(env_name):
    """Return extra gymnasium.make() kwargs for the given environment.

    Args:
        env_name: One of "hover", "waypoints", "dogfight".

    Returns:
        Dict of kwargs to pass to gymnasium.make(**env_kwargs).
    """
    if env_name == "waypoints":
        return WAYPOINT_ENV_KWARGS.copy()
    return {}
