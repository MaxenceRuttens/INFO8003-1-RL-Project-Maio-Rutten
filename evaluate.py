"""
Evaluation script for trained RL agents on PyFlyt environments.
Computes statistics and optionally renders episodes.

Usage:
  python scripts/evaluate.py --model results/models/final_PPO_QuadX-Hover-v4_seed42 --env hover
  python scripts/evaluate.py --model results/models/final_SAC_QuadX-Waypoints-v4_seed42 --env waypoints --render
"""

import argparse
import json
import os
import sys

import gymnasium
import numpy as np
import PyFlyt.gym_envs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_config import get_env_kwargs
from wrappers import FlattenWaypointEnv


def make_env(env_id, flight_mode=0, render_mode=None, env_kwargs=None):
    """Create a PyFlyt environment."""
    env = gymnasium.make(env_id, flight_mode=flight_mode, render_mode=render_mode,
                         **(env_kwargs or {}))
    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenWaypointEnv(env, max_waypoints=4)
    return env


def load_model(model_path):
    """Load a model from a .py submission module or .zip SB3 checkpoint."""
    if model_path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_model()
    # Legacy SB3 .zip fallback
    from stable_baselines3 import PPO, SAC
    for cls in [PPO, SAC]:
        try:
            return cls.load(model_path)
        except Exception:
            continue
    raise ValueError(f"Could not load model: {model_path}")


def evaluate_model(model_path, env_id, n_episodes=20, flight_mode=0,
                   render=False, deterministic=True, env_kwargs=None):
    """Evaluate a trained model and return detailed statistics."""
    model = load_model(model_path)

    render_mode = "human" if render else None
    env = make_env(env_id, flight_mode=flight_mode, render_mode=render_mode,
                   env_kwargs=env_kwargs)

    episode_rewards, episode_lengths, episode_crashes = [], [], []
    episode_waypoints = []

    for i in range(n_episodes):
        obs, info = env.reset(seed=100 + i)
        total_reward, steps, crashed = 0.0, 0, False

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated:
                crashed = reward <= -50
                break
            if truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_crashes.append(crashed)
        episode_waypoints.append(info.get("num_targets_reached", 0))

        print(f"  Episode {i+1}/{n_episodes}: reward={total_reward:.2f}, "
              f"steps={steps}, crashed={crashed}", end="")
        if "Waypoints" in env_id:
            print(f", waypoints={episode_waypoints[-1]}", end="")
        print()

    env.close()

    results = {
        "model_path": model_path,
        "env_id": env_id,
        "n_episodes": n_episodes,
        "flight_mode": flight_mode,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "crash_rate": float(np.mean(episode_crashes)),
        "episode_rewards": [float(r) for r in episode_rewards],
    }
    if "Waypoints" in env_id:
        results["mean_waypoints"] = float(np.mean(episode_waypoints))

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Evaluation: {os.path.basename(results['model_path'])}")
    print(f"Env: {results['env_id']}")
    print(f"{'='*60}")
    print(f"  Episodes:      {results['n_episodes']}")
    print(f"  Mean reward:   {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  Median reward: {results['median_reward']:.2f}")
    print(f"  Min/Max:       {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean length:   {results['mean_length']:.1f}")
    print(f"  Crash rate:    {results['crash_rate']*100:.1f}%")
    if "mean_waypoints" in results:
        print(f"  Mean waypoints: {results['mean_waypoints']:.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--env", type=str, required=True,
                        choices=["hover", "waypoints"])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--flight_mode", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    env_map = {
        "hover": "PyFlyt/QuadX-Hover-v4",
        "waypoints": "PyFlyt/QuadX-Waypoints-v4",
    }

    env_kwargs = get_env_kwargs(args.env)
    results = evaluate_model(
        args.model, env_map[args.env], args.n_episodes, args.flight_mode, args.render,
        env_kwargs=env_kwargs,
    )
    print_results(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
