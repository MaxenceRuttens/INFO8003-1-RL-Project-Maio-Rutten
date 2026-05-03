"""
Dogfight Tournament System with Elo Rating.

Loads student submissions (.py modules or legacy .zip SB3 checkpoints) and runs
a round-robin tournament using PyFlyt MAFixedwingDogfightEnvV2 (1v1).

Usage:
  python scripts/tournament.py submissions/         # dir of .py / .zip files
  python scripts/tournament.py group01.py group02.py group03.zip
  python scripts/tournament.py submissions/ --matches 10 --render
  python scripts/tournament.py submissions/ --output results.json

Submission format (.py — preferred):
  A Python module with a load_model() function that returns an object with
  .predict(obs, deterministic=True) -> (action, info).
  See scripts/submission_template.py for details.

Legacy format (.zip — SB3 checkpoint):
  SB3 PPO or SAC .zip file. Must be trained for obs_space=Box(37,),
  act_space=Box(4,) (1v1 dogfight).
"""

import argparse
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- Elo Rating System ----

class EloSystem:
    """Standard Elo rating system."""

    def __init__(self, k=32, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}
        self.match_history = []

    def add_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, player_a, player_b, score_a):
        """Update ratings after a match. score_a: 1=win, 0=loss, 0.5=draw."""
        ra, rb = self.ratings[player_a], self.ratings[player_b]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        score_b = 1.0 - score_a

        self.ratings[player_a] = ra + self.k * (score_a - ea)
        self.ratings[player_b] = rb + self.k * (score_b - eb)

        self.match_history.append({
            "player_a": player_a,
            "player_b": player_b,
            "score_a": score_a,
            "score_b": score_b,
            "rating_a_before": ra,
            "rating_b_before": rb,
            "rating_a_after": self.ratings[player_a],
            "rating_b_after": self.ratings[player_b],
        })

    def get_rankings(self):
        """Return sorted rankings."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ---- Model Loading ----

def _load_sb3_checkpoint(path):
    """Load a legacy SB3 .zip checkpoint (PPO or SAC)."""
    from stable_baselines3 import PPO, SAC
    for cls in [PPO, SAC]:
        try:
            return cls.load(path)
        except Exception:
            continue
    raise ValueError(f"Could not load SB3 checkpoint: {path}")


def _load_py_submission(path):
    """Load a .py submission module and call its load_model() function."""
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_model"):
        raise ValueError(f"Submission {path} has no load_model() function")
    return mod.load_model()


def _validate_model(model, name):
    """Validate that a model has the correct predict() interface."""
    dummy_obs = np.zeros(37, dtype=np.float32)
    try:
        action, _ = model.predict(dummy_obs, deterministic=True)
        action = np.asarray(action)
        if action.shape != (4,):
            raise ValueError(
                f"Model {name}: predict() returned action shape {action.shape}, "
                f"expected (4,)"
            )
    except Exception as e:
        raise ValueError(f"Model {name}: predict() validation failed: {e}") from e


def load_submission(path):
    """Load a submission from .py module or .zip SB3 checkpoint.

    Args:
        path: Path to .py submission module or .zip SB3 checkpoint.

    Returns:
        Model object with .predict(obs, deterministic=True) -> (action, info).
    """
    if path.endswith(".py"):
        return _load_py_submission(path)
    elif path.endswith(".zip"):
        return _load_sb3_checkpoint(path)
    else:
        raise ValueError(f"Unsupported file type: {path}. Use .py or .zip")


# ---- Match Runner ----

def run_match(model_a, model_b, n_games=5, max_steps=1800, render_mode=None,
              seed_offset=0):
    """Run n_games between two models. Returns (wins_a, wins_b, draws).

    model_a always plays as uav_0, model_b as uav_1.
    The environment is symmetric (circular spawn), so no side-swapping needed.
    """
    from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2

    wins_a, wins_b, draws = 0, 0, 0
    total_reward_a, total_reward_b = 0.0, 0.0
    game_details = []

    for game_idx in range(n_games):
        try:
            env = MAFixedwingDogfightEnvV2(
                team_size=1,
                assisted_flight=True,
                flatten_observation=True,
                render_mode=render_mode,
                max_duration_seconds=60.0,
                agent_hz=30,
            )

            observations, infos = env.reset(seed=seed_offset + game_idx)
            rewards_acc = {agent: 0.0 for agent in env.agents}

            for step in range(max_steps):
                actions = {}
                for agent in env.agents:
                    policy = model_a if agent == "uav_0" else model_b
                    action, _ = policy.predict(observations[agent], deterministic=True)
                    actions[agent] = action

                observations, rewards, terminations, truncations, infos = env.step(actions)

                for agent in rewards:
                    rewards_acc[agent] = rewards_acc.get(agent, 0.0) + rewards.get(agent, 0.0)

                all_done = all(
                    terminations.get(a, True) or truncations.get(a, False)
                    for a in env.agents
                ) or len(env.agents) == 0
                if all_done:
                    break

            try:
                env.close()
            except Exception:
                pass

            # Determine winner based on infos
            a_won, b_won = False, False
            for agent, info in infos.items():
                if info.get("team_win", False):
                    if agent == "uav_0":
                        a_won = True
                    else:
                        b_won = True

            # If no team_win signal, use rewards
            if not a_won and not b_won:
                r_a = rewards_acc.get("uav_0", 0.0)
                r_b = rewards_acc.get("uav_1", 0.0)
                if r_a > r_b + 50:
                    a_won = True
                elif r_b > r_a + 50:
                    b_won = True

            if a_won and not b_won:
                wins_a += 1
                result = "A"
            elif b_won and not a_won:
                wins_b += 1
                result = "B"
            else:
                draws += 1
                result = "draw"

            r_a = rewards_acc.get("uav_0", 0.0)
            r_b = rewards_acc.get("uav_1", 0.0)
            total_reward_a += r_a
            total_reward_b += r_b

            game_details.append({
                "game": game_idx,
                "result": result,
                "reward_a": float(r_a),
                "reward_b": float(r_b),
                "steps": step + 1,
            })

        except Exception as e:
            print(f"    Game {game_idx} failed: {e}")
            draws += 1
            game_details.append({
                "game": game_idx,
                "result": "error",
                "error": str(e),
            })

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "mean_reward_a": total_reward_a / max(n_games, 1),
        "mean_reward_b": total_reward_b / max(n_games, 1),
        "games": game_details,
    }


# ---- Tournament ----

def run_tournament(submission_paths, matches_per_pair=5, render=False):
    """Run round-robin tournament between all submissions."""
    # Load all submissions
    print("Loading submissions...")
    models = {}
    for path in submission_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            model = load_submission(path)
            _validate_model(model, name)
            models[name] = model
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  FAILED: {name} — {e}")

    if len(models) < 2:
        print("Need at least 2 valid submissions for a tournament.")
        return None

    # Initialize Elo
    elo = EloSystem(k=32, initial_rating=1500)
    for name in models:
        elo.add_player(name)

    # Round-robin
    pairs = list(itertools.combinations(models.keys(), 2))
    total_matches = len(pairs)
    all_match_results = []

    print(f"\nRunning {total_matches} matches ({matches_per_pair} games each)...")
    print("=" * 60)

    render_mode = "human" if render else None

    for i, (name_a, name_b) in enumerate(pairs):
        print(f"\n  Match {i+1}/{total_matches}: {name_a} vs {name_b}")

        result = run_match(
            models[name_a], models[name_b],
            n_games=matches_per_pair,
            render_mode=render_mode,
            seed_offset=i * 100,
        )

        # Compute Elo score: proportion of wins for A
        total = result["wins_a"] + result["wins_b"] + result["draws"]
        if total > 0:
            score_a = (result["wins_a"] + 0.5 * result["draws"]) / total
        else:
            score_a = 0.5
        elo.update(name_a, name_b, score_a)

        print(f"    {name_a}: {result['wins_a']}W  |  "
              f"{name_b}: {result['wins_b']}W  |  "
              f"Draws: {result['draws']}  |  "
              f"Avg reward: {result['mean_reward_a']:.0f} vs {result['mean_reward_b']:.0f}")

        all_match_results.append({
            "player_a": name_a,
            "player_b": name_b,
            **result,
        })

    # Final rankings
    rankings = elo.get_rankings()

    print("\n" + "=" * 60)
    print("FINAL RANKINGS")
    print("=" * 60)
    print(f"{'Rank':<6} {'Player':<35} {'Elo':>8}")
    print("-" * 60)
    for rank, (name, rating) in enumerate(rankings, 1):
        print(f"  {rank:<4} {name:<35} {rating:>8.1f}")
    print("=" * 60)

    return {
        "rankings": [{"rank": i+1, "name": n, "elo": round(r, 1)}
                      for i, (n, r) in enumerate(rankings)],
        "matches": all_match_results,
        "match_history": elo.match_history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dogfight Tournament with Elo Rating",
        epilog="Provide .py submissions, .zip checkpoints, or a directory.",
    )
    parser.add_argument("checkpoints", nargs="+",
                        help="Submission .py/.zip files or directory containing them")
    parser.add_argument("--matches", type=int, default=5,
                        help="Number of games per pair (default: 5)")
    parser.add_argument("--render", action="store_true",
                        help="Render matches visually")
    parser.add_argument("--output", type=str, default="tournament_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    # Collect submission paths
    valid_extensions = (".py", ".zip")
    paths = []
    for p in args.checkpoints:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(valid_extensions) and f != "submission_template.py":
                    paths.append(os.path.join(p, f))
        elif os.path.isfile(p) and p.endswith(valid_extensions):
            paths.append(p)
        else:
            print(f"Warning: Skipping {p} (not a .py/.zip file or directory)")

    if not paths:
        print("No submissions found (.py or .zip).")
        sys.exit(1)

    print(f"Found {len(paths)} submissions:")
    for p in paths:
        print(f"  {p}")

    results = run_tournament(paths, matches_per_pair=args.matches, render=args.render)

    if results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
