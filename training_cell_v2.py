"""Subprocess-based PPO training driver for PyFlyt/QuadX-Waypoints-v4.

This module exists to work around a `multiprocessing` + `spawn` failure mode
where the worker process cannot reimport functions from the IPython
kernel's `__main__`. Instead, we write a self-contained training script to
disk and run it via `subprocess.run([sys.executable, ...])`. Each subprocess
is a fresh Python interpreter, so PyBullet socket leaks die with the
process and the parent stays clean.

Usage from a notebook:

    from training_cell_v2 import TrainingConfig, setup_training_script, train_all

    setup_training_script(PROJECT_ROOT)

    cfg = TrainingConfig(
        algo_name="PPO", env_short="QuadX-Waypoints-v4",
        total_timesteps=1_000_000, n_envs=4,
        eval_freq=25_000, n_eval_episodes=5, max_waypoints=4,
        ppo_kwargs=PPO_KWARGS, env_kwargs=WAYPOINT_KWARGS,
        scripts_dir=SCRIPTS_DIR, models_dir=MODELS_DIR, logs_dir=LOGS_DIR,
        project_root=PROJECT_ROOT,
        force_retrain=False, timeout_per_seed=45 * 60,
    )

    checkpoint_paths, failed = train_all(MODES, SEEDS, cfg)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# Configuration object
# =============================================================================

@dataclass
class TrainingConfig:
    """All settings required by the subprocess training driver.

    Anything notebook-level the worker needs is passed through here, never
    reimported from `__main__`.
    """
    # Identity
    algo_name: str
    env_short: str
    # Compute
    total_timesteps: int
    n_envs: int
    eval_freq: int
    n_eval_episodes: int
    max_waypoints: int
    # Hyperparameters and env config (JSON-serialisable; PPO_KWARGS and
    # WAYPOINT_KWARGS are passed straight through to the subprocess)
    ppo_kwargs: dict[str, Any]
    env_kwargs: dict[str, Any]
    # Paths
    scripts_dir: Path
    models_dir: Path
    logs_dir: Path
    project_root: Path
    # Behaviour
    force_retrain: bool = False
    timeout_per_seed: int = 45 * 60   # seconds

    def run_name(self, mode: int, seed: int) -> str:
        return f"{self.algo_name}_{self.env_short}_mode{mode}_seed{seed}"

    def model_path(self, mode: int, seed: int) -> Path:
        return self.models_dir / f"final_{self.run_name(mode, seed)}"

    def log_path(self, mode: int, seed: int) -> Path:
        return self.logs_dir / self.run_name(mode, seed)

    @property
    def train_script_path(self) -> Path:
        return self.project_root / "_train_worker.py"


# =============================================================================
# Embedded training script
# =============================================================================
# This is the body of `_train_worker.py`. It reads all configuration from
# CLI arguments — zero coupling to the parent process / IPython kernel.

_TRAIN_SCRIPT_BODY = r'''"""Standalone PPO trainer for one (seed, flight_mode) on PyFlyt Waypoints.

Invoked by training_cell_v2.train_one via subprocess.run. All configuration is
passed through CLI args so the script has no dependence on the parent kernel.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch
import PyFlyt.gym_envs  # noqa: F401  (env registration side effect)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",            type=int, required=True)
    p.add_argument("--flight-mode",     type=int, required=True)
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--n-envs",          type=int, required=True)
    p.add_argument("--eval-freq",       type=int, required=True)
    p.add_argument("--n-eval-episodes", type=int, required=True)
    p.add_argument("--max-waypoints",   type=int, required=True)
    p.add_argument("--ppo-kwargs",      type=str, required=True, help="JSON")
    p.add_argument("--env-kwargs",      type=str, required=True, help="JSON")
    p.add_argument("--scripts-dir",     type=str, required=True)
    p.add_argument("--ckpt-path",       type=str, required=True, help="WITHOUT .zip")
    p.add_argument("--log-path",        type=str, required=True)
    args = p.parse_args()

    sys.path.insert(0, args.scripts_dir)
    from wrappers import FlattenWaypointEnv

    ppo_kwargs = json.loads(args.ppo_kwargs)
    env_kwargs = json.loads(args.env_kwargs)

    # ---- Seeding ----
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_random_seed(args.seed)

    # ---- Env factory ----
    def make_thunk(rank, mode, base_seed):
        def _init():
            env = gymnasium.make(
                "PyFlyt/QuadX-Waypoints-v4",
                flight_mode=mode,
                **env_kwargs,
            )
            env = FlattenWaypointEnv(env, max_waypoints=args.max_waypoints)
            env = Monitor(env, info_keywords=("num_targets_reached",))
            env.reset(seed=base_seed + rank)
            env.action_space.seed(base_seed + rank)
            return env
        return _init

    train_env = VecMonitor(DummyVecEnv(
        [make_thunk(i, args.flight_mode, args.seed) for i in range(args.n_envs)]))
    eval_env = VecMonitor(DummyVecEnv(
        [make_thunk(0, args.flight_mode, args.seed + 10_000)]))

    # ---- Train ----
    log_path = Path(args.log_path); log_path.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(log_path / "best"),
        log_path=str(log_path),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True, render=False,
    )

    model = PPO(env=train_env, seed=args.seed,
                tensorboard_log=str(log_path / "tb"), **ppo_kwargs)
    model.learn(total_timesteps=args.total_timesteps,
                callback=eval_cb, progress_bar=False)
    Path(args.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.ckpt_path)
    print(f"[done] saved {args.ckpt_path}.zip", flush=True)

    train_env.close(); eval_env.close()


if __name__ == "__main__":
    main()
'''


# =============================================================================
# Public API
# =============================================================================

def setup_training_script(project_root: Path | str) -> Path:
    """Write _train_worker.py into project_root/. Idempotent. Returns its path."""
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    script_path = project_root / "_train_worker.py"
    script_path.write_text(_TRAIN_SCRIPT_BODY)
    print(f"[wrote] {script_path}")
    return script_path


def train_one(seed: int, flight_mode: int, cfg: TrainingConfig) -> Path | None:
    """Train a single (seed, mode) via a fresh-interpreter subprocess.

    Returns the checkpoint path (without .zip) on success, or None on failure /
    timeout. Existing checkpoints are skipped unless cfg.force_retrain.
    """
    expected = cfg.model_path(flight_mode, seed)
    if expected.with_suffix(".zip").exists() and not cfg.force_retrain:
        print(f"[skip]  mode {flight_mode} seed {seed}: checkpoint exists.")
        return expected

    cmd = [
        sys.executable, str(cfg.train_script_path),
        "--seed",            str(seed),
        "--flight-mode",     str(flight_mode),
        "--total-timesteps", str(cfg.total_timesteps),
        "--n-envs",          str(cfg.n_envs),
        "--eval-freq",       str(cfg.eval_freq),
        "--n-eval-episodes", str(cfg.n_eval_episodes),
        "--max-waypoints",   str(cfg.max_waypoints),
        "--ppo-kwargs",      json.dumps(cfg.ppo_kwargs),
        "--env-kwargs",      json.dumps(cfg.env_kwargs),
        "--scripts-dir",     str(cfg.scripts_dir),
        "--ckpt-path",       str(expected),
        "--log-path",        str(cfg.log_path(flight_mode, seed)),
    ]

    print(f"[start] mode {flight_mode} seed {seed}  (timeout={cfg.timeout_per_seed}s)")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, timeout=cfg.timeout_per_seed, check=False)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] mode {flight_mode} seed {seed}: exceeded {cfg.timeout_per_seed}s")
        return None

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL]  mode {flight_mode} seed {seed} (rc={result.returncode}, "
              f"{elapsed/60:.1f} min)")
        return None
    if not expected.with_suffix(".zip").exists():
        print(f"[FAIL]  mode {flight_mode} seed {seed}: rc=0 but checkpoint missing.")
        return None

    print(f"[done]  mode {flight_mode} seed {seed} ({elapsed/60:.1f} min)")
    return expected


def train_all(modes: list[int], seeds: list[int], cfg: TrainingConfig
              ) -> tuple[dict[int, list[Path]], list[tuple[int, int]]]:
    """Run train_one for every (mode, seed) pair, mode-major.

    Returns (checkpoint_paths_by_mode, failed_pairs).
    """
    checkpoints: dict[int, list[Path]] = {m: [] for m in modes}
    failed: list[tuple[int, int]] = []

    for mode in modes:
        print(f"\n========== flight mode {mode} ==========")
        for seed in seeds:
            ckpt = train_one(seed, mode, cfg)
            if ckpt is not None:
                checkpoints[mode].append(ckpt)
            else:
                failed.append((mode, seed))

    print("\nAll checkpoints:")
    for mode in modes:
        print(f"  mode {mode}:")
        for p in checkpoints[mode]:
            print(f"    {p}.zip   exists={p.with_suffix('.zip').exists()}")
    if failed:
        print(f"\nFailed/timed-out: {failed}")
        print("Re-run train_all to retry only those (existing checkpoints are skipped).")

    return checkpoints, failed
