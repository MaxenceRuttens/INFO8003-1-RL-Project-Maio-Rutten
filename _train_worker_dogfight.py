"""Standalone PPO three-phase self-play trainer for one dogfight seed.

v2: adds ProgressCallback for visibility into throughput, and per-phase resume.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import PyFlyt.gym_envs  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",            type=int, required=True)
    p.add_argument("--phase-a-steps",   type=int, required=True)
    p.add_argument("--phase-b-steps",   type=int, required=True)
    p.add_argument("--phase-c-steps",   type=int, required=True)
    p.add_argument("--snapshot-freq",   type=int, required=True)
    p.add_argument("--n-envs",          type=int, required=True)
    p.add_argument("--progress-every",  type=int, required=True)
    p.add_argument("--ppo-kwargs",      type=str, required=True, help="JSON")
    p.add_argument("--scripts-dir",     type=str, required=True)
    p.add_argument("--model-path-a",    type=str, required=True)
    p.add_argument("--model-path-b",    type=str, required=True)
    p.add_argument("--model-path-c",    type=str, required=True)
    p.add_argument("--snapshot-dir",    type=str, required=True)
    p.add_argument("--log-path",        type=str, required=True)
    return p.parse_args()


class ProgressCallback(BaseCallback):
    """Print num_timesteps, elapsed, and steps/s every `print_every` env steps.

    The 'steps/s' number is what tells you whether the timeout will be enough.
    """
    def __init__(self, print_every: int, phase_label: str):
        super().__init__()
        self.print_every = print_every
        self.phase_label = phase_label
        self.next_print = print_every
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.model.num_timesteps >= self.next_print:
            elapsed = time.time() - self.start_time
            rate = self.model.num_timesteps / elapsed if elapsed > 0 else 0.0
            print(f"  [Phase {self.phase_label}][progress] "
                  f"num_timesteps={self.model.num_timesteps:,} "
                  f"elapsed={elapsed:.0f}s "
                  f"rate={rate:.1f} steps/s",
                  flush=True)
            self.next_print += self.print_every
        return True


def main():
    args = parse_args()
    sys.path.insert(0, args.scripts_dir)
    from dogfight_wrapper import DogfightSelfPlayEnv

    ppo_kwargs = json.loads(args.ppo_kwargs)

    # ---- Seed everything ----
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_random_seed(args.seed)

    log_path = Path(args.log_path); log_path.mkdir(parents=True, exist_ok=True)
    snap_dir = Path(args.snapshot_dir); snap_dir.mkdir(parents=True, exist_ok=True)
    for mp in [args.model_path_a, args.model_path_b, args.model_path_c]:
        Path(mp).parent.mkdir(parents=True, exist_ok=True)

    # ---- LeagueDogfightEnv ----
    class LeagueDogfightEnv(DogfightSelfPlayEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.opponent_pool = []
            self._league_rng = random.Random()
        def set_opponent_pool(self, pool):
            self.opponent_pool = list(pool)
        def reset(self, seed=None, options=None):
            if self.opponent_pool:
                self.opponent_policy = self._league_rng.choice(self.opponent_pool)
            return super().reset(seed=seed, options=options)

    def make_env_thunk(rank, base_seed, league=False):
        def _init():
            EnvClass = LeagueDogfightEnv if league else DogfightSelfPlayEnv
            env = EnvClass(
                team_size=1,
                flatten_observation=True,
                max_duration_seconds=60.0,
                agent_hz=30,
            )
            env = Monitor(env)
            env.reset(seed=base_seed + rank)
            env.action_space.seed(base_seed + rank)
            return env
        return _init

    def make_vec(base_seed, league=False, phase_tag='A'):
        fns = [make_env_thunk(i, base_seed, league=league) for i in range(args.n_envs)]
        log_subdir = os.path.join(args.log_path, f"phase_{phase_tag}")
        os.makedirs(log_subdir, exist_ok=True)
        return VecMonitor(DummyVecEnv(fns), filename=log_subdir)

    def underlying(vec):
        for monitor_env in vec.venv.envs:
            yield monitor_env.unwrapped

    def set_opponent(vec, opponent):
        for env in underlying(vec):
            env.set_opponent_policy(opponent)

    def set_pool(vec, pool):
        for env in underlying(vec):
            env.set_opponent_pool(pool)

    print(f"[seed {args.seed}] worker started", flush=True)

    # ===== Phase A: random opponent =====
    if Path(args.model_path_a + ".zip").exists():
        print(f"[Phase A] skip (checkpoint exists at {args.model_path_a}.zip)", flush=True)
    else:
        print(f"[Phase A] random opponent, {args.phase_a_steps:,} steps", flush=True)
        env_a = make_vec(args.seed, phase_tag="A")
        model = PPO(env=env_a, seed=args.seed,
                    tensorboard_log=str(log_path / "tb_A"), **ppo_kwargs)
        prog_a = ProgressCallback(args.progress_every, "A")
        model.learn(total_timesteps=args.phase_a_steps, callback=prog_a, progress_bar=False)
        model.save(args.model_path_a)
        env_a.close()
        print(f"[Phase A] saved {args.model_path_a}.zip", flush=True)

    # ===== Phase B: self-snapshot opponent =====
    snapshots_json = snap_dir / "snapshots.json"
    if Path(args.model_path_b + ".zip").exists():
        print(f"[Phase B] skip (checkpoint exists at {args.model_path_b}.zip)", flush=True)
        if snapshots_json.exists():
            with open(snapshots_json) as f:
                snapshots = json.load(f)
            print(f"[Phase B] loaded {len(snapshots)} snapshot paths from disk", flush=True)
        else:
            # Phase B checkpoint exists but snapshot index missing — reconstruct.
            snapshots = sorted(str(p) for p in snap_dir.glob("snapshot_*.zip"))
            print(f"[Phase B] reconstructed snapshot list from disk: "
                  f"{len(snapshots)} files", flush=True)
    else:
        print(f"[Phase B] self-snapshot opponent, {args.phase_b_steps:,} steps, "
              f"snapshot every {args.snapshot_freq:,}", flush=True)
        env_b = make_vec(args.seed + 1_000_000)
        model = PPO.load(args.model_path_a, env=env_b)

        class SnapshotCallback(BaseCallback):
            def __init__(self, save_freq, save_dir, vec_env):
                super().__init__()
                self.save_freq = save_freq
                self.save_dir = save_dir
                self.vec_env = vec_env
                self.next_threshold = save_freq
                self.snapshot_paths = []
            def _on_step(self):
                if self.model.num_timesteps >= self.next_threshold:
                    ckpt = self.save_dir / f"snapshot_{self.next_threshold:09d}.zip"
                    self.model.save(str(ckpt))
                    opponent = PPO.load(str(ckpt))
                    set_opponent(self.vec_env, opponent)
                    self.snapshot_paths.append(str(ckpt))
                    self.next_threshold += self.save_freq
                    print(f"  [Phase B][snapshot] {ckpt.name} "
                          f"(num_timesteps={self.model.num_timesteps})", flush=True)
                return True

        snap_cb = SnapshotCallback(args.snapshot_freq, snap_dir, env_b)
        prog_b = ProgressCallback(args.progress_every, "B")
        model.learn(total_timesteps=args.phase_b_steps,
                    callback=[snap_cb, prog_b], progress_bar=False)
        model.save(args.model_path_b)
        env_b.close()
        snapshots = snap_cb.snapshot_paths
        print(f"[Phase B] saved {args.model_path_b}.zip "
              f"and {len(snapshots)} snapshots", flush=True)
        with open(snapshots_json, "w") as f:
            json.dump(snapshots, f, indent=2)

    # ===== Phase C: league opponent =====
    if Path(args.model_path_c + ".zip").exists():
        print(f"[Phase C] skip (checkpoint exists at {args.model_path_c}.zip)", flush=True)
    else:
        if not snapshots:
            raise RuntimeError("Phase B produced no snapshots; cannot run Phase C.")
        print(f"[Phase C] league opponent, {args.phase_c_steps:,} steps, "
              f"pool size = {len(snapshots)}", flush=True)
        league_pool = [PPO.load(p) for p in snapshots]
        env_c = make_vec(args.seed + 2_000_000, league=True)
        set_pool(env_c, league_pool)
        model = PPO.load(args.model_path_b, env=env_c)
        prog_c = ProgressCallback(args.progress_every, "C")
        model.learn(total_timesteps=args.phase_c_steps,
                    callback=prog_c, progress_bar=False)
        model.save(args.model_path_c)
        env_c.close()
        print(f"[Phase C] saved {args.model_path_c}.zip", flush=True)

    print(f"[seed {args.seed}] worker done", flush=True)


if __name__ == "__main__":
    main()
