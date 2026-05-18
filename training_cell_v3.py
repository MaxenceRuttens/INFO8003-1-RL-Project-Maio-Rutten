from __future__ import annotations
import os
"""Subprocess-based PPO three-phase self-play training driver for the PyFlyt dogfight.

v2 of the module: addresses the 'silent timeout' failure mode by
  * streaming subprocess stdout/stderr live to the notebook (with timestamps),
  * printing throughput inside each phase from a ProgressCallback,
  * skipping any phase whose checkpoint already exists on disk (per-phase resume),
  * defaulting timeout_per_seed to 3 hours.
"""


import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Configuration object
# =============================================================================

@dataclass
class DogfightTrainingConfig:
    """All settings for one seed of three-phase dogfight self-play."""
    # Identity
    algo_name: str = "PPO"
    env_short: str = "Dogfight"
    # Phase durations (env steps)
    phase_a_steps: int = 200_000
    phase_b_steps: int = 500_000
    phase_c_steps: int = 300_000
    snapshot_freq: int = 100_000
    # Compute
    n_envs: int = 4
    timeout_per_seed: int = 180 * 60        # 3 hours (was 2h in v1)
    progress_print_every: int = 20_000      # steps between in-phase progress prints
    # Hyperparameters
    ppo_kwargs: dict[str, Any] = field(default_factory=dict)
    # Paths
    scripts_dir: Path = None
    models_dir: Path = None
    logs_dir: Path = None
    project_root: Path = None
    # Behaviour
    force_retrain: bool = False

    def run_name(self, seed: int) -> str:
        return f"{self.algo_name}_{self.env_short}_seed{seed}"

    def model_path(self, seed: int, phase: str) -> Path:
        return self.models_dir / f"final_{self.run_name(seed)}_phase{phase}"

    def snapshot_dir(self, seed: int) -> Path:
        return self.models_dir / f"snapshots_{self.run_name(seed)}"

    def log_path(self, seed: int) -> Path:
        return self.logs_dir / self.run_name(seed)

    @property
    def train_script_path(self) -> Path:
        return self.project_root / "_train_worker_dogfight.py"


# =============================================================================
# Embedded worker script
# =============================================================================

_TRAIN_SCRIPT_BODY = r'''"""Standalone PPO three-phase self-play trainer for one dogfight seed.

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
'''


# =============================================================================
# Public API
# =============================================================================

def setup_training_script(project_root: Path | str) -> Path:
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    script_path = project_root / "_train_worker_dogfight.py"
    script_path.write_text(_TRAIN_SCRIPT_BODY)
    print(f"[wrote] {script_path}")
    return script_path


def _stream_subprocess(cmd: list[str], timeout: int) -> int | None:
    """Run cmd; stream each output line to the notebook with a timestamp.

    Returns the process return code, or None if it timed out and was killed.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,    # merge stderr into stdout so we get it all
        text=True,
        bufsize=1,                   # line-buffered
    )

    def reader():
        try:
            for line in proc.stdout:
                ts = datetime.now().strftime("%H:%M:%S")
                # rstrip then add a newline; print() with flush=True keeps order.
                print(f"[{ts}] {line.rstrip()}", flush=True)
        except Exception as e:
            print(f"[stream-reader exception: {e}]", flush=True)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader_thread.join(timeout=5)
        return None

    reader_thread.join(timeout=5)
    return rc


def train_one(seed: int, cfg: DogfightTrainingConfig) -> Path | None:
    paths_no_ext = {p: cfg.model_path(seed, p) for p in ("A", "B", "C")}
    all_exist = all(p.with_suffix(".zip").exists() for p in paths_no_ext.values())
    if all_exist and not cfg.force_retrain:
        print(f"[skip] seed {seed}: all three phase checkpoints exist.")
        return paths_no_ext["C"]

    cmd = [
        sys.executable, str(cfg.train_script_path),
        "--seed",            str(seed),
        "--phase-a-steps",   str(cfg.phase_a_steps),
        "--phase-b-steps",   str(cfg.phase_b_steps),
        "--phase-c-steps",   str(cfg.phase_c_steps),
        "--snapshot-freq",   str(cfg.snapshot_freq),
        "--n-envs",          str(cfg.n_envs),
        "--progress-every",  str(cfg.progress_print_every),
        "--ppo-kwargs",      json.dumps(cfg.ppo_kwargs),
        "--scripts-dir",     str(cfg.scripts_dir),
        "--model-path-a",    str(paths_no_ext["A"]),
        "--model-path-b",    str(paths_no_ext["B"]),
        "--model-path-c",    str(paths_no_ext["C"]),
        "--snapshot-dir",    str(cfg.snapshot_dir(seed)),
        "--log-path",        str(cfg.log_path(seed)),
    ]

    total_budget = cfg.phase_a_steps + cfg.phase_b_steps + cfg.phase_c_steps
    print(f"[start] seed {seed} "
          f"(timeout={cfg.timeout_per_seed}s = {cfg.timeout_per_seed/60:.0f} min, "
          f"budget = {total_budget:,} steps)")
    t0 = time.time()
    rc = _stream_subprocess(cmd, cfg.timeout_per_seed)
    elapsed = time.time() - t0

    if rc is None:
        print(f"[TIMEOUT] seed {seed}: exceeded {cfg.timeout_per_seed}s "
              f"(actual elapsed {elapsed/60:.1f} min)")
        return None
    if rc != 0:
        print(f"[FAIL] seed {seed} (rc={rc}, {elapsed/60:.1f} min)")
        return None
    if not paths_no_ext["C"].with_suffix(".zip").exists():
        print(f"[FAIL] seed {seed}: rc=0 but Phase-C checkpoint missing.")
        return None

    print(f"[done] seed {seed} ({elapsed/60:.1f} min)")
    return paths_no_ext["C"]


def train_all(seeds: list[int], cfg: DogfightTrainingConfig
              ) -> tuple[dict[int, Path], list[int]]:
    final: dict[int, Path] = {}
    failed: list[int] = []
    for seed in seeds:
        print(f"\n========== seed {seed} ==========")
        ckpt = train_one(seed, cfg)
        if ckpt is not None:
            final[seed] = ckpt
        else:
            failed.append(seed)

    print("\nFinal Phase-C checkpoints:")
    for s, p in final.items():
        print(f"  seed {s}: {p}.zip   exists={p.with_suffix('.zip').exists()}")
    if failed:
        print(f"\nFailed/timed-out seeds: {failed}")
        print("Re-run train_all to retry only those "
              "(existing per-phase checkpoints are skipped).")

    return final, failed
