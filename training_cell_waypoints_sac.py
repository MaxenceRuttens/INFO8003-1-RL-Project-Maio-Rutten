from __future__ import annotations

"""Self-contained SAC Waypoints training helper for the notebook workflow.

This module mirrors the final Hover notebook/helper structure while keeping
all main Waypoints SAC experiments in one place: mode comparison, delayed
learning, and reward-shaping trials.
"""

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyFlyt.gym_envs  # noqa: F401
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

CURRENT_DIR = Path(__file__).resolve().parent
SHARED_SCRIPTS_DIR = CURRENT_DIR / "scripts"

if not SHARED_SCRIPTS_DIR.exists():
    SHARED_SCRIPTS_DIR = CURRENT_DIR.parent

if str(SHARED_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS_DIR))

from env_config import get_env_kwargs
from waypoints_reward_variants import (
    FlattenWaypointEnv,
    RewardVariantWaypointEnv,
    list_reward_variants,
)


@dataclass
class WaypointsTrainingConfig:
    """Configuration for SAC Waypoints runs."""

    algo_name: str = "SAC"
    env_name: str = "waypoints"
    flight_mode: int = 0
    timesteps: int = 1_000_000

    learning_rate: float = 5e-5
    buffer_size: int = 1_000_000
    batch_size: int = 512
    gamma: float = 0.995
    ent_coef: str = "auto"
    learning_starts: int = 50_000
    gradient_steps: int = 1

    seeds: list[int] = field(default_factory=lambda: [0])
    eval_freq: int = 10_000
    n_eval_episodes: int = 30
    final_eval_episodes: int = 50

    reward_variant: str | None = None
    project_root: Path | None = None
    results_dir: Path | None = None
    save_name: str | None = None

    def resolved_project_root(self) -> Path:
        if self.project_root is None:
            if (CURRENT_DIR / "scripts").exists():
                return CURRENT_DIR
            return CURRENT_DIR.parents[1]
        return Path(self.project_root).resolve()

    def resolved_results_dir(self) -> Path:
        if self.results_dir is None:
            return self.resolved_project_root() / "results" / "waypoints_sac"
        return Path(self.results_dir).resolve()

    def resolved_save_name(self) -> str:
        if self.save_name is not None:
            return self.save_name
        ts = f"{self.timesteps:,}".replace(",", "_")
        return f"{self.algo_name}_{self.env_name}_mode{self.flight_mode}_{ts}steps"

    def model_zip_path(self, seed: int) -> Path:
        return self.resolved_results_dir() / "models" / f"{self.resolved_save_name()}_seed{seed}.zip"

    def results_path(self, *parts: str) -> Path:
        return self.resolved_results_dir().joinpath(*parts)


def get_env_title(env_name: str) -> str:
    return "QuadX-Hover-v4" if env_name == "hover" else "QuadX-Waypoints-v4"


def make_env(cfg: WaypointsTrainingConfig, seed: int | None = None):
    env = gymnasium.make(
        "PyFlyt/QuadX-Waypoints-v4",
        flight_mode=cfg.flight_mode,
        **get_env_kwargs("waypoints"),
    )

    if cfg.reward_variant is not None:
        env = RewardVariantWaypointEnv(env, variant=cfg.reward_variant)

    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenWaypointEnv(env, max_waypoints=4)

    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)


def train_one_seed(cfg: WaypointsTrainingConfig, seed: int) -> str:
    seed_everything(seed)
    run_name = f"{cfg.resolved_save_name()}_seed{seed}"

    env = make_env(cfg, seed=seed)
    eval_env = make_env(cfg, seed=10_000 + seed)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=0.005,
        ent_coef=cfg.ent_coef,
        train_freq=1,
        gradient_steps=cfg.gradient_steps,
        learning_starts=cfg.learning_starts,
        use_sde=True,
        use_sde_at_warmup=True,
        seed=seed,
        verbose=1,
        tensorboard_log=str(cfg.results_path("tensorboard_sac")),
    )

    cfg.results_path("models").mkdir(parents=True, exist_ok=True)
    cfg.results_path("best_models").mkdir(parents=True, exist_ok=True)
    cfg.results_path("evaluations").mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(cfg.results_path("best_models", run_name)),
        log_path=str(cfg.results_path("evaluations", run_name)),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=cfg.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    save_path = str(cfg.results_path("models", run_name))
    model.save(save_path)

    env.close()
    eval_env.close()

    return save_path + ".zip"


def evaluate_model(
    cfg: WaypointsTrainingConfig,
    model_path: str,
    eval_seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = make_env(cfg)
    model = SAC.load(model_path, env=env)

    returns = []
    lengths = []
    crashes = []

    for seed in eval_seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False
        ep_return = 0.0
        ep_len = 0
        last_info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            last_info = info

        returns.append(ep_return)
        lengths.append(ep_len)
        crashed = bool(
            last_info.get("collision", False)
            or last_info.get("crashed", False)
            or last_info.get("out_of_bounds", False)
        )
        crashes.append(crashed)

    env.close()
    return np.array(returns), np.array(lengths), np.array(crashes)


def bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 123,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    return tuple(np.percentile(boot_means, [2.5, 97.5]))


def iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(values[lo:hi]))


def load_learning_curves(cfg: WaypointsTrainingConfig) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    curves = {}

    for seed in cfg.seeds:
        path = cfg.results_path(
            "evaluations",
            f"{cfg.resolved_save_name()}_seed{seed}",
            "evaluations.npz",
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation file: {path}")

        data = np.load(path)
        timesteps = data["timesteps"]
        mean_returns = data["results"].mean(axis=1)
        curves[seed] = (timesteps, mean_returns)

    return curves


def plot_learning_curve(cfg: WaypointsTrainingConfig, out_path: Path) -> None:
    curves = load_learning_curves(cfg)
    common_steps = curves[cfg.seeds[0]][0]
    returns = np.vstack([curves[seed][1] for seed in cfg.seeds])

    mean_returns = returns.mean(axis=0)

    ci_low = []
    ci_high = []
    for j in range(returns.shape[1]):
        lo, hi = bootstrap_ci_mean(returns[:, j])
        ci_low.append(lo)
        ci_high.append(hi)

    plt.figure(figsize=(7.0, 4.8))

    for i, seed in enumerate(cfg.seeds):
        label = f"seed {seed}" if i == 0 else None
        plt.plot(common_steps, returns[i], linewidth=0.9, alpha=0.25, label=label)

    plt.plot(common_steps, mean_returns, linewidth=2.0, label="mean across seeds")
    plt.fill_between(common_steps, ci_low, ci_high, alpha=0.20, label="95 % bootstrap CI")

    plt.title(f"SAC on {get_env_title(cfg.env_name)} (flight mode {cfg.flight_mode}, {len(cfg.seeds)} seeds)")
    plt.xlabel("Environment steps")
    plt.ylabel("Mean evaluation return")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_final_boxplot(
    cfg: WaypointsTrainingConfig,
    checkpoint_returns: Dict[int, np.ndarray],
    out_path: Path,
) -> None:
    data = [checkpoint_returns[seed] for seed in cfg.seeds]
    means = [np.mean(x) for x in data]

    plt.figure(figsize=(7.0, 4.8))
    plt.boxplot(data, labels=[f"seed {seed}" for seed in cfg.seeds], showfliers=True)
    plt.scatter(np.arange(1, len(cfg.seeds) + 1), means, marker="^", label="per-seed mean")

    plt.title(
        f"SAC on {get_env_title(cfg.env_name)} — final eval ({cfg.final_eval_episodes} episodes per seed)"
    )
    plt.ylabel("Episode return")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate_saved_checkpoints(
    cfg: WaypointsTrainingConfig,
    model_paths: Dict[int, str],
    eval_seeds: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    checkpoint_returns = {}
    checkpoint_lengths = {}
    checkpoint_crashes = {}

    for seed in cfg.seeds:
        returns, lengths, crashes = evaluate_model(
            cfg=cfg,
            model_path=model_paths[seed],
            eval_seeds=eval_seeds,
        )
        checkpoint_returns[seed] = returns
        checkpoint_lengths[seed] = lengths
        checkpoint_crashes[seed] = crashes

    return checkpoint_returns, checkpoint_lengths, checkpoint_crashes


def build_checkpoint_path_map(
    cfg: WaypointsTrainingConfig,
    checkpoint_kind: str,
) -> Dict[int, str]:
    if checkpoint_kind == "final_model":
        return {
            seed: str(cfg.results_path("models", f"{cfg.resolved_save_name()}_seed{seed}.zip"))
            for seed in cfg.seeds
        }

    if checkpoint_kind == "best_checkpoint":
        return {
            seed: str(
                cfg.results_path(
                    "best_models",
                    f"{cfg.resolved_save_name()}_seed{seed}",
                    "best_model.zip",
                )
            )
            for seed in cfg.seeds
        }

    raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")


def save_final_stats(
    final_returns: Dict[int, np.ndarray],
    final_lengths: Dict[int, np.ndarray],
    final_crashes: Dict[int, np.ndarray],
    out_path: Path,
) -> pd.DataFrame:
    per_seed_means = np.array([np.mean(final_returns[seed]) for seed in final_returns])
    all_lengths = np.concatenate([final_lengths[seed] for seed in final_lengths])
    all_crashes = np.concatenate([final_crashes[seed] for seed in final_crashes])

    ci_low, ci_high = bootstrap_ci_mean(per_seed_means)
    std_value = np.std(per_seed_means, ddof=1) if len(per_seed_means) > 1 else np.nan

    table = pd.DataFrame(
        {
            "Statistic": [
                "Per-seed mean returns",
                "Mean across seeds",
                "IQM across seeds",
                "Bootstrap 95 % CI",
                "Mean episode length",
                "Crash rate",
            ],
            "Value": [
                ", ".join(f"{x:.1f}" for x in per_seed_means),
                f"{np.mean(per_seed_means):.1f} ± {std_value:.1f}",
                f"{iqm(per_seed_means):.1f}",
                f"[{ci_low:.1f}, {ci_high:.1f}]",
                f"{np.mean(all_lengths):.1f}",
                f"{100.0 * np.mean(all_crashes):.1f} %",
            ],
        }
    )

    table.to_csv(out_path, index=False)
    return table


def make_all_plots_and_stats(cfg: WaypointsTrainingConfig) -> None:
    fig_dir = cfg.results_path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_name = cfg.resolved_save_name()
    learning_curve_path = fig_dir / f"{save_name}_learning_curve.png"
    final_boxplot_path = fig_dir / f"{save_name}_final_boxplot.png"
    final_stats_path = fig_dir / f"{save_name}_final_stats.csv"
    best_stats_path = fig_dir / f"{save_name}_best_stats.csv"
    comparison_stats_path = fig_dir / f"{save_name}_checkpoint_comparison.csv"

    plot_learning_curve(cfg, learning_curve_path)

    eval_seeds = list(range(100, 100 + cfg.final_eval_episodes))
    final_model_paths = build_checkpoint_path_map(cfg, "final_model")
    best_model_paths = build_checkpoint_path_map(cfg, "best_checkpoint")

    final_returns, final_lengths, final_crashes = evaluate_saved_checkpoints(
        cfg=cfg,
        model_paths=final_model_paths,
        eval_seeds=eval_seeds,
    )
    best_returns, best_lengths, best_crashes = evaluate_saved_checkpoints(
        cfg=cfg,
        model_paths=best_model_paths,
        eval_seeds=eval_seeds,
    )

    plot_final_boxplot(cfg, final_returns, final_boxplot_path)
    final_stats = save_final_stats(final_returns, final_lengths, final_crashes, final_stats_path)
    best_stats = save_final_stats(best_returns, best_lengths, best_crashes, best_stats_path)

    comparison_table = pd.concat(
        [
            final_stats.assign(Checkpoint="final_model"),
            best_stats.assign(Checkpoint="best_checkpoint"),
        ],
        ignore_index=True,
    )[["Checkpoint", "Statistic", "Value"]]
    comparison_table.to_csv(comparison_stats_path, index=False)

    print(f"\nSaved learning curve: {learning_curve_path}")
    print(f"Saved final boxplot:  {final_boxplot_path}")
    print(f"Saved final stats:    {final_stats_path}")
    print(f"Saved best stats:     {best_stats_path}")
    print(f"Saved comparison:     {comparison_stats_path}\n")
    print("Final model stats:")
    print(final_stats.to_string(index=False))
    print("\nBest checkpoint stats:")
    print(best_stats.to_string(index=False))


def train_all(
    cfg: WaypointsTrainingConfig,
    plot_only: bool = False,
    force_retrain: bool = False,
) -> dict[str, Path]:
    if cfg.env_name != "waypoints":
        raise ValueError(f"WaypointsTrainingConfig expects env_name='waypoints', got {cfg.env_name!r}")
    if cfg.reward_variant is not None and cfg.reward_variant not in list_reward_variants():
        raise ValueError(f"Unknown reward variant: {cfg.reward_variant!r}")

    if not plot_only:
        for seed in cfg.seeds:
            if cfg.model_zip_path(seed).exists() and not force_retrain:
                print(f"[skip] seed {seed}: final model already exists at {cfg.model_zip_path(seed)}")
                continue

            tag = cfg.reward_variant if cfg.reward_variant is not None else "baseline"
            print(f"\n========== Training SAC Waypoints seed {seed} ({tag}) ==========")
            train_one_seed(cfg, seed)

    make_all_plots_and_stats(cfg)
    return output_paths(cfg)


def output_paths(cfg: WaypointsTrainingConfig) -> dict[str, Path]:
    save_name = cfg.resolved_save_name()
    figures_dir = cfg.results_path("figures")

    return {
        "learning_curve": figures_dir / f"{save_name}_learning_curve.png",
        "final_boxplot": figures_dir / f"{save_name}_final_boxplot.png",
        "final_stats": figures_dir / f"{save_name}_final_stats.csv",
        "best_stats": figures_dir / f"{save_name}_best_stats.csv",
        "checkpoint_comparison": figures_dir / f"{save_name}_checkpoint_comparison.csv",
    }
