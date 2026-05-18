# INFO8003-1 RL Project — Maio-Rutten

This repository packages our final code, trained models, and Dogfight
submissions for the PyFlyt UAV reinforcement learning project.

## Main Contents

- `task1_hover_ppo_RL_1805.ipynb`: PPO on Hover
- `task1_hover_sac.ipynb`: SAC on Hover
- `task2_waypoints_ppo_RL_final_1805.ipynb`: PPO on Waypoints
- `task2_waypoints_sac.ipynb`: SAC on Waypoints
- `task3_dogfight_ppo.ipynb`: PPO three-phase self-play / league training
- `training_cell_*.py`, `_train_worker_dogfight.py`: notebook helper modules
- `scripts/`: required shared support files (`env_config.py`, `wrappers.py`,
  `dogfight_wrapper.py`, `evaluate.py`, `tournament.py`,
  `submission_template.py`)
- `results/selected_models/`: selected best SAC Hover and SAC Waypoints
  checkpoints
- `results/submissions/`: final Dogfight submission files
- `project_statement/`: project statement PDF and LaTeX source

All shared utilities live under `scripts/`.
The `results/` folder is intentionally minimal and only keeps the final
submission checkpoints.

## Installation

```bash
pip install -r requirements.txt
```

## Notes on Models

- Best SAC Hover checkpoint:
  `results/selected_models/SAC_hover_best_model.zip`
- Best SAC Waypoints checkpoint:
  `results/selected_models/SAC_waypoints_best_model.zip`
- Best Dogfight submission:
  `results/submissions/groupMaioRutten_league.zip`

The best PPO checkpoints for Hover and Waypoints are missing from this
repository.
Those runs were executed in Colab, and the best checkpoint files were
unintentionally not saved back to the repository. The PPO notebooks, helper
code, and reported figures are still included.

## Evaluation

```bash
python scripts/evaluate.py --model results/selected_models/SAC_hover_best_model.zip --env hover
python scripts/evaluate.py --model results/selected_models/SAC_waypoints_best_model.zip --env waypoints --flight_mode 0
python scripts/tournament.py results/submissions
```
