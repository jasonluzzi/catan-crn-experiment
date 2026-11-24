# Checkpoints (not included)

Trained agent weights are **not** shipped with this repository.

Place your trained policy checkpoints here (or point `CATAN_CHECKPOINT_DIR` at
another directory):

- `rl_ppo_shaped_iter17_beats_bc.pt` — `iter17` (PPO, primary agent)
- `bc_v5_trading_policy.pt` — `v5` (behavioral cloning)
- `bc_v4_full_policy.pt` — `v4` (behavioral cloning)

The greedy/random anchor agents (`crn/agent.py`) need no checkpoint. The
checkpoint-free tests (`tests/test_rng_streams.py`, `tests/test_phase3_arms.py`)
run without any of the above.

The `.pt` files themselves are git-ignored; this directory is kept via its README.
