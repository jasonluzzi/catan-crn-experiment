# Catan Testbed for Variable Importance

A self-play Catan agent and a controlled simulation design for asking whether a feature of a starting position actually drives win rate, or only appears to.

> **Project status.** The agent, the swap/CRN infrastructure, and the pip calibration that sets the minimum effect of interest are complete; the calibration result below is real and reproducible from the committed data. The diversity pilot and the main across-arms experiment have **not yet been run** — there is no diversity effect estimate yet. See [Status](#status).

## Why this exists

Variable importance numbers from observational data often can't separate a real driver from a feature that moves alongside one. I started this project to test importance measures in a setting where I control the data-generating process. Catan works well for that. Starting positions vary in measurable ways (production, resource spread, port access), the outcome is clear, and every game can be replayed under the same randomness.

## The agent

- An attention-based policy network: per-tile and per-corner multi-head self-attention plus a graph-attention network over the board graph, with hierarchical action heads (`engine/src/models/`).
- Trained with behavioral cloning, then fine-tuned with PPO through self-play (`engine/src/training/`). The training code is vendored here; the trained weights are **not** shipped (see *Running it*).
- Several trained agents (a BC-to-PPO ladder: `v4`, `v5`, `iter17`), with the top agent chosen through deterministic-policy evaluation against fixed baselines. `iter17` (PPO) is the primary agent. (The per-agent win-rates noted in `crn/agent.py` are reported from the original training runs; the evaluation output and weights are not shipped here, so they are not reproducible from this repo alone.)
- Simulation runs in parallel on CPU across worker processes. The policies are small, so self-play stays on the CPU rather than the GPU.

## The experimental design

**Question.** Does resource diversity (the entropy of a player's starting resources) affect win rate once production is held fixed?

**Equal-pip swaps.** To change diversity without changing production, the design swaps tiles that carry the same pip count. A player's expected production stays the same while their resource mix changes. Number tokens and the desert never move, so every corner's pip total — and board-level resource scarcity — is held fixed.

**Common random numbers.** Paired games share the same seed for dice, dev-card deck order, hidden-card steal draws, and snake-draft order, so the swap is the only thing that differs in how they start. This removes most of the game-to-game noise from each comparison. (CRN cancels chance only until the two games' play diverges; how much coupling survives is itself measured.)

**No unswapped baseline.** Playing the placed board without a swap would introduce placement bias, so effects are estimated as contrasts between swap arms played on the same seeds. The reference board is kept only as a placement-fit diagnostic, not in the main estimate.

**Arms.** Each arm is a single equal-pip resource swap applied to a shared reference board (numbers/desert fixed; the swap selector minimizes how many opponents it perturbs, breaking ties at random from the seed):

| arm | what it changes | focal distinct-resource count |
|---|---|---|
| **INC** | turns a *duplicated* focal resource into one the focal doesn't yet have | **+1** |
| **DEC** | turns a *uniquely held* focal resource into one the focal already has | **−1** |
| **NEU1** | swaps a duplicated focal resource for another already in the focal's set | **unchanged** |
| **NEU2** | a second, distinct neutral swap of the same kind | **unchanged** |

The primary causal contrast is **INC vs DEC** (a two-step, +2 distinct-count change), with placement recorded on the reference board and replayed identically on both. The **NEU1 vs NEU2** contrast is a dose-zero placebo: it moves composition but carries no diversity signal, so it estimates the "swap a focal tile at all" effect.

**Conditional question.** A swap changes which resources a player receives, so diversity never moves on its own. Asking whether diversity matters means asking whether it still matters given which resources were traded, which is a question of conditional independence.

**Minimum effect of interest.** Set from the win-rate (rank) change caused by a one-pip production change, estimated in a separate calibration run (`CALIBRATION.md`), so that a diversity effect is judged against an effect size that matters in play. The calibration gives δ = 0.317 rank (95% CI [0.245, 0.389]).

## Status

- [x] Agent trained
- [x] Swap and CRN infrastructure
- [x] Minimum effect of interest set (pip calibration; `CALIBRATION.md`)
- [ ] Pilot comparing variance of CRN-paired vs ordinary rollouts — **not yet run.** The infrastructure is in place (`crn` vs `independent` modes in `crn/rng.py`, with a test confirming the independent control decouples the shared streams), but no variance-ratio pilot has been executed and no result is stored.
- [ ] Full experiment across arms — not yet run. The arms are implemented and validated at demo scale (`scripts/phase3_demo.py`); a production batch runner analogous to `scripts/run_calibration.py` is not yet written.

## Running it

Run from the repository root.

```bash
# 1. Install dependencies (pin versions to match your trained checkpoints).
pip install -r requirements.txt

# 2. Provide agent checkpoints (NOT included in this repo).
#    Place your trained .pt files in ./checkpoints/ (or set CATAN_CHECKPOINT_DIR):
#      checkpoints/rl_ppo_shaped_iter17_beats_bc.pt   (iter17, PPO)
#      checkpoints/bc_v5_trading_policy.pt            (v5, BC)
#      checkpoints/bc_v4_full_policy.pt               (v4, BC)
#    The greedy/random anchor agents need no checkpoint.

# Determinism / CRN stream-separation tests (no checkpoint needed):
python tests/test_rng_streams.py          # 7/7
python tests/test_phase3_arms.py          # 5/5 (uses the scripted agent)

# Design report: diversity definitions, confounders, swap interference:
python scripts/phase2_report.py --games 300 --seed 0 --agent iter17

# Self-play / evaluation: run paired self-play games through the arms
# (Arm A and Arm B), check invariants, and time it:
python scripts/phase3_demo.py --pairs 30

# Pip calibration (sets the minimum effect of interest), CPU-parallel:
python scripts/run_calibration.py --cap 10000 --batch 500 --workers 7
python tests/test_calibration.py          # calibration validation (needs iter17)
```

Note on checkpoints. The trained agent weights are intentionally **not** included in
this repository. The scripts that use a neural agent (anything with `--agent iter17`,
and `tests/test_calibration.py`) will raise a clear "checkpoint not found" error until
you supply your own `.pt` files as described above — this is expected, not a broken
build. You do not need any weights to read the code, to run the determinism / arm-
invariant tests (`tests/test_rng_streams.py`, `tests/test_phase3_arms.py`), to use the
greedy/random anchor agents, or to inspect the committed calibration results under
`results/`.

Note on training. The pipeline that trains these agents is vendored here — behavioral
cloning, shaped-reward self-play PPO, curriculum, opponent pool, and evaluation
(`engine/src/training/`, with a runnable driver in `engine/src/models/rl_agent.py` and
`training/pipeline.py`).

## Repository layout

```
engine/src/   vendored game engine: rules, board generation + seeding, the
              attention policy, and behavioral-cloning / PPO / self-play / eval
              code. Source commit pinned in engine/PROVENANCE.txt.
crn/
  rng.py          CRN seed pairing: stream-separated numpy Generators; crn vs
                  independent modes
  layout.py       Layout (resource/number per tile) + equal-pip swap primitives
  features.py     diversity features + the equal-pip swap selectors
  arms.py         arm definitions (Arm A: INC vs DEC; Arm B: free placement; placebo)
  agent.py        deterministic (argmax) neural agent loader + greedy/random anchors
  runner.py       paired-game runner + a deterministic scripted agent
  calibration.py  pip calibration (number-token swaps) that sets the min effect
scripts/        phase2_report, phase3_demo, run_calibration
tests/          determinism/CRN, arm-invariant, and calibration tests
results/        calibration outputs (small, committed)
notes/          phase findings
configs/        (reserved)
PREREGISTRATION.md  DESIGN_SCRIPT.md  LIMITATIONS.md  CALIBRATION.md
```

## Acknowledgments

The research design is my own — the question, the CRN paired-simulation design, the
equal-pip swap treatment, the model and tensor architecture, the game rules and reward
shaping, and the pre-registration and analysis plan. The implementation was written
largely with AI assistance (Claude), working to my specification, with me reviewing the
code and verifying behavior and outputs. Responsibility for the design, the claims, and
any errors is mine.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Jason Luzzi
