# Phase 3 — arms, placement replay, invariants (findings)

Source: `python scripts/phase3_demo.py --pairs 30` (iter17) + `tests/test_phase3_arms.py` (5/5).

## The two arms (as built)
- **Arm A (primary, causal).** Agent places on a REFERENCE board; the setup action
  sequence is recorded and **replayed identically** onto two **equal-pip-swapped**
  boards — game A = increase-swap (focal div_count d+1), game B = decrease-swap
  (d−1). Neither played board is the reference ⇒ the optimal-placement advantage is
  balanced and cancels in the within-pair difference. Treatment contrast = **+2
  div_count**, verified +2 in 100% of usable pairs.
- **Arm B (descriptive).** Same two boards, agent places freely on each. Realized
  focal contrast ≈ +1.8 (not the forced +2): the agent re-optimizes, so diversity
  is endogenous. Shows the board still biases realized diversity even under free
  placement, but this arm is not a causal estimate.

## Invariants verified live (with the real agent)
- Board scarcity (per-resource board pips) **identical within every Arm-A pair**.
- **Total board pips == reference** in every pair (= 58 dots; numbers never move).
- Placement **replayed identically** on both treated boards (same settlement corners).
- CRN pair shares dice/deck/steal; independent pair shares only setup (so replay
  still aligns in the control set).

## Timing & yield
- **~1.37 s per Arm-A pair** (2 full iter17 games + a setup run) on CPU.
- **~50% of seeds usable** (symmetric equal-pip swap available for iter17 placement;
  matches the 56% pre-estimate within noise). We generate ~2× seeds to hit a target.

## Mediator logging (post-treatment, logged not adjusted)
- maritime/bank trades: ~12 per game (all 4:1 — iter17 holds no port).
- bank-scarcity (19-cap) events: ~0.07 per game — rare.
- diversity trajectory (every 10 turns): captured; e.g. focal div_count can hold at
  5 while div_entropy drifts (2.18 → ~2.0) as new builds reshape the production mix.
- port acquisition: captured via trajectory, but ≈0 for iter17 (never settles coastally).

## Suggestive (NOT inferential — demo n=15)
Focal mean rank on the increase-board 2.33 vs decrease-board 3.13 (lower = better):
a ~0.8-rank within-pair improvement for +2 diversity. Direction is plausible;
the powered estimate + CIs come in Phase 4/5.
