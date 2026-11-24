# LIMITATIONS — running log

Anything that weakens the causal interpretation, breaks CRN coupling, or deviates
from the design. Updated as issues are found. (Started Phase 1, 2026-11-03.)

## Coupling / CRN caveats
- **"Steal" is only the hidden-card draw.** The robber's destination tile and the
  victim are AGENT actions (not seeded) and may diverge freely between paired games.
  The seeded steal stream governs ONLY which card is pulled from the chosen victim's
  hand.
- **Steal coupling decays with hand divergence.** The steal draw maps one uniform
  (shared stream) onto `available` = the victim's cards ordered by resource type.
  The same uniform picks the same *resource* across paired games only while the
  victim's hand composition matches. Once hands diverge, the same uniform maps to
  possibly different resources. This is an expected, measured form of coupling
  decay, not a bug.
- **Dev-card draws diverge even though deck ORDER is shared.** Both games in a CRN
  pair get a bit-identical shuffled deck, but players buy cards at different times
  once play diverges, so the actual cards drawn diverge. Expected and part of what
  Phase 4 measures.
- **Agent-level randomness for baseline anchors is coupled only at the per-game
  level, not via injected generators.** `baseline.py` (greedy/random) uses the
  global `np.random` pervasively (tie-break jitter, exploration, choice). Rather
  than rewrite it, the runner will seed `np.random` (and torch) once per game from
  the pair's shared stream — identical seed for both CRN games, different for the
  independent control. The *focal* experiment agents are neural **deterministic
  argmax**, which consume NO randomness, so this only affects Phase 6 anchors.

## Engine / scope
- **MCTS is out of scope.** Only flat deterministic policies are used. If MCTS
  agents are added later, determinization re-randomizes hidden info (opponent
  hands, unseen deck) on its own streams and would partially break CRN coupling.
- **`GameState.copy()` shares the rng object by reference.** The CRN runner never
  copies state during play, so this is harmless here; but a copied state would
  advance the original's streams. Would need per-stream state cloning if MCTS
  (which copies heavily) is ever wired in.

## Outcome definition
- **Final VP uses `total_victory_points`** (includes hidden VP dev cards revealed
  at game end), matching the engine's win check (`check_winner` at >= 10). The
  visible VP is logged separately as `vp_visible`. The in-game observation does NOT
  reveal opponents' hidden VP cards; using total VP as the *outcome* is correct
  (it's the end-of-game truth) but note it is not what agents saw while playing.
- **Rank ties broken by lower seat id** — arbitrary but deterministic. Ties at the
  top (shared max VP without a declared winner) are rare but possible if a game
  hits the step cap; such games are flagged `completed=False` and should be
  excluded or handled explicitly in analysis.

## Phase 2 findings that constrain analysis
- **Port access is ~always zero under iter17 placement.** Across 1,200 player-rows
  (300 games), iter17 never settled on a port corner (the feature is verified
  correct: the random agent hits ports in ~22–34% of rows). So in any arm where
  iter17 chooses placement (Arm B, and Arm A seeded from iter17 placements), the
  port covariates are degenerate (no variation) and cannot be used. Ports only
  become an informative covariate if a coastal-settling (weaker) agent is used.
- **Interference is real and measured, not hypothetical.** A targeted focal swap
  changes a non-focal player's div_count in ~0.7 of 3 opponents on average
  (mean |delta| ≈ 0.24); a full reshuffle changes everyone. The estimator must
  include opponents' diversity (it does, per design) — single-player analysis
  would be confounded.
- **~39% of full reshuffles leave the focal player's div_count unchanged** (given
  fixed placement). div_count is coarse (integer 2–5); div_entropy is continuous
  and moves more often. This caps power when using div_count as the treatment and
  motivates reporting both definitions (design robustness check).

## Phase 3 compromises & notes
- **Overlap restriction (primary estimand).** The equal-pip symmetric swap exists
  for only ~50% of (board, placement) configs under iter17. We skip the rest, so the
  Arm-A primary estimand is the diversity effect **over the sub-population admitting
  the contrast**, not all boards. Need to check (Phase 4/5) whether the usable subset
  differs systematically (e.g. in reference div_count or total pips) and report it.
- **Placement replay is valid** because setup legality (distance rule, road
  adjacency) is resource-independent; only the 2nd settlement's starting resources
  differ across the pair, which is the treatment correctly flowing through. Verified
  identical settlement corners on both treated boards.
- **Mediators are near-constant for iter17.** Port acquisition ≈ 0 (never settles
  coastally), maritime trades are all 4:1, bank-scarcity events ~0.07/game. These
  become informative only for weaker/coastal agents (Phase 6). Logged regardless.
- **Interference persists under equal-pip swaps.** The swap holds board *totals*
  fixed, but the specific non-focal tile changes resource, so whoever shares that
  tile is affected. Logged via all-four-player initial features.
- **Unit note:** `board_*_pips` are in raw dots; player `total_pips`/`exp_prod_*`
  are in production_probability (dots/36). Different scales, each internally
  consistent; entropies are scale-invariant.

## The A-B contrast moves composition too, not just diversity (primary caveat)
Changing the focal player's diversity REQUIRES changing its composition, so A and B
differ by more than "diversity +1 vs -1": on board A the focal gains a missing
resource and loses a duplicate; on board B it loses a unique resource and gains a
duplicate. Which specific resources are involved (and any port match) travel with
the treatment. This cannot be fully removed. How we handle it, honestly:
- **Log it.** Each pair records the resource gained/lost on each board, the swap's
  full candidate set, port matches, and the pip-weighted entropy change
  (`treatment` + `swap_records`). The pilot reports a balance table across pairs; if
  ore/grain show up disproportionately on one side, it will be visible.
- **Dose-zero placebo arm** (`run_placebo`): two distinct equal-pip neutral swaps
  that change the focal's resources but leave div_count unchanged. Its A-B
  difference has zero diversity signal, so it estimates the "swap a focal tile at
  all" (pure composition) effect. If that is large relative to the Arm-A effect,
  much of A-B is composition, not diversity. This is the closest thing to a placebo.

## Honest-framing corrections (do NOT overclaim in the write-up)
- NOT "the sole systematic difference is the focal's diversity." The boards differ
  by the constructed swaps (designed to hold pip totals + board-level scarcity fixed
  while moving the focal's diversity); composition and the second (non-focal) tile
  also move.
- NOT placement advantage "cancels." Both analyzed boards are equally DISTANT (one
  swap each) from the placement-optimization board, but the two departures aren't
  interchangeable (B removes a resource the agent chose; A adds one it didn't plan
  around). Say "symmetric in distance," not "cancels."
- NOT "purged of chance variation." CRN cancels chance only UNTIL the games diverge;
  Phase 4 measures how much survives. Don't assume it in the methods section.

## Estimand scope & dose (report explicitly)
- **Eligibility.** Only boards admitting the symmetric equal-pip contrast are kept
  (~50% under iter17). Excluded: focal div 5 (can't raise), div 1 (can't lower), or
  no equal-pip candidate. We log `baseline_div` + skip `reason` for ALL draws so the
  pilot can report included-vs-excluded baseline-diversity distributions. Effective
  estimand: the d-1 vs d+1 effect for players starting at MODERATE d (mostly 3-4).
- **Nonlinearity.** A d+1 vs d-1 contrast averages "gain a resource" and "lose a
  resource," which need not be equal in size (4->5 differs from 3->2). Report the
  estimate stratified by `baseline_div`.
- **Secondary dose.** div_count change is always +/-2, but the pip-weighted entropy
  change varies by swap (`treatment.delta_div_entropy`); logged for a dose-response
  check.
- **Agent mode.** The flat BC/PPO ladder (v4/v5/iter17) is correctly run at argmax
  (the project's "always eval deterministically" rule); these are NOT search agents.
  We exclude the MCTS/ExIt line precisely because it needs search to be strong, so
  "argmax isn't the strongest mode" does not apply to the agents we use.

## Design confounds (tracked, not hidden — see the design doc)
- **Composition:** a shuffle changes *which* resources a player gets, not only how
  diverse they are. Phase 2 logs per-resource expected production so the analysis
  can adjust for it.
- **Interference:** one shuffle changes every player's diversity at once. Phase 2
  logs all four players' treatment values per game.

## Fixed during the refactor (for the record)
- The stale engine coupled resource-shuffle -> desert position -> number placement,
  which would have changed pip totals when resources were reshuffled. The `Layout`
  object holds the desert tile + number tokens fixed and permutes only resources,
  so corner pip totals are invariant within a pair (verified by
  `test_pip_totals_invariant_under_reshuffle`).
- The original `env.reset(seed)` seeded only `np.random`, while the engine drew
  board/dice/deck/steal from Python's `random` — so the seed never actually
  reproduced a game. The refactor injects explicit per-stream numpy Generators.
