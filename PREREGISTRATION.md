# Pre-Registration — Resource-Diversity Causal Effect (Catan CRN study)

**Status: design locked before the calibration and main runs.** Dated 2026-11-07, in
the private development repository (commit "Lock design + pre-registration before main
run", which precedes the calibration-complete commit). This public repository is a
squashed single-commit release, so that ordering is attested here rather than provable
from this repo's git history; the development history is retained privately and can be
shared on request. The diversity main run and the calibration run use disjoint seed
ranges fixed below. Design narrative: `DESIGN_SCRIPT.md`. Known caveats: `LIMITATIONS.md`.

## 1. Setup (fixed in advance)
- Engine: vendored from a private training repo @ git `ede3135` (`engine/PROVENANCE.txt`).
- Agent: deterministic **argmax**; all four seats the same policy. Primary agent =
  **iter17** (PPO). H3 also uses **v5** (best BC) and **v4** (BC base); greedy/random as
  weak anchors for H3a.
- CRN board set per seed = **quintet** `{reference, div+1 (INC), div−1 (DEC), NEU1, NEU2}`,
  all built from a shared reference board with the agent's placement recorded and
  replayed; INC/DEC/NEU1/NEU2 are each a single **equal-pip** resource swap off the
  reference (board-level scarcity + all pip totals held fixed). The two NEU boards are
  neutral swaps (distinct count unchanged) enabling a zero-diversity placebo contrast.
- One **focal player per seed, rotated** (`focal = seed % 4`). Shared streams: dice,
  dev-deck order, hidden-card steal; robber tile/victim are agent decisions.
- Seeds: diversity pilot **[0, 1,000,000)**, diversity main **[1,000,000, 5,000,000)**,
  calibration **[5,000,000, 6,000,000)** (disjoint).

## 2. Outcomes
- **Primary: final rank** (1 = best … 4 = worst), focal player.
- Secondary (pre-specified): final VP (`total_victory_points`), win (rank==1).

## 3. Estimator & inference (fixed)
Board-level regression with **seed fixed effects** over the 4 one-swap boards
(INC, DEC, NEU1, NEU2 — the reference is EXCLUDED from the main slope because its
placement perfectly fits it; it is kept only as a placement-fit diagnostic):

  outcome_{s,b} = α_s + β·divcount_{s,b} + (adjustments) + ε_{s,b}

- α_s absorbs everything CRN holds fixed (board, dice, deck, steal, placement).
- **Adjustments:** a composition summary (focal expected ore+grain production, and the
  identity of the gained/lost resource) and opponents' mean diversity (interference).
  These are pre-treatment / design covariates, NOT mediators.
- **Interference handling (added 2026-11-09).** A resource swap also changes the
  non-focal ("other") tile, perturbing the composition of any opponent adjacent to it
  (empirically ~97% of pairs touch >=1 opponent). We do NOT exclude opponent-touching
  swaps (a clean, opponent-free swap exists only ~19% of the time; requiring it would
  make eligibility ~infeasible and the estimand unrepresentative). Instead, among
  eligible swaps the selector MINIMIZES opponent interference (fewest distinct
  opponents touched), breaking remaining ties uniformly at random from the seed. This
  cuts mean opponents-affected from ~1.42 (uniform) to ~1.02 (chosen == min possible).
  Per-swap interference (count + opponent ids + candidate set) is logged.
  **Robustness subset:** the main estimate is reported on all eligible pairs AND on the
  opponent-free subset (interference == 0 on both boards), mirroring the calibration's
  clean-partner check, to confirm the result is not driven by opponent perturbation.
- **Inference: cluster bootstrap over seeds** (resample base seeds; B = 10,000),
  report 95% CIs. **Two-sided** throughout (report the sign).
- Mediators (port/maritime trades, bank-scarcity events, diversity trajectory) are
  logged for mechanism only and are NOT in the adjustment set.

## 4. Hypotheses (pre-registered)

**H1a — PRIMARY / confirmatory.** Higher focal distinct-resource count → better (lower)
rank. Test: **β < 0** (two-sided CI) in the model above. Also reported on VP (β>0) and
win (β>0). This is the single confirmatory test; power is set to detect δ (§6).

**H1b — secondary.** Diminishing returns as distinct count rises (e.g. 2→3 > 4→5). Test:
add a quadratic `div²` (or compare the d−1→d vs d→d+1 slopes); the NEU boards supply the
within-seed mid-level anchor. Also reported by stratifying β on baseline d (3 vs 4).

**H1c — exploratory.** The diversity effect is weaker when the swap breaks an OWS
(ore/grain/wool) set or removes a resource the focal has a 2:1 port for. Test:
interaction `β·divcount × {OWS-break / port-removal}` from logged gained/lost resources.
NOTE: the port moderator is ~untestable with iter17 (never holds ports); OWS moderator
is testable. Low power (interaction).

**H1d — secondary.** Change in pip-weighted entropy predicts outcomes better than change
in distinct count. Test: non-nested comparison of the seed-FE model with `diventropy`
vs `divcount` as the predictor (CV residual / AIC; or both included). The NEU boards
(same count, different entropy) provide the identifying variation; the two measures are
~0.89 correlated.

**H2 — secondary.** Among swaps involving comparable resource changes, count-changing
swaps (INC/DEC) shift outcomes more than non-count-changing swaps. Test: compare the
count-changing contrasts to the **neutral placebo contrast NEU1−NEU2** (zero diversity
change), adjusting for composition. Exact resource-matching is approximate (logged).

**H3 — separate family (policy dependence).**
- **(a) secondary:** the diversity effect (β) is negative (helps) for *each* agent in the
  ladder (v4, v5, iter17; greedy/random anchors). Per-agent β with CIs.
- **(b) two-sided interaction:** the effect differs between the BC agent and the PPO
  agent. Primary contrast **v5 (BC) vs iter17 (PPO)**; test `β·divcount × agent`.
  POWER: an interaction of a given size needs ~4× the N of a main effect → H3b uses a
  larger run than H1a (sized in §6).

## 5. Placebo / falsification (pre-specified)
- **Dose-zero placebo** (NEU1−NEU2): zero diversity contrast; a large effect here would
  indicate the H1a effect is composition, not diversity. Reported alongside H1a.
- **Pip calibration** (`CALIBRATION.md`): separate number-token study estimating δ.

## 6. Effect size, power, eligibility
- **δ (smallest effect of interest)** = the focal-rank effect of a two-step (+1 vs −1)
  **pip** contrast, from the calibration run (`CALIBRATION.md`), reviewed before it
  enters the power calc. Rationale: a diversity effect smaller than a ±1-pip swing is
  not practically meaningful.
- **Pilot:** 300 usable pairs → estimate the CRN variance ratio and the within-pair rank
  SD → power H1a to detect δ, and H3b (≈4×) → set the main N. No outcome-dependent
  stopping for the confirmatory test; N fixed from the power calc before the main run.
- **Eligibility / overlap:** only boards admitting the symmetric equal-pip quintet are
  analyzed (~50–54% under iter17; usable baseline d ∈ {3,4}). The estimand is the
  d−1↔d+1 effect for players at **moderate baseline diversity**. Included-vs-excluded
  baseline-d distributions are reported (`baseline_div` + skip reason logged for all).

## 7. Locked analysis decisions
Two-sided tests; rank primary; seed-FE estimator; cluster bootstrap over seeds
(B=10,000); H1a sole confirmatory with Holm control within {H1b–d, H2} as a secondary
family and H3 a separate family; reference board excluded from the main slope; mediators
excluded from adjustment; fixed N from the pilot power calc (no peeking for H1a).
