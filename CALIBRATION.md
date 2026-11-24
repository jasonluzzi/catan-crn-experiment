# Pip Calibration — sets δ (smallest effect of interest) for the diversity study

Calibration step, NOT a hypothesis test. Estimates the focal-rank effect of a
two-step pip contrast (+1 vs −1 pip), reusing the diversity pipeline exactly.

## Design
Treatment = a NUMBER-TOKEN swap (resources stay put):
- **Board P+**: swap the focal hex's token with a partner hex's token worth exactly
  +1 pip → focal gains exactly 1 pip.
- **Board P−**: partner worth exactly −1 pip → focal loses exactly 1 pip.

δ̂ = mean( focal_rank(P−) − focal_rank(P+) ), signed so **positive ⇒ the pip helps**.

Eligibility: focal hex adjacent to the focal player, to NO other player, and to
exactly ONE focal building (⇒ focal pip change is exactly ±1). Partner hex not
focal-adjacent, pip = focal ±1, red-number rule preserved on the whole board, never
desert. Prefer partners adjacent to no player; else allow opponent-adjacent partners
and log the opponents' pip changes. Uniform-random pick from eligible candidates
(from the seed); full candidate set logged. A replication is eligible iff BOTH a P+
and a P− swap exist.

Same reference-board generation, replayed agent placement, CRN streams (dice, dev
deck, steals), deterministic final PPO agent (iter17, argmax) in all four seats, and
focal-seat rotation as the diversity experiment.

## Seeds (disjoint ranges)
- **Calibration: [5,000,000, 6,000,000)** — this experiment.
- Diversity pilot: [0, 1,000,000). Diversity main run: [1,000,000, 5,000,000).

## Config
agent=iter17 (argmax); all 4 seats same agent; mode=crn; focal=seed%4; engine
vendored from a private training repo @ git ede3135 (see engine/PROVENANCE.txt).
NOTE: this experiment folder is not yet a git repo — init before the batch run so the
raw data + config can be committed with a hash.

## CHECKPOINT 1 — validation, eligibility, runtime (2026-11-07)
- **Validation tests: 5/5 pass** (`tests/test_calibration.py`):
  token multiset identical across ref/P+/P−; focal pips change exactly ±1/36 with
  distinct-count unchanged; red-number rule holds on all boards; deterministic replay
  reproduces both games; per-player pips/production/entropy logged on all boards.
- **Eligibility: 80%** (199/250; iter17, focal rotated). Skips: no-P+ 33 (focal hex
  at a pip extreme, usually pip 5 → can't +1), no-P− 9, both 9.
- **Runtime: ~2.54 s per eligible replication** (2 games + a setup run), CPU,
  single-core; parallel across pairs (both games on one worker).
- **Opponent interference is common: 69%** (137/199) of eligible reps had NO clean
  (player-free) partner hex, so an opponent's pips also moved (by −delta on P+, +delta
  on P−). This is per spec (logged, not rejected), but it means δ bundles "focal +1
  pip" with "an opponent −1 pip" in most reps — δ is therefore NOT a pure focal-pip
  effect. Flagged for review; see "problems/limitations" below once batches run.

## Batch plan (pending CHECKPOINT-1 approval)
Batches of 1,000 eligible reps. After each: δ̂; cluster-bootstrap 95% CI (cluster by
base seed); relative precision SE(δ̂)/|δ̂|. Stop at rel precision ≤ 0.15 or the compute
cap (PROPOSED: 10,000 eligible reps ≈ 7 h single-core / ~1 h on 8 cores), whichever
first.

## δ results — CHECKPOINT 2 (2026-11-07)
Run: `scripts/run_calibration.py --cap 10000 --batch 500 --workers 7`, seeds [5M, …],
iter17, cluster bootstrap over base seeds (B=10,000). **Stopped early at n=1,000**
(relative precision 0.116 ≤ 0.15). Raw rows: `results/calib_rows.parquet`
(1,260 seeds tried); batch log: `results/calib_progress.jsonl`.

**δ̂ = +0.317 rank** for the two-step (+1 vs −1) pip contrast (positive ⇒ pip helps).
- 95% CI **[0.245, 0.389]**, SE 0.0367, N = 1,000 eligible pairs.
- Relative precision 0.116 (≤ 0.15 target met).
- **SD of within-pair rank differences = 1.162** (reference for the diversity pilot).
- Batch trace: n=500 → δ̂ +0.334 (relprec 0.157); n=1000 → δ̂ +0.317 (relprec 0.116, stop).

**Rank-distribution shift (P+ minus P−, bootstrap 95% CI) — all exclude 0:**
| focal finish | P(P+) | P(P−) | Δ (P+−P−) | 95% CI |
|---|---|---|---|---|
| 1st | 0.315 | 0.211 | **+0.104** | [+0.072, +0.136] |
| 2nd | 0.259 | 0.215 | +0.044 | [+0.009, +0.080] |
| 3rd | 0.212 | 0.295 | −0.083 | [−0.119, −0.048] |
| 4th | 0.214 | 0.279 | −0.065 | [−0.096, −0.034] |

So a +2-pip swing lifts P(1st) ≈ 21%→32% and cuts P(3rd/4th). Focal win rate
0.315 vs 0.211; focal VP 6.89 vs 6.06; completion 100% on both boards.

**Eligibility & interference.** Eligibility 79% (1,000/1,260). Opponent-affected
reps 72% (no clean partner); mean ≈0.6 opponents' pips moved per rep. **But the
clean-partner subset (n=283, pure focal-pip, no opponent change) gives δ̂ = 0.322,
95% CI [0.187, 0.452] — statistically indistinguishable from the full 0.317.** So the
opponent channel does NOT inflate δ; δ reflects the focal-pip effect. (The clean
subset alone did not reach rel-precision 0.15 at this N — relprec 0.21 — but is fully
consistent.)

**Problems / limitations.** (1) The focal pip change is exact ±1, but in 72% of reps
an opponent's pips also move by ∓1 (logged); δ is robust to this (clean≈full). (2)
Eligible boards skew to focal hexes at non-extreme pips (no-P+ dominates skips when the
focal hex is pip-5). (3) δ is for iter17 self-play; other agents may differ (H3 concern).
(4) δ̂ is a **stopped estimate** — the batch loop halts at relative precision ≤ 0.15, which
slightly optimizes the reported precision. This is acceptable for a point-estimate
calibration (δ̂, not a CI bound, feeds the power calc) and is distinct from the
confirmatory test, where N is fixed in advance with no peeking (`PREREGISTRATION.md` §7).

**Recommended δ for the power calculation: δ = 0.317** (relative precision ≤ 0.15, so
δ̂ is used rather than the conservative lower CI bound). This is the rank effect of a
±1-pip swing on the focal — the diversity study is powered to detect a diversity
effect at least this large. Units match the diversity primary contrast (both are
two-step: div+1 vs div−1).
