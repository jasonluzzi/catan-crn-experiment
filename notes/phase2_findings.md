# Phase 2 findings — treatment definitions & confounders

Source: `python scripts/phase2_report.py --games 300` (iter17 self-play, setup-only,
reference layouts, 1,200 player-rows). Reproducible from seed 0.

## Two diversity definitions
- **div_count** — # distinct resource types adjacent to a player's settlements (2–5).
- **div_entropy** — Shannon entropy (bits) of expected production over the 5
  resources, weighted by pips (production_probability). Range 0–log2(5)=2.32.

**Correlation:** Pearson 0.891, Spearman 0.800. Strongly related but NOT redundant:
div_count ignores how balanced production is, div_entropy captures balance (e.g.
touching 4 resources with one dominant tile scores lower entropy than 4 balanced).
Keep both as the design's robustness check; use div_count as the headline treatment
(interpretable) and div_entropy as the continuous alternative.

## Distributions (reference, end of initial placement)
- div_count: mean 3.74, sd 0.70; histogram over 0..5 = [0,0,35,385,634,146].
  Nearly all players touch 3–5 resources; 2 is rare, 5 uncommon.
- div_entropy: mean 1.69 bits, sd 0.31.
- total_pips (production_probability units): mean 0.512 (~18.4 raw pips over 2
  settlements), sd 0.076.
- Ports: **0** generic and **0** matching across all 1,200 rows (see below).

## Treatment variation (what the shuffle actually moves)
- **Full reshuffle, fixed placement:** |Δ div_count| mean 0.72 (max 3); **38.9%
  unchanged**. |Δ div_entropy| mean 0.334 (moves more often, being continuous).
- **Targeted swap** (chooses 2 tiles to shift focal div_count by ±1):
  - available in 87.3% (increase) / 95.3% (decrease) of games,
  - focal Δ div_count exactly ±1 in **100%** of available cases,
  - interference on the other 3 players: ~0.7/3 change on average, mean |Δ| ≈ 0.24.
  So the targeted swap is a far more surgical instrument than the full reshuffle,
  but interference is nonzero (swapping a tile touches any adjacent player).

## Consequences for the analysis
1. **Ports are degenerate under iter17 placement** — iter17 never settles coastally.
   Drop port covariates in agent-placement arms; they only matter for weaker agents.
2. **Interference must be adjusted for** (opponents' diversity in the estimator) —
   confirmed non-negligible, especially under full reshuffle.
3. **div_count coarseness caps power** (39% of reshuffles don't move it); report
   div_entropy alongside, and prefer the targeted swap when a clean ±1 focal
   contrast is wanted.
