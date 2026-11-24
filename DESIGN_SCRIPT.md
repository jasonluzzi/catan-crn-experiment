# Design Script

<!-- MAINTENANCE RULE: This file is curated by hand. Only add to or edit it when
     Jason explicitly instructs. Do NOT auto-update it during phase work. -->

Running narrative for how the experiment's design is explained (committee / paper).

---

## Current design statement (as of 2026-11-04)

We estimate the causal effect of a player's initial resource diversity on game
outcomes using paired simulations with common random numbers (CRN). Each replication
draws a seed fixing a rules-valid reference board and snake-draft order. A fixed
deterministic (argmax) agent places all initial settlements on the reference board;
that placement is recorded and replayed unchanged in both analyzed games. From the
reference we construct two analyzed boards, each a single equal-pip tile swap away:
on board A the swap raises the focal player's count of distinct adjacent resources by
one, on board B it lowers it by one (a within-pair contrast of two). Because each
swap exchanges resources between two tiles of equal pip value, every resource's
board-wide pip total — and thus board-level resource scarcity — is identical across
the reference and both analyzed boards, and the fixed number-token layout holds all
pip totals constant. Within the pair, the dice sequence, dev-card deck order, and the
hidden-card steal draw are shared via CRN; CRN cancels chance variation only until
the games' play diverges, and Phase 4 measures how much coupling survives. The
robber's destination and target remain agent decisions. Neither analyzed board is the
one the agent optimized placement against, so both are equally distant — one swap
each — from it; this is a symmetric departure, not a self-cancelling one, since board
B removes a resource the agent chose to have while board A adds one it did not plan
around. The two boards differ only by the constructed swaps, designed to hold pip
totals and board-level scarcity fixed while moving the focal player's diversity;
because changing diversity necessarily changes composition, the focal's gained/lost
resources (and the second, non-focal tile) travel with the treatment, so we log them
and include a dose-zero placebo arm and a composition balance table to bound that
channel. The estimand is the effect of a one-step increase vs. decrease in focal
diversity, over the sub-population of boards admitting the symmetric contrast (players
at moderate baseline diversity), adjusting for opponents' diversity to address
cross-player interference.
