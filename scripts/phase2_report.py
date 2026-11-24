#!/usr/bin/env python3
"""Phase 2 report: diversity definitions, their correlation, confounders, and the
targeted-swap generator (+ interference).

Runs N games through INITIAL PLACEMENT only (fast) with iter17 self-play on random
reference layouts, computes both diversity measures + confounders for all players,
and reports:
  1. Correlation (Pearson + Spearman) between div_count and div_entropy.
  2. Distributions of div_count, div_entropy, total_pips, port access.
  3. Full-reshuffle vs reference: how much each player's diversity moves.
  4. Targeted-swap validation: does it change focal div_count by exactly +/-1, and
     what is the collateral (interference) effect on the other three players?

Usage: python scripts/phase2_report.py [--games 300] [--seed 0]
"""

import sys
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from crn.rng import spawn_pair
from crn.layout import generate_reference_layout, full_reshuffle
from crn.runner import run_setup
from crn.agent import make_agent
from crn.features import (
    compute_all_features, features_on_layout, focal_tiles, choose_diversity_swap,
)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def summarize(name, vals):
    v = np.asarray(vals, float)
    print(f"  {name:16s} n={len(v):5d}  mean={v.mean():.3f}  sd={v.std():.3f}  "
          f"min={v.min():.3f}  p50={np.median(v):.3f}  max={v.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", default="iter17")
    args = ap.parse_args()

    agent = make_agent(args.agent)
    agents = [agent] * 4

    div_count, div_entropy, total_pips = [], [], []
    generic_port, matching_ports = [], []
    reshuffle_dcount_abschg, reshuffle_dentropy_abschg = [], []

    swap_avail = {"increase": 0, "decrease": 0}
    swap_focal_delta = {"increase": [], "decrease": []}
    swap_interf_nchanged = {"increase": [], "decrease": []}   # # of other players changed
    swap_interf_absdelta = {"increase": [], "decrease": []}   # mean |delta| over others
    n_tried = 0

    base = args.seed * 1_000_000
    for g in range(args.games):
        rng_a, rng_b, layout_gen = spawn_pair(base + g, mode="crn")
        ref = generate_reference_layout(layout_gen)

        state = run_setup(agents, rng_a, ref)
        feats_ref = compute_all_features(state)
        settlements = [list(p.settlements) for p in state.players]
        cities = [list(p.cities) for p in state.players]

        for f in feats_ref:
            div_count.append(f["div_count"])
            div_entropy.append(f["div_entropy"])
            total_pips.append(f["total_pips"])
            generic_port.append(int(f["has_generic_port"]))
            matching_ports.append(f["n_matching_ports"])

        # (3) Full reshuffle under the SAME placements (Arm-A style): diversity move.
        reshuffled = full_reshuffle(ref, layout_gen)
        feats_resh = features_on_layout(settlements, cities, reshuffled)
        for fr, fs in zip(feats_ref, feats_resh):
            reshuffle_dcount_abschg.append(abs(fs["div_count"] - fr["div_count"]))
            reshuffle_dentropy_abschg.append(abs(fs["div_entropy"] - fr["div_entropy"]))

        # (4) Targeted swap on focal = seat (g % 4), both directions.
        n_tried += 1
        focal = g % 4
        ftiles = focal_tiles(state, focal)
        for direction in ("increase", "decrease"):
            res = choose_diversity_swap(ref, ftiles, direction, layout_gen)
            if res is None:
                continue
            swap_avail[direction] += 1
            new_layout, _rec = res
            feats_new = features_on_layout(settlements, cities, new_layout)
            swap_focal_delta[direction].append(
                feats_new[focal]["div_count"] - feats_ref[focal]["div_count"])
            others = [i for i in range(4) if i != focal]
            deltas = [feats_new[i]["div_count"] - feats_ref[i]["div_count"] for i in others]
            swap_interf_nchanged[direction].append(sum(1 for d in deltas if d != 0))
            swap_interf_absdelta[direction].append(float(np.mean(np.abs(deltas))))

    print(f"\n=== Phase 2 report: {args.games} games, agent={args.agent}, "
          f"{len(div_count)} player-rows ===\n")

    print("[1] Correlation between the two diversity definitions (player-level):")
    print(f"  Pearson(div_count, div_entropy)  = {pearson(div_count, div_entropy):.3f}")
    print(f"  Spearman(div_count, div_entropy) = {spearman(div_count, div_entropy):.3f}\n")

    print("[2] Distributions (reference layouts, end of initial placement):")
    summarize("div_count", div_count)
    summarize("div_entropy(bits)", div_entropy)
    summarize("total_pips", total_pips)
    summarize("has_generic_port", generic_port)
    summarize("n_matching_ports", matching_ports)
    bc = np.bincount(np.asarray(div_count), minlength=6)
    print(f"  div_count histogram (0..5): {list(bc[:6])}\n")

    print("[3] Full reshuffle vs reference, SAME placements (how much treatment moves):")
    summarize("|d div_count|", reshuffle_dcount_abschg)
    summarize("|d div_entropy|", reshuffle_dentropy_abschg)
    frac0 = np.mean(np.asarray(reshuffle_dcount_abschg) == 0)
    print(f"  fraction with UNCHANGED div_count after reshuffle: {frac0:.3f}\n")

    print("[4] Targeted swap validation + interference (focal seat rotated):")
    for direction in ("increase", "decrease"):
        avail = swap_avail[direction]
        print(f"  direction={direction}: available in {avail}/{n_tried} games "
              f"({avail/n_tried:.1%})")
        if avail:
            fd = np.asarray(swap_focal_delta[direction])
            want = +1 if direction == "increase" else -1
            print(f"    focal div_count delta: mean={fd.mean():+.3f} "
                  f"(target {want:+d}); exact-hit rate={np.mean(fd == want):.1%}")
            print(f"    interference: mean #other-players-changed="
                  f"{np.mean(swap_interf_nchanged[direction]):.2f}/3, "
                  f"mean |delta| over others={np.mean(swap_interf_absdelta[direction]):.3f}")
    print()


if __name__ == "__main__":
    main()
