#!/usr/bin/env python3
"""Phase 3 end-to-end demo with the real iter17 agent: run Arm A and Arm B pairs,
confirm the invariants hold live, show a diversity trajectory, mediator logging,
and timing for Phase-4 planning.

Usage: python scripts/phase3_demo.py [--pairs 30]
"""
import sys, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "engine"))

from crn.agent import make_agent
from crn.arms import run_arm_a, run_arm_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=30)
    args = ap.parse_args()

    agent = make_agent("iter17")

    # ---- Arm A ----
    t0 = time.time()
    skips = 0; contrasts = []; scar_ok = True; totalpips_ok = True
    maritime = []; scarcity_events = []; ranks_A = []; ranks_B = []
    completed = 0; a_sample = None
    for g in range(args.pairs):
        r = run_arm_a(g, agent, focal_seat=g % 4, mode="crn", feature_every=10)
        if r["skipped"]:
            skips += 1; continue
        contrasts.append(r["focal_div_A"] - r["focal_div_B"])
        scar_ok &= (r["board_scarcity"]["A"] == r["board_scarcity"]["B"])
        totalpips_ok &= (r["board_scarcity"]["A"]["board_total_pips"]
                         == r["board_scarcity"]["ref"]["board_total_pips"])
        f = r["focal"]
        ranks_A.append(r["gameA"]["rank"][f]); ranks_B.append(r["gameB"]["rank"][f])
        maritime += r["gameA"]["maritime_trades"] + r["gameB"]["maritime_trades"]
        scarcity_events += [r["gameA"]["bank_scarcity_events"],
                            r["gameB"]["bank_scarcity_events"]]
        completed += int(r["gameA"]["completed"]) + int(r["gameB"]["completed"])
        if a_sample is None:
            a_sample = r
    dt = time.time() - t0
    usable = args.pairs - skips

    print(f"\n=== Arm A: {args.pairs} pairs (iter17), {dt:.1f}s "
          f"({dt/max(args.pairs,1):.2f}s/pair) ===")
    print(f"  usable (symmetric equal-pip available): {usable}/{args.pairs} "
          f"({usable/args.pairs:.0%}); skipped {skips}")
    print(f"  focal div_count contrast (A-B): all == +2? "
          f"{all(c == 2 for c in contrasts)}  (mean {np.mean(contrasts):.2f})")
    print(f"  board scarcity identical within pair: {scar_ok}; "
          f"total pips == reference: {totalpips_ok}")
    print(f"  games completed (reached a winner): {completed}/{2*usable}")
    print(f"  focal mean rank: increase-board {np.mean(ranks_A):.3f} vs "
          f"decrease-board {np.mean(ranks_B):.3f} (lower=better; DEMO n, not inferential)")
    print(f"  mediators: maritime trades/game mean {np.mean(maritime):.2f}; "
          f"bank-scarcity events/game mean {np.mean(scarcity_events):.2f}")

    # Diversity trajectory of the focal in the sample pair's increase game.
    if a_sample is not None:
        f = a_sample["focal"]
        traj = a_sample["gameA"]["trajectory"]
        pts = [(s["turn"], s["players"][f]["div_count"],
                round(s["players"][f]["div_entropy"], 2)) for s in traj]
        print(f"\n  sample focal diversity trajectory (increase game), "
              f"(turn, div_count, div_entropy):")
        print("   ", pts)

    # ---- Arm B (descriptive): how different is free placement? ----
    t0 = time.time(); b_focal_div = []; a_focal_div = []; nb = min(args.pairs, 12)
    for g in range(nb):
        rb = run_arm_b(g, agent, focal_seat=g % 4, mode="crn", feature_every=0)
        if rb["skipped"]:
            continue
        b_focal_div.append((rb["focal_div_A"], rb["focal_div_B"]))
        ra = run_arm_a(g, agent, focal_seat=g % 4, mode="crn", feature_every=0)
        if not ra["skipped"]:
            a_focal_div.append((ra["focal_div_A"], ra["focal_div_B"]))
    dt = time.time() - t0
    print(f"\n=== Arm B (free placement), {nb} pairs, {dt:.1f}s ===")
    print(f"  Arm A forces focal (A,B) div_count = (d+1, d-1), contrast +2 by design.")
    if b_focal_div:
        bc = [a - b for a, b in b_focal_div]
        print(f"  Arm B realized focal (A,B) div_count contrast: mean {np.mean(bc):+.2f}, "
              f"values {b_focal_div[:6]}")
        print("  -> under free placement the agent re-optimizes, so the forced "
              "diversity contrast is NOT preserved (endogenous; descriptive only).")


if __name__ == "__main__":
    main()
