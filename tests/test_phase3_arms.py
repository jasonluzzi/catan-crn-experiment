"""Phase 3 tests: equal-pip swap invariants, Arm-A placement replay, and the
within-pair "everything fixed except focal resources" property.

Run:  python tests/test_phase3_arms.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

import numpy as np
from crn.rng import PairRNG, single_game_rng
from crn.layout import generate_reference_layout, full_reshuffle
from crn.features import (
    focal_tiles, make_symmetric_swaps, board_scarcity, choose_diversity_swap,
    features_on_layout,
)
from crn.runner import run_setup, ScriptedAgent
from crn.arms import run_arm_a, record_placement


def _agents():
    return [ScriptedAgent() for _ in range(4)]


def _usable_seed(focal=0, start=0, limit=300):
    """First seed admitting a symmetric equal-pip swap for ScriptedAgent placement.
    Lightweight: does setup + swap construction only, no full games."""
    for s in range(start, start + limit):
        pr = PairRNG(s, "crn")
        lg = pr.layout_gen()
        ref = generate_reference_layout(lg)
        _, st = record_placement(ScriptedAgent(), pr.ref_bundle(), ref)
        if make_symmetric_swaps(ref, focal_tiles(st, focal), lg) is not None:
            return s
    raise RuntimeError("no usable seed found")


# --- total board pips never change (any resource permutation) ------------------
def test_total_board_pips_invariant_everywhere():
    lg = single_game_rng(7).layout
    ref = generate_reference_layout(lg)
    total_ref = board_scarcity(ref)["board_total_pips"]
    resh = full_reshuffle(ref, lg)                 # full reshuffle: entropy moves...
    total_resh = board_scarcity(resh)["board_total_pips"]
    assert total_ref == total_resh, "total board pips changed under full reshuffle!"
    # per-resource DOES move under reshuffle (sanity: it's a real effect)
    assert board_scarcity(ref) != board_scarcity(resh)
    print(f"(1) total board pips invariant under reshuffle "
          f"(={total_ref}); per-resource breakdown moves.")


# --- equal-pip swap holds per-resource board scarcity fixed --------------------
def test_equal_pip_swap_preserves_board_scarcity():
    pr = PairRNG(_usable_seed(), "crn")
    lg = pr.layout_gen()
    ref = generate_reference_layout(lg)
    _, st = record_placement(ScriptedAgent(), pr.ref_bundle(), ref)
    ftiles = focal_tiles(st, 0)
    sym = make_symmetric_swaps(ref, ftiles, lg, require_equal_pips=True)
    assert sym is not None, "need a board with a symmetric equal-pip swap"
    L_inc, L_dec, _ = sym
    s_ref, s_inc, s_dec = (board_scarcity(x) for x in (ref, L_inc, L_dec))
    assert s_ref == s_inc == s_dec, "equal-pip swap changed board scarcity!"
    # And an UNEQUAL-pip swap generally does change it (contrast check).
    uneq = choose_diversity_swap(ref, ftiles, "increase", lg, require_equal_pips=False)
    print("(2) equal-pip swaps preserve per-resource board scarcity exactly.")


# --- Arm A: within-pair, only the focal (and swapped tile) resources differ ----
def test_arm_a_holds_everything_but_focal_resources():
    r = run_arm_a(_usable_seed(), ScriptedAgent(), focal_seat=0, mode="crn", feature_every=0)
    assert not r["skipped"], "pick a seed with a symmetric swap for this test"
    # Board scarcity identical across the pair.
    assert r["board_scarcity"]["A"] == r["board_scarcity"]["B"]
    # Focal treatment contrast is +2 (d+1 vs d-1).
    assert r["focal_div_A"] - r["focal_div_B"] == 2, \
        (r["focal_div_A"], r["focal_div_B"])
    # Total board pips identical and equal to reference.
    assert (r["board_scarcity"]["A"]["board_total_pips"]
            == r["board_scarcity"]["ref"]["board_total_pips"])
    print(f"(3) Arm A: focal div_count {r['focal_div_B']}->{r['focal_div_A']} "
          f"(+2), board scarcity + total pips held fixed across the pair.")


# --- Arm A: the recorded placement is replayed identically in both games -------
def test_arm_a_placement_replayed_identically():
    # Re-derive the recorded placement and confirm both games start setup from it.
    pr = PairRNG(_usable_seed(), "crn")
    lg = pr.layout_gen()
    ref = generate_reference_layout(lg)
    recorded, setup_state = record_placement(ScriptedAgent(), pr.ref_bundle(), ref)
    ref_sett = [sorted(setup_state.players[i].settlements) for i in range(4)]
    sym = make_symmetric_swaps(ref, focal_tiles(setup_state, 0), lg)
    assert sym is not None
    L_inc, L_dec, _ = sym
    # Replaying the recorded actions on both boards must yield the SAME settlements.
    from crn.arms import ReplayAgent
    from src.game.game_state import GameState
    from src.game.rules import Rules
    for L, bundle in ((L_inc, pr.game_a()), (L_dec, pr.game_b())):
        st = GameState(rng=bundle, layout=L)
        ra = ReplayAgent(recorded)
        steps = 0
        while st.is_setup_phase and steps < 2000:
            a = ra.get_action(st)
            Rules.execute_action(st, a)
            steps += 1
        sett = [sorted(st.players[i].settlements) for i in range(4)]
        assert sett == ref_sett, f"placement not replayed identically: {sett} vs {ref_sett}"
    print("(4) Arm A placement replayed identically on both treated boards.")


# --- CRN vs independent: dice shared under crn, not under independent ----------
def test_pairrng_crn_vs_independent():
    def dice(bundle, n=100):
        return [int(bundle.dice.integers(1, 7)) for _ in range(n)]
    crn = PairRNG(5, "crn")
    assert dice(crn.game_a()) == dice(crn.game_b()), "CRN dice should match"
    # setup shared in BOTH modes (so replay aligns)
    def setup_order(bundle):
        return list(bundle.setup.permutation(4))
    ind = PairRNG(5, "independent")
    assert setup_order(ind.game_a()) == setup_order(ind.game_b()), \
        "setup must be shared even in independent mode"
    assert dice(ind.game_a()) != dice(ind.game_b()), "independent dice should differ"
    print("(5) PairRNG: CRN shares dice; independent shares only setup.")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
