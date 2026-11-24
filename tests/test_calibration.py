"""Pip-calibration validation tests (run BEFORE any calibration batches).

Checks, per spec:
  - token multiset identical across reference / P+ / P-
  - focal pips change by exactly +1 / -1 (dots); distinct resource count unchanged
  - red-number rule holds on all three boards
  - deterministic replay (rerunning a seed reproduces the games exactly)
  - per-player pips / production / entropy are logged on all boards

Run:  python tests/test_calibration.py
Uses the real iter17 agent (placement matters). Calibration seeds live at 5_000_000+,
disjoint from the diversity pilot/main run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "engine"))

import numpy as np
from crn.agent import make_agent
from crn.calibration import (
    build_calibration_boards, run_calibration, red_number_ok,
    creates_same_number_adjacency,
)
from crn.features import compute_player_features, features_on_layout

CALIB_BASE = 5_000_000
AGENT = None  # set in __main__


def _usable(start=0, limit=400):
    for s in range(start, start + limit):
        b = build_calibration_boards(CALIB_BASE + s, AGENT, (s % 4))
        if not b["skipped"]:
            return CALIB_BASE + s, (s % 4), b
    raise RuntimeError("no usable calibration seed found")


def test_token_multiset_identical():
    _, _, b = _usable()
    mref = sorted(b["ref"].number_per_tile)
    assert sorted(b["L_plus"].number_per_tile) == mref
    assert sorted(b["L_minus"].number_per_tile) == mref
    print("(1) token multiset identical across ref / P+ / P-.")


def test_focal_pip_change_exact_and_count_unchanged():
    seed, focal, b = _usable()
    st = b["setup_state"]
    sett = [list(st.players[i].settlements) for i in range(4)]
    cities = [list(st.players[i].cities) for i in range(4)]
    fr = features_on_layout(sett, cities, b["ref"])[focal]
    fp = features_on_layout(sett, cities, b["L_plus"])[focal]
    fm = features_on_layout(sett, cities, b["L_minus"])[focal]
    one_pip = 1.0 / 36.0   # features store total_pips rounded to 6 dp -> tol 1e-5
    assert abs((fp["total_pips"] - fr["total_pips"]) - one_pip) < 1e-5, \
        (fp["total_pips"], fr["total_pips"])
    assert abs((fm["total_pips"] - fr["total_pips"]) + one_pip) < 1e-5
    assert fp["div_count"] == fr["div_count"] == fm["div_count"]  # resources didn't move
    print(f"(2) focal pips +1/-1 exactly (={one_pip:.5f}); distinct count unchanged "
          f"(={fr['div_count']}).")


def test_red_number_rule_all_boards():
    _, _, b = _usable()
    assert red_number_ok(b["ref"]) and red_number_ok(b["L_plus"]) and red_number_ok(b["L_minus"])
    # same-number adjacency is logged, not rejected (per spec); just report it.
    sn = any(creates_same_number_adjacency(x) for x in (b["L_plus"], b["L_minus"]))
    print(f"(3) red-number rule holds on all boards. (same-number adjacency created: {sn})")


def test_deterministic_replay():
    seed, focal, _ = _usable()
    r1 = run_calibration(seed, AGENT, focal)
    r2 = run_calibration(seed, AGENT, focal)
    for key in ("rank_plus", "rank_minus"):
        assert r1[key] == r2[key]
    assert r1["gameP"]["dice_seq"] == r2["gameP"]["dice_seq"]
    assert r1["gameM"]["dice_seq"] == r2["gameM"]["dice_seq"]
    assert r1["gameP"]["vp"] == r2["gameP"]["vp"] and r1["gameM"]["vp"] == r2["gameM"]["vp"]
    print(f"(4) deterministic replay: ranks P+={r1['rank_plus']} P-={r1['rank_minus']} "
          f"reproduced exactly.")


def test_per_player_logging_present():
    seed, focal, _ = _usable()
    r = run_calibration(seed, AGENT, focal)
    for board in ("ref", "P+", "P-"):
        feats = r["init_features"][board]
        assert len(feats) == 4
        for f in feats:
            assert "total_pips" in f and "div_entropy" in f and "exp_prod_ore" in f
    print("(5) per-player pips / production / entropy logged on ref / P+ / P-.")


if __name__ == "__main__":
    import traceback
    AGENT = make_agent("iter17")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
