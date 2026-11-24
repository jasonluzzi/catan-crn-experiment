"""Phase 1 determinism / stream-separation tests.

Checkpoint requirements:
  (a) same seed + same layout  -> bit-identical game twice
  (b) same seed + shuffled layout -> identical dice sequence and dev-deck order
  (c) changing the layout stream changes nothing else
Plus the design-critical invariant:
  (d) reshuffling resources keeps every number token and every corner pip total
      fixed within a pair.

Run:  python -m pytest tests/test_rng_streams.py -v
 or:  python tests/test_rng_streams.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from crn.rng import spawn_pair, single_game_rng, STREAM_NAMES
from crn.layout import (
    generate_reference_layout, full_reshuffle, targeted_swap, board_to_layout,
)
from crn.runner import ScriptedAgent, play_game
from src.game.game_state import GameState


def _agents():
    return [ScriptedAgent() for _ in range(4)]


def _dice_stream(gen, n=200):
    return [int(gen.integers(1, 7)) for _ in range(n)]


# --------------------------------------------------------------------------- (a)
def test_bit_identical_same_seed_same_layout():
    """Same engine streams + same layout -> identical trace, dice, and VP."""
    layout = generate_reference_layout(single_game_rng(999).layout)

    r1 = play_game(_agents(), single_game_rng(42), layout, record_trace=True)
    r2 = play_game(_agents(), single_game_rng(42), layout, record_trace=True)

    assert r1["trace"] == r2["trace"], "action traces diverged"
    assert r1["dice_seq"] == r2["dice_seq"], "dice sequences diverged"
    assert r1["vp"] == r2["vp"] and r1["winner"] == r2["winner"]
    assert r1["steps"] == r2["steps"]
    print(f"(a) bit-identical: {r1['steps']} steps, winner={r1['winner']}, "
          f"VP={r1['vp']}, completed={r1['completed']}, rolls={len(r1['dice_seq'])}")


# --------------------------------------------------------------------------- (b)
def test_crn_shares_dice_and_deck_across_layouts():
    """A CRN pair with different layouts shares dice + deck + setup streams."""
    rng_a, rng_b, layout_gen = spawn_pair(7, mode="crn")
    ref = generate_reference_layout(layout_gen)
    treated = full_reshuffle(ref, layout_gen)

    # The treatment actually changed resources but not numbers (sanity).
    assert ref.resource_per_tile != treated.resource_per_tile
    assert ref.number_per_tile == treated.number_per_tile

    # Engine streams are bit-identical between the two games.
    assert _dice_stream(rng_a.dice) == _dice_stream(rng_b.dice), "dice stream differs"
    assert list(rng_a.dev_deck.permutation(25)) == list(rng_b.dev_deck.permutation(25))
    assert list(rng_a.setup.permutation(4)) == list(rng_b.setup.permutation(4))
    for pa, pb in zip(rng_a.players, rng_b.players):
        assert _dice_stream(pa, 50) == _dice_stream(pb, 50)

    # And at the game level: the two states get an identical shuffled deck + order,
    # despite different boards.
    rng_a, rng_b, layout_gen = spawn_pair(7, mode="crn")  # fresh (streams consumed above)
    ref = generate_reference_layout(layout_gen)
    treated = full_reshuffle(ref, layout_gen)
    sa = GameState(rng=rng_a, layout=ref)
    sb = GameState(rng=rng_b, layout=treated)
    assert sa.dev_card_deck == sb.dev_card_deck, "dev deck order differs across pair"
    assert sa.setup_order == sb.setup_order, "setup order differs across pair"
    assert [int(t.resource) for t in sa.board.tiles] != [int(t.resource) for t in sb.board.tiles]
    print("(b) CRN pair shares dice/deck/setup streams; boards differ.")


# --------------------------------------------------------------------------- (c)
def test_layout_stream_isolated():
    """A targeted swap changes only the board; dice/deck/setup untouched."""
    rng_a, rng_b, layout_gen = spawn_pair(13, mode="crn")
    ref = generate_reference_layout(layout_gen)
    nd = ref.non_desert_tiles
    swapped = targeted_swap(ref, nd[0], nd[1])

    sa = GameState(rng=rng_a, layout=ref)
    sb = GameState(rng=rng_b, layout=swapped)

    assert sa.dev_card_deck == sb.dev_card_deck
    assert sa.setup_order == sb.setup_order
    # Numbers identical, resources differ only where we swapped.
    assert [t.number for t in sa.board.tiles] == [t.number for t in sb.board.tiles]
    ra = [int(t.resource) for t in sa.board.tiles]
    rb = [int(t.resource) for t in sb.board.tiles]
    diff = [i for i in range(19) if ra[i] != rb[i]]
    assert set(diff) == {nd[0], nd[1]}, f"swap touched unexpected tiles: {diff}"
    print(f"(c) targeted swap isolated to tiles {diff}; streams unchanged.")


def test_independent_control_decouples():
    """The independent control pair does NOT share engine streams."""
    rng_a, rng_b, _ = spawn_pair(7, mode="independent")
    assert _dice_stream(rng_a.dice) != _dice_stream(rng_b.dice)
    assert list(rng_a.dev_deck.permutation(25)) != list(rng_b.dev_deck.permutation(25))
    print("(d-control) independent pair streams differ, as intended.")


def test_spawn_pair_reproducible():
    """spawn_pair is a pure function of (seed, mode)."""
    a1, b1, l1 = spawn_pair(55, "crn")
    a2, b2, l2 = spawn_pair(55, "crn")
    assert _dice_stream(a1.dice) == _dice_stream(a2.dice)
    assert _dice_stream(b1.dice) == _dice_stream(b2.dice)
    assert list(l1.permutation(19)) == list(l2.permutation(19))
    print("(e) spawn_pair reproducible across calls.")


# --------------------------------------------------------------------------- (d)
def _corner_pip_totals(board):
    return [round(sum(board.tiles[t].production_probability
                      for t in c.adjacent_tiles if t < len(board.tiles)), 6)
            for c in board.corners]


def test_pip_totals_invariant_under_reshuffle():
    """Full reshuffle fixes number tokens AND every corner's pip total."""
    layout_gen = single_game_rng(2024).layout
    ref = generate_reference_layout(layout_gen)
    treated = full_reshuffle(ref, layout_gen)

    assert ref.number_per_tile == treated.number_per_tile
    assert ref.desert_tile == treated.desert_tile

    pips_ref = _corner_pip_totals(ref.to_board())
    pips_trt = _corner_pip_totals(treated.to_board())
    assert pips_ref == pips_trt, "corner pip totals changed under reshuffle!"
    # Resource composition is a permutation of the same multiset.
    assert sorted(ref.resource_per_tile) == sorted(treated.resource_per_tile)
    print("(f) pips fixed: number tokens + all 54 corner pip totals invariant.")


def test_roundtrip_layout_board():
    """board_to_layout o to_board is the identity on a generated layout."""
    ref = generate_reference_layout(single_game_rng(1).layout)
    back = board_to_layout(ref.to_board())
    assert back.resource_per_tile == ref.resource_per_tile
    assert back.number_per_tile == ref.number_per_tile
    print("(g) layout<->board round-trip stable.")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
