"""Separated, seedable RNG streams for the CRN paired-simulation experiment.

Every source of randomness in the game engine is given its own independent
numpy Generator, all derived from a single pair seed via
``numpy.random.SeedSequence.spawn`` (statistically independent by construction).

Streams
-------
layout   : board generation (resource shuffle, number-token placement). Only
           consumed when *generating* a layout; during paired play the board is
           built from a fixed Layout, so this stream is idle at play time.
dice     : the 2d6 roll each turn.
dev_deck : the one-time dev-card deck shuffle (draws are deterministic pops).
steal    : ONLY the hidden card drawn from the victim's hand on a robber steal.
           The robber's destination tile and the choice of victim are AGENT
           actions (not seeded); they may diverge freely between paired games.
setup    : the snake-draft player-order shuffle.
players  : one Generator per seat, reserved for policy sampling. The focal
           experiment uses deterministic (argmax) agents, so these are currently
           unused; kept so a future stochastic-sampling arm is stream-separated.

Common Random Numbers (CRN)
---------------------------
Within a CRN pair, the two games share identical dice/dev_deck/steal/setup
streams (and player streams); only the board Layout differs (the treatment).
In an *independent* control pair the second game gets freshly-spawned streams,
so nothing is shared except the layouts. Comparing paired-outcome variance
between the two regimes is the headline measurement (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

# Engine-consumed streams, in a fixed order (order defines the spawn mapping).
STREAM_NAMES = ("layout", "dice", "dev_deck", "steal", "setup")
NUM_PLAYERS = 4


@dataclass
class GameRNG:
    """Bundle of per-stream numpy Generators for a single game.

    The game engine duck-types this: it reads ``.layout``, ``.dice``,
    ``.dev_deck``, ``.steal``, ``.setup`` (and does not import this class).
    """
    layout: np.random.Generator
    dice: np.random.Generator
    dev_deck: np.random.Generator
    steal: np.random.Generator
    setup: np.random.Generator
    players: List[np.random.Generator]


def _game_rng_from_children(children) -> GameRNG:
    """Build a GameRNG from a list of child SeedSequences.

    ``children`` has len == len(STREAM_NAMES) + NUM_PLAYERS. Generators are
    created with ``default_rng`` (does not consume/advance the SeedSequence), so
    building twice from the *same* children yields bit-identical streams.
    """
    gens = [np.random.default_rng(c) for c in children]
    named = dict(zip(STREAM_NAMES, gens[: len(STREAM_NAMES)]))
    players = gens[len(STREAM_NAMES):]
    return GameRNG(players=players, **named)


def spawn_pair(pair_seed: int, mode: str = "crn"):
    """Return (rng_A, rng_B, layout_gen) for one pair.

    layout_gen : a Generator used to create the *reference* board layout for this
                 pair (independent of the per-game streams).
    rng_A/rng_B: GameRNG bundles for game A and game B.
      - mode="crn"        : rng_B's engine/player streams are bit-identical to
                            rng_A's (only the Layout differs between games).
      - mode="independent": rng_B is freshly spawned, independent of rng_A.

    Determinism: fully a function of (pair_seed, mode). A SeedSequence is reusable
    (default_rng does not consume it); only .spawn() advances its child counter,
    which is why rng_A and the CRN rng_B share the same child SeedSequences.
    """
    if mode not in ("crn", "independent"):
        raise ValueError(f"mode must be 'crn' or 'independent', got {mode!r}")

    root = np.random.SeedSequence(pair_seed)
    layout_ss, game_ss = root.spawn(2)
    layout_gen = np.random.default_rng(layout_ss)

    n = len(STREAM_NAMES) + NUM_PLAYERS
    children_a = game_ss.spawn(n)              # game A's streams
    rng_a = _game_rng_from_children(children_a)

    if mode == "crn":
        # Same child SeedSequences -> identical streams for game B (CRN).
        rng_b = _game_rng_from_children(children_a)
    else:
        children_b = game_ss.spawn(n)          # fresh, independent streams
        rng_b = _game_rng_from_children(children_b)

    return rng_a, rng_b, layout_gen


class PairRNG:
    """Phase-3 pair RNG with FRESH-but-identical bundles on demand.

    Constructing a GameState consumes its rng (setup/deck shuffles), so the
    reference-placement run, game A, and game B each need their OWN generator
    instances. Because ``default_rng(seed_sequence)`` does not consume the
    SeedSequence, minting a bundle repeatedly from the same child SeedSequences
    yields bit-identical streams — exactly what we need.

    Stream-sharing policy (this is the heart of the Arm-A design):
      * setup   : ALWAYS shared across reference/A/B, so the snake-draft order
                  matches and the recorded placement can be replayed on both
                  treated boards.
      * layout  : one generator per pair (reference layout generation + swap
                  candidate selection draw from it, in sequence).
      * dice/dev_deck/steal : shared between A and B under mode="crn"; freshly
                  spawned for B under mode="independent" (the control set).
      * players : reserved per-seat streams (unused by deterministic agents).
    """

    def __init__(self, pair_seed: int, mode: str = "crn"):
        if mode not in ("crn", "independent"):
            raise ValueError(f"mode must be 'crn' or 'independent', got {mode!r}")
        self.pair_seed = pair_seed
        self.mode = mode
        root = np.random.SeedSequence(pair_seed)
        c = root.spawn(12)
        self.layout_ss = c[0]
        self.setup_ss = c[1]
        self._A = (c[2], c[3], c[4])            # (dice, dev_deck, steal) for A
        self._B = (c[2], c[3], c[4]) if mode == "crn" else (c[5], c[6], c[7])
        self.player_sss = c[8:12]
        self._layout_gen = None                 # single generator, cached

    def layout_gen(self) -> np.random.Generator:
        """One generator for layout generation + swap selection (sequential)."""
        if self._layout_gen is None:
            self._layout_gen = np.random.default_rng(self.layout_ss)
        return self._layout_gen

    def _bundle(self, engine_sss) -> GameRNG:
        dice_ss, deck_ss, steal_ss = engine_sss
        return GameRNG(
            layout=np.random.default_rng(self.layout_ss),   # idle during play
            dice=np.random.default_rng(dice_ss),
            dev_deck=np.random.default_rng(deck_ss),
            steal=np.random.default_rng(steal_ss),
            setup=np.random.default_rng(self.setup_ss),
            players=[np.random.default_rng(s) for s in self.player_sss],
        )

    def ref_bundle(self) -> GameRNG:
        """Bundle for the reference-placement run (only its setup stream matters)."""
        return self._bundle(self._A)

    def game_a(self) -> GameRNG:
        return self._bundle(self._A)

    def game_b(self) -> GameRNG:
        return self._bundle(self._B)


def single_game_rng(seed: int) -> GameRNG:
    """A standalone GameRNG for one game (used by determinism tests)."""
    root = np.random.SeedSequence(seed)
    children = root.spawn(len(STREAM_NAMES) + NUM_PLAYERS)
    return _game_rng_from_children(children)
