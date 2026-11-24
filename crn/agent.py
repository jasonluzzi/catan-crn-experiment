"""Agent loaders for the experiment (deterministic argmax).

Neural policies run in eval() with argmax selection (`deterministic=True`), so they
consume NO randomness -- within a CRN pair the only divergence source is the layout
treatment.

Baseline anchors (greedy/random) DO use global np.random; their per-game noise is
coupled via a global seed set by the runner (see LIMITATIONS.md), not via the
injected GameRNG.

Checkpoints are NOT shipped with this repository. The three neural agents load from
`CATAN_CHECKPOINT_DIR` (default: `<repo>/checkpoints/`); supply your own trained
`.pt` files there, or run the greedy/random anchors, which need no checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from src.game.rules import Rules                               # noqa: E402
from src.environment.state_encoder import StateEncoder        # noqa: E402
from src.models.baseline import RandomAgent, GreedyAgent      # noqa: E402
from crn.runner import mask_trading_actions                   # noqa: E402

# Bring-your-own-checkpoint: point CATAN_CHECKPOINT_DIR at a directory holding the
# three .pt files below (defaults to ./checkpoints/ at the repo root).
CHECKPOINT_DIR = Path(os.environ.get(
    "CATAN_CHECKPOINT_DIR", Path(__file__).resolve().parent.parent / "checkpoints"))
# Strength ladder. The percentages are the deterministic win-rate vs 3x greedy as
# REPORTED from the original training runs; the eval output and weights are not shipped
# with this repo, so they are not reproducible here.
CHECKPOINTS = {
    "iter17": CHECKPOINT_DIR / "rl_ppo_shaped_iter17_beats_bc.pt",  # 52.3%
    "v5":     CHECKPOINT_DIR / "bc_v5_trading_policy.pt",           # 48.3%
    "v4":     CHECKPOINT_DIR / "bc_v4_full_policy.pt",              # 38.8%
}


def load_hier_policy(checkpoint: str, device: str = "cpu"):
    """Load a HierarchicalCatanPolicy checkpoint in eval mode."""
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Checkpoints are not shipped with this repo. Place your trained .pt files "
            "in the checkpoint directory (set CATAN_CHECKPOINT_DIR, default ./checkpoints/), "
            "or use the greedy/random anchor agents, which need no checkpoint.")
    import torch
    from gymnasium import spaces
    from src.models.hierarchical_policy import HierarchicalCatanPolicy
    obs_space = spaces.Box(low=0, high=1, shape=(176, 65), dtype=np.float32)
    action_space = spaces.Discrete(475)
    policy = HierarchicalCatanPolicy(obs_space, action_space, lr_schedule=lambda _: 3e-4)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"] if "policy_state_dict" in ckpt else ckpt)
    policy.eval()
    return policy.to(device)


class HierPolicyAgent:
    """Pure-policy hierarchical agent; deterministic argmax by default."""

    def __init__(self, policy, device: str = "cpu", no_trading: bool = True,
                 deterministic: bool = True):
        self.policy = policy
        self.device = device
        self.no_trading = no_trading
        self.deterministic = deterministic
        self.encoder = StateEncoder()

    def get_action(self, state) -> int:
        import torch
        p = state.current_player_idx
        obs = self.encoder.encode(state, p)
        mask = Rules.get_legal_actions(state)
        if self.no_trading:
            m = mask_trading_actions(mask)
            if m.any():
                mask = m
        ot = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mt = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.policy.get_distribution(ot, mt)
            action = (dist.mode() if self.deterministic else dist.sample()).item()
        return int(action)


class BaselineAdapter:
    """Wrap a baseline BaseAgent (predict(obs, mask, det)) as get_action(state)."""

    def __init__(self, agent, no_trading: bool = True, deterministic: bool = False):
        self.agent = agent
        self.encoder = StateEncoder()
        self.no_trading = no_trading
        self.deterministic = deterministic

    def get_action(self, state) -> int:
        p = state.current_player_idx
        obs = self.encoder.encode(state, p)
        mask = Rules.get_legal_actions(state)
        if self.no_trading:
            m = mask_trading_actions(mask)
            if m.any():
                mask = m
        return int(self.agent.predict(obs, mask, deterministic=self.deterministic))


def make_agent(kind: str, device: str = "cpu", no_trading: bool = True):
    """Factory: 'iter17'|'v5'|'v4' (neural, argmax) or 'greedy'|'random' (anchors)."""
    if kind in CHECKPOINTS:
        policy = load_hier_policy(str(CHECKPOINTS[kind]), device)
        return HierPolicyAgent(policy, device=device, no_trading=no_trading,
                               deterministic=True)
    if kind == "greedy":
        return BaselineAdapter(GreedyAgent(exploration_prob=0.1), no_trading, False)
    if kind == "random":
        return BaselineAdapter(RandomAgent(), no_trading, False)
    raise ValueError(f"unknown agent kind {kind!r}")
