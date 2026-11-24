"""
Hierarchical Actor for two-stage action selection.

Stage 1: Select action type (20 categories)
Stage 2: Select location/parameter based on type (6 heads)

This structure separates "what to do" from "where/how to do it" for better
learning signal and reduced output space.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .action_decomposer import (
    ActionDecomposer,
    HierarchicalActionType,
    LocationHead,
    TYPE_TO_LOCATION_HEAD,
    NUM_ACTION_TYPES,
    LOCATION_HEAD_SIZES,
    FLAT_RANGES,
)
from ..game.actions import TOTAL_ACTIONS


class HierarchicalActor(nn.Module):
    """
    Two-stage hierarchical actor for Catan.

    Architecture:
        enriched_features (531) → shared_mlp (256) → action_type_head (20)
                                                  → location_features (256)
        location_features → corner_head (54)
                         → edge_head (72)
                         → tile_head (19)
                         → player_head (4)
                         → trade_template_head (50)
                         → small_select_head (25)
    """

    def __init__(
        self,
        feature_dim: int = 531,
        hidden_dim: int = 256,
        log_std_init: float = 0.0,
    ):
        """
        Initialize the hierarchical actor.

        Args:
            feature_dim: Input feature dimension (default 531 = 512 + 19 strategic)
            hidden_dim: Hidden layer dimension
            log_std_init: Initial log standard deviation (unused, kept for compatibility)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.decomposer = ActionDecomposer()

        # Shared feature processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Stage 1: Action type head
        self.action_type_head = nn.Linear(hidden_dim, NUM_ACTION_TYPES)

        # Stage 2: Location heads
        # Each head outputs logits for its specific location space
        self.corner_head = nn.Linear(hidden_dim, LOCATION_HEAD_SIZES[LocationHead.CORNER])  # 54
        self.edge_head = nn.Linear(hidden_dim, LOCATION_HEAD_SIZES[LocationHead.EDGE])      # 72
        self.tile_head = nn.Linear(hidden_dim, LOCATION_HEAD_SIZES[LocationHead.TILE])      # 19
        self.player_head = nn.Linear(hidden_dim, LOCATION_HEAD_SIZES[LocationHead.PLAYER])  # 4
        self.trade_template_head = nn.Linear(hidden_dim, LOCATION_HEAD_SIZES[LocationHead.TRADE_TEMPLATE])  # 50
        self.small_select_head = nn.Linear(hidden_dim, LOCATION_HEAD_SIZES[LocationHead.SMALL_SELECT])      # 25

        # Map from LocationHead enum to actual head modules
        self._location_heads = {
            LocationHead.CORNER: self.corner_head,
            LocationHead.EDGE: self.edge_head,
            LocationHead.TILE: self.tile_head,
            LocationHead.PLAYER: self.player_head,
            LocationHead.TRADE_TEMPLATE: self.trade_template_head,
            LocationHead.SMALL_SELECT: self.small_select_head,
        }

        # Pre-compute mappings for efficient action reconstruction
        self._build_action_mappings()

    def _build_action_mappings(self):
        """Build lookup tables for action conversion."""
        # For each action type, store (flat_start, flat_end, location_head)
        self._type_info = {}
        for start, end, action_type in FLAT_RANGES:
            location_head = TYPE_TO_LOCATION_HEAD[action_type]
            self._type_info[int(action_type)] = {
                'start': start,
                'end': end,
                'location_head': location_head,
                'size': end - start,
            }

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all head logits.

        Args:
            features: (batch_size, feature_dim) enriched features

        Returns:
            Dict containing:
                - type_logits: (batch_size, 20) action type logits
                - corner_logits: (batch_size, 54)
                - edge_logits: (batch_size, 72)
                - tile_logits: (batch_size, 19)
                - player_logits: (batch_size, 4)
                - trade_template_logits: (batch_size, 50)
                - small_select_logits: (batch_size, 25)
        """
        # Shared feature processing
        shared = self.shared_mlp(features)

        # Compute all logits
        return {
            'type_logits': self.action_type_head(shared),
            'corner_logits': self.corner_head(shared),
            'edge_logits': self.edge_head(shared),
            'tile_logits': self.tile_head(shared),
            'player_logits': self.player_head(shared),
            'trade_template_logits': self.trade_template_head(shared),
            'small_select_logits': self.small_select_head(shared),
        }

    def get_action(
        self,
        features: torch.Tensor,
        flat_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample or select action from hierarchical policy.

        Args:
            features: (batch_size, feature_dim) enriched features
            flat_mask: (batch_size, 475) flat action mask
            deterministic: If True, select most probable action

        Returns:
            Tuple of:
                - flat_action: (batch_size,) flat action indices
                - log_prob: (batch_size,) log probabilities
        """
        batch_size = features.shape[0]
        device = features.device

        # Get all logits
        logits = self.forward(features)

        # Convert flat mask to type mask
        type_mask = self._get_batch_type_mask(flat_mask)

        # Apply mask and sample/select action type
        type_logits = logits['type_logits']
        type_logits = type_logits.masked_fill(~type_mask, float('-inf'))
        type_probs = F.softmax(type_logits, dim=-1)

        if deterministic:
            type_ids = type_probs.argmax(dim=-1)
        else:
            type_dist = Categorical(probs=type_probs)
            type_ids = type_dist.sample()

        type_log_probs = F.log_softmax(type_logits, dim=-1)
        selected_type_log_prob = type_log_probs.gather(1, type_ids.unsqueeze(-1)).squeeze(-1)

        # For each sample, get location from appropriate head
        flat_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        location_log_probs = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            type_id = type_ids[i].item()
            type_info = self._type_info[type_id]
            location_head = type_info['location_head']

            if location_head == LocationHead.NONE:
                # No location needed
                flat_actions[i] = type_info['start']
                location_log_probs[i] = 0.0
            else:
                # Get location logits and mask
                head_logits = self._get_location_logits(logits, location_head, i)
                location_mask = self._get_location_mask_for_type(flat_mask[i], type_id)

                # Ensure mask is the right size (may need truncation for variable-size heads)
                actual_size = type_info['size']
                if len(location_mask) > actual_size:
                    location_mask = location_mask[:actual_size]
                if head_logits.shape[-1] > actual_size:
                    head_logits = head_logits[:actual_size]

                # Apply mask
                masked_logits = head_logits.masked_fill(~location_mask, float('-inf'))

                if deterministic:
                    location_id = masked_logits.argmax()
                else:
                    location_probs = F.softmax(masked_logits, dim=-1)
                    location_dist = Categorical(probs=location_probs)
                    location_id = location_dist.sample()

                location_log_prob = F.log_softmax(masked_logits, dim=-1)
                location_log_probs[i] = location_log_prob[location_id]

                flat_actions[i] = type_info['start'] + location_id

        # Total log prob = log(type) + log(location)
        total_log_prob = selected_type_log_prob + location_log_probs

        return flat_actions, total_log_prob

    def evaluate_action(
        self,
        features: torch.Tensor,
        flat_action: torch.Tensor,
        flat_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.

        Args:
            features: (batch_size, feature_dim) enriched features
            flat_action: (batch_size,) flat action indices
            flat_mask: (batch_size, 475) flat action mask

        Returns:
            Tuple of:
                - log_prob: (batch_size,) log probabilities
                - entropy: (batch_size,) entropy values
        """
        batch_size = features.shape[0]
        device = features.device

        # Get all logits
        logits = self.forward(features)

        # Convert flat actions to hierarchical
        type_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        location_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            t, l = self.decomposer.flat_to_hierarchical(flat_action[i].item())
            type_ids[i] = t
            location_ids[i] = l

        # Type probabilities
        type_mask = self._get_batch_type_mask(flat_mask)
        type_logits = logits['type_logits'].masked_fill(~type_mask, float('-inf'))
        type_log_probs = F.log_softmax(type_logits, dim=-1)
        selected_type_log_prob = type_log_probs.gather(1, type_ids.unsqueeze(-1)).squeeze(-1)

        # Type entropy
        type_probs = F.softmax(type_logits, dim=-1)
        type_entropy = -(type_probs * type_log_probs).sum(dim=-1)
        # Mask out invalid probs for entropy (they contribute 0 * -inf = nan)
        type_entropy = torch.where(
            torch.isnan(type_entropy),
            torch.zeros_like(type_entropy),
            type_entropy
        )

        # Location probabilities and entropy
        location_log_probs = torch.zeros(batch_size, device=device)
        location_entropy = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            type_id = type_ids[i].item()
            type_info = self._type_info[type_id]
            location_head = type_info['location_head']

            if location_head == LocationHead.NONE:
                location_log_probs[i] = 0.0
                location_entropy[i] = 0.0
            else:
                # Get location logits and mask
                head_logits = self._get_location_logits(logits, location_head, i)
                location_mask = self._get_location_mask_for_type(flat_mask[i], type_id)

                # Truncate if needed
                actual_size = type_info['size']
                if len(location_mask) > actual_size:
                    location_mask = location_mask[:actual_size]
                if head_logits.shape[-1] > actual_size:
                    head_logits = head_logits[:actual_size]

                masked_logits = head_logits.masked_fill(~location_mask, float('-inf'))
                loc_log_probs = F.log_softmax(masked_logits, dim=-1)
                loc_probs = F.softmax(masked_logits, dim=-1)

                loc_id = location_ids[i].item()
                location_log_probs[i] = loc_log_probs[loc_id] if loc_id < len(loc_log_probs) else 0.0

                ent = -(loc_probs * loc_log_probs).sum()
                location_entropy[i] = ent if not torch.isnan(ent) else 0.0

        total_log_prob = selected_type_log_prob + location_log_probs
        total_entropy = type_entropy + location_entropy

        return total_log_prob, total_entropy

    def _get_batch_type_mask(self, flat_mask: torch.Tensor) -> torch.Tensor:
        """Convert batch of flat masks to type masks."""
        batch_size = flat_mask.shape[0]
        device = flat_mask.device

        type_mask = torch.zeros(batch_size, NUM_ACTION_TYPES, dtype=torch.bool, device=device)

        for start, end, action_type in FLAT_RANGES:
            # A type is valid if any action in its range is valid
            type_valid = flat_mask[:, start:end].any(dim=-1)
            type_mask[:, int(action_type)] = type_valid

        return type_mask

    def _get_location_logits(
        self,
        logits: Dict[str, torch.Tensor],
        location_head: LocationHead,
        batch_idx: int
    ) -> torch.Tensor:
        """Get location logits for a specific head and batch index."""
        head_name = {
            LocationHead.CORNER: 'corner_logits',
            LocationHead.EDGE: 'edge_logits',
            LocationHead.TILE: 'tile_logits',
            LocationHead.PLAYER: 'player_logits',
            LocationHead.TRADE_TEMPLATE: 'trade_template_logits',
            LocationHead.SMALL_SELECT: 'small_select_logits',
        }[location_head]

        return logits[head_name][batch_idx]

    def _get_location_mask_for_type(
        self,
        flat_mask: torch.Tensor,
        type_id: int
    ) -> torch.Tensor:
        """Get location mask for a specific action type."""
        type_info = self._type_info[type_id]
        start, end = type_info['start'], type_info['end']
        return flat_mask[start:end]

    def get_all_logits_flat(
        self,
        features: torch.Tensor,
        flat_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get logits in flat 475-action format for compatibility.

        This reconstructs the flat action logits from hierarchical logits.
        Useful for visualization and debugging.

        Args:
            features: (batch_size, feature_dim) enriched features
            flat_mask: (batch_size, 475) flat action mask

        Returns:
            (batch_size, 475) flat action logits
        """
        batch_size = features.shape[0]
        device = features.device

        logits = self.forward(features)
        flat_logits = torch.full((batch_size, TOTAL_ACTIONS), float('-inf'), device=device)

        # Get type probabilities (log scale)
        type_mask = self._get_batch_type_mask(flat_mask)
        type_log_probs = F.log_softmax(
            logits['type_logits'].masked_fill(~type_mask, float('-inf')),
            dim=-1
        )

        # For each action type, combine type prob with location prob
        for type_id, type_info in self._type_info.items():
            start, end = type_info['start'], type_info['end']
            location_head = type_info['location_head']

            if location_head == LocationHead.NONE:
                # Single action - just use type prob
                flat_logits[:, start] = type_log_probs[:, type_id]
            else:
                # Get location logits
                head_name = {
                    LocationHead.CORNER: 'corner_logits',
                    LocationHead.EDGE: 'edge_logits',
                    LocationHead.TILE: 'tile_logits',
                    LocationHead.PLAYER: 'player_logits',
                    LocationHead.TRADE_TEMPLATE: 'trade_template_logits',
                    LocationHead.SMALL_SELECT: 'small_select_logits',
                }[location_head]

                head_logits = logits[head_name][:, :end - start]

                # Apply location mask
                location_mask = flat_mask[:, start:end]
                masked_location_logits = head_logits.masked_fill(~location_mask, float('-inf'))
                location_log_probs = F.log_softmax(masked_location_logits, dim=-1)

                # Combined log prob = log(type) + log(location)
                flat_logits[:, start:end] = (
                    type_log_probs[:, type_id].unsqueeze(-1) +
                    location_log_probs
                )

        return flat_logits


class HierarchicalActorHead(nn.Module):
    """
    Wrapper that makes HierarchicalActor compatible with SB3's action_net interface.

    This handles the conversion between SB3's expected interface and our
    hierarchical structure.
    """

    def __init__(
        self,
        feature_dim: int = 531,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.hierarchical_actor = HierarchicalActor(feature_dim, hidden_dim)
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all head logits."""
        return self.hierarchical_actor.forward(features)

    def get_action(
        self,
        features: torch.Tensor,
        flat_mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action with log probability."""
        return self.hierarchical_actor.get_action(features, flat_mask, deterministic)

    def evaluate_action(
        self,
        features: torch.Tensor,
        flat_action: torch.Tensor,
        flat_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate action log probability and entropy."""
        return self.hierarchical_actor.evaluate_action(features, flat_action, flat_mask)
