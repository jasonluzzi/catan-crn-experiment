"""
Hierarchical Behavioral Cloning Model.

Uses two-stage action prediction:
1. Action type (20 categories)
2. Location/parameter based on type (6 heads)

This structure provides better learning signal by separating "what to do"
from "where to do it", addressing the position memorization problem.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Set

from .hierarchical_actor import HierarchicalActor
from .action_decomposer import (
    ActionDecomposer,
    HierarchicalActionType,
    LocationHead,
    TYPE_TO_LOCATION_HEAD,
    NUM_ACTION_TYPES,
    FLAT_RANGES,
)
from .action_mask_extractor import (
    get_action_type_mask,
    get_full_hierarchical_mask,
    extract_corner_validity,
    extract_edge_validity,
    extract_corner_ownership,
    NUM_CORNERS,
    NUM_EDGES,
)


# Constants
TOTAL_ACTIONS = 475
NUM_PLAYERS = 4
OBS_SHAPE = (176, 65)

# Phase detection constants
ACTION_SETUP_SETTLEMENT_START = 235
ACTION_SETUP_SETTLEMENT_END = 289
ACTION_SETUP_ROAD_START = 289
ACTION_SETUP_ROAD_END = 361


class HierarchicalBCModel(nn.Module):
    """
    Behavioral cloning model with hierarchical action structure.

    Architecture:
        observation → feature_extractor → features (512)
                                       ↓
                   feature_projection → projected_features (512)
                                       ↓
                   hierarchical_actor → action_type_logits (20)
                                      → corner_logits (54)
                                      → edge_logits (72)
                                      → tile_logits (19)
                                      → player_logits (4)
                                      → trade_template_logits (50)
                                      → small_select_logits (25)
    """

    def __init__(
        self,
        use_gnn: bool = True,
        features_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.use_gnn = use_gnn
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.decomposer = ActionDecomposer()

        # Import and build feature extractor
        if use_gnn:
            from .graph_module import CatanBoardGNN
            from .player_module import PlayerModule

            self.board_gnn = CatanBoardGNN(
                input_dim=65,
                hidden_dim=128,
                output_dim=128,
                num_layers=2,
                num_heads=4,
            )
            self.player_module = PlayerModule(
                input_dim=65,
                hidden_dims=[128, 128],
                output_dim=128,
            )

            # Context, bank, roll history, trade encoders
            self.context_encoder = nn.Sequential(
                nn.Linear(65, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.bank_encoder = nn.Sequential(
                nn.Linear(5 * 65, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            self.roll_history_encoder = nn.Sequential(
                nn.Linear(20 * 65, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.trade_encoder = nn.Sequential(
                nn.Linear(65, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )

            # GNN outputs: 320 + 256 + 64 + 32 + 64 + 32 = 768
            combined_dim = 320 + 256 + 64 + 32 + 64 + 32

        else:
            # Transformer-based - use existing extractor
            from .catan_policy import CatanFeaturesExtractor
            from gymnasium import spaces

            obs_space = spaces.Box(low=0, high=1, shape=OBS_SHAPE, dtype=np.float32)
            self.features_extractor = CatanFeaturesExtractor(
                obs_space,
                features_dim=features_dim,
            )
            # CatanFeaturesExtractor outputs features_dim + strategic_context_dim (19)
            combined_dim = features_dim + 19

        # Feature projection.
        # GNN path: project the concatenated module outputs (768) -> features_dim.
        # Transformer path: feed the CatanFeaturesExtractor's enriched output
        # (features_dim + 19 = 531) STRAIGHT THROUGH (Identity), so the actor takes 531 —
        # identical to the live HierarchicalCatanPolicy (which feeds 531 to a 256x531 actor).
        # This makes features_extractor.* and hierarchical_actor.* transfer 1:1 into the live
        # policy for eval + RL fine-tuning (no re-train). See value_head_experiment/bc_policy_pivot notes.
        if use_gnn:
            self.feature_projection = nn.Sequential(
                nn.Linear(combined_dim, features_dim),
                nn.ReLU(),
                nn.LayerNorm(features_dim),
            )
            actor_in = features_dim
        else:
            self.feature_projection = nn.Identity()
            actor_in = combined_dim  # = features_dim + 19 = 531, matches the live actor

        # Hierarchical actor head
        self.hierarchical_actor = HierarchicalActor(
            feature_dim=actor_in,
            hidden_dim=hidden_dim,
        )

        # Auxiliary heads (BC-only; not transferred to the live policy)
        self.win_head = nn.Sequential(
            nn.Linear(actor_in, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_PLAYERS),
        )

        self.lr_head = nn.Sequential(
            nn.Linear(actor_in, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_PLAYERS),
        )

        self.la_head = nn.Sequential(
            nn.Linear(actor_in, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_PLAYERS),
        )

    def extract_features(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observation.

        Args:
            observation: (batch, 176, 65) tensor

        Returns:
            (batch, features_dim) projected features
        """
        batch_size = observation.shape[0]

        if self.use_gnn:
            # Extract components from observation
            context = observation[:, 0, :]
            tiles = observation[:, 1:20, :]
            corners = observation[:, 20:74, :]
            edges = observation[:, 74:146, :]
            players = observation[:, 146:150, :]
            bank = observation[:, 150:155, :].reshape(batch_size, -1)
            roll_history = observation[:, 155:175, :].reshape(batch_size, -1)
            trade_state = observation[:, 175, :]

            # Process through GNN
            _, _, _, gnn_aggregated = self.board_gnn(tiles, corners, edges)

            # Process players
            _, player_features = self.player_module(players)

            # Process other components
            context_features = self.context_encoder(context)
            bank_features = self.bank_encoder(bank)
            roll_features = self.roll_history_encoder(roll_history)
            trade_features = self.trade_encoder(trade_state)

            # Concatenate all features
            combined = torch.cat([
                gnn_aggregated,
                player_features,
                context_features,
                bank_features,
                roll_features,
                trade_features,
            ], dim=1)

        else:
            # Use transformer-based extractor
            combined = self.features_extractor(observation)

        # Project features
        features = self.feature_projection(combined)
        return features

    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all outputs.

        Args:
            observation: (batch, 176, 65) tensor

        Returns:
            Dict containing all head logits and features
        """
        # Extract features
        features = self.extract_features(observation)

        # Get hierarchical action logits
        actor_outputs = self.hierarchical_actor.forward(features)

        # Auxiliary predictions
        win_logits = self.win_head(features)
        lr_logits = self.lr_head(features)
        la_logits = self.la_head(features)

        return {
            'features': features,
            'type_logits': actor_outputs['type_logits'],
            'corner_logits': actor_outputs['corner_logits'],
            'edge_logits': actor_outputs['edge_logits'],
            'tile_logits': actor_outputs['tile_logits'],
            'player_logits': actor_outputs['player_logits'],
            'trade_template_logits': actor_outputs['trade_template_logits'],
            'small_select_logits': actor_outputs['small_select_logits'],
            'win_logits': win_logits,
            'lr_logits': lr_logits,
            'la_logits': la_logits,
        }

    def predict_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Predict flat action from observation.

        Args:
            observation: (batch, 176, 65) tensor
            deterministic: If True, select most probable action

        Returns:
            (batch,) flat action indices
        """
        features = self.extract_features(observation)

        # Create dummy mask (all valid) since we don't have mask info
        batch_size = observation.shape[0]
        device = observation.device
        dummy_mask = torch.ones(batch_size, TOTAL_ACTIONS, dtype=torch.bool, device=device)

        flat_actions, _ = self.hierarchical_actor.get_action(
            features, dummy_mask, deterministic
        )
        return flat_actions


def compute_hierarchical_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    decomposer: ActionDecomposer,
    type_weight: float = 1.0,
    location_weight: float = 1.0,
    aux_weight: float = 0.5,
    setup_weight_multiplier: float = 2.0,
    use_action_masking: bool = True,
    skill_weight_win: float = 1.0,
    road_loc_weight: float = 1.0,
    open_loc_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute hierarchical BC loss.

    Loss = type_ce + location_ce + auxiliary_losses

    Setup actions (SETUP_SETTLEMENT, SETUP_ROAD) can be weighted higher
    to address the imbalance problem.

    Args:
        outputs: Model outputs dict
        batch: Batch dict with 'action', 'winner_idx', 'observation', etc.
        decomposer: ActionDecomposer instance
        type_weight: Weight for action type loss
        location_weight: Weight for location loss
        aux_weight: Weight for auxiliary losses
        setup_weight_multiplier: Extra weight for setup phase actions
        use_action_masking: If True, mask invalid action types based on game phase

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    batch_size = batch['action'].shape[0]
    device = batch['action'].device

    # Convert flat actions to hierarchical labels
    flat_actions = batch['action'].cpu().numpy()
    type_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    location_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    location_heads = []
    is_setup = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Skill-weighted cloning: upweight the loss on moves made by the game's
    # eventual winner (skill_weight_win) vs losers (1.0). Neutral (=1.0) → no
    # effect. Applied multiplicatively to both the type and location losses.
    skill_weights = None
    if skill_weight_win != 1.0 and 'player_won' in batch:
        pw = batch['player_won'].to(device=device, dtype=torch.float32)
        skill_weights = 1.0 + (skill_weight_win - 1.0) * pw  # winners → W, losers → 1

    for i, action in enumerate(flat_actions):
        type_id, loc_id = decomposer.flat_to_hierarchical(int(action))
        type_labels[i] = type_id
        location_labels[i] = loc_id
        location_heads.append(TYPE_TO_LOCATION_HEAD[HierarchicalActionType(type_id)])
        # Check if this is a setup action
        is_setup[i] = bool(
            (ACTION_SETUP_SETTLEMENT_START <= action < ACTION_SETUP_SETTLEMENT_END) or
            (ACTION_SETUP_ROAD_START <= action < ACTION_SETUP_ROAD_END)
        )

    # Get type logits and optionally apply action masking
    type_logits = outputs['type_logits']

    if use_action_masking and 'observation' in batch:
        # Get comprehensive action type mask (phase + resources)
        type_mask = get_action_type_mask(batch['observation'], use_resource_masking=True)

        # Mask invalid types by setting their logits to -inf
        # Safety: for samples where ground truth type is masked, skip masking
        masked_type_logits = type_logits.clone()
        for i in range(batch_size):
            gt_type = type_labels[i].item()
            if type_mask[i, gt_type]:
                # Ground truth is valid, apply mask
                masked_type_logits[i, ~type_mask[i]] = float('-inf')
            # else: ground truth would be masked, keep original logits for this sample

        # Extract location validity masks
        corner_valid = extract_corner_validity(batch['observation'])
        edge_valid = extract_edge_validity(batch['observation'])
        corner_owner, building_type = extract_corner_ownership(batch['observation'])
    else:
        masked_type_logits = type_logits
        corner_valid = None
        edge_valid = None
        corner_owner = None
        building_type = None

    # Type classification loss - per sample with setup weighting
    type_losses_per_sample = F.cross_entropy(
        masked_type_logits, type_labels, reduction='none'
    )
    # Apply setup weight multiplier to type loss for setup actions
    type_weights = torch.ones(batch_size, device=device)
    type_weights[is_setup] = setup_weight_multiplier
    if skill_weights is not None:
        type_weights = type_weights * skill_weights
    type_loss = (type_losses_per_sample * type_weights).mean()

    # Location losses - compute per head, then aggregate
    location_losses = []
    per_head_losses = {head: [] for head in LocationHead if head != LocationHead.NONE}

    for i in range(batch_size):
        head = location_heads[i]
        if head == LocationHead.NONE:
            continue

        # Get the appropriate logits and truncate to actual size
        type_id = type_labels[i].item()
        action_type = HierarchicalActionType(type_id)
        actual_size = decomposer.get_location_size(type_id)

        if head == LocationHead.CORNER:
            logits = outputs['corner_logits'][i, :actual_size].clone()
            # Apply corner location mask if available
            if use_action_masking and corner_valid is not None:
                if action_type in (HierarchicalActionType.BUILD_SETTLEMENT,
                                  HierarchicalActionType.SETUP_SETTLEMENT):
                    loc_mask = corner_valid[i, :actual_size]
                elif action_type == HierarchicalActionType.BUILD_CITY:
                    # Can only upgrade own settlements
                    loc_mask = (corner_owner[i, :actual_size] == 0) & (building_type[i, :actual_size] == 1)
                else:
                    loc_mask = None
                if loc_mask is not None and loc_mask.any():
                    # Safety: don't mask if target would be masked out
                    target = location_labels[i].item()
                    if target < actual_size and loc_mask[target]:
                        logits[~loc_mask] = float('-inf')
        elif head == LocationHead.EDGE:
            logits = outputs['edge_logits'][i, :actual_size].clone()
            # Apply edge location mask if available
            if use_action_masking and edge_valid is not None:
                if action_type in (HierarchicalActionType.BUILD_ROAD,
                                  HierarchicalActionType.SETUP_ROAD):
                    loc_mask = edge_valid[i, :actual_size]
                    # Safety: don't mask if target would be masked out
                    target = location_labels[i].item()
                    if loc_mask.any() and target < actual_size and loc_mask[target]:
                        logits[~loc_mask] = float('-inf')
        elif head == LocationHead.TILE:
            logits = outputs['tile_logits'][i, :actual_size]
        elif head == LocationHead.PLAYER:
            logits = outputs['player_logits'][i, :actual_size]
        elif head == LocationHead.TRADE_TEMPLATE:
            logits = outputs['trade_template_logits'][i, :actual_size]
        elif head == LocationHead.SMALL_SELECT:
            logits = outputs['small_select_logits'][i, :actual_size]
        else:
            continue

        # Cross-entropy loss for this location
        target = location_labels[i]
        if target >= actual_size:
            # Safety check - skip invalid targets
            continue

        loc_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

        # Apply setup weight multiplier if this is a setup action
        if is_setup[i]:
            loc_loss = loc_loss * setup_weight_multiplier
        if skill_weights is not None:
            loc_loss = loc_loss * skill_weights[i]
        # Location-targeted weighting: upweight the LOCATION loss for the weak
        # compounding heads identified by the per-head diagnostic — midgame road
        # direction (BUILD_ROAD) and opening settlement (SETUP_SETTLEMENT).
        if action_type == HierarchicalActionType.BUILD_ROAD:
            loc_loss = loc_loss * road_loc_weight
        elif action_type == HierarchicalActionType.SETUP_SETTLEMENT:
            loc_loss = loc_loss * open_loc_weight

        location_losses.append(loc_loss)
        per_head_losses[head].append(loc_loss.item())

    # Average location loss
    if location_losses:
        location_loss = torch.stack(location_losses).mean()
    else:
        location_loss = torch.tensor(0.0, device=device)

    # Auxiliary losses (same as original BC)
    win_mask = batch['winner_idx'] >= 0
    if win_mask.any():
        win_loss = F.cross_entropy(
            outputs['win_logits'][win_mask],
            batch['winner_idx'][win_mask],
        )
    else:
        win_loss = torch.tensor(0.0, device=device)

    lr_mask = batch['lr_holder_idx'] >= 0
    if lr_mask.any():
        lr_loss = F.cross_entropy(
            outputs['lr_logits'][lr_mask],
            batch['lr_holder_idx'][lr_mask],
        )
    else:
        lr_loss = torch.tensor(0.0, device=device)

    la_mask = batch['la_holder_idx'] >= 0
    if la_mask.any():
        la_loss = F.cross_entropy(
            outputs['la_logits'][la_mask],
            batch['la_holder_idx'][la_mask],
        )
    else:
        la_loss = torch.tensor(0.0, device=device)

    # Total loss
    total_loss = (
        type_weight * type_loss +
        location_weight * location_loss +
        aux_weight * (win_loss + lr_loss + la_loss) / 3
    )

    # Compute per-head average losses for metrics
    head_loss_avgs = {}
    for head, losses in per_head_losses.items():
        if losses:
            head_loss_avgs[f'{head.name.lower()}_loss'] = sum(losses) / len(losses)
        else:
            head_loss_avgs[f'{head.name.lower()}_loss'] = 0.0

    metrics = {
        'loss': total_loss.item(),
        'type_loss': type_loss.item(),
        'location_loss': location_loss.item(),
        'win_loss': win_loss.item() if isinstance(win_loss, torch.Tensor) else win_loss,
        'lr_loss': lr_loss.item() if isinstance(lr_loss, torch.Tensor) else lr_loss,
        'la_loss': la_loss.item() if isinstance(la_loss, torch.Tensor) else la_loss,
        **head_loss_avgs,
    }

    return total_loss, metrics


def compute_hierarchical_accuracy(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    decomposer: ActionDecomposer,
    use_action_masking: bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy metrics for hierarchical model.

    Metrics:
    - type_acc: Action type prediction accuracy
    - location_acc: Location prediction accuracy (given correct type)
    - joint_acc: Both type AND location correct (equivalent to flat accuracy)
    - Per-head location accuracies

    Args:
        outputs: Model outputs dict
        batch: Batch dict with 'action', 'winner_idx', 'observation', etc.
        decomposer: ActionDecomposer instance
        use_action_masking: If True, mask invalid action types based on game phase

    Returns:
        Dict of accuracy metrics
    """
    batch_size = batch['action'].shape[0]
    device = batch['action'].device

    # Convert flat actions to hierarchical labels
    flat_actions = batch['action'].cpu().numpy()
    type_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    location_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    location_heads = []

    for i, action in enumerate(flat_actions):
        type_id, loc_id = decomposer.flat_to_hierarchical(int(action))
        type_labels[i] = type_id
        location_labels[i] = loc_id
        location_heads.append(TYPE_TO_LOCATION_HEAD[HierarchicalActionType(type_id)])

    # Get type logits and optionally apply action masking
    type_logits = outputs['type_logits']

    if use_action_masking and 'observation' in batch:
        # Get comprehensive action type mask (phase + resources)
        type_mask = get_action_type_mask(batch['observation'], use_resource_masking=True)

        # Mask invalid types by setting their logits to -inf before argmax
        masked_type_logits = type_logits.clone()
        masked_type_logits[~type_mask] = float('-inf')
        type_preds = masked_type_logits.argmax(dim=1)

        # Extract location validity masks
        corner_valid = extract_corner_validity(batch['observation'])
        edge_valid = extract_edge_validity(batch['observation'])
        corner_owner, building_type = extract_corner_ownership(batch['observation'])
    else:
        type_preds = type_logits.argmax(dim=1)
        corner_valid = None
        edge_valid = None
        corner_owner = None
        building_type = None

    type_correct = (type_preds == type_labels).float()
    type_acc = type_correct.mean().item()

    # Per-type accuracy for key types
    roll_dice_mask = type_labels == HierarchicalActionType.ROLL_DICE
    end_turn_mask = type_labels == HierarchicalActionType.END_TURN
    setup_settle_mask = type_labels == HierarchicalActionType.SETUP_SETTLEMENT
    setup_road_mask = type_labels == HierarchicalActionType.SETUP_ROAD
    build_settle_mask = type_labels == HierarchicalActionType.BUILD_SETTLEMENT
    build_road_mask = type_labels == HierarchicalActionType.BUILD_ROAD

    # Accuracy excluding trivial actions (ROLL_DICE, END_TURN)
    non_trivial_mask = ~(roll_dice_mask | end_turn_mask)
    if non_trivial_mask.any():
        type_acc_nontrivial = type_correct[non_trivial_mask].mean().item()
    else:
        type_acc_nontrivial = 0.0

    # Location accuracy (per sample, only for samples with location)
    location_correct = torch.zeros(batch_size, dtype=torch.float, device=device)
    per_head_correct = {head: [] for head in LocationHead if head != LocationHead.NONE}
    per_head_total = {head: 0 for head in LocationHead if head != LocationHead.NONE}

    for i in range(batch_size):
        head = location_heads[i]
        if head == LocationHead.NONE:
            location_correct[i] = 1.0  # No location to predict
            continue

        type_id = type_labels[i].item()
        action_type = HierarchicalActionType(type_id)
        actual_size = decomposer.get_location_size(type_id)

        target = location_labels[i].item()

        if head == LocationHead.CORNER:
            logits = outputs['corner_logits'][i, :actual_size].clone()
            # Apply corner location mask if available
            if use_action_masking and corner_valid is not None:
                if action_type in (HierarchicalActionType.BUILD_SETTLEMENT,
                                  HierarchicalActionType.SETUP_SETTLEMENT):
                    loc_mask = corner_valid[i, :actual_size]
                elif action_type == HierarchicalActionType.BUILD_CITY:
                    loc_mask = (corner_owner[i, :actual_size] == 0) & (building_type[i, :actual_size] == 1)
                else:
                    loc_mask = None
                if loc_mask is not None and loc_mask.any():
                    # Safety: don't mask if target would be masked out
                    if target < actual_size and loc_mask[target]:
                        logits[~loc_mask] = float('-inf')
        elif head == LocationHead.EDGE:
            logits = outputs['edge_logits'][i, :actual_size].clone()
            # Apply edge location mask if available
            if use_action_masking and edge_valid is not None:
                if action_type in (HierarchicalActionType.BUILD_ROAD,
                                  HierarchicalActionType.SETUP_ROAD):
                    loc_mask = edge_valid[i, :actual_size]
                    # Safety: don't mask if target would be masked out
                    if loc_mask.any() and target < actual_size and loc_mask[target]:
                        logits[~loc_mask] = float('-inf')
        elif head == LocationHead.TILE:
            logits = outputs['tile_logits'][i, :actual_size]
        elif head == LocationHead.PLAYER:
            logits = outputs['player_logits'][i, :actual_size]
        elif head == LocationHead.TRADE_TEMPLATE:
            logits = outputs['trade_template_logits'][i, :actual_size]
        elif head == LocationHead.SMALL_SELECT:
            logits = outputs['small_select_logits'][i, :actual_size]
        else:
            continue

        pred = logits.argmax().item()
        correct = float(pred == target)

        location_correct[i] = correct
        per_head_correct[head].append(correct)
        per_head_total[head] += 1

    location_acc = location_correct.mean().item()

    # Location accuracy excluding NONE types (actual location predictions)
    has_location_mask = torch.tensor(
        [h != LocationHead.NONE for h in location_heads],
        dtype=torch.bool, device=device
    )
    if has_location_mask.any():
        location_acc_real = location_correct[has_location_mask].mean().item()
    else:
        location_acc_real = 0.0

    # Joint accuracy (both type and location correct)
    joint_correct = type_correct * location_correct
    joint_acc = joint_correct.mean().item()

    # Setup action accuracy
    setup_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i, action in enumerate(flat_actions):
        if (ACTION_SETUP_SETTLEMENT_START <= action < ACTION_SETUP_SETTLEMENT_END or
            ACTION_SETUP_ROAD_START <= action < ACTION_SETUP_ROAD_END):
            setup_mask[i] = True

    setup_count = setup_mask.sum().item()
    if setup_mask.any():
        setup_acc = joint_correct[setup_mask].mean().item()
        # Also track type accuracy specifically for setup actions
        setup_type_acc = type_correct[setup_mask].mean().item()
        # Debug: what types is the model predicting for setup actions?
        setup_pred_types = type_preds[setup_mask].cpu().numpy()
        setup_true_types = type_labels[setup_mask].cpu().numpy()
        # Count predictions by type for debugging
        from collections import Counter
        setup_pred_counts = Counter(setup_pred_types.tolist())
        setup_true_counts = Counter(setup_true_types.tolist())
    else:
        setup_acc = 0.0
        setup_type_acc = 0.0
        setup_pred_counts = {}
        setup_true_counts = {}

    # Per-head accuracies
    head_accs = {}
    for head, corrects in per_head_correct.items():
        if corrects:
            head_accs[f'{head.name.lower()}_acc'] = sum(corrects) / len(corrects)
        else:
            head_accs[f'{head.name.lower()}_acc'] = 0.0

    # Auxiliary accuracies
    win_mask = batch['winner_idx'] >= 0
    if win_mask.any():
        win_preds = outputs['win_logits'][win_mask].argmax(dim=1)
        win_acc = (win_preds == batch['winner_idx'][win_mask]).float().mean().item()
    else:
        win_acc = 0.0

    lr_mask = batch['lr_holder_idx'] >= 0
    if lr_mask.any():
        lr_preds = outputs['lr_logits'][lr_mask].argmax(dim=1)
        lr_acc = (lr_preds == batch['lr_holder_idx'][lr_mask]).float().mean().item()
    else:
        lr_acc = 0.0

    la_mask = batch['la_holder_idx'] >= 0
    if la_mask.any():
        la_preds = outputs['la_logits'][la_mask].argmax(dim=1)
        la_acc = (la_preds == batch['la_holder_idx'][la_mask]).float().mean().item()
    else:
        la_acc = 0.0

    return {
        'type_acc': type_acc,
        'type_acc_nontrivial': type_acc_nontrivial,  # Excludes ROLL_DICE, END_TURN
        'location_acc': location_acc,
        'location_acc_real': location_acc_real,  # Excludes NONE types
        'joint_acc': joint_acc,  # Equivalent to flat top-1 accuracy
        'setup_acc': setup_acc,
        'setup_type_acc': setup_type_acc,  # Type accuracy on setup actions only
        'setup_count': setup_count,  # Number of setup actions in batch
        'setup_pred_counts': dict(setup_pred_counts),  # Debug: what model predicted
        'setup_true_counts': dict(setup_true_counts),  # Debug: ground truth types
        'win_acc': win_acc,
        'lr_acc': lr_acc,
        'la_acc': la_acc,
        **head_accs,
    }
