"""Custom policy for Catan RL agent with attention-based features.

This module defines a custom features extractor that uses attention modules
for processing tile and player features, integrated with MaskablePPO.

Supports both flat (475-action) and hierarchical (20-type + location heads)
action structures.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict, List, Type, Optional, Tuple, Union

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskableActorCriticPolicy = object  # Placeholder

from .tile_module import TileModule
from .player_module import PlayerModule
from .corner_module import CornerModule
from .edge_module import EdgeModule
from .roll_history_module import RollHistoryModule, BankResourceEncoder, TradeStateEncoder
from .graph_module import CatanBoardGNN


class CatanFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor combining tile, corner, edge, player, and trading modules.

    This extractor processes the Catan observation by:
    1. Extracting and encoding game context (row 0)
    2. Processing tile features through TileModule with attention
    3. Processing corner features through CornerModule with attention
    4. Processing edge features through EdgeModule with MLP
    5. Processing player features through PlayerModule with MLP
    6. Processing bank resources through BankResourceEncoder
    7. Processing roll history through RollHistoryModule with attention
    8. Processing trade state through TradeStateEncoder
    9. Concatenating all features for the policy network

    Observation structure (176 x 65):
        - Row 0: Game context
        - Rows 1-19: Tile features (19 tiles)
        - Rows 20-73: Corner features (54 corners)
        - Rows 74-145: Edge features (72 edges)
        - Rows 146-149: Player features (4 players)
        - Rows 150-154: Bank resources (5 resource types)
        - Rows 155-174: Roll history (last 20 rolls)
        - Row 175: Pending trade state
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        tile_embed_dim: int = 128,
        tile_num_heads: int = 8,
        tile_num_layers: int = 2,
        corner_embed_dim: int = 64,
        corner_num_heads: int = 4,
        corner_num_layers: int = 1,
        edge_hidden_dim: int = 64,
        edge_output_dim: int = 32,
        player_hidden_dims: Optional[list] = None,
        player_output_dim: int = 128,
        context_embed_dim: int = 64,
        roll_history_embed_dim: int = 64,
        roll_history_num_heads: int = 4,
        roll_history_num_layers: int = 1,
        bank_embed_dim: int = 32,
        trade_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        """Initialize the feature extractor.

        Args:
            observation_space: The observation space.
            features_dim: Output feature dimension.
            tile_embed_dim: Embedding dimension for tile module.
            tile_num_heads: Number of attention heads for tiles.
            tile_num_layers: Number of transformer layers for tiles.
            corner_embed_dim: Embedding dimension for corner module.
            corner_num_heads: Number of attention heads for corners.
            corner_num_layers: Number of transformer layers for corners.
            edge_hidden_dim: Hidden dimension for edge module.
            edge_output_dim: Output dimension for edge module.
            player_hidden_dims: Hidden dimensions for player MLP.
            player_output_dim: Output dimension for player module.
            context_embed_dim: Embedding dimension for game context.
            roll_history_embed_dim: Embedding dimension for roll history.
            roll_history_num_heads: Number of attention heads for roll history.
            roll_history_num_layers: Number of transformer layers for roll history.
            bank_embed_dim: Embedding dimension for bank resources.
            trade_embed_dim: Embedding dimension for trade state.
            dropout: Dropout rate.
        """
        # PHASE 3: Output dimension includes strategic context (19 additional features)
        # Base features_dim (512) + strategic_context_dim (19) = 531
        strategic_context_dim = 19
        total_output_dim = features_dim + strategic_context_dim
        super().__init__(observation_space, total_output_dim)

        if player_hidden_dims is None:
            player_hidden_dims = [128, 128]

        # Get observation dimensions
        obs_shape = observation_space.shape
        self.obs_rows = obs_shape[0]  # 176
        self.feature_dim = obs_shape[1]  # 65

        # Tile module (processes rows 1-19)
        self.tile_module = TileModule(
            input_dim=self.feature_dim,
            embed_dim=tile_embed_dim,
            num_heads=tile_num_heads,
            num_layers=tile_num_layers,
            dropout=dropout,
            num_tiles=19,
        )

        # Corner module (processes rows 20-73)
        self.corner_module = CornerModule(
            input_dim=self.feature_dim,
            embed_dim=corner_embed_dim,
            num_heads=corner_num_heads,
            num_layers=corner_num_layers,
            dropout=dropout,
            num_corners=54,
        )

        # Edge module (processes rows 74-145)
        self.edge_module = EdgeModule(
            input_dim=self.feature_dim,
            hidden_dim=edge_hidden_dim,
            output_dim=edge_output_dim,
            dropout=dropout,
            num_edges=72,
        )

        # Player module (processes rows 146-149)
        self.player_module = PlayerModule(
            input_dim=self.feature_dim,
            hidden_dims=player_hidden_dims,
            output_dim=player_output_dim,
            dropout=dropout,
            num_players=4,
        )

        # Bank resource encoder (processes rows 150-154)
        self.bank_encoder = BankResourceEncoder(
            input_dim=self.feature_dim,
            output_dim=bank_embed_dim,
            num_resources=5,
            dropout=dropout,
        )

        # Roll history module (processes rows 155-174)
        self.roll_history_module = RollHistoryModule(
            input_dim=self.feature_dim,
            embed_dim=roll_history_embed_dim,
            num_heads=roll_history_num_heads,
            num_layers=roll_history_num_layers,
            dropout=dropout,
            max_rolls=20,
        )

        # Trade state encoder (processes row 175)
        self.trade_encoder = TradeStateEncoder(
            input_dim=self.feature_dim,
            output_dim=trade_embed_dim,
            dropout=dropout,
        )

        # Context encoder (processes row 0)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, context_embed_dim * 2),
            nn.LayerNorm(context_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_embed_dim * 2, context_embed_dim),
            nn.LayerNorm(context_embed_dim),
        )

        # Calculate total input dimension to final projection
        # tile + corner + edge + player*2 + context + bank + roll_history + trade
        # 128 + 64 + 32 + 256 + 64 + 32 + 64 + 32 = 672
        total_features = (
            tile_embed_dim +
            corner_embed_dim +
            edge_output_dim +
            player_output_dim * 2 +
            context_embed_dim +
            bank_embed_dim +
            roll_history_embed_dim +
            trade_embed_dim
        )

        # Final feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
        )

        # Store feature dimensions for debugging
        self._total_features = total_features

        # === Auxiliary Prediction Heads ===

        # Win probability head: predicts P(win) for each of 4 players
        # Output is softmax over 4 players, summing to 1.0
        self.win_prob_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),  # 4 players
        )

        # Final-VP regression head (#2): predicts each player's FINAL victory points
        # (perspective-relative, normalized), a denser/smoother economic value target than
        # the binary winner. Trained via MSE alongside win_prob_head when the trainer's
        # --vp-regression flag is set; its gradient also shapes the shared trunk toward
        # economic structure.
        self.vp_regression_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),  # 4 players' final VP
        )

        # Victory path head: predicts P(army) and P(road) for each of 4 players
        # Output is sigmoid for each (independent probabilities, player can have both)
        # PHASE 3: Used for strategic context and decision-making
        self.victory_path_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 8),  # 4 players × 2 paths (army, road)
        )

        # LEGACY: threat_level_head is superseded by competition features in strategic context
        # Competition features (my_P(path) * opp_P(path)) provide richer threat signal
        # Keeping the head definition for backward compatibility when loading old checkpoints
        self.threat_level_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),  # 3 opponents
        )

        # === Strategic Context ===
        # PHASE 3: Strategic context is concatenated to features for actor/critic
        # Total: 19 values (win_probs:4 + army_probs:4 + road_probs:4 +
        #                   competition_army:3 + competition_road:3 + turn_progress:1)
        self.strategic_context_dim = 19
        self._base_features_dim = features_dim  # Store base features dim (before strategic context)

        # Store auxiliary outputs for training and retrieval
        self._win_probs: Optional[torch.Tensor] = None
        self._army_probs: Optional[torch.Tensor] = None
        self._road_probs: Optional[torch.Tensor] = None
        self._threat_levels: Optional[torch.Tensor] = None
        self._strategic_context: Optional[torch.Tensor] = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.

        PHASE 3: Now returns enriched features (512 + 19 = 531 dim) with
        strategic context concatenated for actor/critic decision-making.

        Args:
            observations: Observation tensor of shape (batch, 176, 65).

        Returns:
            Feature tensor of shape (batch, features_dim + strategic_context_dim).
            Base features_dim is 512, strategic_context_dim is 19, total is 531.
        """
        # Ensure input is clean (replace NaN/Inf with 0)
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)

        # Parse observation components
        context = observations[:, 0, :]  # (batch, 65)
        tiles = observations[:, 1:20, :]  # (batch, 19, 65)
        corners = observations[:, 20:74, :]  # (batch, 54, 65)
        edges = observations[:, 74:146, :]  # (batch, 72, 65)
        players = observations[:, 146:150, :]  # (batch, 4, 65)
        bank_resources = observations[:, 150:155, :]  # (batch, 5, 65)
        roll_history = observations[:, 155:175, :]  # (batch, 20, 65)
        trade_state = observations[:, 175, :]  # (batch, 65)

        # Extract turn progress from context (feature 0 is normalized turn number)
        # Context encoding: idx 0 = turn_number * 0.01 (clamped to 1.0)
        turn_progress = context[:, 0:1]  # (batch, 1)

        # Process through modules
        _, tile_aggregated = self.tile_module(tiles)  # (batch, tile_embed_dim)
        _, corner_aggregated = self.corner_module(corners)  # (batch, corner_embed_dim)
        _, edge_aggregated = self.edge_module(edges)  # (batch, edge_output_dim)
        _, player_aggregated = self.player_module(players)  # (batch, player_output_dim * 2)
        context_encoded = self.context_encoder(context)  # (batch, context_embed_dim)
        bank_encoded = self.bank_encoder(bank_resources)  # (batch, bank_embed_dim)
        _, roll_history_aggregated = self.roll_history_module(roll_history)  # (batch, roll_history_embed_dim)
        trade_encoded = self.trade_encoder(trade_state)  # (batch, trade_embed_dim)

        # Concatenate all features
        combined = torch.cat([
            tile_aggregated,
            corner_aggregated,
            edge_aggregated,
            player_aggregated,
            context_encoded,
            bank_encoded,
            roll_history_aggregated,
            trade_encoded,
        ], dim=-1)  # (batch, total_features)

        # Project to output dimension
        features = self.feature_projection(combined)  # (batch, features_dim)

        # Ensure output is clean and bounded (prevents downstream numerical issues)
        # Use wider bounds [-50, 50] to avoid gradient saturation while still preventing overflow
        features = torch.nan_to_num(features, nan=0.0, posinf=50.0, neginf=-50.0)
        features = torch.clamp(features, min=-50.0, max=50.0)

        # Compute auxiliary predictions (stored for later retrieval)
        self._compute_auxiliary_outputs(features)

        # PHASE 3: Compute strategic context and concatenate to features
        strategic_context = self.compute_strategic_context(features, turn_progress)

        # Concatenate base features with strategic context
        enriched_features = torch.cat([features, strategic_context], dim=-1)
        # (batch, 512 + 19 = 531)

        return enriched_features

    def _compute_auxiliary_outputs(self, features: torch.Tensor) -> None:
        """Compute win probability, victory path, and threat level predictions.

        These are stored internally and can be retrieved via get_auxiliary_outputs().

        Args:
            features: Feature tensor of shape (batch, features_dim).
        """
        # Win probability: softmax over 4 players
        win_logits = self.win_prob_head(features)  # (batch, 4)
        self._win_probs = torch.softmax(win_logits, dim=-1)

        # Victory path: sigmoid for independent probabilities
        # PHASE 3: P(army) and P(road) for each player
        path_logits = self.victory_path_head(features)  # (batch, 8)
        path_probs = torch.sigmoid(path_logits)
        self._army_probs = path_probs[:, 0:4]  # (batch, 4)
        self._road_probs = path_probs[:, 4:8]  # (batch, 4)

        # LEGACY: threat_level_head no longer computed - superseded by competition features
        # Competition features (my_P(path) * opp_P(path)) in strategic context are better
        # The head weights exist for checkpoint compatibility but aren't used
        self._threat_levels = None

    def compute_strategic_context(
        self,
        features: torch.Tensor,
        turn_progress: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute strategic context for actor/critic decision-making.

        PHASE 3: This context vector is concatenated to features before
        passing to actor/critic, enabling strategic reasoning about:
        - Who is likely to win
        - What victory path each player is pursuing
        - Who is competing with the agent for the same victory condition

        Args:
            features: Feature tensor of shape (batch, features_dim).
            turn_progress: Optional normalized turn number (batch, 1).
                If None, defaults to 0.5 (mid-game assumption).

        Returns:
            Strategic context tensor of shape (batch, 19).
        """
        batch_size = features.shape[0]
        device = features.device

        # Ensure auxiliary outputs are computed
        if self._win_probs is None:
            self._compute_auxiliary_outputs(features)

        # Default turn progress if not provided
        if turn_progress is None:
            turn_progress = torch.full((batch_size, 1), 0.5, device=device)

        # Competition features: explicit interaction terms
        # Agent is always player 0 in relative encoding
        my_army = self._army_probs[:, 0:1]  # (batch, 1)
        my_road = self._road_probs[:, 0:1]  # (batch, 1)
        opp_army = self._army_probs[:, 1:4]  # (batch, 3)
        opp_road = self._road_probs[:, 1:4]  # (batch, 3)

        competition_army = my_army * opp_army  # (batch, 3)
        competition_road = my_road * opp_road  # (batch, 3)

        # Concatenate all strategic context
        strategic_context = torch.cat([
            self._win_probs,      # 4: who is winning
            self._army_probs,     # 4: P(army) per player
            self._road_probs,     # 4: P(road) per player
            competition_army,     # 3: army competition per opponent
            competition_road,     # 3: road competition per opponent
            turn_progress,        # 1: game phase indicator
        ], dim=-1)  # (batch, 19)

        self._strategic_context = strategic_context
        return strategic_context

    def get_auxiliary_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get the most recent auxiliary predictions.

        Returns:
            Dict with all auxiliary prediction tensors.
        """
        return {
            'win_probs': self._win_probs,
            'army_probs': self._army_probs,
            'road_probs': self._road_probs,
            'threat_levels': self._threat_levels,
            'strategic_context': self._strategic_context,
        }

    def get_win_probabilities(self) -> Optional[torch.Tensor]:
        """Get win probability predictions from last forward pass.

        Returns:
            Tensor of shape (batch, 4) with P(win) for each player.
        """
        return self._win_probs

    def get_victory_path_probs(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get victory path predictions from last forward pass.

        PHASE 3: Returns P(army) and P(road) for each player.

        Returns:
            Tuple of (army_probs, road_probs), each shape (batch, 4).
        """
        return self._army_probs, self._road_probs

    def get_strategic_context(self) -> Optional[torch.Tensor]:
        """Get strategic context from last forward pass.

        PHASE 3: This is the 19-dim vector concatenated to features.

        Returns:
            Tensor of shape (batch, 19) with strategic context.
        """
        return self._strategic_context

    def get_threat_levels(self) -> Optional[torch.Tensor]:
        """Get threat level predictions from last forward pass.

        Returns:
            Tensor of shape (batch, 3) with threat level for each opponent.
        """
        return self._threat_levels


class CatanGNNFeaturesExtractor(BaseFeaturesExtractor):
    """GNN-based feature extractor for Catan.

    Uses Graph Neural Networks to process the board as a heterogeneous graph,
    capturing tile-corner-edge relationships explicitly.

    Architecture v2: Replaces separate Transformer modules with unified GNN.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        gnn_num_layers: int = 2,
        gnn_num_heads: int = 4,
        player_hidden_dims: Optional[list] = None,
        player_output_dim: int = 128,
        context_embed_dim: int = 64,
        roll_history_embed_dim: int = 64,
        roll_history_num_heads: int = 4,
        roll_history_num_layers: int = 1,
        bank_embed_dim: int = 32,
        trade_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        """Initialize the GNN feature extractor.

        Args:
            observation_space: The observation space.
            features_dim: Output feature dimension.
            gnn_hidden_dim: Hidden dimension for GNN.
            gnn_output_dim: Output dimension for each GNN node type.
            gnn_num_layers: Number of GNN message passing layers.
            gnn_num_heads: Number of attention heads in GNN.
            player_hidden_dims: Hidden dimensions for player MLP.
            player_output_dim: Output dimension for player module.
            context_embed_dim: Embedding dimension for game context.
            roll_history_embed_dim: Embedding dimension for roll history.
            roll_history_num_heads: Number of attention heads for roll history.
            roll_history_num_layers: Number of transformer layers for roll history.
            bank_embed_dim: Embedding dimension for bank resources.
            trade_embed_dim: Embedding dimension for trade state.
            dropout: Dropout rate.
        """
        strategic_context_dim = 19
        total_output_dim = features_dim + strategic_context_dim
        super().__init__(observation_space, total_output_dim)

        if player_hidden_dims is None:
            player_hidden_dims = [128, 128]

        obs_shape = observation_space.shape
        self.obs_rows = obs_shape[0]
        self.feature_dim = obs_shape[1]

        # GNN for board (tiles + corners + edges)
        self.board_gnn = CatanBoardGNN(
            input_dim=self.feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            dropout=dropout,
        )

        # Player module (same as v1)
        self.player_module = PlayerModule(
            input_dim=self.feature_dim,
            hidden_dims=player_hidden_dims,
            output_dim=player_output_dim,
            dropout=dropout,
            num_players=4,
        )

        # Bank resource encoder
        self.bank_encoder = BankResourceEncoder(
            input_dim=self.feature_dim,
            output_dim=bank_embed_dim,
            num_resources=5,
            dropout=dropout,
        )

        # Roll history module
        self.roll_history_module = RollHistoryModule(
            input_dim=self.feature_dim,
            embed_dim=roll_history_embed_dim,
            num_heads=roll_history_num_heads,
            num_layers=roll_history_num_layers,
            dropout=dropout,
            max_rolls=20,
        )

        # Trade state encoder
        self.trade_encoder = TradeStateEncoder(
            input_dim=self.feature_dim,
            output_dim=trade_embed_dim,
            dropout=dropout,
        )

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, context_embed_dim * 2),
            nn.LayerNorm(context_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_embed_dim * 2, context_embed_dim),
            nn.LayerNorm(context_embed_dim),
        )

        # Total features from all modules
        # GNN: gnn_output_dim + gnn_output_dim + gnn_output_dim//2 = 128+128+64 = 320
        # Player: output_dim * 2 = 256
        # Context: 64, Bank: 32, Roll: 64, Trade: 32
        # Total: 320 + 256 + 64 + 32 + 64 + 32 = 768
        gnn_total = self.board_gnn.aggregated_dim
        total_features = (
            gnn_total +
            player_output_dim * 2 +
            context_embed_dim +
            bank_embed_dim +
            roll_history_embed_dim +
            trade_embed_dim
        )

        # Final feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
        )

        self._total_features = total_features

        # Auxiliary heads (same as v1)
        self.win_prob_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

        self.victory_path_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 8),
        )

        self.threat_level_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        self.strategic_context_dim = 19
        self._base_features_dim = features_dim

        self._win_probs: Optional[torch.Tensor] = None
        self._army_probs: Optional[torch.Tensor] = None
        self._road_probs: Optional[torch.Tensor] = None
        self._threat_levels: Optional[torch.Tensor] = None
        self._strategic_context: Optional[torch.Tensor] = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations using GNN.

        Args:
            observations: Observation tensor of shape (batch, 176, 65).

        Returns:
            Feature tensor of shape (batch, features_dim + strategic_context_dim).
        """
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)

        # Parse observation components
        context = observations[:, 0, :]
        tiles = observations[:, 1:20, :]
        corners = observations[:, 20:74, :]
        edges = observations[:, 74:146, :]
        players = observations[:, 146:150, :]
        bank_resources = observations[:, 150:155, :]
        roll_history = observations[:, 155:175, :]
        trade_state = observations[:, 175, :]

        turn_progress = context[:, 0:1]

        # Process board through GNN
        _, _, _, board_aggregated = self.board_gnn(tiles, corners, edges)

        # Process other modules
        _, player_aggregated = self.player_module(players)
        context_encoded = self.context_encoder(context)
        bank_encoded = self.bank_encoder(bank_resources)
        _, roll_history_aggregated = self.roll_history_module(roll_history)
        trade_encoded = self.trade_encoder(trade_state)

        # Concatenate all features
        combined = torch.cat([
            board_aggregated,
            player_aggregated,
            context_encoded,
            bank_encoded,
            roll_history_aggregated,
            trade_encoded,
        ], dim=-1)

        # Project to output dimension
        features = self.feature_projection(combined)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = torch.clamp(features, min=-10.0, max=10.0)

        # Compute auxiliary predictions
        self._compute_auxiliary_outputs(features)

        # Compute strategic context
        strategic_context = self.compute_strategic_context(features, turn_progress)

        # Concatenate base features with strategic context
        enriched_features = torch.cat([features, strategic_context], dim=-1)

        return enriched_features

    def _compute_auxiliary_outputs(self, features: torch.Tensor) -> None:
        """Compute auxiliary predictions."""
        win_logits = self.win_prob_head(features)
        self._win_probs = torch.softmax(win_logits, dim=-1)

        path_logits = self.victory_path_head(features)
        path_probs = torch.sigmoid(path_logits)
        self._army_probs = path_probs[:, 0:4]
        self._road_probs = path_probs[:, 4:8]

        self._threat_levels = None

    def compute_strategic_context(
        self,
        features: torch.Tensor,
        turn_progress: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute strategic context for actor/critic."""
        batch_size = features.shape[0]
        device = features.device

        if self._win_probs is None:
            self._compute_auxiliary_outputs(features)

        if turn_progress is None:
            turn_progress = torch.full((batch_size, 1), 0.5, device=device)

        my_army = self._army_probs[:, 0:1]
        my_road = self._road_probs[:, 0:1]
        opp_army = self._army_probs[:, 1:4]
        opp_road = self._road_probs[:, 1:4]

        competition_army = my_army * opp_army
        competition_road = my_road * opp_road

        strategic_context = torch.cat([
            self._win_probs,
            self._army_probs,
            self._road_probs,
            competition_army,
            competition_road,
            turn_progress,
        ], dim=-1)

        self._strategic_context = strategic_context
        return strategic_context

    def get_auxiliary_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get the most recent auxiliary predictions."""
        return {
            'win_probs': self._win_probs,
            'army_probs': self._army_probs,
            'road_probs': self._road_probs,
            'threat_levels': self._threat_levels,
            'strategic_context': self._strategic_context,
        }

    def get_win_probabilities(self) -> Optional[torch.Tensor]:
        """Get win probability predictions from last forward pass."""
        return self._win_probs

    def get_victory_path_probs(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get victory path predictions from last forward pass."""
        return self._army_probs, self._road_probs

    def get_strategic_context(self) -> Optional[torch.Tensor]:
        """Get strategic context from last forward pass."""
        return self._strategic_context

    def get_threat_levels(self) -> Optional[torch.Tensor]:
        """Get threat level predictions from last forward pass."""
        return self._threat_levels


class CatanPolicy(MaskableActorCriticPolicy):
    """Custom MaskablePPO policy for Catan with attention-based features.

    This policy uses the CatanFeaturesExtractor to process observations
    before passing to the actor and critic networks.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        net_arch: Optional[Dict] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize the policy.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            lr_schedule: Learning rate schedule.
            net_arch: Network architecture for actor/critic.
            activation_fn: Activation function.
            features_extractor_kwargs: Arguments for feature extractor.
        """
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        if net_arch is None:
            # Default network architecture
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=CatanFeaturesExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs,
        )


def create_catan_policy_kwargs(
    features_dim: int = 512,
    tile_embed_dim: int = 128,
    tile_num_heads: int = 8,
    tile_num_layers: int = 2,
    corner_embed_dim: int = 64,
    corner_num_heads: int = 4,
    corner_num_layers: int = 1,
    edge_hidden_dim: int = 64,
    edge_output_dim: int = 32,
    player_hidden_dims: Optional[list] = None,
    player_output_dim: int = 128,
    context_embed_dim: int = 64,
    roll_history_embed_dim: int = 64,
    roll_history_num_heads: int = 4,
    roll_history_num_layers: int = 1,
    bank_embed_dim: int = 32,
    trade_embed_dim: int = 32,
    dropout: float = 0.1,
    pi_hidden_dims: Optional[list] = None,
    vf_hidden_dims: Optional[list] = None,
) -> Dict:
    """Create policy keyword arguments for MaskablePPO.

    This helper function creates the policy_kwargs dictionary needed
    to instantiate MaskablePPO with the custom CatanPolicy.

    Args:
        features_dim: Output dimension of feature extractor.
        tile_embed_dim: Embedding dimension for tile module.
        tile_num_heads: Number of attention heads for tiles.
        tile_num_layers: Number of transformer layers for tiles.
        corner_embed_dim: Embedding dimension for corner module.
        corner_num_heads: Number of attention heads for corners.
        corner_num_layers: Number of transformer layers for corners.
        edge_hidden_dim: Hidden dimension for edge module.
        edge_output_dim: Output dimension for edge module.
        player_hidden_dims: Hidden dimensions for player MLP.
        player_output_dim: Output dimension for player module.
        context_embed_dim: Embedding dimension for game context.
        roll_history_embed_dim: Embedding dimension for roll history.
        roll_history_num_heads: Number of attention heads for roll history.
        roll_history_num_layers: Number of transformer layers for roll history.
        bank_embed_dim: Embedding dimension for bank resources.
        trade_embed_dim: Embedding dimension for trade state.
        dropout: Dropout rate.
        pi_hidden_dims: Hidden dimensions for policy network.
        vf_hidden_dims: Hidden dimensions for value network.

    Returns:
        Dictionary of policy keyword arguments.
    """
    if pi_hidden_dims is None:
        pi_hidden_dims = [256, 256]
    if vf_hidden_dims is None:
        vf_hidden_dims = [256, 256]
    if player_hidden_dims is None:
        player_hidden_dims = [128, 128]

    return {
        'features_extractor_class': CatanFeaturesExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'tile_embed_dim': tile_embed_dim,
            'tile_num_heads': tile_num_heads,
            'tile_num_layers': tile_num_layers,
            'corner_embed_dim': corner_embed_dim,
            'corner_num_heads': corner_num_heads,
            'corner_num_layers': corner_num_layers,
            'edge_hidden_dim': edge_hidden_dim,
            'edge_output_dim': edge_output_dim,
            'player_hidden_dims': player_hidden_dims,
            'player_output_dim': player_output_dim,
            'context_embed_dim': context_embed_dim,
            'roll_history_embed_dim': roll_history_embed_dim,
            'roll_history_num_heads': roll_history_num_heads,
            'roll_history_num_layers': roll_history_num_layers,
            'bank_embed_dim': bank_embed_dim,
            'trade_embed_dim': trade_embed_dim,
            'dropout': dropout,
        },
        'net_arch': dict(pi=pi_hidden_dims, vf=vf_hidden_dims),
        'activation_fn': nn.Tanh,
    }


def create_gnn_policy_kwargs(
    features_dim: int = 512,
    gnn_hidden_dim: int = 128,
    gnn_output_dim: int = 128,
    gnn_num_layers: int = 2,
    gnn_num_heads: int = 4,
    player_hidden_dims: Optional[list] = None,
    player_output_dim: int = 128,
    context_embed_dim: int = 64,
    roll_history_embed_dim: int = 64,
    roll_history_num_heads: int = 4,
    roll_history_num_layers: int = 1,
    bank_embed_dim: int = 32,
    trade_embed_dim: int = 32,
    dropout: float = 0.1,
    pi_hidden_dims: Optional[list] = None,
    vf_hidden_dims: Optional[list] = None,
) -> Dict:
    """Create policy keyword arguments for GNN-based MaskablePPO.

    This helper function creates the policy_kwargs dictionary needed
    to instantiate MaskablePPO with the GNN-based CatanGNNFeaturesExtractor.

    Args:
        features_dim: Output dimension of feature extractor.
        gnn_hidden_dim: Hidden dimension for GNN layers.
        gnn_output_dim: Output dimension for each GNN node type.
        gnn_num_layers: Number of GNN message passing layers.
        gnn_num_heads: Number of attention heads in GNN.
        player_hidden_dims: Hidden dimensions for player MLP.
        player_output_dim: Output dimension for player module.
        context_embed_dim: Embedding dimension for game context.
        roll_history_embed_dim: Embedding dimension for roll history.
        roll_history_num_heads: Number of attention heads for roll history.
        roll_history_num_layers: Number of transformer layers for roll history.
        bank_embed_dim: Embedding dimension for bank resources.
        trade_embed_dim: Embedding dimension for trade state.
        dropout: Dropout rate.
        pi_hidden_dims: Hidden dimensions for policy network.
        vf_hidden_dims: Hidden dimensions for value network.

    Returns:
        Dictionary of policy keyword arguments.
    """
    if pi_hidden_dims is None:
        pi_hidden_dims = [256, 256]
    if vf_hidden_dims is None:
        vf_hidden_dims = [256, 256]
    if player_hidden_dims is None:
        player_hidden_dims = [128, 128]

    return {
        'features_extractor_class': CatanGNNFeaturesExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'gnn_hidden_dim': gnn_hidden_dim,
            'gnn_output_dim': gnn_output_dim,
            'gnn_num_layers': gnn_num_layers,
            'gnn_num_heads': gnn_num_heads,
            'player_hidden_dims': player_hidden_dims,
            'player_output_dim': player_output_dim,
            'context_embed_dim': context_embed_dim,
            'roll_history_embed_dim': roll_history_embed_dim,
            'roll_history_num_heads': roll_history_num_heads,
            'roll_history_num_layers': roll_history_num_layers,
            'bank_embed_dim': bank_embed_dim,
            'trade_embed_dim': trade_embed_dim,
            'dropout': dropout,
        },
        'net_arch': dict(pi=pi_hidden_dims, vf=vf_hidden_dims),
        'activation_fn': nn.Tanh,
    }


# =============================================================================
# Hierarchical Policy Support
# =============================================================================

def get_policy_class(hierarchical: bool = False):
    """Get the appropriate policy class based on action structure.

    Args:
        hierarchical: If True, return HierarchicalCatanPolicy, else CatanPolicy.

    Returns:
        Policy class to use with MaskablePPO.
    """
    if hierarchical:
        from .hierarchical_policy import HierarchicalCatanPolicy
        return HierarchicalCatanPolicy
    else:
        return CatanPolicy


def create_policy_kwargs_auto(
    hierarchical: bool = False,
    use_gnn: bool = False,
    **kwargs
) -> Dict:
    """Create policy kwargs with automatic selection of flat vs hierarchical.

    Args:
        hierarchical: If True, create hierarchical policy kwargs.
        use_gnn: If True, use GNN-based feature extractor.
        **kwargs: Additional arguments passed to the appropriate helper function.

    Returns:
        Dictionary of policy keyword arguments.
    """
    if hierarchical:
        from .hierarchical_policy import create_hierarchical_policy_kwargs
        return create_hierarchical_policy_kwargs(**kwargs)
    elif use_gnn:
        return create_gnn_policy_kwargs(**kwargs)
    else:
        return create_catan_policy_kwargs(**kwargs)
