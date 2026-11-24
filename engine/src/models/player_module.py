"""Player state encoder module for processing player features.

This module encodes player states using shared MLPs, with special handling
for the current agent versus opponents.
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class PlayerModule(nn.Module):
    """MLP encoder for player states.

    This module processes player features using a shared MLP architecture,
    with the ability to distinguish the current agent from opponents.
    """

    def __init__(
        self,
        input_dim: int = 65,
        hidden_dims: List[int] = None,
        output_dim: int = 128,
        dropout: float = 0.1,
        num_players: int = 4,
    ):
        """Initialize the player module.

        Args:
            input_dim: Input feature dimension per player (default 65).
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output embedding dimension.
            dropout: Dropout rate.
            num_players: Number of players (default 4).
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_players = num_players

        # Learnable embedding to mark the current agent
        self.agent_marker = nn.Parameter(torch.randn(1, input_dim) * 0.02)

        # Player position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(num_players, input_dim) * 0.02
        )

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)

        # Attention-based aggregation for opponents
        self.opponent_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )

        # Final projection for self vs opponents
        self.self_projection = nn.Linear(output_dim, output_dim)
        self.opponent_projection = nn.Linear(output_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize module weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        player_features: torch.Tensor,
        current_player_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process player features through MLP.

        The player features are already ordered with the perspective player first
        (index 0), followed by opponents in relative order.

        Args:
            player_features: Player features of shape (batch, 4, 65).
            current_player_idx: Index of current agent in the features (default 0,
                since StateEncoder already reorders from agent's perspective).

        Returns:
            Tuple of:
                - Per-player embeddings: (batch, 4, output_dim)
                - Aggregated embedding: (batch, output_dim * 2) - self + aggregated opponents
        """
        batch_size = player_features.size(0)

        # Add position embeddings
        pos_embed = self.position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = player_features + pos_embed

        # Add agent marker to current player
        agent_marker = self.agent_marker.unsqueeze(0).expand(batch_size, -1, -1)
        x[:, current_player_idx:current_player_idx+1, :] = (
            x[:, current_player_idx:current_player_idx+1, :] + agent_marker
        )

        # Apply MLP to each player
        # Reshape for batch processing
        x = x.view(batch_size * self.num_players, -1)  # (batch * 4, input_dim)
        x = self.mlp(x)  # (batch * 4, output_dim)
        player_embeddings = x.view(batch_size, self.num_players, -1)  # (batch, 4, output_dim)

        # Extract self embedding
        self_embedding = player_embeddings[:, current_player_idx, :]  # (batch, output_dim)
        self_embedding = self.self_projection(self_embedding)

        # Get opponent embeddings (all players except current)
        opponent_indices = [i for i in range(self.num_players) if i != current_player_idx]
        opponent_embeddings = player_embeddings[:, opponent_indices, :]  # (batch, 3, output_dim)

        # Attention-based aggregation of opponents with numerical stability
        attn_logits = self.opponent_attention(opponent_embeddings)  # (batch, 3, 1)
        # Clamp attention logits to prevent overflow in softmax
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)
        attn_weights = torch.softmax(attn_logits, dim=1)
        aggregated_opponents = torch.sum(opponent_embeddings * attn_weights, dim=1)  # (batch, output_dim)
        aggregated_opponents = self.opponent_projection(aggregated_opponents)

        # Concatenate self and opponent embeddings
        aggregated = torch.cat([self_embedding, aggregated_opponents], dim=-1)  # (batch, output_dim * 2)

        return player_embeddings, aggregated

    def extract_player_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract player features from full observation.

        The observation has shape (batch, 150, 65) where:
        - Row 0: Game context
        - Rows 1-19: Tile features
        - Rows 20-73: Corner features
        - Rows 74-145: Edge features
        - Rows 146-149: Player features

        Args:
            observation: Full observation tensor of shape (batch, 150, 65).

        Returns:
            Player features of shape (batch, 4, 65).
        """
        return observation[:, 146:150, :]
