"""Edge processing module for Catan board edges.

This module uses MLP with pooling to process edge (road) features,
capturing road connectivity and placement information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class EdgeModule(nn.Module):
    """MLP module for processing edge features.

    This module processes the 72 Catan board edges using shared MLP,
    allowing the model to learn road connectivity patterns and valid
    road placements.

    The Catan board has 72 edges where roads can be placed.
    """

    def __init__(
        self,
        input_dim: int = 65,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
        num_edges: int = 72,
        use_positional_embeddings: bool = False,  # PHASE 3: Disabled to prevent edge memorization
    ):
        """Initialize the edge module.

        Args:
            input_dim: Input feature dimension per edge (default 65).
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            dropout: Dropout rate.
            num_edges: Number of edges (default 72).
            use_positional_embeddings: Whether to add positional embeddings.
                Disabled by default to prevent edge ID memorization. Edge
                features already encode connectivity and ownership.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_edges = num_edges
        self.use_positional_embeddings = use_positional_embeddings

        # Learnable positional embeddings for each edge
        # (kept for checkpoint compatibility, but optionally disabled)
        self.position_embeddings = nn.Parameter(
            torch.randn(num_edges, input_dim) * 0.02
        )

        # MLP for edge processing
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Attention-based pooling for aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )

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
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process edge features through MLP.

        Args:
            edge_features: Edge features of shape (batch, 72, 65).

        Returns:
            Tuple of:
                - Per-edge embeddings: (batch, 72, output_dim)
                - Aggregated embedding: (batch, output_dim)
        """
        batch_size = edge_features.size(0)

        # Add positional embeddings (optional - disabled by default to prevent memorization)
        if self.use_positional_embeddings:
            pos_embed = self.position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            x = edge_features + pos_embed
        else:
            x = edge_features

        # Apply MLP to each edge
        # Reshape for batch processing (use reshape for non-contiguous tensors)
        x = x.reshape(batch_size * self.num_edges, -1)  # (batch * 72, input_dim)
        x = self.mlp(x)  # (batch * 72, output_dim)
        edge_embeddings = x.view(batch_size, self.num_edges, -1)  # (batch, 72, output_dim)

        # Attention-based aggregation with numerical stability
        attn_logits = self.attention_pool(edge_embeddings)  # (batch, 72, 1)
        # Clamp attention logits to prevent overflow in softmax
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Weighted sum
        aggregated = torch.sum(edge_embeddings * attn_weights, dim=1)  # (batch, output_dim)

        return edge_embeddings, aggregated

    def extract_edge_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract edge features from full observation.

        The observation has shape (batch, 150, 65) where:
        - Row 0: Game context
        - Rows 1-19: Tile features
        - Rows 20-73: Corner features
        - Rows 74-145: Edge features (72 edges)
        - Rows 146-149: Player features

        Args:
            observation: Full observation tensor of shape (batch, 150, 65).

        Returns:
            Edge features of shape (batch, 72, 65).
        """
        return observation[:, 74:146, :]
