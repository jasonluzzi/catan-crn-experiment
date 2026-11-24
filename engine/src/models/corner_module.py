"""Corner attention module for processing Catan board corners.

This module uses multi-head self-attention to process corner (vertex) features,
capturing relationships between settlement/city placement locations on the board.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CornerModule(nn.Module):
    """Multi-head attention module for processing corner features.

    This module processes the 54 Catan board corners using self-attention,
    allowing the model to learn relationships between corners (e.g., adjacent
    corners for road connectivity, production value of settlements).

    The Catan board has 54 corners where settlements and cities can be placed.
    """

    def __init__(
        self,
        input_dim: int = 65,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_corners: int = 54,
        use_positional_embeddings: bool = False,  # PHASE 3: Disabled to prevent corner memorization
    ):
        """Initialize the corner module.

        Args:
            input_dim: Input feature dimension per corner (default 65).
            embed_dim: Embedding dimension for attention.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            num_corners: Number of corners (default 54).
            use_positional_embeddings: Whether to add positional embeddings.
                Disabled by default to prevent corner ID memorization (e.g., always
                placing settlements at corner 0). Position-invariant features in
                the input encode production rankings, resource diversity, etc.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_corners = num_corners
        self.use_positional_embeddings = use_positional_embeddings

        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Learnable positional embeddings for each corner
        # (kept for checkpoint compatibility, but optionally disabled)
        self.position_embeddings = nn.Parameter(
            torch.randn(num_corners, embed_dim) * 0.02
        )

        # Layer normalization before attention
        self.input_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layer normalization
        self.output_norm = nn.LayerNorm(embed_dim)

        # Attention pooling for aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
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
        corner_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process corner features through attention.

        Args:
            corner_features: Corner features of shape (batch, 54, 65).

        Returns:
            Tuple of:
                - Per-corner embeddings: (batch, 54, embed_dim)
                - Aggregated embedding: (batch, embed_dim)
        """
        batch_size = corner_features.size(0)

        # Project input to embedding dimension
        x = self.input_projection(corner_features)  # (batch, 54, embed_dim)

        # Add positional embeddings (optional - disabled by default to prevent memorization)
        if self.use_positional_embeddings:
            pos_embed = self.position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pos_embed

        # Normalize
        x = self.input_norm(x)

        # Apply transformer encoder
        corner_embeddings = self.transformer_encoder(x)  # (batch, 54, embed_dim)

        # Apply output normalization
        corner_embeddings = self.output_norm(corner_embeddings)

        # Attention pooling for aggregation with numerical stability
        attn_logits = self.attention_pool(corner_embeddings)  # (batch, 54, 1)
        # Clamp attention logits to prevent overflow in softmax
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Weighted sum
        aggregated = torch.sum(corner_embeddings * attn_weights, dim=1)  # (batch, embed_dim)

        return corner_embeddings, aggregated

    def extract_corner_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract corner features from full observation.

        The observation has shape (batch, 150, 65) where:
        - Row 0: Game context
        - Rows 1-19: Tile features
        - Rows 20-73: Corner features (54 corners)
        - Rows 74-145: Edge features
        - Rows 146-149: Player features

        Args:
            observation: Full observation tensor of shape (batch, 150, 65).

        Returns:
            Corner features of shape (batch, 54, 65).
        """
        return observation[:, 20:74, :]
