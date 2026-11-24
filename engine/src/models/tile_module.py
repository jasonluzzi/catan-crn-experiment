"""Tile attention module for processing Catan board tiles.

This module uses multi-head self-attention to process tile features,
capturing relationships between different hexagonal tiles on the board.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class HexPositionalEncoding(nn.Module):
    """Learnable positional embeddings for hexagonal grid positions.

    The Catan board has 19 tiles arranged in a hexagonal pattern:
        Row 0: 3 tiles
        Row 1: 4 tiles
        Row 2: 5 tiles (center)
        Row 3: 4 tiles
        Row 4: 3 tiles
    """

    # Hex grid coordinates for each tile (q, r axial coordinates)
    # Center tile is (0, 0), using axial coordinate system
    HEX_COORDS = [
        # Row 0 (top)
        (-1, -2), (0, -2), (1, -2),
        # Row 1
        (-2, -1), (-1, -1), (0, -1), (1, -1),
        # Row 2 (center)
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        # Row 3
        (-1, 1), (0, 1), (1, 1), (2, 1),
        # Row 4 (bottom)
        (0, 2), (1, 2), (2, 2),
    ]

    def __init__(self, embed_dim: int, num_tiles: int = 19):
        """Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension.
            num_tiles: Number of tiles (default 19 for Catan).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tiles = num_tiles

        # Create learnable embeddings for each position
        self.position_embeddings = nn.Parameter(
            torch.randn(num_tiles, embed_dim) * 0.02
        )

        # Register hex coordinates as a buffer (not trainable)
        coords = torch.tensor(self.HEX_COORDS, dtype=torch.float32)
        self.register_buffer('hex_coords', coords)

        # Optional: project hex coordinates to embedding space
        self.coord_projection = nn.Linear(2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, num_tiles, embed_dim).

        Returns:
            Tensor with positional encoding added.
        """
        batch_size = x.size(0)

        # Use learnable positional embeddings
        pos_embed = self.position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Optionally add hex coordinate features
        coord_embed = self.coord_projection(self.hex_coords)
        coord_embed = coord_embed.unsqueeze(0).expand(batch_size, -1, -1)

        return x + pos_embed + coord_embed


class TileModule(nn.Module):
    """Multi-head attention module for processing tile features.

    This module processes the 19 Catan board tiles using self-attention,
    allowing the model to learn relationships between tiles (e.g., adjacent
    high-value tiles, robber placement effects).
    """

    def __init__(
        self,
        input_dim: int = 65,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_tiles: int = 19,
        use_positional_embeddings: bool = False,  # PHASE 3: Disabled to prevent tile memorization
    ):
        """Initialize the tile module.

        Args:
            input_dim: Input feature dimension per tile (default 65).
            embed_dim: Embedding dimension for attention.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            num_tiles: Number of tiles (default 19).
            use_positional_embeddings: Whether to add positional embeddings.
                Disabled by default to prevent tile ID memorization (e.g., always
                placing robber on tile 16). Position-invariant features in the
                input already encode production rankings.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tiles = num_tiles
        self.use_positional_embeddings = use_positional_embeddings

        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Positional encoding (kept for checkpoint compatibility, but optionally disabled)
        self.positional_encoding = HexPositionalEncoding(embed_dim, num_tiles)

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
        tile_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process tile features through attention.

        Args:
            tile_features: Tile features of shape (batch, 19, 65).

        Returns:
            Tuple of:
                - Per-tile embeddings: (batch, 19, embed_dim)
                - Aggregated embedding: (batch, embed_dim)
        """
        batch_size = tile_features.size(0)

        # Project input to embedding dimension
        x = self.input_projection(tile_features)  # (batch, 19, embed_dim)

        # Add positional encoding (optional - disabled by default to prevent memorization)
        if self.use_positional_embeddings:
            x = self.positional_encoding(x)

        # Normalize
        x = self.input_norm(x)

        # Apply transformer encoder
        tile_embeddings = self.transformer_encoder(x)  # (batch, 19, embed_dim)

        # Apply output normalization
        tile_embeddings = self.output_norm(tile_embeddings)

        # Attention pooling for aggregation with numerical stability
        attn_logits = self.attention_pool(tile_embeddings)  # (batch, 19, 1)
        # Clamp attention logits to prevent overflow in softmax
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Weighted sum
        aggregated = torch.sum(tile_embeddings * attn_weights, dim=1)  # (batch, embed_dim)

        return tile_embeddings, aggregated

    def extract_tile_features(self, observation: torch.Tensor) -> torch.Tensor:
        """Extract tile features from full observation.

        The observation has shape (batch, 150, 65) where:
        - Row 0: Game context
        - Rows 1-19: Tile features
        - Rows 20-73: Corner features
        - Rows 74-145: Edge features
        - Rows 146-149: Player features

        Args:
            observation: Full observation tensor of shape (batch, 150, 65).

        Returns:
            Tile features of shape (batch, 19, 65).
        """
        return observation[:, 1:20, :]
