"""Roll History Module for processing dice roll history with attention.

Processes the last 20 dice rolls using temporal attention to capture
patterns in resource production and game dynamics.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RollHistoryModule(nn.Module):
    """Attention-based module for processing dice roll history.

    Uses transformer-style attention over the roll history sequence,
    with temporal positional encoding to capture recency effects.
    """

    def __init__(
        self,
        input_dim: int = 65,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_rolls: int = 20,
    ):
        """Initialize the roll history module.

        Args:
            input_dim: Input feature dimension per roll entry.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            max_rolls: Maximum number of rolls in history.
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_rolls = max_rolls

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal positional encoding (learnable)
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, max_rolls, embed_dim) * 0.02
        )

        # Transformer encoder for roll sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Attention pooling for aggregation
        # Use scaled initialization to prevent large initial attention scores
        self.attention_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) / math.sqrt(embed_dim))
        self.attention_scale = math.sqrt(embed_dim)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, roll_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process roll history through attention.

        Args:
            roll_history: Input tensor of shape (batch, max_rolls, input_dim).

        Returns:
            Tuple of:
                - Encoded sequence: (batch, max_rolls, embed_dim)
                - Aggregated features: (batch, embed_dim)
        """
        batch_size = roll_history.shape[0]

        # Project input
        x = self.input_projection(roll_history)  # (batch, max_rolls, embed_dim)

        # Add temporal positional encoding
        x = x + self.temporal_pos_encoding

        # Create padding mask for empty roll entries (all zeros)
        # Roll entry is empty if all features are zero
        roll_sums = roll_history.abs().sum(dim=-1)  # (batch, max_rolls)
        padding_mask = roll_sums == 0  # True where roll is empty

        # CRITICAL FIX: Handle all-masked case BEFORE transformer
        # When all positions are masked (setup phase with no rolls),
        # transformer produces NaN from softmax([-inf, -inf, ...])
        all_masked = padding_mask.all(dim=1)  # (batch,)

        if all_masked.all():
            # All items in batch have empty roll history
            # Skip transformer and use projected input directly
            encoded = x
            # Use mean of positional encoding as aggregated representation
            aggregated = x.mean(dim=1)  # (batch, embed_dim)
            aggregated = self.output_projection(aggregated)
            return encoded, aggregated

        # Apply transformer (with padding mask) - safe because not all positions are masked
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)  # (batch, max_rolls, embed_dim)

        # Handle mixed batches where some items are fully masked
        if all_masked.any():
            # Replace NaN entries with projected input for fully-masked items
            all_masked_expanded = all_masked.unsqueeze(-1).unsqueeze(-1).expand_as(encoded)
            encoded = torch.where(all_masked_expanded, x, encoded)

        # Attention pooling for aggregation
        query = self.attention_pool_query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        keys = encoded  # (batch, max_rolls, embed_dim)
        values = encoded

        # Compute attention scores
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / self.attention_scale  # (batch, 1, max_rolls)

        # Handle all-masked case BEFORE masking to avoid -inf gradient issues
        all_masked = padding_mask.all(dim=1, keepdim=True).unsqueeze(1)  # (batch, 1, 1)

        # Use large negative value instead of -inf to prevent gradient explosion
        # -inf causes NaN gradients; -1e4 is enough for softmax to produce ~0
        attention_scores = attention_scores.masked_fill(padding_mask.unsqueeze(1), -1e4)

        # For fully-masked items, use uniform attention (all zeros before softmax = uniform)
        attention_scores = attention_scores.masked_fill(all_masked, 0.0)

        # Apply softmax with numerical stability
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, 1, max_rolls)

        # Clamp to prevent extreme values that cause gradient issues
        attention_weights = attention_weights.clamp(min=1e-8, max=1.0 - 1e-8)

        # Weighted sum
        aggregated = torch.matmul(attention_weights, values)  # (batch, 1, embed_dim)
        aggregated = aggregated.squeeze(1)  # (batch, embed_dim)

        # Output projection
        aggregated = self.output_projection(aggregated)

        return encoded, aggregated


class BankResourceEncoder(nn.Module):
    """Simple encoder for bank resource state."""

    def __init__(
        self,
        input_dim: int = 65,
        output_dim: int = 32,
        num_resources: int = 5,
        dropout: float = 0.1,
    ):
        """Initialize the bank resource encoder.

        Args:
            input_dim: Input feature dimension per resource row.
            output_dim: Output embedding dimension.
            num_resources: Number of resource types.
            dropout: Dropout rate.
        """
        super().__init__()

        self.num_resources = num_resources

        # Per-resource encoder
        self.resource_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Aggregation MLP
        self.aggregation = nn.Sequential(
            nn.Linear(output_dim * num_resources, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, bank_resources: torch.Tensor) -> torch.Tensor:
        """Encode bank resource state.

        Args:
            bank_resources: Input tensor of shape (batch, num_resources, input_dim).

        Returns:
            Encoded features of shape (batch, output_dim).
        """
        batch_size = bank_resources.shape[0]

        # Encode each resource row
        encoded = self.resource_encoder(bank_resources)  # (batch, num_resources, output_dim)

        # Flatten and aggregate
        flattened = encoded.view(batch_size, -1)  # (batch, num_resources * output_dim)
        aggregated = self.aggregation(flattened)  # (batch, output_dim)

        return aggregated


class TradeStateEncoder(nn.Module):
    """Encoder for pending trade state."""

    def __init__(
        self,
        input_dim: int = 65,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        """Initialize the trade state encoder.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, trade_state: torch.Tensor) -> torch.Tensor:
        """Encode trade state.

        Args:
            trade_state: Input tensor of shape (batch, input_dim).

        Returns:
            Encoded features of shape (batch, output_dim).
        """
        return self.encoder(trade_state)
