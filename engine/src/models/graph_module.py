"""Graph Neural Network module for Catan board.

This module uses Graph Attention Networks (GAT) to process the Catan board
as a heterogeneous graph with tiles, corners, and edges as nodes.

Graph Structure:
- Tiles (19 nodes): Connected to adjacent corners (3 each)
- Corners (54 nodes): Connected to adjacent tiles (3 each) and edges (3 each)
- Edges (72 nodes): Connected to endpoint corners (2 each)

Message passing allows:
- Corners to aggregate production from adjacent tiles
- Edges to understand road connectivity through corners
- Tiles to see building patterns on adjacent corners
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer.

    Implements attention over neighbors in a graph, allowing nodes to
    weight which neighbors are most important.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        """Initialize GAT layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension (per head if concat=True).
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            concat: If True, concatenate heads; if False, average them.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear transformation for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features))

        # Initialize attention parameters
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if concat:
            self.output_dim = out_features * num_heads
        else:
            self.output_dim = out_features

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through GAT layer.

        Args:
            x: Node features (batch, num_nodes, in_features).
            adj: Adjacency matrix (num_nodes, num_nodes) or (batch, num_nodes, num_nodes).

        Returns:
            Updated node features (batch, num_nodes, output_dim).
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation: (batch, num_nodes, num_heads * out_features)
        h = self.W(x)
        h = h.view(batch_size, num_nodes, self.num_heads, self.out_features)
        # (batch, num_nodes, num_heads, out_features)

        # Compute attention scores
        # Source attention: (batch, num_nodes, num_heads)
        attn_src = (h * self.a_src).sum(dim=-1)
        # Destination attention: (batch, num_nodes, num_heads)
        attn_dst = (h * self.a_dst).sum(dim=-1)

        # Compute pairwise attention: (batch, num_nodes, num_nodes, num_heads)
        # attn[i,j] = attn_src[i] + attn_dst[j]
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        attn = self.leaky_relu(attn)

        # Mask non-adjacent nodes
        if adj.dim() == 2:
            # Same adjacency for all batches
            mask = adj.unsqueeze(0).unsqueeze(-1)  # (1, num_nodes, num_nodes, 1)
        else:
            mask = adj.unsqueeze(-1)  # (batch, num_nodes, num_nodes, 1)

        # Set non-adjacent attention to -inf
        attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows
        attn = self.dropout(attn)

        # Aggregate neighbor features
        # attn: (batch, num_nodes, num_nodes, num_heads)
        # h: (batch, num_nodes, num_heads, out_features)
        # Want: (batch, num_nodes, num_heads, out_features)
        h = h.permute(0, 2, 1, 3)  # (batch, num_heads, num_nodes, out_features)
        attn = attn.permute(0, 3, 1, 2)  # (batch, num_heads, num_nodes, num_nodes)

        out = torch.matmul(attn, h)  # (batch, num_heads, num_nodes, out_features)
        out = out.permute(0, 2, 1, 3)  # (batch, num_nodes, num_heads, out_features)

        if self.concat:
            out = out.reshape(batch_size, num_nodes, -1)
        else:
            out = out.mean(dim=2)

        return out


class HeterogeneousGATLayer(nn.Module):
    """Graph Attention for heterogeneous graph with multiple node/edge types.

    In Catan:
    - Node types: tiles, corners, edges
    - Edge types: tile-corner, corner-edge
    """

    def __init__(
        self,
        tile_dim: int,
        corner_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize heterogeneous GAT layer.

        Args:
            tile_dim: Tile feature dimension.
            corner_dim: Corner feature dimension.
            edge_dim: Edge feature dimension.
            hidden_dim: Hidden dimension for message passing.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project all node types to same dimension
        self.tile_proj = nn.Linear(tile_dim, hidden_dim)
        self.corner_proj = nn.Linear(corner_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Tile → Corner message passing
        self.tile_to_corner = GraphAttentionLayer(
            hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True
        )

        # Corner → Edge message passing
        self.corner_to_edge = GraphAttentionLayer(
            hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True
        )

        # Edge → Corner message passing (road connectivity)
        self.edge_to_corner = GraphAttentionLayer(
            hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True
        )

        # Corner → Tile message passing (building patterns)
        self.corner_to_tile = GraphAttentionLayer(
            hidden_dim, hidden_dim // num_heads, num_heads, dropout, concat=True
        )

        # Layer norms
        self.tile_norm = nn.LayerNorm(hidden_dim)
        self.corner_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tile_features: torch.Tensor,
        corner_features: torch.Tensor,
        edge_features: torch.Tensor,
        tile_corner_adj: torch.Tensor,
        corner_edge_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through heterogeneous GAT.

        Args:
            tile_features: (batch, 19, tile_dim)
            corner_features: (batch, 54, corner_dim)
            edge_features: (batch, 72, edge_dim)
            tile_corner_adj: (19, 54) adjacency between tiles and corners
            corner_edge_adj: (54, 72) adjacency between corners and edges

        Returns:
            Updated (tile, corner, edge) features.
        """
        batch_size = tile_features.shape[0]

        # Project to common dimension
        tiles = self.tile_proj(tile_features)
        corners = self.corner_proj(corner_features)
        edges = self.edge_proj(edge_features)

        # Message passing: Tiles → Corners
        # Create combined node tensor for tile→corner attention
        tile_corner_nodes = torch.cat([tiles, corners], dim=1)  # (batch, 19+54, hidden)

        # Build adjacency for tile→corner (tiles are source, corners are destination)
        # Full adjacency: (19+54, 19+54)
        num_tiles, num_corners = 19, 54
        full_tc_adj = torch.zeros(num_tiles + num_corners, num_tiles + num_corners,
                                   device=tile_features.device)
        # Tiles (0:19) connect to corners (19:73)
        full_tc_adj[:num_tiles, num_tiles:] = tile_corner_adj
        full_tc_adj[num_tiles:, :num_tiles] = tile_corner_adj.T

        tc_out = self.tile_to_corner(tile_corner_nodes, full_tc_adj)
        corners_updated = corners + self.dropout(tc_out[:, num_tiles:])
        corners_updated = self.corner_norm(corners_updated)

        # Message passing: Corners → Edges
        corner_edge_nodes = torch.cat([corners_updated, edges], dim=1)  # (batch, 54+72, hidden)

        num_edges = 72
        full_ce_adj = torch.zeros(num_corners + num_edges, num_corners + num_edges,
                                   device=tile_features.device)
        full_ce_adj[:num_corners, num_corners:] = corner_edge_adj
        full_ce_adj[num_corners:, :num_corners] = corner_edge_adj.T

        ce_out = self.corner_to_edge(corner_edge_nodes, full_ce_adj)
        edges_updated = edges + self.dropout(ce_out[:, num_corners:])
        edges_updated = self.edge_norm(edges_updated)

        # Message passing: Edges → Corners (road connectivity)
        corner_edge_nodes = torch.cat([corners_updated, edges_updated], dim=1)
        ec_out = self.edge_to_corner(corner_edge_nodes, full_ce_adj)
        corners_updated = corners_updated + self.dropout(ec_out[:, :num_corners])
        corners_updated = self.corner_norm(corners_updated)

        # Message passing: Corners → Tiles (building patterns)
        tile_corner_nodes = torch.cat([tiles, corners_updated], dim=1)
        ct_out = self.corner_to_tile(tile_corner_nodes, full_tc_adj)
        tiles_updated = tiles + self.dropout(ct_out[:, :num_tiles])
        tiles_updated = self.tile_norm(tiles_updated)

        return tiles_updated, corners_updated, edges_updated


class CatanBoardGNN(nn.Module):
    """Complete GNN module for Catan board processing.

    Replaces separate Tile/Corner/Edge modules with unified graph processing.
    """

    def __init__(
        self,
        input_dim: int = 65,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize Catan Board GNN.

        Args:
            input_dim: Input feature dimension (65 for Catan observations).
            hidden_dim: Hidden dimension for message passing.
            output_dim: Output embedding dimension.
            num_layers: Number of GNN layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projections
        self.tile_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.corner_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            HeterogeneousGATLayer(
                hidden_dim, hidden_dim, hidden_dim,
                hidden_dim, num_heads, dropout
            )
            for _ in range(num_layers)
        ])

        # Output projections
        self.tile_output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.corner_output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.edge_output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
        )

        # Attention pooling for aggregation
        self.tile_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )
        self.corner_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )
        self.edge_pool = nn.Sequential(
            nn.Linear(output_dim // 2, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, 1),
        )

        # Build static adjacency matrices
        self._build_adjacency_matrices()

    def _build_adjacency_matrices(self):
        """Build adjacency matrices for Catan board graph.

        Standard Catan board topology:
        - Each tile has 6 corners
        - Each corner touches 1-3 tiles
        - Each corner has 2-3 adjacent edges
        - Each edge connects 2 corners
        """
        # Tile-Corner adjacency (19 tiles × 54 corners)
        # This encodes which corners are adjacent to which tiles
        tile_corner = torch.zeros(19, 54)

        # Standard Catan tile-corner mapping
        # Tile corners are numbered consistently
        TILE_CORNERS = [
            [0, 1, 2, 8, 9, 10],      # Tile 0
            [2, 3, 4, 10, 11, 12],    # Tile 1
            [4, 5, 6, 12, 13, 14],    # Tile 2
            [7, 8, 9, 17, 18, 19],    # Tile 3
            [9, 10, 11, 19, 20, 21],  # Tile 4
            [11, 12, 13, 21, 22, 23], # Tile 5
            [13, 14, 15, 23, 24, 25], # Tile 6
            [16, 17, 18, 27, 28, 29], # Tile 7
            [18, 19, 20, 29, 30, 31], # Tile 8
            [20, 21, 22, 31, 32, 33], # Tile 9 (center)
            [22, 23, 24, 33, 34, 35], # Tile 10
            [24, 25, 26, 35, 36, 37], # Tile 11
            [28, 29, 30, 38, 39, 40], # Tile 12
            [30, 31, 32, 40, 41, 42], # Tile 13
            [32, 33, 34, 42, 43, 44], # Tile 14
            [34, 35, 36, 44, 45, 46], # Tile 15
            [39, 40, 41, 47, 48, 49], # Tile 16
            [41, 42, 43, 49, 50, 51], # Tile 17
            [43, 44, 45, 51, 52, 53], # Tile 18
        ]

        for tile_id, corners in enumerate(TILE_CORNERS):
            for corner_id in corners:
                if corner_id < 54:
                    tile_corner[tile_id, corner_id] = 1.0

        self.register_buffer('tile_corner_adj', tile_corner)

        # Corner-Edge adjacency (54 corners × 72 edges)
        # Each edge connects exactly 2 corners
        corner_edge = torch.zeros(54, 72)

        # Standard Catan edge-corner mapping
        # Edges connect pairs of adjacent corners
        EDGE_CORNERS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # Top row horizontal
            (0, 8), (2, 10), (4, 12), (6, 14),  # Top row vertical down
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # Row 2
            (7, 17), (9, 19), (11, 21), (13, 23), (15, 25),  # Vertical
            (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),  # Row 3
            (16, 27), (18, 29), (20, 31), (22, 33), (24, 35), (26, 37),  # Vertical
            (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37),  # Row 4
            (28, 38), (30, 40), (32, 42), (34, 44), (36, 46),  # Vertical
            (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46),  # Row 5
            (39, 47), (41, 49), (43, 51), (45, 53),  # Vertical
            (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53),  # Bottom row
        ]

        for edge_id, (c1, c2) in enumerate(EDGE_CORNERS):
            if edge_id < 72 and c1 < 54 and c2 < 54:
                corner_edge[c1, edge_id] = 1.0
                corner_edge[c2, edge_id] = 1.0

        self.register_buffer('corner_edge_adj', corner_edge)

    def forward(
        self,
        tile_features: torch.Tensor,
        corner_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process board through GNN.

        Args:
            tile_features: (batch, 19, 65)
            corner_features: (batch, 54, 65)
            edge_features: (batch, 72, 65)

        Returns:
            Tuple of:
                - tile_embeddings: (batch, 19, output_dim)
                - corner_embeddings: (batch, 54, output_dim)
                - edge_embeddings: (batch, 72, output_dim//2)
                - aggregated: (batch, output_dim + output_dim + output_dim//2)
        """
        # Input projections
        tiles = self.tile_input(tile_features)
        corners = self.corner_input(corner_features)
        edges = self.edge_input(edge_features)

        # GNN layers
        for gnn_layer in self.gnn_layers:
            tiles, corners, edges = gnn_layer(
                tiles, corners, edges,
                self.tile_corner_adj, self.corner_edge_adj
            )

        # Output projections
        tile_out = self.tile_output(tiles)
        corner_out = self.corner_output(corners)
        edge_out = self.edge_output(edges)

        # Attention pooling
        tile_attn = torch.softmax(
            torch.clamp(self.tile_pool(tile_out), -50, 50), dim=1
        )
        corner_attn = torch.softmax(
            torch.clamp(self.corner_pool(corner_out), -50, 50), dim=1
        )
        edge_attn = torch.softmax(
            torch.clamp(self.edge_pool(edge_out), -50, 50), dim=1
        )

        tile_agg = (tile_out * tile_attn).sum(dim=1)
        corner_agg = (corner_out * corner_attn).sum(dim=1)
        edge_agg = (edge_out * edge_attn).sum(dim=1)

        aggregated = torch.cat([tile_agg, corner_agg, edge_agg], dim=-1)

        return tile_out, corner_out, edge_out, aggregated

    @property
    def aggregated_dim(self) -> int:
        """Return dimension of aggregated output."""
        return self.output_dim + self.output_dim + self.output_dim // 2
