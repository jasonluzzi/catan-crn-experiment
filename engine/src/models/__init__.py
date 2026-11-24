"""Phase 3 Models package with attention-based and GNN architecture.

This package contains:

Architecture v1 (Transformer-based):
- TileModule: Multi-head attention for processing tile features
- CornerModule: Multi-head attention for processing corner/vertex features
- EdgeModule: MLP encoder for edge/road features
- PlayerModule: MLP encoder for player states
- CatanFeaturesExtractor: Combines all modules

Architecture v2 (GNN-based):
- CatanBoardGNN: Graph Neural Network for tile-corner-edge graph
- CatanGNNFeaturesExtractor: GNN-based feature extractor

Common:
- CatanPolicy: Custom MaskablePPO policy
- Baseline agents: Random and Greedy agents

Note: CatanRLAgent is imported separately to avoid circular dependencies.
Use: from src.models.rl_agent import CatanRLAgent
"""

from .tile_module import TileModule, HexPositionalEncoding
from .corner_module import CornerModule
from .edge_module import EdgeModule
from .player_module import PlayerModule
from .graph_module import CatanBoardGNN, GraphAttentionLayer, HeterogeneousGATLayer
from .catan_policy import (
    CatanFeaturesExtractor,
    CatanGNNFeaturesExtractor,
    CatanPolicy,
    create_catan_policy_kwargs,
    create_gnn_policy_kwargs,
)
from .baseline import BaseAgent, RandomAgent, GreedyAgent

__all__ = [
    # v1 Transformer modules
    'TileModule',
    'HexPositionalEncoding',
    'CornerModule',
    'EdgeModule',
    'PlayerModule',
    # v2 GNN modules
    'CatanBoardGNN',
    'GraphAttentionLayer',
    'HeterogeneousGATLayer',
    # Feature extractors
    'CatanFeaturesExtractor',
    'CatanGNNFeaturesExtractor',
    # Policy
    'CatanPolicy',
    'create_catan_policy_kwargs',
    'create_gnn_policy_kwargs',
    # Baselines
    'BaseAgent',
    'RandomAgent',
    'GreedyAgent',
]
