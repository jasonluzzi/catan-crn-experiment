"""
Pre-trained auxiliary oracle for strategic predictions.

This module provides a wrapper around the pre-trained auxiliary heads model
trained on Colonist.io data. It can be used during RL training to provide
"oracle" strategic predictions that guide the policy.

Usage:
    oracle = PretrainedOracle.load("models/pretrained_auxiliary_heads_v2.pt")

    # During RL rollout:
    simple_features = oracle.extract_features_from_obs(observation)
    predictions = oracle.predict(simple_features)
    # predictions contains win_probs, lr_probs, la_probs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class AuxiliaryOracle(nn.Module):
    """
    Pre-trained auxiliary prediction model.

    Architecture matches the training script:
    - encoder: 64 → hidden_dim → hidden_dim
    - win_prob_head: hidden_dim → 64 → 4
    - victory_path_head: hidden_dim → 64 → 8
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 256):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Win probability head
        self.win_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Victory path head (LR + LA)
        self.victory_path_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [batch, 64] input features

        Returns:
            Dict with win_probs, lr_probs, la_probs
        """
        encoded = self.encoder(features)

        # Win probability
        win_logits = self.win_prob_head(encoded)
        win_probs = F.softmax(win_logits, dim=-1)

        # Victory path
        vp_logits = self.victory_path_head(encoded)
        lr_probs = torch.sigmoid(vp_logits[:, :4])
        la_probs = torch.sigmoid(vp_logits[:, 4:])

        return {
            'win_probs': win_probs,
            'lr_probs': lr_probs,
            'la_probs': la_probs,
        }


class PretrainedOracle:
    """
    Wrapper for the pre-trained auxiliary oracle.

    Provides methods to:
    1. Extract simple features from full 176x65 observations
    2. Run predictions through the pre-trained model
    3. Use predictions as strategic context during RL
    """

    def __init__(self, model: AuxiliaryOracle, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Freeze all weights
        for param in self.model.parameters():
            param.requires_grad = False

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "PretrainedOracle":
        """
        Load from checkpoint.

        Args:
            checkpoint_path: Path to the .pt file
            device: Device to load on

        Returns:
            PretrainedOracle instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        config = checkpoint.get('config', {'feature_dim': 64, 'hidden_dim': 256})
        model = AuxiliaryOracle(
            feature_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded pre-trained oracle from {checkpoint_path}")
        print(f"  Best validation accuracy: {checkpoint['metrics'].get('best_val_acc', 'N/A'):.4f}")

        return cls(model, device)

    def extract_features_from_obs(self, observation: np.ndarray) -> torch.Tensor:
        """
        Extract simple features from full 176x65 observation.

        Observation structure:
        - Row 0: Context (turn number, phase, etc.)
        - Rows 146-149: Player features

        We extract features matching the Colonist.io training format.

        Args:
            observation: [batch, 176, 65] or [176, 65] numpy array

        Returns:
            [batch, 64] tensor of features
        """
        if observation.ndim == 2:
            observation = observation[np.newaxis, ...]  # Add batch dim

        batch_size = observation.shape[0]
        features = np.zeros((batch_size, 64), dtype=np.float32)

        for b in range(batch_size):
            obs = observation[b]

            # Turn progress (from context row 0)
            # Feature 0 is normalized turn number (turn * 0.01)
            features[b, 0] = obs[0, 0] / 100.0  # Re-normalize to 0-1

            # Player features from rows 146-149
            player_rows = obs[146:150, :]

            for p in range(4):
                player_row = player_rows[p]

                # VP (index varies by encoding, using common positions)
                # In state_encoder.py, player row features include:
                # settlements, cities, roads, dev cards, VP, etc.
                # Let's use the relative features we can access

                # These indices are based on src/environment/state_encoder.py
                # settlements: normalized count (typically at early indices)
                # We'll use a simplified mapping that approximates the training data

                # Settlements (index ~0-3 area)
                features[b, 5 + p] = player_row[0]  # Normalized settlements

                # Cities (index ~4-7 area)
                features[b, 9 + p] = player_row[1]  # Normalized cities

                # Dev cards (estimated)
                features[b, 13 + p] = player_row[4]  # Dev cards

                # Roads built count (estimated)
                features[b, 17 + p] = player_row[2]  # Roads

                # VP (often stored in player encoding)
                vp = player_row[5] if player_row.shape[0] > 5 else 0
                features[b, 1 + p] = vp

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def extract_features_from_state(
        self,
        player_vps: list,
        player_settlements: list,
        player_cities: list,
        player_roads: list,
        player_dev_cards: list,
        turn_progress: float,
    ) -> torch.Tensor:
        """
        Extract features from game state directly.

        This is more accurate than extracting from observations since
        it matches the Colonist.io training format exactly.

        Args:
            player_vps: VPs for each player [p0, p1, p2, p3]
            player_settlements: Settlement counts
            player_cities: City counts
            player_roads: Road counts
            player_dev_cards: Dev card counts
            turn_progress: Normalized game progress (0-1)

        Returns:
            [1, 64] tensor of features
        """
        features = torch.zeros(1, 64, device=self.device)

        # Turn progress
        features[0, 0] = turn_progress

        # VPs (normalized by 10)
        for i, vp in enumerate(player_vps[:4]):
            features[0, 1 + i] = vp / 10.0

        # Settlements (normalized by 5)
        for i, s in enumerate(player_settlements[:4]):
            features[0, 5 + i] = s / 5.0

        # Cities (normalized by 4)
        for i, c in enumerate(player_cities[:4]):
            features[0, 9 + i] = c / 4.0

        # Dev cards (normalized by 5)
        for i, d in enumerate(player_dev_cards[:4]):
            features[0, 13 + i] = d / 5.0

        # Roads (normalized by 15)
        for i, r in enumerate(player_roads[:4]):
            features[0, 17 + i] = r / 15.0

        # VP differences from leader
        max_vp = max(player_vps[:4]) if player_vps else 0
        for i, vp in enumerate(player_vps[:4]):
            features[0, 21 + i] = (vp - max_vp) / 10.0

        # Is each player in the lead?
        for i, vp in enumerate(player_vps[:4]):
            features[0, 25 + i] = 1.0 if vp == max_vp and max_vp > 0 else 0.0

        return features

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run prediction through the frozen oracle.

        Args:
            features: [batch, 64] features

        Returns:
            Dict with win_probs, lr_probs, la_probs tensors
        """
        return self.model(features)

    @torch.no_grad()
    def get_strategic_context(
        self,
        features: torch.Tensor,
        agent_player_id: int = 0,
    ) -> torch.Tensor:
        """
        Get strategic context vector for the actor/critic.

        Returns a vector suitable for concatenation to policy features:
        - win_probs: 4 values (relative to agent perspective)
        - lr_probs: 4 values
        - la_probs: 4 values
        - competition_lr: 3 values (agent vs each opponent)
        - competition_la: 3 values
        - Plus any additional strategic signals

        Args:
            features: [batch, 64] features
            agent_player_id: Which player is the agent (for relative encoding)

        Returns:
            [batch, 17] strategic context tensor
        """
        predictions = self.predict(features)

        batch_size = features.shape[0]

        # Rotate to agent perspective (agent always appears as player 0)
        win_probs = predictions['win_probs']
        lr_probs = predictions['lr_probs']
        la_probs = predictions['la_probs']

        if agent_player_id != 0:
            # Rotate so agent is first
            indices = [(agent_player_id + i) % 4 for i in range(4)]
            win_probs = win_probs[:, indices]
            lr_probs = lr_probs[:, indices]
            la_probs = la_probs[:, indices]

        # Competition features
        my_lr = lr_probs[:, 0:1]
        my_la = la_probs[:, 0:1]
        opp_lr = lr_probs[:, 1:4]
        opp_la = la_probs[:, 1:4]

        competition_lr = my_lr * opp_lr  # [batch, 3]
        competition_la = my_la * opp_la  # [batch, 3]

        # Concatenate strategic context
        strategic_context = torch.cat([
            win_probs,        # 4
            lr_probs,         # 4
            la_probs,         # 4
            competition_lr,   # 3
            competition_la,   # 3
        ], dim=-1)  # Total: 18

        return strategic_context


def create_oracle_from_checkpoint(checkpoint_path: str = "models/pretrained_auxiliary_heads_v2.pt") -> PretrainedOracle:
    """
    Convenience function to create an oracle from the default checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        PretrainedOracle instance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PretrainedOracle.load(checkpoint_path, device)
