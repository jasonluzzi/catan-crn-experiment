"""
Hierarchical Policy for MaskablePPO.

Extends MaskableActorCriticPolicy to use a two-stage hierarchical action structure:
1. Action type selection (20 categories)
2. Location/parameter selection (6 heads)

This policy is compatible with MaskablePPO from sb3-contrib and provides
proper action masking at both the type and location levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Type, Union

try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.maskable.distributions import MaskableDistribution
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskableActorCriticPolicy = object

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from .hierarchical_actor import HierarchicalActor
from .action_decomposer import (
    ActionDecomposer,
    NUM_ACTION_TYPES,
    FLAT_RANGES,
)
from ..game.actions import TOTAL_ACTIONS


class HierarchicalDistribution:
    """
    Custom distribution that wraps hierarchical action sampling.

    This provides a distribution interface compatible with SB3's PPO while
    using our hierarchical action structure internally.
    """

    def __init__(
        self,
        hierarchical_actor: HierarchicalActor,
        features: torch.Tensor,
        flat_mask: torch.Tensor,
    ):
        """
        Initialize the hierarchical distribution.

        Args:
            hierarchical_actor: The HierarchicalActor module
            features: Feature tensor (batch_size, feature_dim)
            flat_mask: Action mask (batch_size, 475)
        """
        self.actor = hierarchical_actor
        self.features = features
        self.flat_mask = flat_mask
        self._last_actions = None
        self._last_log_probs = None

    def sample(self) -> torch.Tensor:
        """Sample actions from the hierarchical policy."""
        actions, log_probs = self.actor.get_action(
            self.features, self.flat_mask, deterministic=False
        )
        self._last_actions = actions
        self._last_log_probs = log_probs
        return actions

    def mode(self) -> torch.Tensor:
        """Get the most likely actions (deterministic)."""
        actions, log_probs = self.actor.get_action(
            self.features, self.flat_mask, deterministic=True
        )
        self._last_actions = actions
        self._last_log_probs = log_probs
        return actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given actions."""
        log_probs, _ = self.actor.evaluate_action(
            self.features, actions, self.flat_mask
        )
        return log_probs

    def entropy(self) -> torch.Tensor:
        """Compute entropy of the distribution."""
        if self._last_actions is None:
            # Need to sample first to get actions
            _ = self.sample()

        _, entropy = self.actor.evaluate_action(
            self.features, self._last_actions, self.flat_mask
        )
        return entropy


class HierarchicalMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO policy with hierarchical action structure.

    This policy uses the CatanFeaturesExtractor for feature processing
    and HierarchicalActor for two-stage action selection.

    Architecture:
        observation → features_extractor → features (531)
                                         ↓
        features → HierarchicalActor → action_type_logits (20)
                                     → location_logits (per head)
                                         ↓
        features → value_net → value
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        actor_hidden_dim: int = 256,
        **kwargs,
    ):
        """
        Initialize the hierarchical policy.

        Args:
            observation_space: The observation space
            action_space: The action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture (ignored for actor, used for value)
            activation_fn: Activation function
            features_extractor_class: Feature extractor class
            features_extractor_kwargs: Feature extractor kwargs
            share_features_extractor: Whether actor/critic share features
            normalize_images: Whether to normalize images
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            actor_hidden_dim: Hidden dimension for hierarchical actor
        """
        # Store actor hidden dim before super().__init__ creates networks
        self.actor_hidden_dim = actor_hidden_dim

        # Import here to avoid circular imports
        if features_extractor_class is None:
            from .catan_policy import CatanFeaturesExtractor
            features_extractor_class = CatanFeaturesExtractor

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        # Default network architecture for value function
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    def _build(self, lr_schedule) -> None:
        """Build networks and optimizer.

        Override completely to avoid parent's action_net construction.
        """
        # Features extractor is already built in parent's __init__
        features_dim = self.features_extractor.features_dim

        # Build hierarchical actor instead of flat MLP
        self.hierarchical_actor = HierarchicalActor(
            feature_dim=features_dim,
            hidden_dim=self.actor_hidden_dim,
        )

        # Build value function MLP
        if isinstance(self.net_arch, dict) and 'vf' in self.net_arch:
            vf_layers = self.net_arch['vf']
        else:
            vf_layers = [256, 256]

        value_layers = []
        last_dim = features_dim
        for dim in vf_layers:
            value_layers.extend([
                nn.Linear(last_dim, dim),
                self.activation_fn(),
            ])
            last_dim = dim

        self.value_mlp = nn.Sequential(*value_layers)
        self.value_net = nn.Linear(last_dim, 1)

        # Placeholders for compatibility with parent class
        self.mlp_extractor = nn.Identity()
        self.action_net = nn.Identity()

        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """Not used - we build everything in _build()."""
        pass

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute actions, log probs, and values.

        Args:
            obs: Observation tensor
            deterministic: Whether to sample or take mode
            action_masks: Optional action mask

        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
        features = self.extract_features(obs)

        # Handle action masks
        if action_masks is None:
            action_masks = torch.ones(
                obs.shape[0], TOTAL_ACTIONS,
                dtype=torch.bool, device=obs.device
            )

        # Get actions and log probs from hierarchical actor
        actions, log_probs = self.hierarchical_actor.get_action(
            features, action_masks, deterministic=deterministic
        )

        # Compute values
        value_features = self.value_mlp(features)
        values = self.value_net(value_features)

        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observation tensor
            actions: Actions to evaluate
            action_masks: Optional action mask

        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features
        features = self.extract_features(obs)

        # Handle action masks
        if action_masks is None:
            action_masks = torch.ones(
                obs.shape[0], TOTAL_ACTIONS,
                dtype=torch.bool, device=obs.device
            )

        # Evaluate actions using hierarchical actor
        log_probs, entropy = self.hierarchical_actor.evaluate_action(
            features, actions, action_masks
        )

        # Compute values
        value_features = self.value_mlp(features)
        values = self.value_net(value_features)

        return values, log_probs, entropy

    def get_distribution(
        self,
        obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> HierarchicalDistribution:
        """
        Get the action distribution for given observations.

        Args:
            obs: Observation tensor
            action_masks: Optional action mask

        Returns:
            HierarchicalDistribution instance
        """
        features = self.extract_features(obs)

        if action_masks is None:
            action_masks = torch.ones(
                obs.shape[0], TOTAL_ACTIONS,
                dtype=torch.bool, device=obs.device
            )

        return HierarchicalDistribution(
            self.hierarchical_actor,
            features,
            action_masks,
        )

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for observations."""
        features = self.extract_features(obs)
        value_features = self.value_mlp(features)
        return self.value_net(value_features)

    def _predict(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the action according to the policy.

        Args:
            observation: Observation tensor
            deterministic: Whether to sample or take mode
            action_masks: Optional action mask

        Returns:
            Action tensor
        """
        features = self.extract_features(observation)

        if action_masks is None:
            action_masks = torch.ones(
                observation.shape[0], TOTAL_ACTIONS,
                dtype=torch.bool, device=observation.device
            )

        actions, _ = self.hierarchical_actor.get_action(
            features, action_masks, deterministic=deterministic
        )
        return actions

    def predict(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Get action from policy.

        Args:
            observation: Observation tensor
            state: RNN state (unused)
            episode_start: Episode start flags (unused)
            deterministic: Whether to sample or take mode
            action_masks: Optional action mask

        Returns:
            Tuple of (actions, state)
        """
        self.set_training_mode(False)

        with torch.no_grad():
            actions = self._predict(observation, deterministic, action_masks)

        return actions, state


class HierarchicalCatanPolicy(HierarchicalMaskableActorCriticPolicy):
    """
    Convenience class with Catan-specific defaults.

    Uses CatanFeaturesExtractor with standard settings.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args,
        net_arch: Optional[Dict] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        from .catan_policy import CatanFeaturesExtractor

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        if net_arch is None:
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


def create_hierarchical_policy_kwargs(
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
    actor_hidden_dim: int = 256,
    vf_hidden_dims: Optional[list] = None,
) -> Dict:
    """
    Create policy kwargs for MaskablePPO with hierarchical policy.

    Args:
        features_dim: Output dimension of feature extractor
        tile_embed_dim: Embedding dimension for tile module
        tile_num_heads: Number of attention heads for tiles
        tile_num_layers: Number of transformer layers for tiles
        corner_embed_dim: Embedding dimension for corner module
        corner_num_heads: Number of attention heads for corners
        corner_num_layers: Number of transformer layers for corners
        edge_hidden_dim: Hidden dimension for edge module
        edge_output_dim: Output dimension for edge module
        player_hidden_dims: Hidden dimensions for player MLP
        player_output_dim: Output dimension for player module
        context_embed_dim: Embedding dimension for game context
        roll_history_embed_dim: Embedding dimension for roll history
        roll_history_num_heads: Number of attention heads for roll history
        roll_history_num_layers: Number of transformer layers for roll history
        bank_embed_dim: Embedding dimension for bank resources
        trade_embed_dim: Embedding dimension for trade state
        dropout: Dropout rate
        actor_hidden_dim: Hidden dimension for hierarchical actor
        vf_hidden_dims: Hidden dimensions for value network

    Returns:
        Dictionary of policy keyword arguments
    """
    if player_hidden_dims is None:
        player_hidden_dims = [128, 128]
    if vf_hidden_dims is None:
        vf_hidden_dims = [256, 256]

    from .catan_policy import CatanFeaturesExtractor

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
        'net_arch': dict(pi=[256, 256], vf=vf_hidden_dims),
        'activation_fn': nn.Tanh,
        'actor_hidden_dim': actor_hidden_dim,
    }
