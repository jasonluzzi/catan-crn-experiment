"""Utilities for extending/reinitializing action heads when action space changes.

This module provides functions to handle loading pretrained models into environments
with different action configurations (e.g., enabling P2P trading after training without it).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


# Action ranges for trading actions
TRADING_ACTION_START = 366  # First trading action index
TRADING_ACTION_END = 475    # Last trading action index + 1


def detect_untrained_actions(
    model,
    threshold_std: float = 0.005,
) -> Tuple[bool, int, int]:
    """Detect if action head has untrained (near-zero) weights for certain actions.

    Args:
        model: MaskablePPO model to check.
        threshold_std: Standard deviation threshold below which weights are considered untrained.

    Returns:
        Tuple of (has_untrained, start_idx, end_idx) where start_idx and end_idx
        define the range of untrained actions.
    """
    action_weights = model.policy.action_net.weight.data

    # Check trading action weight statistics
    trading_weights = action_weights[TRADING_ACTION_START:TRADING_ACTION_END]
    trading_std = trading_weights.std().item()

    # Check non-trading for comparison
    non_trading_weights = action_weights[:TRADING_ACTION_START]
    non_trading_std = non_trading_weights.std().item()

    # If trading weights have much lower variance, they're untrained
    has_untrained = trading_std < threshold_std and non_trading_std > threshold_std * 2

    return has_untrained, TRADING_ACTION_START, TRADING_ACTION_END


def reinitialize_action_weights(
    model,
    start_idx: int,
    end_idx: int,
    init_method: str = "xavier",
    negative_bias: float = -2.0,
) -> None:
    """Reinitialize action head weights for a range of actions.

    Args:
        model: MaskablePPO model to modify.
        start_idx: Start index of actions to reinitialize.
        end_idx: End index (exclusive) of actions to reinitialize.
        init_method: Initialization method ("xavier", "normal", "zeros").
        negative_bias: Initial bias value (negative to discourage selection initially).
    """
    action_net = model.policy.action_net

    # Get dimensions
    in_features = action_net.in_features
    num_actions = end_idx - start_idx

    # Create temporary layer for initialization
    temp_layer = nn.Linear(in_features, num_actions)

    if init_method == "xavier":
        nn.init.xavier_uniform_(temp_layer.weight)
    elif init_method == "normal":
        nn.init.normal_(temp_layer.weight, mean=0.0, std=0.02)
    elif init_method == "zeros":
        nn.init.zeros_(temp_layer.weight)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    # Copy initialized weights to action_net
    with torch.no_grad():
        action_net.weight.data[start_idx:end_idx] = temp_layer.weight.data
        action_net.bias.data[start_idx:end_idx] = negative_bias

    print(f"Reinitialized action weights for actions {start_idx}-{end_idx}")
    print(f"  Method: {init_method}")
    print(f"  Initial bias: {negative_bias} (discourages selection until trained)")


def prepare_model_for_trading(
    model,
    init_method: str = "xavier",
    negative_bias: float = -2.0,
    freeze_feature_extractor_steps: int = 0,
) -> dict:
    """Prepare a model trained without trading for training with trading enabled.

    This function:
    1. Detects untrained trading action weights
    2. Reinitializes them with proper initialization
    3. Optionally sets up feature extractor freezing

    Args:
        model: MaskablePPO model to prepare.
        init_method: Weight initialization method.
        negative_bias: Initial bias for trading actions (negative to discourage initially).
        freeze_feature_extractor_steps: Number of steps to freeze feature extractor
            (0 = don't freeze).

    Returns:
        Dict with preparation info:
        - 'trading_reinitialized': bool
        - 'freeze_until_step': int
    """
    result = {
        'trading_reinitialized': False,
        'freeze_until_step': freeze_feature_extractor_steps,
    }

    # Check for untrained trading actions
    has_untrained, start_idx, end_idx = detect_untrained_actions(model)

    if has_untrained:
        print("\nDetected untrained trading action weights.")
        print("This model was likely trained with P2P trading disabled.")
        print("Reinitializing trading actions for proper learning...\n")

        reinitialize_action_weights(
            model,
            start_idx,
            end_idx,
            init_method=init_method,
            negative_bias=negative_bias,
        )
        result['trading_reinitialized'] = True
    else:
        print("Trading action weights appear to be trained. No reinitialization needed.")

    # Set up feature extractor freezing if requested
    if freeze_feature_extractor_steps > 0:
        print(f"\nFeature extractor will be frozen for {freeze_feature_extractor_steps} steps")
        _freeze_feature_extractor(model)

    return result


def _freeze_feature_extractor(model) -> None:
    """Freeze the feature extractor parameters."""
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
    print("Feature extractor frozen (requires_grad=False)")


def _unfreeze_feature_extractor(model) -> None:
    """Unfreeze the feature extractor parameters."""
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = True
    print("Feature extractor unfrozen (requires_grad=True)")


class TradingWarmupCallback:
    """Callback to manage trading action warmup during training.

    This callback:
    1. Unfreezes feature extractor after warmup period
    2. Optionally adjusts trading action bias over time
    """

    def __init__(
        self,
        freeze_until_step: int = 0,
        bias_warmup_steps: int = 0,
        initial_bias: float = -2.0,
        final_bias: float = 0.0,
    ):
        """Initialize the callback.

        Args:
            freeze_until_step: Step at which to unfreeze feature extractor.
            bias_warmup_steps: Steps over which to warm up trading action bias.
            initial_bias: Starting bias for trading actions.
            final_bias: Final bias for trading actions.
        """
        self.freeze_until_step = freeze_until_step
        self.bias_warmup_steps = bias_warmup_steps
        self.initial_bias = initial_bias
        self.final_bias = final_bias
        self.unfrozen = freeze_until_step == 0

    def on_step(self, model, step: int) -> None:
        """Called at each training step.

        Args:
            model: The model being trained.
            step: Current training step.
        """
        # Unfreeze feature extractor if needed
        if not self.unfrozen and step >= self.freeze_until_step:
            _unfreeze_feature_extractor(model)
            self.unfrozen = True

        # Warm up trading action bias if configured
        if self.bias_warmup_steps > 0 and step < self.bias_warmup_steps:
            progress = step / self.bias_warmup_steps
            current_bias = self.initial_bias + (self.final_bias - self.initial_bias) * progress

            with torch.no_grad():
                model.policy.action_net.bias.data[TRADING_ACTION_START:TRADING_ACTION_END] = current_bias
