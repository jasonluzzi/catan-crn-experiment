"""Numerically stable probability distributions for MaskablePPO.

This module provides wrappers around sb3-contrib's MaskableCategorical that add
numerical stability measures to prevent the Simplex constraint violation error:
"Expected parameter probs to satisfy the constraint Simplex()"

The issue occurs when logits become too large (causing softmax overflow) or when
NaN/Inf values propagate through the network.
"""

import torch as th
import numpy as np
from typing import Optional, Union

from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

MaybeMasks = Union[th.Tensor, np.ndarray, None]


class StableMaskableCategorical(Categorical):
    """
    Numerically stable version of MaskableCategorical.

    Key improvements:
    1. Clamps logits to prevent overflow in softmax
    2. Ensures at least one valid action before applying mask
    3. Adds small epsilon to prevent log(0)
    4. Replaces NaN/Inf values

    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like log_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set to a
        large negative value, resulting in near 0 probability.
    """

    # Maximum logit magnitude to prevent overflow in softmax
    # exp(88) is approximately the max float32 value
    MAX_LOGIT = 50.0

    # Large negative value for masked actions (but not so large it causes issues)
    MASK_VALUE = -1e8

    # Small epsilon to prevent log(0)
    EPS = 1e-8

    def __init__(
        self,
        probs: Optional[th.Tensor] = None,
        logits: Optional[th.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks: MaybeMasks = None,
    ):
        self.masks: Optional[th.Tensor] = None

        # Pre-process logits for numerical stability before parent init
        if logits is not None:
            logits = self._stabilize_logits(logits)
        if probs is not None:
            probs = self._stabilize_probs(probs)

        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits.clone()
        self.apply_masking(masks)

    def _stabilize_logits(self, logits: th.Tensor) -> th.Tensor:
        """Ensure logits are numerically stable."""
        # Replace NaN/Inf with zeros
        logits = th.nan_to_num(logits, nan=0.0, posinf=self.MAX_LOGIT, neginf=-self.MAX_LOGIT)
        # Clamp to prevent overflow
        logits = th.clamp(logits, min=-self.MAX_LOGIT, max=self.MAX_LOGIT)
        return logits

    def _stabilize_probs(self, probs: th.Tensor) -> th.Tensor:
        """Ensure probabilities are numerically stable and sum to 1."""
        # Replace NaN/Inf
        probs = th.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        # Clamp to valid range
        probs = th.clamp(probs, min=0.0, max=1.0)
        # Ensure non-negative
        probs = th.abs(probs)
        # Add epsilon and renormalize
        probs = probs + self.EPS
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def apply_masking(self, masks: MaybeMasks) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(self.logits.shape)

            # Safety check: ensure at least one action is valid per batch element
            # If all actions are masked, allow all actions (this shouldn't happen but prevents crashes)
            any_valid = self.masks.any(dim=-1, keepdim=True)
            safe_masks = th.where(any_valid, self.masks, th.ones_like(self.masks))

            HUGE_NEG = th.tensor(self.MASK_VALUE, dtype=self.logits.dtype, device=device)

            logits = th.where(safe_masks, self._original_logits, HUGE_NEG)

            # Apply stability measures to masked logits
            logits = self._stabilize_logits(logits)
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)

        # Force update of probs with stability measures
        self.probs = self._stable_logits_to_probs(self.logits)

    def _stable_logits_to_probs(self, logits: th.Tensor) -> th.Tensor:
        """Convert logits to probabilities with numerical stability."""
        # Use log_softmax for numerical stability, then exp
        # log_softmax subtracts max internally for stability
        log_probs = th.nn.functional.log_softmax(logits, dim=-1)
        probs = th.exp(log_probs)

        # Ensure probs are valid
        probs = th.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = th.clamp(probs, min=0.0, max=1.0)

        # Renormalize to ensure sum to 1
        probs = probs / (probs.sum(dim=-1, keepdim=True) + self.EPS)

        return probs

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device

        # Use stable computation: avoid log(0) by using log_softmax
        log_probs = th.nn.functional.log_softmax(self.logits, dim=-1)
        p_log_p = self.probs * log_probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))

        # Replace any NaN in entropy
        entropy = -p_log_p.sum(-1)
        entropy = th.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

        return entropy

    def sample(self) -> th.Tensor:
        """Sample with fallback for numerical issues."""
        try:
            return super().sample()
        except RuntimeError:
            # Fallback: sample uniformly from valid actions
            if self.masks is not None:
                # Get valid indices
                valid_indices = th.where(self.masks)
                if len(valid_indices[0]) > 0:
                    # Random selection from valid actions
                    idx = th.randint(0, len(valid_indices[0]), (self.masks.shape[0],), device=self.masks.device)
                    return valid_indices[-1][idx]
            # Ultimate fallback: return zeros
            return th.zeros(self.probs.shape[:-1], dtype=th.long, device=self.probs.device)

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        """Compute log probability with numerical stability."""
        log_probs = th.nn.functional.log_softmax(self.logits, dim=-1)
        # Gather the log prob for the selected action
        value = value.long().unsqueeze(-1)
        log_prob = log_probs.gather(-1, value).squeeze(-1)
        # Replace NaN/Inf
        log_prob = th.nan_to_num(log_prob, nan=-100.0, posinf=0.0, neginf=-100.0)
        return log_prob


def patch_maskable_categorical():
    """
    Monkey-patch sb3-contrib's MaskableCategorical with our stable version.

    Call this at the start of training to replace all MaskableCategorical usage
    with the numerically stable version.

    Usage:
        from stable_distributions import patch_maskable_categorical
        patch_maskable_categorical()

        # Now train normally - MaskableCategorical is replaced
        model = MaskablePPO(...)
        model.learn(...)
    """
    try:
        import sb3_contrib.common.maskable.distributions as dist_module
        dist_module.MaskableCategorical = StableMaskableCategorical
        print("Successfully patched MaskableCategorical with StableMaskableCategorical")
    except ImportError:
        print("Warning: Could not patch MaskableCategorical - sb3-contrib not found")
