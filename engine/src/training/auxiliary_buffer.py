"""Extended rollout buffer for auxiliary head training.

Stores additional targets needed for training auxiliary prediction heads:
- Winner ID for win probability training
- Victory path targets for victory path training (largest army, longest road)
"""

from typing import Optional, Generator, NamedTuple
import numpy as np
import torch
from gymnasium import spaces

from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class AuxiliaryRolloutBufferSamples(NamedTuple):
    """Extended samples including auxiliary targets."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor
    # Auxiliary targets
    winner_targets: torch.Tensor  # One-hot of winner (batch, 4)
    winner_mask: torch.Tensor     # 1 if winner known, 0 if episode incomplete
    # Victory path targets (largest army, longest road)
    victory_path_targets: torch.Tensor  # (batch, 8) - 4 players × 2 paths (army, road)
    victory_path_mask: torch.Tensor     # 1 if victory path known, 0 if incomplete


class AuxiliaryRolloutBuffer(MaskableRolloutBuffer):
    """Rollout buffer that stores auxiliary training targets.

    Extends MaskableRolloutBuffer to store:
    - winner_id: The player who won the game (backfilled at episode end)
    - winner_mask: Whether this timestep has a known winner
    - victory_path: Who achieved largest army / longest road (backfilled at episode end)
    - victory_path_mask: Whether victory path info is known

    The win probability head is trained with cross-entropy loss against
    the one-hot encoded winner for all timesteps in completed episodes.

    The victory path head is trained with BCE loss against binary targets
    indicating whether each player achieved largest army or longest road.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        num_players: int = 4,
    ):
        # Set num_players before super().__init__ because parent's __init__ calls reset()
        self.num_players = num_players

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        # Note: reset() is called by parent, which initializes our auxiliary storage

    def reset(self) -> None:
        """Reset buffer and auxiliary storage."""
        super().reset()
        self.winner_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.winner_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.victory_path_targets = np.zeros(
            (self.buffer_size, self.n_envs, self.num_players * 2), dtype=np.float32
        )
        self.victory_path_mask = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._episode_starts = np.zeros(self.n_envs, dtype=np.int32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
        infos: Optional[list] = None,
    ) -> None:
        """Add a transition with auxiliary info extraction.

        Args:
            obs: Observations
            action: Actions taken
            reward: Rewards received
            episode_start: Whether this is start of new episode
            value: Value estimates
            log_prob: Log probabilities
            action_masks: Valid action masks
            infos: Info dicts from environment (contains winner)
        """
        # Track episode starts for backfilling
        for env_idx in range(self.n_envs):
            if episode_start[env_idx]:
                self._episode_starts[env_idx] = self.pos

        # Extract winner and victory path info from infos and backfill completed episodes
        if infos is not None:
            for env_idx, info in enumerate(infos):
                winner = info.get("winner")
                if winner is not None:
                    # Episode completed - backfill winner for all steps
                    self._backfill_winner(env_idx, winner)

                    # Extract victory path info (largest army, longest road holders)
                    largest_army_holder = info.get("largest_army_holder")
                    longest_road_holder = info.get("longest_road_holder")
                    self._backfill_victory_path(
                        env_idx, largest_army_holder, longest_road_holder
                    )

        # Call parent add (action_masks must be keyword argument)
        super().add(obs, action, reward, episode_start, value, log_prob, action_masks=action_masks)

    def _backfill_winner(self, env_idx: int, winner_id: int) -> None:
        """Backfill winner ID for all timesteps in the completed episode.

        Args:
            env_idx: Environment index
            winner_id: ID of the winning player
        """
        start_pos = self._episode_starts[env_idx]
        end_pos = self.pos

        # Handle buffer wraparound
        if end_pos >= start_pos:
            # Normal case: episode doesn't wrap
            self.winner_ids[start_pos:end_pos + 1, env_idx] = winner_id
            self.winner_mask[start_pos:end_pos + 1, env_idx] = 1.0
        else:
            # Wrapped case: episode spans buffer boundary
            self.winner_ids[start_pos:, env_idx] = winner_id
            self.winner_ids[:end_pos + 1, env_idx] = winner_id
            self.winner_mask[start_pos:, env_idx] = 1.0
            self.winner_mask[:end_pos + 1, env_idx] = 1.0

    def _backfill_victory_path(
        self,
        env_idx: int,
        largest_army_holder: Optional[int],
        longest_road_holder: Optional[int],
    ) -> None:
        """Backfill victory path targets for all timesteps in completed episode.

        Sets binary targets indicating whether each player achieved:
        - Largest army (2+ knights played and more than any other player)
        - Longest road (5+ connected roads and more than any other player)

        Args:
            env_idx: Environment index
            largest_army_holder: Player ID who holds largest army, or None
            longest_road_holder: Player ID who holds longest road, or None
        """
        start_pos = self._episode_starts[env_idx]
        end_pos = self.pos

        # Create victory path target array: [p0_army, p0_road, p1_army, p1_road, ...]
        target = np.zeros(self.num_players * 2, dtype=np.float32)

        if largest_army_holder is not None:
            # Set army target for the holder
            target[largest_army_holder * 2] = 1.0

        if longest_road_holder is not None:
            # Set road target for the holder
            target[longest_road_holder * 2 + 1] = 1.0

        # Handle buffer wraparound
        if end_pos >= start_pos:
            # Normal case: episode doesn't wrap
            self.victory_path_targets[start_pos:end_pos + 1, env_idx] = target
            self.victory_path_mask[start_pos:end_pos + 1, env_idx] = 1.0
        else:
            # Wrapped case: episode spans buffer boundary
            self.victory_path_targets[start_pos:, env_idx] = target
            self.victory_path_targets[:end_pos + 1, env_idx] = target
            self.victory_path_mask[start_pos:, env_idx] = 1.0
            self.victory_path_mask[:end_pos + 1, env_idx] = 1.0

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[AuxiliaryRolloutBufferSamples, None, None]:
        """Get samples with auxiliary targets.

        Args:
            batch_size: Size of minibatches

        Yields:
            AuxiliaryRolloutBufferSamples with winner and victory path targets
        """
        assert self.full, "Rollout buffer must be full before sampling"

        # Prepare auxiliary targets
        winner_targets = self._prepare_winner_targets()
        winner_mask = self.winner_mask.flatten()

        # Prepare victory path targets (flatten to batch dimension)
        victory_path_targets = self.victory_path_targets.reshape(-1, self.num_players * 2)
        victory_path_mask = self.victory_path_mask.flatten()

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield self._get_samples(
                batch_indices,
                winner_targets,
                winner_mask,
                victory_path_targets,
                victory_path_mask,
            )
            start_idx += batch_size

    def _prepare_winner_targets(self) -> np.ndarray:
        """Prepare one-hot encoded winner targets.

        Returns:
            Array of shape (buffer_size * n_envs, num_players) with one-hot winners
        """
        flat_winners = self.winner_ids.flatten()
        one_hot = np.zeros((len(flat_winners), self.num_players), dtype=np.float32)

        # Only set one-hot for valid winners (mask == 1)
        flat_mask = self.winner_mask.flatten()
        valid_indices = flat_mask > 0
        one_hot[valid_indices, flat_winners[valid_indices]] = 1.0

        return one_hot

    def _get_samples(
        self,
        batch_indices: np.ndarray,
        winner_targets: np.ndarray,
        winner_mask: np.ndarray,
        victory_path_targets: np.ndarray,
        victory_path_mask: np.ndarray,
    ) -> AuxiliaryRolloutBufferSamples:
        """Get a batch of samples with auxiliary targets.

        Args:
            batch_indices: Indices to sample
            winner_targets: One-hot winner targets
            winner_mask: Mask for valid winners
            victory_path_targets: Victory path targets (batch, 8)
            victory_path_mask: Mask for valid victory paths

        Returns:
            AuxiliaryRolloutBufferSamples
        """
        # Get flattened data
        obs = self.observations.reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        values = self.values.flatten()
        log_probs = self.log_probs.flatten()
        advantages = self.advantages.flatten()
        returns = self.returns.flatten()
        action_masks = self.action_masks.reshape(-1, self.action_masks.shape[-1])

        return AuxiliaryRolloutBufferSamples(
            observations=self.to_torch(obs[batch_indices]),
            actions=self.to_torch(actions[batch_indices]),
            old_values=self.to_torch(values[batch_indices]),
            old_log_prob=self.to_torch(log_probs[batch_indices]),
            advantages=self.to_torch(advantages[batch_indices]),
            returns=self.to_torch(returns[batch_indices]),
            action_masks=self.to_torch(action_masks[batch_indices]),
            winner_targets=self.to_torch(winner_targets[batch_indices]),
            winner_mask=self.to_torch(winner_mask[batch_indices]),
            victory_path_targets=self.to_torch(victory_path_targets[batch_indices]),
            victory_path_mask=self.to_torch(victory_path_mask[batch_indices]),
        )
