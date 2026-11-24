"""
Multi-Agent PPO for Catan self-play.

Single policy plays all 4 positions, learning from all players' experiences.
This provides 4x training data per game and enables natural trading dynamics.
"""

import gc
import time
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from src.training.multi_agent_wrapper import (
    MultiAgentCatanEnv,
    MultiAgentVecEnv,
    PlayerTransition,
)
from src.training.game_logger import GameLoggerConfig
from src.models.catan_policy import CatanPolicy


@dataclass
class RolloutBatch:
    """A batch of transitions for PPO training."""
    observations: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class MultiAgentRolloutBuffer:
    """
    Rollout buffer for multi-agent PPO.

    Collects transitions from all 4 players across multiple games,
    then computes advantages and returns for PPO training.
    """

    def __init__(
        self,
        buffer_size: int = 4096,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: Union[str, torch.device] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # Storage
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.action_masks: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

        # Computed values
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

        # Episode tracking for GAE computation
        self._episode_starts: List[int] = [0]

    def add(
        self,
        observation: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add a single transition to the buffer."""
        self.observations.append(observation)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

        if done:
            self._episode_starts.append(len(self.observations))

    def add_transitions(
        self,
        transitions: List[PlayerTransition],
        policy: nn.Module,
        device: torch.device,
        chunk_size: int = 512,
    ):
        """
        Add multiple transitions, computing values and log_probs using the policy.

        Args:
            transitions: List of PlayerTransition from completed games
            policy: The policy network to compute values and log_probs
            device: Device to run policy on
            chunk_size: Process transitions in chunks to avoid OOM on MPS/GPU
        """
        if not transitions:
            return

        # Process in chunks to avoid OOM on MPS
        all_values = []
        all_log_probs = []

        for start in range(0, len(transitions), chunk_size):
            end = min(start + chunk_size, len(transitions))
            chunk = transitions[start:end]

            obs_batch = np.array([t.observation for t in chunk])
            action_batch = np.array([t.action for t in chunk])
            mask_batch = np.array([t.action_mask for t in chunk])

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs_batch).float().to(device)
                action_tensor = torch.from_numpy(action_batch).long().to(device)
                mask_tensor = torch.from_numpy(mask_batch).bool().to(device)

                values, log_probs, _ = policy.evaluate_actions(
                    obs_tensor, action_tensor, action_masks=mask_tensor
                )

                all_values.extend(values.cpu().numpy().flatten().tolist())
                all_log_probs.extend(log_probs.cpu().numpy().flatten().tolist())

                # Clear GPU memory between chunks
                del obs_tensor, action_tensor, mask_tensor, values, log_probs
                if device.type == 'mps':
                    torch.mps.empty_cache()

        # Add to buffer
        for i, t in enumerate(transitions):
            self.add(
                observation=t.observation,
                action=t.action,
                action_mask=t.action_mask,
                reward=t.reward,
                done=t.done,
                value=all_values[i],
                log_prob=all_log_probs[i],
            )

    def compute_returns_and_advantages(self, last_values: Optional[np.ndarray] = None):
        """
        Compute GAE advantages and returns.

        Uses generalized advantage estimation for variance reduction.
        """
        n = len(self.observations)
        if n == 0:
            return

        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)

        # For terminal states, next value is 0
        # For non-terminal, we use the actual next value
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        # Process each episode separately for correct GAE
        episode_starts = self._episode_starts.copy()
        if episode_starts[-1] != n:
            episode_starts.append(n)

        for ep_idx in range(len(episode_starts) - 1):
            start = episode_starts[ep_idx]
            end = episode_starts[ep_idx + 1]

            if start >= end:
                continue

            # Compute GAE for this episode
            last_gae = 0.0
            for t in reversed(range(start, end)):
                if t == end - 1:
                    # Last step of episode
                    next_value = 0.0  # Terminal
                    next_non_terminal = 0.0
                else:
                    next_value = values[t + 1]
                    next_non_terminal = 1.0 - float(dones[t])

                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                self.advantages[t] = last_gae

            # Returns = advantages + values
            self.returns[start:end] = self.advantages[start:end] + values[start:end]

        # Normalize advantages
        if len(self.advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int) -> List[RolloutBatch]:
        """Generate randomized minibatches for training."""
        n = len(self.observations)
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            batch = RolloutBatch(
                observations=torch.from_numpy(
                    np.array([self.observations[i] for i in batch_indices])
                ).float().to(self.device),
                actions=torch.tensor(
                    [self.actions[i] for i in batch_indices]
                ).long().to(self.device),
                action_masks=torch.from_numpy(
                    np.array([self.action_masks[i] for i in batch_indices])
                ).bool().to(self.device),
                old_log_probs=torch.tensor(
                    [self.log_probs[i] for i in batch_indices]
                ).float().to(self.device),
                advantages=torch.from_numpy(
                    self.advantages[batch_indices]
                ).float().to(self.device),
                returns=torch.from_numpy(
                    self.returns[batch_indices]
                ).float().to(self.device),
                values=torch.tensor(
                    [self.values[i] for i in batch_indices]
                ).float().to(self.device),
            )
            batches.append(batch)

        return batches

    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.action_masks = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = None
        self.returns = None
        self._episode_starts = [0]

    def __len__(self):
        return len(self.observations)


class MultiAgentPPO:
    """
    Multi-Agent PPO trainer for Catan self-play.

    A single policy plays all 4 positions. Experiences from all players
    are collected and used for training, providing 4x data per game.
    """

    def __init__(
        self,
        policy: Union[str, type] = CatanPolicy,
        env_kwargs: Optional[Dict] = None,
        n_envs: int = 4,
        n_steps: int = 512,
        batch_size: int = 256,
        n_epochs: int = 8,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.02,
        device: str = "auto",
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        game_log_config: Optional[GameLoggerConfig] = None,
        exploration_fraction: float = 0.0,
    ):
        # Hyperparameters
        self.n_envs = n_envs
        self.exploration_fraction = exploration_fraction  # Fraction of actions to randomize
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.verbose = verbose

        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if verbose:
            print(f"Using device: {self.device}")

        # Environment setup
        env_kwargs = env_kwargs or {}
        self.env = MultiAgentVecEnv(
            num_envs=n_envs,
            game_log_config=game_log_config,
            **env_kwargs
        )

        # Policy setup
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        if isinstance(policy, str):
            if policy == "CatanPolicy":
                policy = CatanPolicy

        self.policy = policy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lambda _: learning_rate,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

        # Rollout buffer
        self.rollout_buffer = MultiAgentRolloutBuffer(
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

        # Logging
        self.tensorboard_log = tensorboard_log
        self.writer: Optional[SummaryWriter] = None
        if tensorboard_log:
            self.writer = SummaryWriter(tensorboard_log)

        # Training state
        self.num_timesteps = 0
        self.num_episodes = 0
        self._last_obs: Optional[np.ndarray] = None

        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.win_counts: Dict[int, int] = {i: 0 for i in range(4)}

        # Game stats tracking (capped to prevent memory growth)
        from src.training.multi_agent_wrapper import GameStats
        self.recent_game_stats: List[GameStats] = []
        self._max_recent_stats = 500  # Keep last 500 games max

    def collect_rollouts(self) -> int:
        """
        Collect rollouts from all environments.

        Returns:
            Number of transitions collected
        """
        self.policy.eval()

        # Reset if needed
        if self._last_obs is None:
            self._last_obs, _ = self.env.reset()

        transitions_collected = 0
        steps_taken = 0

        while steps_taken < self.n_steps:
            # Get action masks
            action_masks = self.env.get_action_masks()

            # Get actions from policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self._last_obs).float().to(self.device)
                mask_tensor = torch.from_numpy(action_masks).bool().to(self.device)

                actions, values, log_probs = self.policy(obs_tensor, action_masks=mask_tensor)

                actions = actions.cpu().numpy()

            # Exploration: randomly override some actions to prevent self-play collapse
            # This forces the agent to handle unexpected opponent behavior
            if self.exploration_fraction > 0:
                for i in range(self.n_envs):
                    if np.random.random() < self.exploration_fraction:
                        # Pick a random legal action
                        legal_actions = np.where(action_masks[i])[0]
                        if len(legal_actions) > 0:
                            actions[i] = np.random.choice(legal_actions)

            # Step environments
            new_obs, rewards, dones, truncateds, infos = self.env.step(actions)

            steps_taken += self.n_envs
            self.num_timesteps += self.n_envs

            # Collect transitions from completed games
            transitions = self.env.collect_transitions()
            if transitions:
                self.rollout_buffer.add_transitions(
                    transitions, self.policy, self.device
                )
                transitions_collected += len(transitions)

                # Track statistics
                for t in transitions:
                    if t.done:
                        self.num_episodes += 1

            # Collect game stats (capped to prevent memory growth)
            game_stats = self.env.collect_game_stats()
            if game_stats:
                self.recent_game_stats.extend(game_stats)
                # Trim to max size to prevent unbounded memory growth
                if len(self.recent_game_stats) > self._max_recent_stats:
                    self.recent_game_stats = self.recent_game_stats[-self._max_recent_stats:]

            self._last_obs = new_obs

        # Memory cleanup after rollout collection
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return transitions_collected

    def get_recent_stats(self, n_games: int = 50) -> Dict[str, float]:
        """Get aggregated stats from recent games."""
        if not self.recent_game_stats:
            return {}

        # Take last n games
        recent = self.recent_game_stats[-n_games:] if len(self.recent_game_stats) > n_games else self.recent_game_stats

        def get_learning_stats(game, stat_list):
            """Extract stats only for learning players."""
            if not game.learning_players or not stat_list:
                return stat_list  # Fallback to all if no learning_players info
            return [stat_list[i] for i in game.learning_players if i < len(stat_list)]

        def learning_mean(game, stat_list):
            """Mean of stats for learning players only."""
            vals = get_learning_stats(game, stat_list)
            return np.mean(vals) if vals else 0.0

        def learning_max(game, stat_list):
            """Max of stats for learning players only."""
            vals = get_learning_stats(game, stat_list)
            return max(vals) if vals else 0

        def learning_won(game):
            """Check if a learning player won."""
            if game.winner is None:
                return 0.0
            return 1.0 if game.learning_players and game.winner in game.learning_players else 0.0

        def learning_has_achievement(game, holder):
            """Check if a learning player has longest road/largest army."""
            if holder is None:
                return 0.0
            return 1.0 if game.learning_players and holder in game.learning_players else 0.0

        # Aggregate stats (learning agents only)
        stats = {
            "game_length": np.mean([g.game_length for g in recent]),
            "timeout_rate": np.mean([1.0 if g.timed_out else 0.0 for g in recent]),
            "win_rate": np.mean([learning_won(g) for g in recent]),
            "avg_vp": np.mean([learning_mean(g, g.final_vps) for g in recent]),
            "max_vp": np.mean([learning_max(g, g.final_vps) for g in recent]),
            "settlements": np.mean([learning_mean(g, g.settlements_built) for g in recent]),
            "cities": np.mean([learning_mean(g, g.cities_built) for g in recent]),
            "roads": np.mean([learning_mean(g, g.roads_built) for g in recent]),
            "dev_cards": np.mean([learning_mean(g, g.dev_cards_bought) for g in recent]),
            "starting_pips": np.mean([learning_mean(g, g.starting_pips) if g.starting_pips else 0 for g in recent]),
            "longest_road_claimed": np.mean([learning_has_achievement(g, g.longest_road_holder) for g in recent]),
            "largest_army_claimed": np.mean([learning_has_achievement(g, g.largest_army_holder) for g in recent]),
        }

        return stats

    def clear_recent_stats(self):
        """Clear recent game stats (keep all_game_stats for full history)."""
        self.recent_game_stats = []

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step using collected rollouts.

        Returns:
            Dictionary of training metrics
        """
        self.policy.train()

        # Compute advantages
        self.rollout_buffer.compute_returns_and_advantages()

        # Training metrics
        all_losses = []
        all_pg_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_approx_kls = []
        all_clip_fractions = []

        # PPO epochs
        continue_training = True

        for epoch in range(self.n_epochs):
            if not continue_training:
                break

            # Get minibatches
            batches = self.rollout_buffer.get_batches(self.batch_size)

            for batch in batches:
                # Forward pass
                values, log_probs, entropy = self.policy.evaluate_actions(
                    batch.observations,
                    batch.actions,
                    action_masks=batch.action_masks,
                )
                values = values.flatten()

                # Compute ratios
                ratio = torch.exp(log_probs - batch.old_log_probs)

                # Clipped surrogate loss
                policy_loss_1 = batch.advantages * ratio
                policy_loss_2 = batch.advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    values_pred = batch.values + torch.clamp(
                        values - batch.values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                else:
                    values_pred = values

                value_loss = F.mse_loss(values_pred, batch.returns)

                # Entropy loss
                entropy_loss = -torch.mean(entropy)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()

                # Standard gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # Extra clipping for attention pool parameters (prone to gradient explosion)
                # These parameters can explode during setup phase when roll history is empty
                for name, param in self.policy.named_parameters():
                    if 'attention_pool' in name and param.grad is not None:
                        param.grad.clamp_(-0.1, 0.1)

                self.optimizer.step()

                # Compute approximate KL for early stopping
                with torch.no_grad():
                    log_ratio = log_probs - batch.old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()

                all_losses.append(loss.item())
                all_pg_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_approx_kls.append(approx_kl)
                all_clip_fractions.append(clip_fraction)

                # Early stopping
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

        # Clear buffer
        self.rollout_buffer.clear()

        # Aggressive memory cleanup (especially important for MPS)
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()  # Force completion of pending operations
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return {
            "loss": np.mean(all_losses),
            "policy_loss": np.mean(all_pg_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "approx_kl": np.mean(all_approx_kls),
            "clip_fraction": np.mean(all_clip_fractions),
        }

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10,
        eval_env: Optional[MultiAgentVecEnv] = None,
        eval_freq: int = 10000,
    ):
        """
        Train the policy for a given number of timesteps.

        Args:
            total_timesteps: Total timesteps to train for
            callback: Optional callback for monitoring
            log_interval: How often to log training progress
            eval_env: Optional evaluation environment
            eval_freq: How often to evaluate
        """
        iteration = 0
        start_time = time.time()

        while self.num_timesteps < total_timesteps:
            iteration += 1

            # Collect rollouts
            n_transitions = self.collect_rollouts()

            if n_transitions == 0:
                continue

            # Train
            train_metrics = self.train_step()

            # Logging
            if self.verbose and iteration % log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.num_timesteps / elapsed

                print(f"Timesteps: {self.num_timesteps:,} | "
                      f"Episodes: {self.num_episodes} | "
                      f"FPS: {fps:.0f} | "
                      f"Loss: {train_metrics['loss']:.4f} | "
                      f"Entropy: {-train_metrics['entropy_loss']:.4f} | "
                      f"KL: {train_metrics['approx_kl']:.4f}")

            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar("train/loss", train_metrics["loss"], self.num_timesteps)
                self.writer.add_scalar("train/policy_loss", train_metrics["policy_loss"], self.num_timesteps)
                self.writer.add_scalar("train/value_loss", train_metrics["value_loss"], self.num_timesteps)
                self.writer.add_scalar("train/entropy", -train_metrics["entropy_loss"], self.num_timesteps)
                self.writer.add_scalar("train/approx_kl", train_metrics["approx_kl"], self.num_timesteps)
                self.writer.add_scalar("train/clip_fraction", train_metrics["clip_fraction"], self.num_timesteps)
                self.writer.add_scalar("rollout/n_transitions", n_transitions, self.num_timesteps)
                self.writer.add_scalar("rollout/n_episodes", self.num_episodes, self.num_timesteps)
                self.writer.add_scalar("time/fps", self.num_timesteps / (time.time() - start_time), self.num_timesteps)

            # Callback
            if callback:
                callback.on_step()

        if self.writer:
            self.writer.close()

        return self

    def save(self, path: Union[str, Path]):
        """Save the policy."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
        }, path)
        # Note: verbose output removed to avoid interfering with tqdm progress bars
        # The training script handles checkpoint logging via tqdm.write()

    def load(self, path: Union[str, Path]):
        """Load a saved policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_timesteps = checkpoint.get("num_timesteps", 0)
        self.num_episodes = checkpoint.get("num_episodes", 0)
        # Note: verbose output removed to avoid interfering with tqdm progress bars
        return self

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Union[str, Path],
        **kwargs
    ) -> "MultiAgentPPO":
        """Load a model from checkpoint."""
        model = cls(**kwargs)
        model.load(path)
        return model

    def evaluate_vs_random(self, n_games: int = 10) -> Dict[str, float]:
        """
        Evaluate the policy against random opponents.

        The trained policy plays from a RANDOM position each game (0-3),
        ensuring fair evaluation across all starting positions.
        CatanEnv automatically handles opponent turns via random play.

        Returns:
            Dictionary with win_rate, avg_vp, avg_game_length, position_distribution
        """
        from src.environment.catan_env import CatanEnv
        from src.game.rules import Rules

        self.policy.eval()

        wins = 0
        total_vp = 0
        total_length = 0
        total_settlements = 0
        total_cities = 0
        position_counts = [0, 0, 0, 0]  # Track which positions agent played

        for _ in range(n_games):
            # CatanEnv plays agent as player 0, opponents are random by default
            env = CatanEnv(max_turns=15000)
            obs, info = env.reset()
            done = False

            # Track proposals to prevent trading loops (matching training behavior)
            MAX_PROPOSALS_PER_TURN = 5
            proposals_this_turn = 0
            last_turn = env.state.turn_number

            while not done:
                # Reset proposal counter on new turn
                if env.state.turn_number != last_turn:
                    proposals_this_turn = 0
                    last_turn = env.state.turn_number

                # CatanEnv only returns control when it's the agent's turn
                # Opponents are played automatically between agent actions
                action_mask = info.get('action_mask', Rules.get_legal_actions(env.state))

                # Apply proposal limit to prevent trading loops
                if proposals_this_turn >= MAX_PROPOSALS_PER_TURN:
                    action_mask[366:416] = False  # Disable PROPOSE_TRADE actions

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)
                    action, _, _ = self.policy(obs_tensor, action_masks=mask_tensor)
                    action = action.item()

                # Track proposals
                if 366 <= action < 416:  # PROPOSE_TRADE action range
                    proposals_this_turn += 1

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Collect results
            agent_id = env.agent_player_id
            position_counts[agent_id] += 1
            agent = env.state.players[agent_id]
            if env.state.winner == agent_id:
                wins += 1
            total_vp += agent.victory_points
            total_length += env.state.turn_number
            total_settlements += len(agent.settlements)
            total_cities += len(agent.cities)

        return {
            "eval_win_rate": wins / n_games,
            "eval_avg_vp": total_vp / n_games,
            "eval_avg_length": total_length / n_games,
            "eval_settlements": total_settlements / n_games,
            "eval_cities": total_cities / n_games,
            "eval_position_dist": position_counts,  # Verify randomization
        }
