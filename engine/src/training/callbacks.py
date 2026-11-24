"""Training callbacks for Catan RL agent.

Includes specialized callbacks for self-play training with ELO tracking.

OPTIMIZATION NOTES:
- Reduced callback frequency where possible
- Use num_timesteps for accurate timing instead of n_calls
- Consolidated logging to reduce overhead
"""

import os
from typing import Any, Dict, Optional, List, Callable
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback as SB3EvalCallback

from .game_logger import GameLogger, GameLoggerConfig, LogLevel
from .elo_tracker import ELOTracker, placement_from_vp
from .match_tracker import MatchTracker, PlayerInfo


class WinRateCallback(BaseCallback):
    """Callback to track and log win rate during training."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        save_best: bool = True,
        best_model_save_path: str = "./models",
        early_stopping: bool = False,
        patience: int = 5,
        min_evals_before_stopping: int = 3,
        verbose: int = 1
    ):
        """Initialize the callback.

        Args:
            eval_env: Environment for evaluation.
            eval_freq: Evaluate every N steps.
            n_eval_episodes: Number of episodes per evaluation.
            save_best: Whether to save the best model.
            best_model_save_path: Path to save the best model.
            early_stopping: Whether to stop training if no improvement.
            patience: Number of evaluations without improvement before stopping.
            min_evals_before_stopping: Minimum evaluations before early stopping can trigger.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_best = save_best
        self.best_model_save_path = best_model_save_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_evals_before_stopping = min_evals_before_stopping

        self.win_rates = []
        self.avg_vps = []
        self.best_win_rate = 0.0
        self.best_avg_vp = 0.0
        self.evals_without_improvement = 0
        self.n_evals = 0
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        # Use num_timesteps (actual timesteps) instead of n_calls (callback invocations)
        # This ensures eval_freq means "every N timesteps" regardless of n_envs
        if self.num_timesteps >= self.last_eval_timestep + self.eval_freq:
            self.last_eval_timestep = self.num_timesteps

            win_rate, avg_vp = self._evaluate()
            self.win_rates.append(win_rate)
            self.avg_vps.append(avg_vp)
            self.n_evals += 1

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Win Rate = {win_rate:.2%}, Avg VP = {avg_vp:.1f}")

            # Log to tensorboard
            self.logger.record("eval/win_rate", win_rate)
            self.logger.record("eval/avg_vp", avg_vp)
            self.logger.record("eval/best_win_rate", self.best_win_rate)

            # Check for improvement (use both win rate and avg VP)
            improved = False
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                improved = True
                if self.verbose > 0:
                    print(f"New best win rate: {win_rate:.2%}")

            # Also consider VP improvement if win rates are equal
            if win_rate >= self.best_win_rate and avg_vp > self.best_avg_vp:
                self.best_avg_vp = avg_vp
                improved = True
                if win_rate == self.best_win_rate and self.verbose > 0:
                    print(f"New best avg VP: {avg_vp:.1f}")

            # Save best model
            if improved and self.save_best:
                self._save_best_model()
                self.evals_without_improvement = 0
            else:
                self.evals_without_improvement += 1

            # Early stopping check
            if self.early_stopping and self.n_evals >= self.min_evals_before_stopping:
                if self.evals_without_improvement >= self.patience:
                    if self.verbose > 0:
                        print(f"\nEarly stopping triggered after {self.patience} evaluations without improvement.")
                        print(f"Best win rate: {self.best_win_rate:.2%}, Best avg VP: {self.best_avg_vp:.1f}")
                    return False  # Stop training

            self.logger.record("eval/evals_without_improvement", self.evals_without_improvement)

        return True

    def _save_best_model(self) -> None:
        """Save the current model as the best model."""
        os.makedirs(self.best_model_save_path, exist_ok=True)
        path = os.path.join(self.best_model_save_path, "catan_agent_best.zip")
        self.model.save(path)
        if self.verbose > 0:
            print(f"Saved new best model to {path}")

    def _evaluate(self) -> tuple:
        """Run evaluation episodes."""
        wins = 0
        total_vp = 0

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False

            while not done:
                action_mask = self.eval_env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=action_mask)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            if info.get("is_agent_winner", False):
                wins += 1
            # Use total_vp (includes hidden VP cards) for accurate tracking
            total_vp += info.get("total_vp", info.get("vp", 0))

        return wins / self.n_eval_episodes, total_vp / self.n_eval_episodes


class EntropyDecayCallback(BaseCallback):
    """Callback to decay entropy coefficient during training."""

    def __init__(
        self,
        initial_ent_coef: float = 0.1,
        final_ent_coef: float = 0.01,
        decay_steps: int = 800000,  # Increased from 500000 for longer exploration
        verbose: int = 0
    ):
        """Initialize the callback.

        Args:
            initial_ent_coef: Starting entropy coefficient.
            final_ent_coef: Final entropy coefficient.
            decay_steps: Number of steps for decay.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        """Update entropy coefficient."""
        # Use num_timesteps for accurate progress tracking
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_ent_coef = self.initial_ent_coef - progress * (self.initial_ent_coef - self.final_ent_coef)

        # Update the model's entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = new_ent_coef
        elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            # For some implementations
            pass

        # Log periodically (approximately every 10K timesteps)
        if self.n_calls % 1000 == 0:
            self.logger.record("train/ent_coef", new_ent_coef)

        return True


class CheckpointCallback(BaseCallback):
    """Callback to save model checkpoints.

    Optionally integrates with self-play opponent pool and ELO tracking.
    """

    def __init__(
        self,
        save_freq: int = 50000,
        save_path: str = "./models",
        name_prefix: str = "catan_agent",
        self_play_manager=None,
        elo_tracker: Optional[ELOTracker] = None,
        verbose: int = 1
    ):
        """Initialize the callback.

        Args:
            save_freq: Save every N timesteps.
            save_path: Directory to save models.
            name_prefix: Prefix for saved model names.
            self_play_manager: Optional SelfPlayManager to add checkpoints to pool.
            elo_tracker: Optional ELOTracker to record model ELO.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.self_play_manager = self_play_manager
        self.elo_tracker = elo_tracker
        self.last_save_timestep = 0

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Save checkpoint if needed."""
        # Use num_timesteps for accurate checkpoint timing
        if self.num_timesteps >= self.last_save_timestep + self.save_freq:
            self.last_save_timestep = self.num_timesteps
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saved model checkpoint to {path}")

            # Add to opponent pool if self-play enabled
            if self.self_play_manager is not None:
                self.self_play_manager.opponent_pool.append(path)
                # Trim pool if too large
                max_size = self.self_play_manager.max_pool_size
                if len(self.self_play_manager.opponent_pool) > max_size:
                    self.self_play_manager.opponent_pool.pop(0)
                if self.verbose > 0:
                    print(f"  Added to opponent pool (size: {len(self.self_play_manager.opponent_pool)})")

            # Record ELO for this checkpoint
            if self.elo_tracker is not None:
                self.elo_tracker.add_model_to_pool(path)
                if self.verbose > 0:
                    print(f"  ELO at checkpoint: {self.elo_tracker.current_elo:.0f}")

        return True


class GameStatsCallback(BaseCallback):
    """Callback to track detailed game statistics.

    OPTIMIZATION: Reduced logging frequency and uses efficient data structures.
    """

    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        """Initialize the callback.

        Args:
            log_freq: Log statistics every N timesteps.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        # OPTIMIZATION: Use fixed-size circular buffers for recent stats
        self._buffer_size = 100
        self.episode_vps = []
        self.wins = 0
        self.total_episodes = 0
        self.last_log_timestep = 0

    def _on_step(self) -> bool:
        """Track episode statistics.

        OPTIMIZATION: Only process infos when episode ends, minimal per-step overhead.
        """
        # Check for episode end in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "is_agent_winner" in info and info["is_agent_winner"] is not None:
                self.total_episodes += 1
                if info["is_agent_winner"]:
                    self.wins += 1
                if "vp" in info:
                    # OPTIMIZATION: Keep only recent VPs
                    self.episode_vps.append(info["vp"])
                    if len(self.episode_vps) > self._buffer_size:
                        self.episode_vps.pop(0)

        # Log statistics periodically (using num_timesteps)
        if self.num_timesteps >= self.last_log_timestep + self.log_freq and self.total_episodes > 0:
            self.last_log_timestep = self.num_timesteps
            win_rate = self.wins / self.total_episodes
            self.logger.record("game/win_rate", win_rate)
            self.logger.record("game/total_episodes", self.total_episodes)
            if self.episode_vps:
                self.logger.record("game/avg_vp", np.mean(self.episode_vps))

        return True

    def _on_training_end(self) -> None:
        """Log final statistics."""
        if self.total_episodes > 0:
            print(f"\nTraining Statistics:")
            print(f"  Total Episodes: {self.total_episodes}")
            print(f"  Win Rate: {self.wins / self.total_episodes:.2%}")
            if self.episode_vps:
                print(f"  Average VP: {np.mean(self.episode_vps):.1f}")


class EvalCallback(SB3EvalCallback):
    """Extended evaluation callback with win rate tracking."""

    def __init__(
        self,
        eval_env,
        best_model_save_path: str = "./models",
        log_path: str = "./logs",
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            eval_env: Environment for evaluation.
            best_model_save_path: Path to save best model.
            log_path: Path for evaluation logs.
            eval_freq: Evaluate every N steps.
            n_eval_episodes: Number of episodes per evaluation.
            deterministic: Whether to use deterministic actions.
            verbose: Verbosity level.
        """
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.win_rates_history = []

    def _on_step(self) -> bool:
        """Evaluate and track win rate."""
        result = super()._on_step()

        # Additional win rate logging after each evaluation
        if self.n_calls % self.eval_freq == 0:
            # Count wins from recent evaluations
            wins = sum(1 for info in getattr(self, '_last_episode_infos', [])
                      if info.get("is_agent_winner", False))
            win_rate = wins / max(1, len(getattr(self, '_last_episode_infos', [])))
            self.win_rates_history.append(win_rate)

        return result


class GameLoggingCallback(BaseCallback):
    """Callback to log games for replay and animation.

    This callback integrates with the environment to log games during training.
    It supports sampling (log every N games) and configurable detail levels.
    """

    def __init__(
        self,
        log_every_n_games: int = 100,
        log_level: str = "key_actions",
        save_dir: str = "./game_logs",
        always_log_setup: bool = True,
        always_log_settlements: bool = True,
        always_log_cities: bool = True,
        always_log_dev_cards: bool = True,
        always_log_robber: bool = True,
        include_state_snapshots: bool = False,
        verbose: int = 1
    ):
        """Initialize the game logging callback.

        Args:
            log_every_n_games: Only log every N games (1 = log all).
            log_level: Detail level - "none", "outcome_only", "key_actions", "full".
            save_dir: Directory to save game logs.
            always_log_setup: Always log initial placements.
            always_log_settlements: Always log settlement builds.
            always_log_cities: Always log city upgrades.
            always_log_dev_cards: Always log dev card purchases/plays.
            always_log_robber: Always log robber moves.
            include_state_snapshots: Include periodic state snapshots.
            verbose: Verbosity level.
        """
        super().__init__(verbose)

        # Parse log level
        level_map = {
            "none": LogLevel.NONE,
            "outcome_only": LogLevel.OUTCOME_ONLY,
            "key_actions": LogLevel.KEY_ACTIONS,
            "full": LogLevel.FULL,
            "analytics": LogLevel.ANALYTICS,
        }
        log_level_enum = level_map.get(log_level.lower(), LogLevel.KEY_ACTIONS)

        # Create config
        config = GameLoggerConfig(
            log_every_n_games=log_every_n_games,
            log_level=log_level_enum,
            always_log_setup=always_log_setup,
            always_log_settlements=always_log_settlements,
            always_log_cities=always_log_cities,
            always_log_dev_cards=always_log_dev_cards,
            always_log_robber=always_log_robber,
            save_dir=save_dir,
            include_state_snapshots=include_state_snapshots,
        )

        self.game_logger = GameLogger(config)
        self.games_completed = 0
        self.games_logged = 0

    def _on_step(self) -> bool:
        """Check for episode completions and log games."""
        # Check for completed episodes in infos
        for i, info in enumerate(self.locals.get("infos", [])):
            # Episode ended
            if info.get("is_agent_winner") is not None:
                self.games_completed += 1

                # Log stats periodically
                if self.verbose > 0 and self.games_completed % 100 == 0:
                    stats = self.game_logger.get_stats()
                    self.logger.record("game_log/games_seen", stats["total_games_seen"])
                    self.logger.record("game_log/games_logged", stats["games_logged"])

        return True

    def _on_training_end(self) -> None:
        """Log final stats."""
        stats = self.game_logger.get_stats()
        if self.verbose > 0:
            print(f"\nGame Logging Statistics:")
            print(f"  Total games seen: {stats['total_games_seen']}")
            print(f"  Games logged: {stats['games_logged']}")
            print(f"  Log file: {stats['current_file']}")

    def get_game_logger(self) -> GameLogger:
        """Get the underlying GameLogger instance."""
        return self.game_logger


class SelfPlayStatsCallback(BaseCallback):
    """Callback for tracking meaningful metrics during self-play training.

    During self-play, raw win rate converges to ~25% (1 in 4 players).
    This callback tracks more meaningful metrics:
    - Win rate vs random baseline (periodic evaluation)
    - Average VP achieved
    - ELO rating progression
    - Placement distribution
    - Detailed match history with opponent info
    """

    def __init__(
        self,
        eval_env_fn: Callable,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        elo_tracker: Optional[ELOTracker] = None,
        match_tracker: Optional[MatchTracker] = None,
        log_freq: int = 10000,
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            eval_env_fn: Function that creates evaluation env with random opponents.
            eval_freq: Evaluate vs random every N timesteps.
            n_eval_episodes: Episodes per evaluation.
            elo_tracker: Optional ELO tracker instance.
            match_tracker: Optional MatchTracker for detailed game logging.
            log_freq: Log training stats every N timesteps.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.elo_tracker = elo_tracker
        self.match_tracker = match_tracker
        self.log_freq = log_freq

        # Evaluation tracking
        self.last_eval_timestep = 0
        self.eval_win_rates = []
        self.eval_avg_vps = []
        self.best_eval_win_rate = 0.0

        # Training stats (reset periodically)
        self.placement_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.training_vps = []
        self.total_training_games = 0  # Completed games only (someone won)
        self.truncated_games = 0  # Games that hit timeout
        self.last_log_timestep = 0

    def _on_step(self) -> bool:
        """Track stats and run periodic evaluation."""
        # Track training game outcomes
        for info in self.locals.get("infos", []):
            # Check for game completion (not truncation)
            game_completed = info.get("game_completed", False)
            game_truncated = info.get("game_truncated", False)

            if game_truncated:
                self.truncated_games += 1

            # Only count completed games for win rate / placement stats
            if game_completed and info.get("is_agent_winner") is not None:
                self.total_training_games += 1
                # Use total_vp (includes hidden VP cards) for accurate tracking
                agent_vp = info.get("total_vp", info.get("vp", 0))
                self.training_vps.append(agent_vp)

                # Determine placement from VP - use total VP for accurate placement
                opponent_vps = info.get("opponent_total_vps", info.get("opponent_vps", [0, 0, 0]))
                placement = placement_from_vp(agent_vp, opponent_vps)
                self.placement_counts[placement] = self.placement_counts.get(placement, 0) + 1

                # Update ELO if tracker provided
                opponent_paths = info.get("opponent_paths", ["random", "random", "random"])
                if self.elo_tracker:
                    self.elo_tracker.record_game(
                        placement=placement,
                        opponent_paths=opponent_paths,
                        timestep=self.num_timesteps,
                    )

                # Record detailed match if tracker provided
                if self.match_tracker:
                    # Set opponents from info
                    opponent_infos = info.get("opponent_infos", [
                        {"agent_type": "random"} for _ in range(3)
                    ])
                    self.match_tracker.set_opponents_for_game(opponent_infos)

                    # Determine winner (who has highest total VP)
                    all_vps = [agent_vp] + opponent_vps
                    winner_idx = all_vps.index(max(all_vps))

                    self.match_tracker.record_match(
                        timestep=self.num_timesteps,
                        winner_idx=winner_idx,
                        victory_points=all_vps,
                        turns=info.get("turn", 0),
                        agent_reward=info.get("episode", {}).get("r", 0.0),
                    )

        # Log training stats periodically
        if self.num_timesteps >= self.last_log_timestep + self.log_freq:
            self.last_log_timestep = self.num_timesteps
            self._log_training_stats()

        # Evaluate vs random baseline periodically
        if self.num_timesteps >= self.last_eval_timestep + self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self._evaluate_vs_random()

        return True

    def _log_training_stats(self) -> None:
        """Log training statistics."""
        if self.total_training_games == 0:
            return

        # Placement distribution (more meaningful than win rate for self-play)
        total = sum(self.placement_counts.values())
        if total > 0:
            for place in [1, 2, 3, 4]:
                rate = self.placement_counts.get(place, 0) / total
                self.logger.record(f"self_play/placement_{place}_rate", rate)

        # Average VP
        if self.training_vps:
            recent_vps = self.training_vps[-100:]
            self.logger.record("self_play/avg_vp", np.mean(recent_vps))
            self.logger.record("self_play/max_vp", np.max(recent_vps))

        # ELO if tracking
        if self.elo_tracker:
            stats = self.elo_tracker.get_stats()
            self.logger.record("self_play/elo", stats["current_elo"])
            self.logger.record("self_play/elo_vs_random", stats["elo_vs_random_baseline"])

        self.logger.record("self_play/total_games", self.total_training_games)
        self.logger.record("self_play/truncated_games", self.truncated_games)

    def _evaluate_vs_random(self) -> None:
        """Evaluate model against random opponents."""
        eval_env = self.eval_env_fn()

        wins = 0
        total_vp = 0

        for _ in range(self.n_eval_episodes):
            obs, info = eval_env.reset()
            done = False

            while not done:
                action_mask = eval_env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=action_mask)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

            if info.get("is_agent_winner", False):
                wins += 1
            # Use total_vp (includes hidden VP cards) for accurate tracking
            total_vp += info.get("total_vp", info.get("vp", 0))

        eval_env.close()

        # Guard against division by zero
        if self.n_eval_episodes == 0:
            return

        win_rate = wins / self.n_eval_episodes
        avg_vp = total_vp / self.n_eval_episodes

        self.eval_win_rates.append(win_rate)
        self.eval_avg_vps.append(avg_vp)

        # Log
        self.logger.record("eval_vs_random/win_rate", win_rate)
        self.logger.record("eval_vs_random/avg_vp", avg_vp)
        self.logger.record("eval_vs_random/best_win_rate", self.best_eval_win_rate)

        if win_rate > self.best_eval_win_rate:
            self.best_eval_win_rate = win_rate

        if self.verbose > 0:
            print(f"Step {self.num_timesteps} - Eval vs Random: "
                  f"Win Rate = {win_rate:.2%}, Avg VP = {avg_vp:.1f}")
            if self.elo_tracker:
                print(f"  ELO: {self.elo_tracker.current_elo:.0f}")

    def _on_training_end(self) -> None:
        """Log final statistics."""
        # Save match tracker data
        if self.match_tracker:
            self.match_tracker.save()

        if self.verbose > 0:
            print(f"\nSelf-Play Training Statistics:")
            print(f"  Completed Games: {self.total_training_games}")
            print(f"  Truncated Games (timeout): {self.truncated_games}")

            total = sum(self.placement_counts.values())
            if total > 0:
                print(f"  Placement Distribution:")
                for place in [1, 2, 3, 4]:
                    rate = self.placement_counts.get(place, 0) / total
                    print(f"    {place}st/nd/rd/th: {rate:.1%}")

            if self.eval_win_rates:
                print(f"  Best Win Rate vs Random: {self.best_eval_win_rate:.2%}")
                print(f"  Final Win Rate vs Random: {self.eval_win_rates[-1]:.2%}")

            if self.elo_tracker:
                stats = self.elo_tracker.get_stats()
                print(f"  Final ELO: {stats['current_elo']:.0f}")
                print(f"  ELO vs Random Baseline: +{stats['elo_vs_random_baseline']:.0f}")

            if self.match_tracker:
                stats = self.match_tracker.get_stats()
                print(f"\nMatch History:")
                print(f"  Total Matches Recorded: {stats['total_matches']}")
                print(f"  Win Rate vs Random Only: {stats['win_rate_vs_random']:.2%}")
                print(f"  Win Rate vs Pool Models: {stats['win_rate_vs_pool']:.2%}")
                print(f"  Avg Placement: {stats['avg_agent_placement']:.2f}")
                print(f"  Saved to: {self.match_tracker.save_dir}/match_summary.csv")


class ELOCallback(BaseCallback):
    """Callback for tracking and logging ELO ratings during training.

    Works with both self-play and regular training to track improvement
    relative to a baseline.
    """

    def __init__(
        self,
        elo_tracker: ELOTracker,
        save_freq: int = 50000,
        save_path: str = "./models",
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            elo_tracker: ELO tracker instance.
            save_freq: Save ELO data every N timesteps.
            save_path: Path to save ELO data.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.elo_tracker = elo_tracker
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_timestep = 0

        # Set save path for tracker
        elo_path = os.path.join(save_path, "elo_data.json")
        self.elo_tracker.save_path = elo_path

    def _on_step(self) -> bool:
        """Log ELO and save periodically."""
        # Log current ELO
        if self.n_calls % 1000 == 0:
            stats = self.elo_tracker.get_stats()
            self.logger.record("elo/current", stats["current_elo"])
            self.logger.record("elo/games_played", stats["current_games"])
            self.logger.record("elo/vs_random_baseline", stats["elo_vs_random_baseline"])

        # Save ELO data periodically
        if self.num_timesteps >= self.last_save_timestep + self.save_freq:
            self.last_save_timestep = self.num_timesteps
            self.elo_tracker.save()

        return True

    def _on_training_end(self) -> None:
        """Save final ELO data."""
        self.elo_tracker.save()

        if self.verbose > 0:
            stats = self.elo_tracker.get_stats()
            print(f"\nELO Statistics:")
            print(f"  Final ELO: {stats['current_elo']:.0f}")
            print(f"  Total Games: {stats['current_games']}")
            print(f"  Win Rate: {stats['current_win_rate']:.2%}")
            print(f"  ELO vs Random: +{stats['elo_vs_random_baseline']:.0f}")
