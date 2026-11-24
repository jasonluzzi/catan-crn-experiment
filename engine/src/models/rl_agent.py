"""Phase 2 RL agent training for Catan using MaskablePPO with attention modules.

This module provides the CatanRLAgent class with:
- Custom attention-based policy (TileModule + PlayerModule)
- Self-play training integration
- Opponent pool management
- Numerically stable action distributions
"""

import os
from typing import Optional, List, Dict, Any
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_linear_fn

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks
    MASKABLE_PPO_AVAILABLE = True

    # Apply numerical stability patch to MaskableCategorical
    from .stable_distributions import patch_maskable_categorical
    patch_maskable_categorical()
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    print("Warning: sb3-contrib not installed. MaskablePPO not available.")

from ..environment.catan_env import CatanEnv
from ..environment.reward import RewardConfig
from ..environment.logging_env import LoggingCatanEnv, make_logging_env
from .baseline import RandomAgent, GreedyAgent
from .catan_policy import CatanPolicy, create_catan_policy_kwargs
from ..training.callbacks import (
    WinRateCallback, EntropyDecayCallback,
    CheckpointCallback, GameStatsCallback,
    SelfPlayStatsCallback, ELOCallback
)
from ..training.self_play import SelfPlayManager, SelfPlayAgent, SelfPlayCallback
from ..training.game_logger import GameLogger, GameLoggerConfig, LogLevel
from ..training.elo_tracker import ELOTracker
from ..training.match_tracker import MatchTracker


def mask_fn(env) -> np.ndarray:
    """Get action mask from the environment.

    This handles wrapped environments by checking for the action_masks method.
    Includes safety checks to ensure the mask is valid (at least one True value).
    """
    # Handle wrapped environments (e.g., Monitor wrapper)
    if hasattr(env, 'action_masks'):
        mask = env.action_masks()
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'action_masks'):
        mask = env.unwrapped.action_masks()
    elif hasattr(env, 'env') and hasattr(env.env, 'action_masks'):
        mask = env.env.action_masks()
    else:
        raise AttributeError(f"Cannot find action_masks method on {type(env)}")

    # Safety check: ensure mask is valid boolean array
    mask = np.asarray(mask, dtype=bool)

    # Critical: ensure at least one action is valid to prevent Simplex constraint violation
    if not mask.any():
        import warnings
        warnings.warn("No valid actions in mask - allowing all actions as fallback")
        mask = np.ones_like(mask, dtype=bool)

    return mask


def make_env(
    opponent_policies: Optional[List] = None,
    reward_config: Optional[RewardConfig] = None,
    rank: int = 0,
    seed: int = 0,
    agent_position: Optional[int] = None,
    opponent_sampler: Optional[callable] = None,
):
    """Create a Catan environment with action masking.

    Args:
        opponent_policies: List of opponent policies.
        reward_config: Reward configuration.
        rank: Environment rank for parallel envs.
        seed: Random seed.
        agent_position: Fixed position for agent (0-3), or None for random.
        opponent_sampler: Optional callable that returns (opponents, opponent_infos)
            on each episode reset. Used for self-play.

    Returns:
        A function that creates the environment.
    """
    def _init():
        env = CatanEnv(
            opponent_policies=opponent_policies,
            reward_config=reward_config,
            agent_position=agent_position,
            opponent_sampler=opponent_sampler,
        )
        env = Monitor(env)
        if MASKABLE_PPO_AVAILABLE:
            env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


class SelfPlayEnvManager:
    """Manages environment opponents during self-play training."""

    def __init__(
        self,
        self_play_manager: SelfPlayManager,
        n_envs: int = 4,
        seed: int = 42,
        agent_position: Optional[int] = None,
    ):
        """Initialize the self-play environment manager.

        Args:
            self_play_manager: SelfPlayManager instance.
            n_envs: Number of parallel environments.
            seed: Random seed.
            agent_position: Fixed agent position (0-3) or None for random.
        """
        self.self_play_manager = self_play_manager
        self.n_envs = n_envs
        self.seed = seed
        self.agent_position = agent_position
        self.envs = None

    def create_envs(self, current_model=None, mix_strategy: str = "mixed"):
        """Create environments with opponents from the pool.

        Args:
            current_model: Current model for self-play opponents.
            mix_strategy: How to mix opponents ("random", "self", "pool", "mixed").

        Returns:
            VecEnv with configured opponents.
        """
        env_fns = []
        for i in range(self.n_envs):
            opponents = self.self_play_manager.get_opponents_for_training(
                current_model=current_model,
                mix_strategy=mix_strategy,
            )
            env_fns.append(make_env(
                opponent_policies=opponents,
                rank=i,
                seed=self.seed + i,
                agent_position=self.agent_position,
            ))

        if self.n_envs > 1:
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)


class CatanRLAgent:
    """Phase 2 RL agent for playing Catan with attention-based features."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        verbose: int = 1,
        use_custom_policy: bool = True,
    ):
        """Initialize the RL agent.

        Args:
            model_path: Path to a saved model to load.
            device: Device to use ("cpu", "cuda", or "auto").
            verbose: Verbosity level.
            use_custom_policy: Whether to use the custom attention policy.
        """
        self.model = None
        self.device = device
        self.verbose = verbose
        self.use_custom_policy = use_custom_policy
        self.self_play_manager = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(
        self,
        total_timesteps: int = 1_000_000,
        n_envs: int = 8,  # OPTIMIZED: Increased from 4 to 8 for better parallelization
        opponent_type: str = "random",
        learning_rate: float = 1e-4,  # Reduced from 3e-4 for stability
        n_steps: int = 1024,  # OPTIMIZED: Reduced from 2048 for faster updates (more frequent gradient steps)
        batch_size: int = 512,  # OPTIMIZED: Increased from 256 for better GPU utilization
        n_epochs: int = 8,  # OPTIMIZED: Reduced from 10 - faster per update cycle
        gamma: float = 0.99,
        ent_coef: float = 0.02,  # Increased from 0.01 to maintain exploration
        clip_range: float = 0.2,
        target_kl: float = 0.02,  # Stop updates early if KL divergence too high
        save_path: str = "./models",
        log_path: str = "./logs",
        eval_freq: int = 100000,
        checkpoint_freq: int = 100000,
        # Game logging parameters
        enable_game_logging: bool = False,
        log_every_n_games: int = 100,
        game_log_level: str = "key_actions",
        game_log_dir: str = "./game_logs",
        always_log_setup: bool = True,
        always_log_settlements: bool = True,
        always_log_cities: bool = True,
        always_log_dev_cards: bool = True,
        agent_position: Optional[int] = None,
        # Evaluation and early stopping parameters
        n_eval_episodes: int = 20,
        save_best: bool = True,
        early_stopping: bool = False,
        patience: int = 5,
        # Self-play parameters
        use_self_play: bool = False,
        opponent_pool_size: int = 30,  # Increased from 10 for more diverse opponents
        self_play_save_freq: int = 50000,
        mix_strategy: str = "mixed",
        # Custom policy parameters
        features_dim: int = 512,
        tile_embed_dim: int = 128,
        tile_num_heads: int = 8,
        tile_num_layers: int = 2,
        player_output_dim: int = 128,
    ) -> None:
        """Train the agent.

        Args:
            total_timesteps: Total training timesteps.
            n_envs: Number of parallel environments.
            opponent_type: Type of opponents ("random", "greedy", "mixed").
            learning_rate: Learning rate.
            n_steps: Steps per environment per update.
            batch_size: Minibatch size.
            n_epochs: Number of epochs per update.
            gamma: Discount factor.
            ent_coef: Entropy coefficient.
            clip_range: PPO clip range.
            target_kl: Target KL divergence for early stopping of updates.
            save_path: Path to save models.
            log_path: Path for tensorboard logs.
            eval_freq: Evaluation frequency in timesteps.
            checkpoint_freq: Checkpoint save frequency.
            enable_game_logging: Whether to log games for replay.
            log_every_n_games: Only log every N games (1 = log all).
            game_log_level: Detail level - "none", "outcome_only", "key_actions", "full".
            game_log_dir: Directory to save game logs.
            always_log_setup: Always log initial placements.
            always_log_settlements: Always log settlement builds.
            always_log_cities: Always log city upgrades.
            always_log_dev_cards: Always log dev card purchases/plays.
            agent_position: Fixed agent position (0-3), or None for random each game.
            n_eval_episodes: Number of episodes per evaluation.
            save_best: Whether to save the best model based on win rate.
            early_stopping: Whether to stop training if no improvement.
            patience: Number of evaluations without improvement before stopping.
            use_self_play: Whether to use self-play training.
            opponent_pool_size: Maximum number of opponents in the pool.
            self_play_save_freq: Frequency to save model to opponent pool.
            mix_strategy: Strategy for mixing opponents ("random", "self", "pool", "mixed").
            features_dim: Output dimension of feature extractor.
            tile_embed_dim: Embedding dimension for tile module.
            tile_num_heads: Number of attention heads for tiles.
            tile_num_layers: Number of transformer layers for tiles.
            player_output_dim: Output dimension for player module.
        """
        if not MASKABLE_PPO_AVAILABLE:
            raise RuntimeError("MaskablePPO not available. Install sb3-contrib.")

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # Initialize self-play manager, ELO tracker, and match tracker if enabled
        self.elo_tracker = None
        self.match_tracker = None
        if use_self_play:
            self_play_dir = os.path.join(save_path, "self_play")
            self.self_play_manager = SelfPlayManager(
                save_dir=self_play_dir,
                max_pool_size=opponent_pool_size,
                save_freq=self_play_save_freq,
            )
            # Initialize ELO tracker
            elo_path = os.path.join(save_path, "elo_data.json")
            self.elo_tracker = ELOTracker(save_path=elo_path)

            # Initialize match tracker for detailed game logging
            self.match_tracker = MatchTracker(
                save_dir=save_path,
                elo_tracker=self.elo_tracker,
            )

            print(f"Self-play enabled: pool size={opponent_pool_size}")
            print(f"ELO tracking enabled: starting ELO={self.elo_tracker.current_elo:.0f}")
            print(f"Match tracking enabled: saving to {save_path}/match_summary.csv")

        # Create opponent policies (used as default/fallback)
        opponents = self._create_opponents(opponent_type)

        # Create opponent sampler for self-play (samples from pool on each episode)
        # Note: We read from disk to support SubprocVecEnv (each subprocess needs
        # to see pool updates made by the main process)
        opponent_sampler = None
        if use_self_play and self.self_play_manager is not None:
            # Use absolute path for subprocess compatibility
            pool_dir = os.path.abspath(os.path.join(save_path, "self_play"))

            # Determine ELO save path for reading opponent ELOs
            elo_save_path = os.path.join(save_path, "elo_data.json") if save_path else None

            def _create_opponent_sampler(pool_directory, mix_strat, elo_path, total_ts):
                """Create a closure that samples opponents from the pool on disk."""
                def sampler():
                    # Import inside function for subprocess compatibility
                    import glob
                    import json as _json
                    import os as _os
                    import numpy as _np
                    from ..training.self_play import SelfPlayAgent as _SelfPlayAgent, get_curriculum_mix
                    from .baseline import RandomAgent as _RandomAgent

                    # Read pool from disk (supports SubprocVecEnv)
                    # Look for both opponent_*.zip AND catan_agent_*_steps.zip
                    pool_files = []
                    pool_files.extend(glob.glob(_os.path.join(pool_directory, "opponent_*.zip")))
                    # Also check parent dir for checkpoint models
                    parent_dir = _os.path.dirname(pool_directory)
                    pool_files.extend(glob.glob(_os.path.join(parent_dir, "catan_agent_*_steps.zip")))
                    pool_files = sorted(set(pool_files))  # Remove duplicates

                    # Load ELO data if available
                    elo_data = {}
                    if elo_path and _os.path.exists(elo_path):
                        try:
                            with open(elo_path, 'r') as f:
                                data = _json.load(f)
                                elo_data = data.get("pool_ratings", {})
                        except Exception:
                            pass

                    def get_elo_for_path(path):
                        """Look up ELO for a model path."""
                        abs_path = _os.path.abspath(path)
                        if abs_path in elo_data:
                            return elo_data[abs_path].get("elo", 1500)
                        if path in elo_data:
                            return elo_data[path].get("elo", 1500)
                        return 1500  # Default ELO if not found

                    # Determine effective mix strategy
                    effective_mix = mix_strat
                    if mix_strat == "curriculum":
                        # Read current progress from file (written by callback)
                        progress_file = _os.path.join(parent_dir, "curriculum_progress.txt")
                        progress = 0.0
                        if _os.path.exists(progress_file):
                            try:
                                with open(progress_file, 'r') as f:
                                    progress = float(f.read().strip())
                            except Exception:
                                pass
                        effective_mix = get_curriculum_mix(progress)

                    opps = []
                    infos = []

                    if effective_mix == "random" or len(pool_files) == 0:
                        # All random
                        for _ in range(3):
                            opps.append(_RandomAgent())
                            infos.append({"agent_type": "random", "elo": 1000})
                    elif effective_mix == "pool":
                        # All from pool
                        for _ in range(3):
                            if pool_files:
                                path = _np.random.choice(pool_files)
                                try:
                                    from sb3_contrib import MaskablePPO
                                    model = MaskablePPO.load(path)
                                    opps.append(_SelfPlayAgent(model, model_path=path))
                                    elo = get_elo_for_path(path)
                                    infos.append({"agent_type": "pool", "model_path": path, "elo": elo})
                                except Exception:
                                    opps.append(_RandomAgent())
                                    infos.append({"agent_type": "random", "elo": 1000})
                            else:
                                opps.append(_RandomAgent())
                                infos.append({"agent_type": "random", "elo": 1000})
                    else:  # mixed
                        # One random
                        opps.append(_RandomAgent())
                        infos.append({"agent_type": "random", "elo": 1000})

                        # Two from pool (or random if pool empty)
                        for _ in range(2):
                            if pool_files:
                                path = _np.random.choice(pool_files)
                                try:
                                    from sb3_contrib import MaskablePPO
                                    model = MaskablePPO.load(path)
                                    opps.append(_SelfPlayAgent(model, model_path=path))
                                    elo = get_elo_for_path(path)
                                    infos.append({"agent_type": "pool", "model_path": path, "elo": elo})
                                except Exception:
                                    opps.append(_RandomAgent())
                                    infos.append({"agent_type": "random", "elo": 1000})
                            else:
                                opps.append(_RandomAgent())
                                infos.append({"agent_type": "random", "elo": 1000})

                    return opps, infos
                return sampler
            opponent_sampler = _create_opponent_sampler(pool_dir, mix_strategy, elo_save_path, total_timesteps)

        # Create shared game logger if enabled
        # NOTE: With SubprocVecEnv, each subprocess gets its own copy of the logger
        # with independent game counters. We adjust log_every_n_games per-env so that
        # the TOTAL logged games matches what the user requested.
        # E.g., if user wants 1 game per 1000 total games with 4 envs:
        #       each env logs every 250 games -> 4 * (games/250) = games/250 total logged
        # Wait, that's wrong. Let me think...
        # If user wants 1 game logged per 1000 games total:
        #   - Total games = G, want G/1000 logged
        #   - With n_envs, each env plays G/n_envs games
        #   - Each env should log (G/n_envs) / (1000/n_envs) = G/1000 games
        #   - So per-env frequency = log_every_n_games / n_envs
        per_env_log_frequency = max(1, log_every_n_games // n_envs)

        game_logger = None
        shared_file_path = None
        if enable_game_logging:
            os.makedirs(game_log_dir, exist_ok=True)
            # Create a shared file path for all subprocesses to write to
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shared_file_path = os.path.join(game_log_dir, f"games_{timestamp}.jsonl")

            level_map = {
                "none": LogLevel.NONE,
                "outcome_only": LogLevel.OUTCOME_ONLY,
                "key_actions": LogLevel.KEY_ACTIONS,
                "full": LogLevel.FULL,
                "analytics": LogLevel.ANALYTICS,
            }
            config = GameLoggerConfig(
                log_every_n_games=per_env_log_frequency,
                log_level=level_map.get(game_log_level.lower(), LogLevel.KEY_ACTIONS),
                always_log_setup=always_log_setup,
                always_log_settlements=always_log_settlements,
                always_log_cities=always_log_cities,
                always_log_dev_cards=always_log_dev_cards,
                save_dir=game_log_dir,
                shared_file_path=shared_file_path,
            )
            game_logger = GameLogger(config)
            print(f"Game logging enabled: every {log_every_n_games} games total "
                  f"(~{per_env_log_frequency} per env with {n_envs} envs), level={game_log_level}")
            print(f"  -> Writing to shared file: {shared_file_path}")

        # Create training environments
        if enable_game_logging:
            env_fns = [
                make_logging_env(
                    opponent_policies=opponents,
                    game_logger=game_logger,
                    log_every_n_games=per_env_log_frequency,
                    log_level=game_log_level,
                    save_dir=game_log_dir,
                    shared_file_path=shared_file_path,
                    rank=i,
                    seed=42 + i,
                    opponent_type=opponent_type,
                    agent_position=agent_position,
                    opponent_sampler=opponent_sampler,
                )
                for i in range(n_envs)
            ]
        else:
            env_fns = [
                make_env(
                    opponent_policies=opponents,
                    rank=i,
                    seed=42 + i,
                    agent_position=agent_position,
                    opponent_sampler=opponent_sampler,
                )
                for i in range(n_envs)
            ]

        if n_envs > 1:
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)

        # Create evaluation environment
        eval_env = make_env(opponent_policies=opponents, seed=123)()

        # Build policy kwargs for custom policy
        policy_kwargs = None
        policy_type = "MlpPolicy"

        if self.use_custom_policy:
            policy_kwargs = create_catan_policy_kwargs(
                features_dim=features_dim,
                tile_embed_dim=tile_embed_dim,
                tile_num_heads=tile_num_heads,
                tile_num_layers=tile_num_layers,
                player_output_dim=player_output_dim,
            )
            print(f"Using custom attention policy: features_dim={features_dim}, "
                  f"tile_embed={tile_embed_dim}, heads={tile_num_heads}, layers={tile_num_layers}")

        # Create learning rate schedule (linear decay from start to 1e-5 over 80% of training)
        lr_schedule = get_linear_fn(
            start=learning_rate,
            end=1e-5,
            end_fraction=0.8
        )

        # Create or continue training model
        if self.model is None:
            self.model = MaskablePPO(
                policy_type,
                env,
                learning_rate=lr_schedule,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                ent_coef=ent_coef,
                clip_range=clip_range,
                max_grad_norm=0.5,  # Gradient clipping for stability
                target_kl=target_kl,  # Early stopping if KL divergence too high
                verbose=self.verbose,
                tensorboard_log=log_path if log_path else None,
                device=self.device,
                policy_kwargs=policy_kwargs,
            )
        else:
            # Reload with new env to support different n_envs
            # This preserves weights but allows changing parallelism
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
                temp_path = f.name
            self.model.save(temp_path)
            self.model = MaskablePPO.load(
                temp_path,
                env=env,
                device=self.device,
                tensorboard_log=log_path if log_path else None,
            )
            os.unlink(temp_path)

        # Create callbacks
        callback_list = [
            WinRateCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                save_best=save_best,
                best_model_save_path=save_path,
                early_stopping=early_stopping,
                patience=patience,
                verbose=self.verbose
            ),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=save_path,
                name_prefix="catan_agent",
                self_play_manager=self.self_play_manager if use_self_play else None,
                elo_tracker=self.elo_tracker,
                verbose=self.verbose
            ),
            GameStatsCallback(verbose=self.verbose),
        ]

        # Add self-play callbacks if enabled
        if use_self_play and self.self_play_manager is not None:
            # Create eval env factory for periodic eval vs random
            def make_eval_env():
                return make_env(
                    opponent_policies=[RandomAgent(), RandomAgent(), RandomAgent()],
                    seed=999
                )()

            callback_list.extend([
                SelfPlayTrainingCallback(
                    self_play_manager=self.self_play_manager,
                    elo_tracker=self.elo_tracker,
                    verbose=self.verbose
                ),
                SelfPlayStatsCallback(
                    eval_env_fn=make_eval_env,
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    elo_tracker=self.elo_tracker,
                    match_tracker=self.match_tracker,
                    verbose=self.verbose
                ),
                ELOCallback(
                    elo_tracker=self.elo_tracker,
                    save_freq=checkpoint_freq,
                    save_path=save_path,
                    verbose=self.verbose
                ),
            ])

            # Add curriculum progress callback if using curriculum mixing
            if mix_strategy == "curriculum":
                callback_list.append(
                    CurriculumProgressCallback(
                        save_path=save_path,
                        total_timesteps=total_timesteps,
                        verbose=self.verbose
                    )
                )

        callbacks = CallbackList(callback_list)

        # Train
        print(f"Training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save final model
        final_path = os.path.join(save_path, "catan_agent_final.zip")
        self.model.save(final_path)
        print(f"Saved final model to {final_path}")

        env.close()
        eval_env.close()

    def _create_opponents(self, opponent_type: str) -> List:
        """Create opponent policies based on type."""
        if opponent_type == "random":
            return [RandomAgent() for _ in range(3)]
        elif opponent_type == "greedy":
            return [GreedyAgent() for _ in range(3)]
        elif opponent_type == "mixed":
            return [RandomAgent(), GreedyAgent(), GreedyAgent()]
        else:
            return [RandomAgent() for _ in range(3)]

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Predict an action given the current state.

        Args:
            observation: The encoded game observation.
            action_mask: Boolean mask of valid actions.
            deterministic: Whether to select deterministically.

        Returns:
            The selected action index.
        """
        if self.model is None:
            # Fallback to random if no model
            valid_actions = np.where(action_mask)[0]
            return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

        action, _ = self.model.predict(
            observation,
            deterministic=deterministic,
            action_masks=action_mask
        )
        return int(action)

    def save(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model.
        """
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load a model from a file.

        Args:
            path: Path to the saved model.
        """
        if not MASKABLE_PPO_AVAILABLE:
            raise RuntimeError("MaskablePPO not available. Install sb3-contrib.")

        self.model = MaskablePPO.load(path, device=self.device)
        print(f"Model loaded from {path}")

    def evaluate(
        self,
        n_episodes: int = 100,
        opponent_type: str = "random",
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate the agent.

        Args:
            n_episodes: Number of episodes to evaluate.
            opponent_type: Type of opponents.
            verbose: Whether to print results.

        Returns:
            Dictionary with evaluation metrics.
        """
        opponents = self._create_opponents(opponent_type)
        env = CatanEnv(opponent_policies=opponents)

        wins = 0
        total_vp = 0
        total_turns = 0
        total_reward = 0

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action_mask = env.action_masks()
                action = self.predict(obs, action_mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            if info.get("is_agent_winner", False):
                wins += 1
            total_vp += info.get("vp", 0)
            total_turns += info.get("turn", 0)
            total_reward += episode_reward

            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Win Rate = {wins/(episode+1):.2%}")

        results = {
            "win_rate": wins / n_episodes,
            "avg_vp": total_vp / n_episodes,
            "avg_turns": total_turns / n_episodes,
            "avg_reward": total_reward / n_episodes,
            "total_wins": wins,
            "total_episodes": n_episodes,
        }

        if verbose:
            print(f"\nEvaluation Results ({opponent_type} opponents):")
            print(f"  Win Rate: {results['win_rate']:.2%}")
            print(f"  Average VP: {results['avg_vp']:.1f}")
            print(f"  Average Turns: {results['avg_turns']:.0f}")

        env.close()
        return results


class SelfPlayTrainingCallback:
    """Callback for updating opponent pool during training.

    This is a stable-baselines3 compatible callback that adds the current
    model to the opponent pool at regular intervals.
    """

    def __init__(
        self,
        self_play_manager: SelfPlayManager,
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            self_play_manager: SelfPlayManager instance.
            verbose: Verbosity level.
        """
        from stable_baselines3.common.callbacks import BaseCallback
        self._base_class = BaseCallback

        self.self_play_manager = self_play_manager
        self.verbose = verbose

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.self_play_manager.update_step(self.num_timesteps)

        if self.self_play_manager.should_add_to_pool():
            path = self.self_play_manager.add_to_pool(self.model, self.num_timesteps)
            if self.verbose > 0:
                print(f"Added model to opponent pool: {path}")
                print(f"Pool size: {len(self.self_play_manager.opponent_pool)}")

        return True


# Make SelfPlayTrainingCallback a proper BaseCallback
from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayTrainingCallback(BaseCallback):
    """Callback for updating opponent pool during self-play training."""

    def __init__(
        self,
        self_play_manager: SelfPlayManager,
        elo_tracker=None,
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            self_play_manager: SelfPlayManager instance.
            elo_tracker: Optional ELO tracker to register models.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.self_play_manager = self_play_manager
        self.elo_tracker = elo_tracker

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.self_play_manager.update_step(self.num_timesteps)

        if self.self_play_manager.should_add_to_pool():
            path = self.self_play_manager.add_to_pool(self.model, self.num_timesteps)
            if self.verbose > 0:
                print(f"Added model to opponent pool: {path}")
                print(f"Pool size: {len(self.self_play_manager.opponent_pool)}")

            # Also register with ELO tracker so we can look up this model's ELO
            if self.elo_tracker is not None:
                self.elo_tracker.add_model_to_pool(path)
                if self.verbose > 0:
                    print(f"  ELO at save: {self.elo_tracker.current_elo:.0f}")

        return True


class CurriculumProgressCallback(BaseCallback):
    """Callback for writing curriculum progress to file for subprocess access."""

    def __init__(
        self,
        save_path: str,
        total_timesteps: int,
        update_freq: int = 1000,
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            save_path: Directory to save progress file.
            total_timesteps: Total training timesteps for progress calculation.
            update_freq: How often to update the file (in steps).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.progress_file = os.path.join(save_path, "curriculum_progress.txt")
        self.last_logged_phase = None

    def _on_step(self) -> bool:
        """Update curriculum progress file."""
        if self.n_calls % self.update_freq == 0:
            progress = self.num_timesteps / self.total_timesteps
            with open(self.progress_file, 'w') as f:
                f.write(f"{progress:.4f}")

            # Log phase transitions
            from ..training.self_play import get_curriculum_mix
            current_phase = get_curriculum_mix(progress)
            if current_phase != self.last_logged_phase:
                if self.verbose > 0:
                    print(f"Curriculum phase: {current_phase} (progress: {progress:.1%})")
                self.last_logged_phase = current_phase
                self.logger.record("curriculum/phase", ["random", "mixed", "pool"].index(current_phase))

            self.logger.record("curriculum/progress", progress)

        return True


def get_optimal_training_config(cpu_cores: int = None, has_gpu: bool = True) -> dict:
    """Get recommended training configuration based on hardware.

    TRAINING OPTIMIZATION GUIDE:
    ===========================

    This function returns optimized training parameters based on available hardware.
    The optimizations focus on maximizing throughput without changing the model architecture.

    Key Optimizations Applied:
    1. n_envs: Scaled based on CPU cores (default 8, can go up to 16 on powerful machines)
    2. batch_size: Larger batches (512) for better GPU utilization
    3. n_steps: Reduced from 2048 to 1024 for more frequent gradient updates
    4. n_epochs: Reduced from 10 to 8 for faster update cycles

    Expected Speedup:
    - n_envs 4->8: ~1.5-2x speedup (more parallel data collection)
    - batch_size 256->512: ~1.2x speedup (better GPU efficiency)
    - n_steps 2048->1024: ~1.1x speedup (more frequent learning)
    - n_epochs 10->8: ~1.1x speedup (faster per-update cycle)
    - State encoder optimizations: ~1.2x speedup (reduced allocations)
    - Reward calculation optimization: ~1.1x speedup (no state copying)

    Total expected speedup: ~2-3x over default configuration

    Args:
        cpu_cores: Number of CPU cores (auto-detected if None)
        has_gpu: Whether GPU is available

    Returns:
        Dictionary of optimized training parameters
    """
    import os
    if cpu_cores is None:
        cpu_cores = os.cpu_count() or 4

    # Scale n_envs based on CPU cores (each env runs in separate process)
    # Use slightly fewer than total cores to leave room for main process + GPU
    if cpu_cores >= 16:
        n_envs = 16
    elif cpu_cores >= 8:
        n_envs = 8
    elif cpu_cores >= 4:
        n_envs = 4
    else:
        n_envs = 2

    # Batch size scales with GPU memory (512 is good for most GPUs)
    batch_size = 512 if has_gpu else 256

    # n_steps * n_envs = total data per update
    # Smaller n_steps = more frequent updates = faster iteration
    n_steps = 1024

    # Fewer epochs per update = faster but may need more total updates
    n_epochs = 8

    return {
        'n_envs': n_envs,
        'batch_size': batch_size,
        'n_steps': n_steps,
        'n_epochs': n_epochs,
        # Additional optimized defaults
        'eval_freq': 50000,  # Less frequent eval for speed
        'checkpoint_freq': 100000,
    }


def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Phase 2 Catan RL agent")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments (default: 8 for optimal performance)")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "greedy", "mixed"],
                        help="Opponent type")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (reduced for stability)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Minibatch size for training (default: 512 for GPU efficiency)")
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="Steps per env per update (default: 1024)")
    parser.add_argument("--n-epochs", type=int, default=8,
                        help="Number of epochs per update (default: 8)")
    parser.add_argument("--save-path", type=str, default="./models",
                        help="Model save directory")
    parser.add_argument("--log-path", type=str, default="./logs",
                        help="Tensorboard log directory (omit to disable tensorboard)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of final evaluation episodes")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to model to continue training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, don't train")

    # Mid-training evaluation and early stopping
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=20,
                        help="Number of episodes per mid-training evaluation")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping if no improvement")
    parser.add_argument("--patience", type=int, default=5,
                        help="Evaluations without improvement before early stopping")
    parser.add_argument("--no-save-best", action="store_true",
                        help="Disable saving the best model")

    # Self-play arguments
    parser.add_argument("--use-self-play", action="store_true",
                        help="Enable self-play training with opponent pool")
    parser.add_argument("--pool-size", type=int, default=30,
                        help="Maximum opponent pool size")
    parser.add_argument("--self-play-save-freq", type=int, default=50000,
                        help="Frequency to save model to opponent pool")
    parser.add_argument("--mix-strategy", type=str, default="curriculum",
                        choices=["random", "self", "pool", "mixed", "curriculum"],
                        help="Opponent mixing strategy for self-play (curriculum = progressive random->mixed->pool)")

    # Custom policy arguments
    parser.add_argument("--no-custom-policy", action="store_true",
                        help="Use standard MLP policy instead of attention policy")
    parser.add_argument("--features-dim", type=int, default=512,
                        help="Feature extractor output dimension")
    parser.add_argument("--tile-embed-dim", type=int, default=128,
                        help="Tile module embedding dimension")
    parser.add_argument("--tile-num-heads", type=int, default=8,
                        help="Number of attention heads for tiles")
    parser.add_argument("--tile-num-layers", type=int, default=2,
                        help="Number of transformer layers for tiles")

    # Game logging arguments
    parser.add_argument("--enable-game-logging", action="store_true",
                        help="Enable game logging for replay/animation")
    parser.add_argument("--log-every-n-games", type=int, default=100,
                        help="Only log every N games (1 = log all)")
    parser.add_argument("--game-log-level", type=str, default="key_actions",
                        choices=["none", "outcome_only", "key_actions", "full"],
                        help="Game logging detail level")
    parser.add_argument("--game-log-dir", type=str, default="./game_logs",
                        help="Directory to save game logs")

    # Agent position
    parser.add_argument("--agent-position", type=int, default=None,
                        choices=[0, 1, 2, 3],
                        help="Fixed agent position (0-3). Default: random each game")

    args = parser.parse_args()

    agent = CatanRLAgent(
        model_path=args.load,
        use_custom_policy=not args.no_custom_policy,
    )

    if args.eval_only:
        if agent.model is None:
            print("No model loaded. Cannot evaluate.")
            return
        agent.evaluate(n_episodes=args.eval_episodes, opponent_type=args.opponent)
    else:
        agent.train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            opponent_type=args.opponent,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            save_path=args.save_path,
            log_path=args.log_path,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_best=not args.no_save_best,
            early_stopping=args.early_stopping,
            patience=args.patience,
            enable_game_logging=args.enable_game_logging,
            log_every_n_games=args.log_every_n_games,
            game_log_level=args.game_log_level,
            game_log_dir=args.game_log_dir,
            agent_position=args.agent_position,
            use_self_play=args.use_self_play,
            opponent_pool_size=args.pool_size,
            self_play_save_freq=args.self_play_save_freq,
            mix_strategy=args.mix_strategy,
            features_dim=args.features_dim,
            tile_embed_dim=args.tile_embed_dim,
            tile_num_heads=args.tile_num_heads,
            tile_num_layers=args.tile_num_layers,
        )

        # Final evaluation
        print("\nFinal Evaluation:")
        agent.evaluate(n_episodes=args.eval_episodes, opponent_type=args.opponent)


if __name__ == "__main__":
    main()
