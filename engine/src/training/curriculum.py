"""Curriculum learning manager for progressive training difficulty.

Implements a multi-stage curriculum where the agent starts against easy
opponents and gradually faces harder ones as it improves.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable, Any, Tuple
import numpy as np
import json
import os


class OpponentMix(Enum):
    """Types of opponent mixtures for training."""
    ALL_RANDOM = auto()      # All random opponents
    MOSTLY_RANDOM = auto()   # 2 random, 1 greedy
    MIXED = auto()           # 1 random, 1 greedy, 1 from pool
    MOSTLY_POOL = auto()     # 1 random, 2 from pool
    ALL_POOL = auto()        # All from opponent pool
    SELF_PLAY = auto()       # All copies of current model
    # PHASE 3: New mix for Stage 2 (1 greedy + 1 random + 1 random choice)
    MIXED_RANDOM_GREEDY = auto()  # 1 greedy, 1 random, 1 (50% random / 50% greedy)


@dataclass
class CurriculumStage:
    """Definition of a single curriculum stage."""
    name: str
    opponent_mix: OpponentMix
    # Advancement conditions (first met triggers advancement)
    min_timesteps: int = 0           # Minimum timesteps before advancing
    target_win_rate: float = 0.0     # Win rate threshold (0 = disabled)
    max_timesteps: int = 0           # Force advancement after this (0 = disabled)
    # Optional stage-specific parameters
    entropy_coef: Optional[float] = None
    learning_rate_multiplier: float = 1.0
    # Evaluation parameters for this stage
    eval_episodes: int = 20
    eval_frequency: int = 10000
    # Action space control
    enable_p2p_trading: bool = False  # Whether to enable player-to-player trading

    def __post_init__(self):
        """Validate stage configuration."""
        if self.min_timesteps < 0:
            raise ValueError("min_timesteps must be non-negative")
        if self.target_win_rate < 0 or self.target_win_rate > 1:
            raise ValueError("target_win_rate must be between 0 and 1")


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum learning manager."""
    stages: List[CurriculumStage] = field(default_factory=list)
    # Global settings
    eval_window_size: int = 10       # Number of evaluations for smoothing
    advancement_patience: int = 3    # Required consecutive evals above threshold
    save_on_stage_change: bool = True
    log_stage_transitions: bool = True

    @classmethod
    def default_curriculum(cls) -> 'CurriculumConfig':
        """Create a sensible default curriculum for Catan training."""
        return cls(
            stages=[
                # Stage 1: Learn basics against random (0-30% of training)
                CurriculumStage(
                    name="basics",
                    opponent_mix=OpponentMix.ALL_RANDOM,
                    min_timesteps=100_000,
                    target_win_rate=0.40,  # Should beat random 40%+ to advance
                    max_timesteps=500_000,
                    entropy_coef=0.05,     # High exploration
                    eval_episodes=20,
                ),
                # Stage 2: Transition with greedy opponents (30-50%)
                CurriculumStage(
                    name="transition",
                    opponent_mix=OpponentMix.MOSTLY_RANDOM,
                    min_timesteps=100_000,
                    target_win_rate=0.50,
                    max_timesteps=500_000,
                    entropy_coef=0.02,
                    eval_episodes=25,
                ),
                # Stage 3: Mixed opponents including pool (50-70%)
                CurriculumStage(
                    name="intermediate",
                    opponent_mix=OpponentMix.MIXED,
                    min_timesteps=100_000,
                    target_win_rate=0.60,
                    max_timesteps=500_000,
                    entropy_coef=0.01,
                    eval_episodes=30,
                ),
                # Stage 4: Mostly pool opponents (70-85%)
                CurriculumStage(
                    name="advanced",
                    opponent_mix=OpponentMix.MOSTLY_POOL,
                    min_timesteps=100_000,
                    target_win_rate=0.70,
                    max_timesteps=500_000,
                    entropy_coef=0.01,
                    eval_episodes=30,
                ),
                # Stage 5: Pure self-play/pool (85-100%)
                CurriculumStage(
                    name="mastery",
                    opponent_mix=OpponentMix.ALL_POOL,
                    min_timesteps=100_000,
                    target_win_rate=0.95,  # Target: 95% vs random
                    max_timesteps=0,       # No max - continue until done
                    entropy_coef=0.005,
                    eval_episodes=50,
                ),
            ]
        )

    @classmethod
    def fast_curriculum(cls) -> 'CurriculumConfig':
        """Create a faster curriculum for testing or quick training runs."""
        return cls(
            stages=[
                CurriculumStage(
                    name="warmup",
                    opponent_mix=OpponentMix.ALL_RANDOM,
                    min_timesteps=50_000,
                    target_win_rate=0.35,
                    max_timesteps=200_000,
                ),
                CurriculumStage(
                    name="training",
                    opponent_mix=OpponentMix.MIXED,
                    min_timesteps=100_000,
                    target_win_rate=0.80,
                    max_timesteps=0,
                ),
            ]
        )

    @classmethod
    def trading_curriculum(cls) -> 'CurriculumConfig':
        """Curriculum for learning P2P trading: random → transition → mixed."""
        return cls(
            stages=[
                # Stage 1: Learn trading basics against random (250K)
                CurriculumStage(
                    name="random",
                    opponent_mix=OpponentMix.ALL_RANDOM,
                    min_timesteps=50_000,
                    target_win_rate=0.60,  # Advance early if doing well
                    max_timesteps=250_000,  # Force advancement at 250K
                    entropy_coef=0.03,
                    eval_episodes=25,
                ),
                # Stage 2: Transition with greedy opponents (250K)
                CurriculumStage(
                    name="transition",
                    opponent_mix=OpponentMix.MOSTLY_RANDOM,  # 2 random, 1 greedy
                    min_timesteps=50_000,
                    target_win_rate=0.55,
                    max_timesteps=250_000,
                    entropy_coef=0.02,
                    eval_episodes=25,
                ),
                # Stage 3: Mixed opponents (250K)
                CurriculumStage(
                    name="mixed",
                    opponent_mix=OpponentMix.MIXED,  # 1 random, 1 greedy, 1 pool
                    min_timesteps=50_000,
                    target_win_rate=0.90,
                    max_timesteps=250_000,
                    entropy_coef=0.01,
                    eval_episodes=30,
                ),
            ]
        )

    @classmethod
    def phase3_curriculum(cls) -> 'CurriculumConfig':
        """PHASE 3: 5-stage curriculum with self-play before trading.

        Stage 0: Random opponents, NO TRADING → target 33% win rate
        Stage 1: Pool/self-play, NO TRADING → target 50% (forces better fundamentals)
        Stage 2: Random opponents, WITH TRADING → target 60% win rate
        Stage 3: Pool/self-play, WITH TRADING → continuous improvement

        This curriculum:
        - Disables P2P trading initially to force learning core mechanics
        - Introduces self-play BEFORE trading to build competitive fundamentals
        - Trading only added after agent can compete against copies of itself
        """
        return cls(
            stages=[
                # Stage 0: Random opponents, NO TRADING
                # Learn basic game mechanics first (build settlements, roads, cities)
                CurriculumStage(
                    name="basics_no_trading",
                    opponent_mix=OpponentMix.ALL_RANDOM,
                    min_timesteps=50_000,
                    target_win_rate=0.33,  # 33% win rate to advance
                    max_timesteps=500_000,  # Force advancement after 500K
                    entropy_coef=0.03,
                    eval_episodes=30,
                    eval_frequency=25000,
                    enable_p2p_trading=False,  # NO TRADING
                ),
                # Stage 1: Pool/self-play, NO TRADING
                # Compete against copies of self to develop better fundamentals
                # Forces expansion and efficient building vs intelligent opponents
                CurriculumStage(
                    name="pool_no_trading",
                    opponent_mix=OpponentMix.ALL_POOL,
                    min_timesteps=100_000,
                    target_win_rate=0.50,  # 50% vs pool (better than average self)
                    max_timesteps=1_000_000,  # Force advancement after 1M
                    entropy_coef=0.02,
                    eval_episodes=30,
                    eval_frequency=25000,
                    enable_p2p_trading=False,  # STILL NO TRADING
                ),
                # Stage 2: Random opponents, WITH TRADING
                # Now learn to use trading effectively against weaker opponents
                # Higher entropy (0.05) to explore trading actions and avoid action loops
                CurriculumStage(
                    name="random_with_trading",
                    opponent_mix=OpponentMix.ALL_RANDOM,
                    min_timesteps=100_000,
                    target_win_rate=0.60,  # 60% win rate with trading
                    max_timesteps=1_000_000,  # Force advancement after 1M
                    entropy_coef=0.05,  # Higher entropy for trading exploration
                    eval_episodes=30,
                    eval_frequency=25000,
                    enable_p2p_trading=True,  # TRADING ENABLED
                ),
                # Stage 3: Pool/self-play, WITH TRADING
                # Continuous improvement through self-play with full game mechanics
                # Moderate entropy (0.03) to continue exploring trading strategies
                CurriculumStage(
                    name="pool_with_trading",
                    opponent_mix=OpponentMix.ALL_POOL,
                    min_timesteps=100_000,
                    target_win_rate=0.0,  # No target - continuous play
                    max_timesteps=0,  # No max - continue until external stop
                    entropy_coef=0.03,  # Moderate entropy for trading
                    eval_episodes=50,
                    eval_frequency=50000,
                    enable_p2p_trading=True,
                ),
            ]
        )


class CurriculumManager:
    """Manages curriculum learning with progressive difficulty.

    Tracks training progress and automatically advances through stages
    based on performance metrics.
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        save_path: Optional[str] = None,
    ):
        """Initialize the curriculum manager.

        Args:
            config: Curriculum configuration. Uses default if None.
            save_path: Path to save/load curriculum state.
        """
        self.config = config or CurriculumConfig.default_curriculum()
        self.save_path = save_path

        # Current state
        self.current_stage_idx: int = 0
        self.stage_start_timestep: int = 0
        self.total_timesteps: int = 0

        # Performance tracking per stage
        self.stage_win_rates: List[float] = []
        self.consecutive_above_threshold: int = 0

        # History for logging
        self.stage_history: List[Dict[str, Any]] = []

        # Load saved state if exists
        if save_path and os.path.exists(save_path):
            self.load(save_path)

    @property
    def current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.config.stages[self.current_stage_idx]

    @property
    def is_final_stage(self) -> bool:
        """Check if we're in the final stage."""
        return self.current_stage_idx >= len(self.config.stages) - 1

    @property
    def stage_timesteps(self) -> int:
        """Get timesteps elapsed in current stage."""
        return self.total_timesteps - self.stage_start_timestep

    def get_opponent_mix(self) -> OpponentMix:
        """Get the opponent mix for the current stage."""
        return self.current_stage.opponent_mix

    def get_opponents(
        self,
        random_agent_cls,
        greedy_agent_cls,
        pool_sample_fn: Optional[Callable[[], List]] = None,
        current_model=None,
    ) -> Tuple[List, List[Dict]]:
        """Get opponents for the current curriculum stage.

        Args:
            random_agent_cls: Class to create random agents.
            greedy_agent_cls: Class to create greedy agents.
            pool_sample_fn: Function to sample from opponent pool.
            current_model: Current model for self-play.

        Returns:
            Tuple of (opponents list, opponent info dicts).
        """
        mix = self.get_opponent_mix()
        opponents = []
        infos = []

        if mix == OpponentMix.ALL_RANDOM:
            for _ in range(3):
                opponents.append(random_agent_cls())
                infos.append({"agent_type": "random", "elo": 1000})

        elif mix == OpponentMix.MOSTLY_RANDOM:
            # 2 random, 1 greedy
            for _ in range(2):
                opponents.append(random_agent_cls())
                infos.append({"agent_type": "random", "elo": 1000})
            opponents.append(greedy_agent_cls())
            infos.append({"agent_type": "greedy", "elo": 1200})

        elif mix == OpponentMix.MIXED:
            # 1 random, 1 greedy, 1 pool (or greedy if pool empty)
            opponents.append(random_agent_cls())
            infos.append({"agent_type": "random", "elo": 1000})
            opponents.append(greedy_agent_cls())
            infos.append({"agent_type": "greedy", "elo": 1200})

            if pool_sample_fn:
                pool_opponents = pool_sample_fn()
                if pool_opponents:
                    opponents.append(pool_opponents[0])
                    infos.append({"agent_type": "pool", "elo": 1400})
                else:
                    opponents.append(greedy_agent_cls())
                    infos.append({"agent_type": "greedy", "elo": 1200})
            else:
                opponents.append(greedy_agent_cls())
                infos.append({"agent_type": "greedy", "elo": 1200})

        elif mix == OpponentMix.MOSTLY_POOL:
            # 1 random, 2 pool
            opponents.append(random_agent_cls())
            infos.append({"agent_type": "random", "elo": 1000})

            pool_opponents = pool_sample_fn() if pool_sample_fn else []
            for i in range(2):
                if pool_opponents and i < len(pool_opponents):
                    opponents.append(pool_opponents[i])
                    infos.append({"agent_type": "pool", "elo": 1400})
                else:
                    opponents.append(greedy_agent_cls())
                    infos.append({"agent_type": "greedy", "elo": 1200})

        elif mix == OpponentMix.ALL_POOL:
            # All from pool
            pool_opponents = pool_sample_fn() if pool_sample_fn else []
            for i in range(3):
                if pool_opponents and i < len(pool_opponents):
                    opponents.append(pool_opponents[i])
                    infos.append({"agent_type": "pool", "elo": 1400})
                else:
                    opponents.append(greedy_agent_cls())
                    infos.append({"agent_type": "greedy", "elo": 1200})

        elif mix == OpponentMix.SELF_PLAY:
            # All copies of current model
            if current_model:
                from .self_play import SelfPlayAgent
                for _ in range(3):
                    opponents.append(SelfPlayAgent(current_model))
                    infos.append({"agent_type": "self", "elo": 1500})
            else:
                for _ in range(3):
                    opponents.append(random_agent_cls())
                    infos.append({"agent_type": "random", "elo": 1000})

        elif mix == OpponentMix.MIXED_RANDOM_GREEDY:
            # PHASE 3: 1 greedy, 1 random, 1 (50% random / 50% greedy)
            opponents.append(greedy_agent_cls())
            infos.append({"agent_type": "greedy", "elo": 1200})

            opponents.append(random_agent_cls())
            infos.append({"agent_type": "random", "elo": 1000})

            # 50/50 random choice for third opponent
            if np.random.random() < 0.5:
                opponents.append(random_agent_cls())
                infos.append({"agent_type": "random", "elo": 1000})
            else:
                opponents.append(greedy_agent_cls())
                infos.append({"agent_type": "greedy", "elo": 1200})

        return opponents, infos

    def update(
        self,
        timesteps: int,
        win_rate: Optional[float] = None,
    ) -> bool:
        """Update curriculum state and check for stage advancement.

        Args:
            timesteps: Current total timesteps.
            win_rate: Current evaluation win rate (if available).

        Returns:
            True if stage advanced, False otherwise.
        """
        self.total_timesteps = timesteps

        # Record win rate if provided
        if win_rate is not None:
            self.stage_win_rates.append(win_rate)
            # Keep only recent win rates
            if len(self.stage_win_rates) > self.config.eval_window_size:
                self.stage_win_rates = self.stage_win_rates[-self.config.eval_window_size:]

        # Check for stage advancement
        if self._should_advance():
            return self._advance_stage()

        return False

    def _should_advance(self) -> bool:
        """Check if conditions are met to advance to next stage."""
        if self.is_final_stage:
            return False

        stage = self.current_stage
        elapsed = self.stage_timesteps

        # Check minimum timesteps
        if elapsed < stage.min_timesteps:
            return False

        # Check max timesteps (force advancement)
        if stage.max_timesteps > 0 and elapsed >= stage.max_timesteps:
            return True

        # Check win rate threshold
        if stage.target_win_rate > 0 and self.stage_win_rates:
            avg_win_rate = np.mean(self.stage_win_rates[-self.config.eval_window_size:])
            if avg_win_rate >= stage.target_win_rate:
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.config.advancement_patience:
                    return True
            else:
                self.consecutive_above_threshold = 0

        return False

    def _advance_stage(self) -> bool:
        """Advance to the next curriculum stage."""
        if self.is_final_stage:
            return False

        # Record history
        self.stage_history.append({
            "stage_idx": self.current_stage_idx,
            "stage_name": self.current_stage.name,
            "start_timestep": self.stage_start_timestep,
            "end_timestep": self.total_timesteps,
            "final_win_rates": list(self.stage_win_rates),
            "avg_win_rate": np.mean(self.stage_win_rates) if self.stage_win_rates else 0,
        })

        # Advance
        self.current_stage_idx += 1
        self.stage_start_timestep = self.total_timesteps
        self.stage_win_rates = []
        self.consecutive_above_threshold = 0

        if self.config.log_stage_transitions:
            print(f"\n{'='*50}")
            print(f"CURRICULUM: Advancing to stage {self.current_stage_idx}: {self.current_stage.name}")
            print(f"  Opponent mix: {self.current_stage.opponent_mix.name}")
            print(f"  Target win rate: {self.current_stage.target_win_rate:.0%}")
            print(f"{'='*50}\n")

        # Save state
        if self.config.save_on_stage_change and self.save_path:
            self.save()

        return True

    def get_entropy_coef(self, default: float = 0.01) -> float:
        """Get entropy coefficient for current stage."""
        return self.current_stage.entropy_coef or default

    def get_learning_rate_multiplier(self) -> float:
        """Get learning rate multiplier for current stage."""
        return self.current_stage.learning_rate_multiplier

    def get_eval_config(self) -> Dict[str, int]:
        """Get evaluation configuration for current stage."""
        return {
            "n_episodes": self.current_stage.eval_episodes,
            "frequency": self.current_stage.eval_frequency,
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get current curriculum progress."""
        return {
            "current_stage": self.current_stage_idx,
            "stage_name": self.current_stage.name,
            "total_stages": len(self.config.stages),
            "stage_progress": self.stage_timesteps / max(1, self.current_stage.min_timesteps),
            "total_timesteps": self.total_timesteps,
            "recent_win_rates": list(self.stage_win_rates),
            "avg_win_rate": np.mean(self.stage_win_rates) if self.stage_win_rates else 0,
            "opponent_mix": self.current_stage.opponent_mix.name,
            "is_final_stage": self.is_final_stage,
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save curriculum state to file."""
        path = path or self.save_path
        if not path:
            return

        state = {
            "current_stage_idx": self.current_stage_idx,
            "stage_start_timestep": self.stage_start_timestep,
            "total_timesteps": self.total_timesteps,
            "stage_win_rates": self.stage_win_rates,
            "consecutive_above_threshold": self.consecutive_above_threshold,
            "stage_history": self.stage_history,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load curriculum state from file."""
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            state = json.load(f)

        self.current_stage_idx = state.get("current_stage_idx", 0)
        self.stage_start_timestep = state.get("stage_start_timestep", 0)
        self.total_timesteps = state.get("total_timesteps", 0)
        self.stage_win_rates = state.get("stage_win_rates", [])
        self.consecutive_above_threshold = state.get("consecutive_above_threshold", 0)
        self.stage_history = state.get("stage_history", [])

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self.current_stage_idx = 0
        self.stage_start_timestep = 0
        self.total_timesteps = 0
        self.stage_win_rates = []
        self.consecutive_above_threshold = 0
        self.stage_history = []


class CurriculumCallback:
    """Stable-baselines3 callback for curriculum learning."""

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        eval_env_factory: Callable,
        n_eval_episodes: int = 20,
        eval_freq: int = 10000,
        verbose: int = 1,
    ):
        """Initialize the curriculum callback.

        Args:
            curriculum_manager: CurriculumManager instance.
            eval_env_factory: Factory function to create evaluation environments.
            n_eval_episodes: Number of episodes per evaluation.
            eval_freq: Evaluation frequency in timesteps.
            verbose: Verbosity level.
        """
        from stable_baselines3.common.callbacks import BaseCallback
        self._base = BaseCallback

        self.curriculum = curriculum_manager
        self.eval_env_factory = eval_env_factory
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.last_eval_timestep = 0


# Create proper callback class
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Stable-baselines3 callback for curriculum learning."""

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        eval_env_factory: Callable,
        n_eval_episodes: int = 20,
        eval_freq: int = 10000,
        on_stage_advance: Optional[Callable[[int, str], None]] = None,
        verbose: int = 1,
    ):
        """Initialize the curriculum callback.

        Args:
            curriculum_manager: CurriculumManager instance.
            eval_env_factory: Factory function to create evaluation environments.
            n_eval_episodes: Number of episodes per evaluation.
            eval_freq: Evaluation frequency in timesteps.
            on_stage_advance: Callback when stage advances (stage_idx, stage_name).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.curriculum = curriculum_manager
        self.eval_env_factory = eval_env_factory
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.on_stage_advance = on_stage_advance
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Apply curriculum entropy coefficient (overrides decay callback)
        stage_entropy = self.curriculum.current_stage.entropy_coef
        if stage_entropy is not None and hasattr(self.model, 'ent_coef'):
            if abs(self.model.ent_coef - stage_entropy) > 1e-6:
                self.model.ent_coef = stage_entropy

        # Check if it's time to evaluate
        if self.num_timesteps >= self.last_eval_timestep + self.eval_freq:
            self.last_eval_timestep = self.num_timesteps

            # Run evaluation
            win_rate = self._evaluate()

            # Update curriculum
            advanced = self.curriculum.update(self.num_timesteps, win_rate)

            # Log metrics
            progress = self.curriculum.get_progress()
            self.logger.record("curriculum/stage", progress["current_stage"])
            self.logger.record("curriculum/stage_name", progress["stage_name"])
            self.logger.record("curriculum/win_rate", win_rate)
            self.logger.record("curriculum/opponent_mix", progress["opponent_mix"])
            if stage_entropy is not None:
                self.logger.record("curriculum/entropy_coef", stage_entropy)

            if advanced and self.on_stage_advance:
                self.on_stage_advance(
                    progress["current_stage"],
                    progress["stage_name"]
                )

        return True

    def _evaluate(self) -> float:
        """Run evaluation and return win rate."""
        eval_env = self.eval_env_factory()
        wins = 0

        for _ in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False

            while not done:
                action_mask = eval_env.action_masks()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=action_mask
                )
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

            if info.get("is_agent_winner", False):
                wins += 1

        eval_env.close()
        return wins / self.n_eval_episodes

    def _on_training_end(self) -> None:
        """Save curriculum state at training end."""
        self.curriculum.save()

        if self.verbose > 0:
            progress = self.curriculum.get_progress()
            print(f"\nCurriculum Learning Summary:")
            print(f"  Final Stage: {progress['stage_name']} ({progress['current_stage']}/{progress['total_stages']})")
            print(f"  Total Timesteps: {progress['total_timesteps']}")
            if progress['recent_win_rates']:
                print(f"  Final Win Rate: {progress['avg_win_rate']:.2%}")
