"""Self-play training manager for Catan RL agent."""

import os
from typing import List, Optional, Dict, Any
import numpy as np

from ..environment.catan_env import CatanMultiAgentEnv
from ..models.baseline import BaseAgent, RandomAgent


def get_curriculum_mix(progress: float) -> str:
    """Get curriculum mixing strategy based on training progress.

    Progressive curriculum for self-play training:
    - Early training (0-30%): Random opponents to learn basics
    - Mid training (30-60%): Mixed opponents for transition
    - Late training (60-100%): Strong pool opponents only

    Args:
        progress: Training progress as fraction (timesteps / total_timesteps).

    Returns:
        Mix strategy string: "random", "mixed", or "pool".
    """
    if progress < 0.3:
        return "random"   # Learn basics against weak opponents
    elif progress < 0.6:
        return "mixed"    # Transition with mixed opponents
    else:
        return "pool"     # Final phase: strong pool opponents only


class SelfPlayAgent(BaseAgent):
    """Wrapper to use trained RL model as an opponent."""

    def __init__(self, model, model_path: Optional[str] = None):
        """Initialize with a trained model.

        Args:
            model: A trained MaskablePPO model.
            model_path: Path to the model file (for tracking/logging).
        """
        self.model = model
        self.model_path = model_path

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Predict an action using the trained model."""
        action, _ = self.model.predict(
            observation,
            deterministic=deterministic,
            action_masks=action_mask
        )
        return int(action)


class SelfPlayManager:
    """Manages self-play training with opponent pool."""

    def __init__(
        self,
        save_dir: str = "./models/self_play",
        max_pool_size: int = 10,
        save_freq: int = 50000,
    ):
        """Initialize the self-play manager.

        Args:
            save_dir: Directory to save opponent checkpoints.
            max_pool_size: Maximum number of opponents to keep.
            save_freq: Save opponent to pool every N steps.
        """
        self.save_dir = save_dir
        self.max_pool_size = max_pool_size
        self.save_freq = save_freq
        self.opponent_pool: List[str] = []  # Paths to saved models
        self.current_step = 0

        os.makedirs(save_dir, exist_ok=True)

    def should_add_to_pool(self) -> bool:
        """Check if it's time to add current model to pool."""
        return self.current_step > 0 and self.current_step % self.save_freq == 0

    def add_to_pool(self, model, step: int) -> str:
        """Add current model to opponent pool.

        Args:
            model: The current model to save.
            step: Current training step.

        Returns:
            Path to saved model.
        """
        path = os.path.join(self.save_dir, f"opponent_{step}.zip")
        model.save(path)
        self.opponent_pool.append(path)

        # Remove oldest if pool is full
        if len(self.opponent_pool) > self.max_pool_size:
            old_path = self.opponent_pool.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        return path

    def sample_opponents(self, n: int = 3, include_random: bool = True) -> List:
        """Sample opponents from the pool.

        Args:
            n: Number of opponents to sample.
            include_random: Whether to always include a random agent.

        Returns:
            List of opponent agents.
        """
        from sb3_contrib import MaskablePPO

        opponents = []

        if include_random:
            opponents.append(RandomAgent())
            n -= 1

        if len(self.opponent_pool) == 0:
            # No trained opponents yet, use random
            opponents.extend([RandomAgent() for _ in range(n)])
        else:
            # Sample from pool
            for _ in range(n):
                path = np.random.choice(self.opponent_pool)
                try:
                    model = MaskablePPO.load(path)
                    opponents.append(SelfPlayAgent(model, model_path=path))
                except Exception:
                    opponents.append(RandomAgent())

        return opponents

    def get_opponents_for_training(
        self,
        current_model=None,
        mix_strategy: str = "mixed"
    ) -> List:
        """Get opponents for a training episode.

        Args:
            current_model: The current model being trained.
            mix_strategy: How to mix opponents:
                - "random": All random agents
                - "self": All use current model
                - "pool": Sample from historical pool
                - "mixed": Mix of all types

        Returns:
            List of 3 opponent agents.
        """
        if mix_strategy == "random":
            return [RandomAgent() for _ in range(3)]

        elif mix_strategy == "self" and current_model is not None:
            return [SelfPlayAgent(current_model) for _ in range(3)]

        elif mix_strategy == "pool":
            return self.sample_opponents(n=3, include_random=False)

        else:  # mixed
            opponents = []

            # One random
            opponents.append(RandomAgent())

            # One from pool (or random if pool empty)
            if len(self.opponent_pool) > 0 and current_model is not None:
                opponents.append(SelfPlayAgent(current_model))
            else:
                opponents.append(RandomAgent())

            # One from pool
            if len(self.opponent_pool) > 0:
                from sb3_contrib import MaskablePPO
                path = np.random.choice(self.opponent_pool)
                try:
                    model = MaskablePPO.load(path)
                    opponents.append(SelfPlayAgent(model, model_path=path))
                except Exception:
                    opponents.append(RandomAgent())
            else:
                opponents.append(RandomAgent())

            return opponents

    def update_step(self, step: int) -> None:
        """Update current training step."""
        self.current_step = step


class SelfPlayCallback:
    """Callback for self-play training updates."""

    def __init__(self, manager: SelfPlayManager, verbose: int = 1):
        """Initialize the callback.

        Args:
            manager: SelfPlayManager instance.
            verbose: Verbosity level.
        """
        self.manager = manager
        self.verbose = verbose

    def on_step(self, step: int, model) -> None:
        """Called at each training step.

        Args:
            step: Current training step.
            model: Current model.
        """
        self.manager.update_step(step)

        if self.manager.should_add_to_pool():
            path = self.manager.add_to_pool(model, step)
            if self.verbose > 0:
                print(f"Added model to opponent pool: {path}")
                print(f"Pool size: {len(self.manager.opponent_pool)}")
