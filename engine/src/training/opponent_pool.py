"""Opponent pool management for self-play training."""

import os
from collections import deque
from typing import List, Optional, Tuple, Dict
import numpy as np


class OpponentPool:
    """Manages a pool of historical opponent policies.

    Uses a prioritized sampling strategy where more recent and
    higher-performing opponents are sampled more frequently.
    """

    def __init__(
        self,
        max_size: int = 20,
        save_dir: str = "./models/opponents",
        prioritize_recent: bool = True,
        prioritize_strong: bool = True,
    ):
        """Initialize the opponent pool.

        Args:
            max_size: Maximum number of opponents to keep.
            save_dir: Directory to save opponent models.
            prioritize_recent: Weight recent models higher.
            prioritize_strong: Weight high win-rate models higher.
        """
        self.max_size = max_size
        self.save_dir = save_dir
        self.prioritize_recent = prioritize_recent
        self.prioritize_strong = prioritize_strong

        self.opponents: deque = deque(maxlen=max_size)
        self.metadata: Dict[str, dict] = {}  # path -> metadata

        os.makedirs(save_dir, exist_ok=True)

    def add(
        self,
        model,
        step: int,
        win_rate: float = 0.5,
        name: Optional[str] = None
    ) -> str:
        """Add a model to the pool.

        Args:
            model: The model to add.
            step: Training step when model was saved.
            win_rate: Model's win rate when saved.
            name: Optional name for the model.

        Returns:
            Path to saved model.
        """
        name = name or f"opponent_step{step}"
        path = os.path.join(self.save_dir, f"{name}.zip")

        # Save model
        model.save(path)

        # Add to pool
        if len(self.opponents) >= self.max_size:
            # Remove oldest
            old_path = self.opponents[0]
            if old_path in self.metadata:
                del self.metadata[old_path]
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception:
                    pass

        self.opponents.append(path)
        self.metadata[path] = {
            "step": step,
            "win_rate": win_rate,
            "name": name,
        }

        return path

    def sample(self, n: int = 1) -> List[str]:
        """Sample opponents from the pool.

        Args:
            n: Number of opponents to sample.

        Returns:
            List of paths to sampled models.
        """
        if len(self.opponents) == 0:
            return []

        n = min(n, len(self.opponents))

        # Calculate sampling weights
        weights = self._calculate_weights()

        # Sample without replacement
        indices = np.random.choice(
            len(self.opponents),
            size=n,
            replace=False,
            p=weights
        )

        return [self.opponents[i] for i in indices]

    def _calculate_weights(self) -> np.ndarray:
        """Calculate sampling weights for each opponent."""
        n = len(self.opponents)
        weights = np.ones(n)

        if self.prioritize_recent:
            # Linear recency weighting
            recency = np.linspace(0.5, 1.0, n)
            weights *= recency

        if self.prioritize_strong:
            # Weight by win rate
            for i, path in enumerate(self.opponents):
                if path in self.metadata:
                    win_rate = self.metadata[path].get("win_rate", 0.5)
                    weights[i] *= (0.5 + win_rate)

        # Normalize
        weights /= weights.sum()
        return weights

    def get_stats(self) -> dict:
        """Get statistics about the pool.

        Returns:
            Dictionary with pool statistics.
        """
        if len(self.opponents) == 0:
            return {"size": 0}

        win_rates = [
            self.metadata[p].get("win_rate", 0.5)
            for p in self.opponents if p in self.metadata
        ]

        return {
            "size": len(self.opponents),
            "avg_win_rate": np.mean(win_rates) if win_rates else 0.5,
            "max_win_rate": max(win_rates) if win_rates else 0.5,
            "min_win_rate": min(win_rates) if win_rates else 0.5,
        }

    def load_opponent(self, path: str):
        """Load an opponent model from path.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded model or None if loading fails.
        """
        try:
            from sb3_contrib import MaskablePPO
            return MaskablePPO.load(path)
        except Exception as e:
            print(f"Failed to load opponent from {path}: {e}")
            return None

    def get_diverse_sample(self, n: int = 3) -> List[str]:
        """Get a diverse sample of opponents.

        Tries to sample opponents from different training stages.

        Args:
            n: Number of opponents to sample.

        Returns:
            List of paths to sampled models.
        """
        if len(self.opponents) == 0:
            return []

        n = min(n, len(self.opponents))

        # Divide pool into segments and sample from each
        segment_size = len(self.opponents) // n
        if segment_size == 0:
            segment_size = 1

        samples = []
        for i in range(n):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(self.opponents))
            if start < end:
                idx = np.random.randint(start, end)
                samples.append(self.opponents[idx])

        return samples

    def update_win_rate(self, path: str, new_win_rate: float) -> None:
        """Update the win rate for an opponent.

        Args:
            path: Path to the opponent model.
            new_win_rate: New win rate.
        """
        if path in self.metadata:
            # Exponential moving average
            old_rate = self.metadata[path].get("win_rate", 0.5)
            self.metadata[path]["win_rate"] = 0.9 * old_rate + 0.1 * new_win_rate

    def clear(self) -> None:
        """Clear the opponent pool."""
        for path in self.opponents:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        self.opponents.clear()
        self.metadata.clear()
