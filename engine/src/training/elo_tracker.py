"""ELO rating tracker for self-play training.

Tracks ELO ratings for models in the opponent pool and the current
training model, adapting standard ELO for 4-player games.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ELORecord:
    """Record of a model's ELO rating."""
    model_path: str
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ELOTracker:
    """Tracks ELO ratings for self-play training.

    Adapts standard ELO for 4-player games by:
    1. Assigning placement scores (1st=1.0, 2nd=0.67, 3rd=0.33, 4th=0.0)
    2. Computing expected score against each opponent
    3. Updating ELO based on actual vs expected performance
    """

    # Placement scores for 4-player game
    PLACEMENT_SCORES = {
        1: 1.0,    # 1st place
        2: 0.67,   # 2nd place
        3: 0.33,   # 3rd place
        4: 0.0,    # 4th place
    }

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_elo: float = 1500.0,
        save_path: Optional[str] = None,
    ):
        """Initialize the ELO tracker.

        Args:
            k_factor: ELO K-factor (higher = more volatile ratings).
            initial_elo: Starting ELO for new models.
            save_path: Path to save/load ELO history.
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.save_path = save_path

        # Current training model's ELO
        self.current_elo = initial_elo
        self.current_games = 0
        self.current_wins = 0

        # Historical ELO for pool models: model_path -> ELORecord
        self.pool_ratings: Dict[str, ELORecord] = {}

        # ELO history over time: [(timestep, elo), ...]
        self.elo_history: List[Tuple[int, float]] = []

        # Random baseline ELO (fixed reference point)
        self.random_baseline_elo = 1000.0

        if save_path and os.path.exists(save_path):
            self.load(save_path)

    def expected_score(self, player_elo: float, opponent_elo: float) -> float:
        """Calculate expected score against an opponent.

        Args:
            player_elo: Player's ELO rating.
            opponent_elo: Opponent's ELO rating.

        Returns:
            Expected score (0.0 to 1.0).
        """
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))

    def update_elo(
        self,
        player_elo: float,
        opponent_elos: List[float],
        placement: int,
    ) -> float:
        """Update ELO based on game result.

        Args:
            player_elo: Current player ELO.
            opponent_elos: List of opponent ELOs (3 opponents).
            placement: Player's placement (1-4).

        Returns:
            New ELO rating.
        """
        actual_score = self.PLACEMENT_SCORES.get(placement, 0.0)

        # Calculate expected score as average against all opponents
        expected = 0.0
        for opp_elo in opponent_elos:
            expected += self.expected_score(player_elo, opp_elo)
        expected /= len(opponent_elos)

        # Update ELO
        new_elo = player_elo + self.k_factor * (actual_score - expected)
        return new_elo

    def record_game(
        self,
        placement: int,
        opponent_paths: List[str],
        timestep: int = 0,
    ) -> float:
        """Record a game result and update ELOs.

        Args:
            placement: Current model's placement (1-4).
            opponent_paths: Paths to opponent models (or "random" for random agents).
            timestep: Current training timestep.

        Returns:
            Updated ELO for current model.
        """
        # Get opponent ELOs
        opponent_elos = []
        for path in opponent_paths:
            if path == "random":
                opponent_elos.append(self.random_baseline_elo)
            else:
                # Normalize path and look up ELO
                elo = self.get_pool_elo(path)
                opponent_elos.append(elo)

        # Update current model's ELO
        old_elo = self.current_elo
        self.current_elo = self.update_elo(old_elo, opponent_elos, placement)
        self.current_games += 1
        if placement == 1:
            self.current_wins += 1

        # Record history
        self.elo_history.append((timestep, self.current_elo))

        return self.current_elo

    def record_game_vs_random(
        self,
        placement: int,
        timestep: int = 0,
    ) -> float:
        """Record a game against random opponents.

        Args:
            placement: Current model's placement (1-4).
            timestep: Current training timestep.

        Returns:
            Updated ELO for current model.
        """
        return self.record_game(
            placement=placement,
            opponent_paths=["random", "random", "random"],
            timestep=timestep,
        )

    def add_model_to_pool(self, model_path: str) -> None:
        """Add current model to the pool with its current ELO.

        Args:
            model_path: Path where model was saved.
        """
        # Normalize to absolute path for consistent lookups
        abs_path = os.path.abspath(model_path)
        self.pool_ratings[abs_path] = ELORecord(
            model_path=abs_path,
            elo=self.current_elo,
            games_played=self.current_games,
            wins=self.current_wins,
        )

    def get_pool_elo(self, model_path: str) -> float:
        """Get ELO for a pool model.

        Args:
            model_path: Path to the model.

        Returns:
            Model's ELO rating.
        """
        # Normalize to absolute path for consistent lookups
        abs_path = os.path.abspath(model_path)
        if abs_path in self.pool_ratings:
            return self.pool_ratings[abs_path].elo
        # Also try the original path (for backwards compatibility)
        if model_path in self.pool_ratings:
            return self.pool_ratings[model_path].elo
        return self.initial_elo

    def get_stats(self) -> Dict:
        """Get current ELO statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "current_elo": self.current_elo,
            "current_games": self.current_games,
            "current_win_rate": self.current_wins / max(1, self.current_games),
            "pool_size": len(self.pool_ratings),
            "pool_avg_elo": (
                sum(r.elo for r in self.pool_ratings.values()) / len(self.pool_ratings)
                if self.pool_ratings else self.initial_elo
            ),
            "elo_vs_random_baseline": self.current_elo - self.random_baseline_elo,
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save ELO data to file.

        Args:
            path: Path to save to (uses self.save_path if not provided).
        """
        path = path or self.save_path
        if not path:
            return

        data = {
            "current_elo": self.current_elo,
            "current_games": self.current_games,
            "current_wins": self.current_wins,
            "pool_ratings": {k: asdict(v) for k, v in self.pool_ratings.items()},
            "elo_history": self.elo_history,
            "k_factor": self.k_factor,
            "initial_elo": self.initial_elo,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load ELO data from file.

        Args:
            path: Path to load from.
        """
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.current_elo = data.get("current_elo", self.initial_elo)
        self.current_games = data.get("current_games", 0)
        self.current_wins = data.get("current_wins", 0)
        self.elo_history = data.get("elo_history", [])
        self.k_factor = data.get("k_factor", self.k_factor)
        self.initial_elo = data.get("initial_elo", self.initial_elo)

        self.pool_ratings = {}
        for k, v in data.get("pool_ratings", {}).items():
            self.pool_ratings[k] = ELORecord(**v)


def placement_from_vp(agent_vp: int, opponent_vps: List[int]) -> int:
    """Determine placement from victory points.

    Args:
        agent_vp: Agent's victory points.
        opponent_vps: List of opponent VPs.

    Returns:
        Placement (1-4).
    """
    all_vps = [agent_vp] + opponent_vps
    sorted_vps = sorted(all_vps, reverse=True)

    # Find placement (1-indexed)
    placement = sorted_vps.index(agent_vp) + 1
    return placement
