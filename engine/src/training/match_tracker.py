"""Match tracker for recording game outcomes with full opponent details.

Tracks which specific agents played in each game, their ELO ratings,
and the game outcomes for analysis.
"""

import os
import json
import csv
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class PlayerInfo:
    """Information about a player in a match."""
    player_idx: int  # Position in game (0-3)
    agent_type: str  # "training", "random", "greedy", "pool"
    model_path: Optional[str] = None  # Path if pool model
    elo: float = 1500.0
    win_rate_vs_random: Optional[float] = None  # If known


@dataclass
class MatchRecord:
    """Record of a single game."""
    match_id: int
    timestep: int
    timestamp: str

    # Players (index 0 is always the training agent)
    players: List[Dict]  # List of PlayerInfo as dicts

    # Outcomes
    winner_idx: int  # Which player won (0-3)
    placements: List[int]  # Final placement for each player
    victory_points: List[int]  # Final VP for each player
    turns: int  # Total game turns

    # Training agent specifics
    agent_placement: int
    agent_vp: int
    agent_reward: float = 0.0


class MatchTracker:
    """Tracks all matches during training with full opponent details.

    Saves match data to CSV for easy analysis and JSON for full details.
    """

    def __init__(
        self,
        save_dir: str = "./models",
        elo_tracker=None,
        save_every_n_matches: int = 100,
    ):
        """Initialize the match tracker.

        Args:
            save_dir: Directory to save match data.
            elo_tracker: Optional ELOTracker for getting ratings.
            save_every_n_matches: Save to disk every N matches.
        """
        self.save_dir = save_dir
        self.elo_tracker = elo_tracker
        self.save_every_n_matches = save_every_n_matches

        self.matches: List[MatchRecord] = []
        self.match_count = 0

        # Current game tracking
        self.current_opponents: List[PlayerInfo] = []

        os.makedirs(save_dir, exist_ok=True)

    def set_opponents_for_game(
        self,
        opponent_infos: List[Dict[str, Any]],
    ) -> None:
        """Set the opponents for the current game.

        Called at the start of each episode to record who's playing.

        Args:
            opponent_infos: List of dicts with opponent details:
                - agent_type: "random", "greedy", or "pool"
                - model_path: Path if pool model (optional)
        """
        self.current_opponents = []

        # Training agent is always player 0
        training_elo = self.elo_tracker.current_elo if self.elo_tracker else 1500.0
        self.current_opponents.append(PlayerInfo(
            player_idx=0,
            agent_type="training",
            model_path=None,
            elo=training_elo,
        ))

        # Add opponents (players 1-3)
        for i, info in enumerate(opponent_infos):
            agent_type = info.get("agent_type", "random")
            model_path = info.get("model_path")

            # Get ELO for this opponent
            if agent_type == "random":
                elo = 1000.0  # Random baseline
                win_rate = None
            elif agent_type == "greedy":
                elo = 1200.0  # Greedy baseline (estimated)
                win_rate = None
            elif model_path and self.elo_tracker:
                elo = self.elo_tracker.get_pool_elo(model_path)
                # Could track win rate from pool records
                pool_record = self.elo_tracker.pool_ratings.get(model_path)
                if pool_record:
                    win_rate = pool_record.wins / max(1, pool_record.games_played)
                else:
                    win_rate = None
            else:
                elo = 1500.0
                win_rate = None

            self.current_opponents.append(PlayerInfo(
                player_idx=i + 1,
                agent_type=agent_type,
                model_path=model_path,
                elo=elo,
                win_rate_vs_random=win_rate,
            ))

    def record_match(
        self,
        timestep: int,
        winner_idx: int,
        victory_points: List[int],
        turns: int,
        agent_reward: float = 0.0,
    ) -> MatchRecord:
        """Record a completed match.

        Args:
            timestep: Current training timestep.
            winner_idx: Index of winning player (0-3).
            victory_points: VP for each player [p0, p1, p2, p3].
            turns: Total game turns.
            agent_reward: Reward received by training agent.

        Returns:
            The recorded MatchRecord.
        """
        self.match_count += 1

        # Calculate placements from VP
        vp_with_idx = [(vp, i) for i, vp in enumerate(victory_points)]
        vp_with_idx.sort(reverse=True)
        placements = [0] * 4
        for place, (vp, idx) in enumerate(vp_with_idx, 1):
            placements[idx] = place

        # Training agent is always index 0
        agent_placement = placements[0]
        agent_vp = victory_points[0]

        record = MatchRecord(
            match_id=self.match_count,
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            players=[asdict(p) for p in self.current_opponents],
            winner_idx=winner_idx,
            placements=placements,
            victory_points=victory_points,
            turns=turns,
            agent_placement=agent_placement,
            agent_vp=agent_vp,
            agent_reward=agent_reward,
        )

        self.matches.append(record)

        # Periodic save
        if self.match_count % self.save_every_n_matches == 0:
            self.save()

        return record

    def get_opponent_summary(self) -> str:
        """Get a human-readable summary of current opponents."""
        if not self.current_opponents:
            return "No opponents set"

        lines = []
        for p in self.current_opponents:
            if p.agent_type == "training":
                lines.append(f"P{p.player_idx}: Training Agent (ELO: {p.elo:.0f})")
            elif p.agent_type == "random":
                lines.append(f"P{p.player_idx}: Random (ELO: {p.elo:.0f})")
            elif p.agent_type == "greedy":
                lines.append(f"P{p.player_idx}: Greedy (ELO: {p.elo:.0f})")
            elif p.agent_type == "pool":
                wr = f", WR: {p.win_rate_vs_random:.1%}" if p.win_rate_vs_random else ""
                lines.append(f"P{p.player_idx}: Pool Model (ELO: {p.elo:.0f}{wr})")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get statistics about tracked matches."""
        if not self.matches:
            return {
                "total_matches": 0,
                "agent_wins": 0,
                "agent_win_rate": 0.0,
            }

        agent_wins = sum(1 for m in self.matches if m.agent_placement == 1)

        # Breakdown by opponent type
        vs_random_only = [m for m in self.matches
                         if all(p["agent_type"] in ["training", "random"]
                               for p in m.players)]
        vs_pool = [m for m in self.matches
                   if any(p["agent_type"] == "pool" for p in m.players)]

        return {
            "total_matches": len(self.matches),
            "agent_wins": agent_wins,
            "agent_win_rate": agent_wins / len(self.matches),
            "avg_agent_vp": sum(m.agent_vp for m in self.matches) / len(self.matches),
            "avg_agent_placement": sum(m.agent_placement for m in self.matches) / len(self.matches),
            "matches_vs_random_only": len(vs_random_only),
            "matches_vs_pool": len(vs_pool),
            "win_rate_vs_random": (
                sum(1 for m in vs_random_only if m.agent_placement == 1) / len(vs_random_only)
                if vs_random_only else 0.0
            ),
            "win_rate_vs_pool": (
                sum(1 for m in vs_pool if m.agent_placement == 1) / len(vs_pool)
                if vs_pool else 0.0
            ),
        }

    def save(self) -> None:
        """Save match data to disk."""
        if not self.matches:
            return

        # Save full JSON
        json_path = os.path.join(self.save_dir, "match_history.json")
        with open(json_path, 'w') as f:
            json.dump([asdict(m) for m in self.matches], f, indent=2)

        # Save summary CSV for easy analysis
        csv_path = os.path.join(self.save_dir, "match_summary.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "match_id", "timestep", "agent_placement", "agent_vp",
                "winner_idx", "turns", "agent_reward",
                "opp1_type", "opp1_elo", "opp2_type", "opp2_elo", "opp3_type", "opp3_elo",
            ])
            for m in self.matches:
                opps = m.players[1:]  # Skip training agent
                writer.writerow([
                    m.match_id, m.timestep, m.agent_placement, m.agent_vp,
                    m.winner_idx, m.turns, m.agent_reward,
                    opps[0]["agent_type"] if len(opps) > 0 else "",
                    opps[0]["elo"] if len(opps) > 0 else "",
                    opps[1]["agent_type"] if len(opps) > 1 else "",
                    opps[1]["elo"] if len(opps) > 1 else "",
                    opps[2]["agent_type"] if len(opps) > 2 else "",
                    opps[2]["elo"] if len(opps) > 2 else "",
                ])

    def load(self, path: Optional[str] = None) -> None:
        """Load match history from disk."""
        json_path = path or os.path.join(self.save_dir, "match_history.json")
        if not os.path.exists(json_path):
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.matches = []
        for d in data:
            # Convert players back from dict
            record = MatchRecord(
                match_id=d["match_id"],
                timestep=d["timestep"],
                timestamp=d["timestamp"],
                players=d["players"],
                winner_idx=d["winner_idx"],
                placements=d["placements"],
                victory_points=d["victory_points"],
                turns=d["turns"],
                agent_placement=d["agent_placement"],
                agent_vp=d["agent_vp"],
                agent_reward=d.get("agent_reward", 0.0),
            )
            self.matches.append(record)

        self.match_count = len(self.matches)
