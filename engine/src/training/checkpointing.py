"""Checkpoint and recovery management for training.

Provides comprehensive state saving and restoration including:
- Model weights
- Training progress
- Optimizer state
- Curriculum state
- Opponent pool
- ELO ratings
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    timesteps: int = 0
    episodes: int = 0
    best_win_rate: float = 0.0
    best_avg_vp: float = 0.0
    current_win_rate: float = 0.0
    current_avg_vp: float = 0.0
    training_time_seconds: float = 0.0


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    timestep: int
    episode: int
    win_rate: float
    avg_vp: float
    curriculum_stage: Optional[int] = None
    elo_rating: Optional[float] = None
    is_best: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 10           # Max regular checkpoints to keep
    keep_best: int = 5                  # Number of best models to keep
    checkpoint_freq: int = 100000       # Save checkpoint every N timesteps
    save_optimizer: bool = True         # Save optimizer state for exact resumption
    save_replay_buffer: bool = False    # Save replay buffer (if applicable)
    compress: bool = False              # Compress checkpoint files


class CheckpointManager:
    """Manages model checkpoints and training state recovery."""

    def __init__(
        self,
        config: Optional[CheckpointConfig] = None,
        verbose: int = 1,
    ):
        """Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration.
            verbose: Verbosity level.
        """
        self.config = config or CheckpointConfig()
        self.verbose = verbose

        # Checkpoint tracking
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoints: List[CheckpointMetadata] = []

        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.checkpoint_dir, "best"), exist_ok=True)

        # Load existing checkpoint index
        self._load_index()

    @property
    def checkpoint_dir(self) -> str:
        return self.config.checkpoint_dir

    def save_checkpoint(
        self,
        model,
        timestep: int,
        episode: int = 0,
        win_rate: float = 0.0,
        avg_vp: float = 0.0,
        curriculum_stage: Optional[int] = None,
        elo_rating: Optional[float] = None,
        curriculum_manager: Optional[Any] = None,
        elo_tracker: Optional[Any] = None,
        opponent_pool: Optional[Any] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        notes: str = "",
        force: bool = False,
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: The model to save.
            timestep: Current training timestep.
            episode: Current episode count.
            win_rate: Current win rate.
            avg_vp: Current average VP.
            curriculum_stage: Current curriculum stage index.
            elo_rating: Current ELO rating.
            curriculum_manager: CurriculumManager to save state.
            elo_tracker: ELOTracker to save state.
            opponent_pool: OpponentPool to save.
            additional_state: Additional state to save.
            notes: Optional notes about this checkpoint.
            force: Save even if not at checkpoint frequency.

        Returns:
            Path to saved checkpoint.
        """
        # Check if we should save
        if not force:
            last_checkpoint_step = self.checkpoints[-1].timestep if self.checkpoints else 0
            if timestep - last_checkpoint_step < self.config.checkpoint_freq:
                return ""

        # Generate checkpoint ID
        checkpoint_id = f"checkpoint_{timestep:010d}"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_id)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model
        model_path = os.path.join(checkpoint_path, "model.zip")
        model.save(model_path)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestep=timestep,
            episode=episode,
            win_rate=win_rate,
            avg_vp=avg_vp,
            curriculum_stage=curriculum_stage,
            elo_rating=elo_rating,
            notes=notes,
        )

        # Save additional components
        state = {
            "metadata": metadata.to_dict(),
            "additional_state": additional_state or {},
        }

        # Save curriculum state
        if curriculum_manager is not None:
            curriculum_path = os.path.join(checkpoint_path, "curriculum.json")
            curriculum_manager.save(curriculum_path)
            state["has_curriculum"] = True

        # Save ELO state
        if elo_tracker is not None:
            elo_path = os.path.join(checkpoint_path, "elo.json")
            elo_tracker.save(elo_path)
            state["has_elo"] = True

        # Save opponent pool references
        if opponent_pool is not None:
            pool_state = {
                "opponents": list(opponent_pool.opponents) if hasattr(opponent_pool, 'opponents') else [],
                "metadata": dict(opponent_pool.metadata) if hasattr(opponent_pool, 'metadata') else {},
            }
            pool_path = os.path.join(checkpoint_path, "opponent_pool.json")
            with open(pool_path, 'w') as f:
                json.dump(pool_state, f, indent=2)
            state["has_opponent_pool"] = True

        # Save state file
        state_path = os.path.join(checkpoint_path, "state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        # Update checkpoint list
        self.checkpoints.append(metadata)

        # Check if this is best model
        if win_rate > 0 and (not self.best_checkpoints or win_rate > self.best_checkpoints[0].win_rate):
            metadata.is_best = True
            self._save_best_checkpoint(checkpoint_path, metadata)

        # Prune old checkpoints
        self._prune_checkpoints()

        # Save index
        self._save_index()

        if self.verbose > 0:
            print(f"Saved checkpoint: {checkpoint_id} (win_rate={win_rate:.2%})")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        load_best: bool = False,
        model_class: Optional[Any] = None,
        device: str = "auto",
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to load. If None, loads latest.
            load_best: Load the best checkpoint instead of latest.
            model_class: Model class for loading (e.g., MaskablePPO).
            device: Device to load model on.

        Returns:
            Dictionary with loaded components:
            - model: Loaded model
            - metadata: CheckpointMetadata
            - curriculum_state: Path to curriculum state file
            - elo_state: Path to ELO state file
            - opponent_pool_state: Opponent pool state dict
            - additional_state: Additional saved state
        """
        # Determine which checkpoint to load
        if load_best:
            if not self.best_checkpoints:
                raise ValueError("No best checkpoints available")
            checkpoint_id = self.best_checkpoints[-1].checkpoint_id
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "best", checkpoint_id)
        elif checkpoint_id is None:
            if not self.checkpoints:
                raise ValueError("No checkpoints available")
            checkpoint_id = self.checkpoints[-1].checkpoint_id
            checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_id)
        else:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_id)

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Load state file
        state_path = os.path.join(checkpoint_path, "state.json")
        with open(state_path, 'r') as f:
            state = json.load(f)

        # Load model
        model = None
        model_path = os.path.join(checkpoint_path, "model.zip")
        if model_class is not None and os.path.exists(model_path):
            model = model_class.load(model_path, device=device)

        # Build result
        result = {
            "model": model,
            "model_path": model_path,
            "metadata": CheckpointMetadata.from_dict(state["metadata"]),
            "additional_state": state.get("additional_state", {}),
        }

        # Add component paths if they exist
        if state.get("has_curriculum"):
            result["curriculum_path"] = os.path.join(checkpoint_path, "curriculum.json")

        if state.get("has_elo"):
            result["elo_path"] = os.path.join(checkpoint_path, "elo.json")

        if state.get("has_opponent_pool"):
            pool_path = os.path.join(checkpoint_path, "opponent_pool.json")
            with open(pool_path, 'r') as f:
                result["opponent_pool_state"] = json.load(f)

        if self.verbose > 0:
            print(f"Loaded checkpoint: {checkpoint_id}")
            print(f"  Timestep: {result['metadata'].timestep}")
            print(f"  Win rate: {result['metadata'].win_rate:.2%}")

        return result

    def _save_best_checkpoint(
        self,
        checkpoint_path: str,
        metadata: CheckpointMetadata,
    ) -> None:
        """Save a copy of the checkpoint as best model."""
        best_dir = os.path.join(self.config.checkpoint_dir, "best")
        best_path = os.path.join(best_dir, metadata.checkpoint_id)

        # Copy checkpoint to best directory
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        shutil.copytree(checkpoint_path, best_path)

        # Update best checkpoints list
        self.best_checkpoints.append(metadata)
        self.best_checkpoints.sort(key=lambda m: m.win_rate, reverse=True)

        # Keep only top N best
        while len(self.best_checkpoints) > self.config.keep_best:
            old_best = self.best_checkpoints.pop()
            old_path = os.path.join(best_dir, old_best.checkpoint_id)
            if os.path.exists(old_path):
                shutil.rmtree(old_path)

        if self.verbose > 0:
            print(f"New best model: {metadata.checkpoint_id} (win_rate={metadata.win_rate:.2%})")

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints to stay under limit."""
        while len(self.checkpoints) > self.config.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = os.path.join(self.config.checkpoint_dir, old_checkpoint.checkpoint_id)
            if os.path.exists(old_path):
                shutil.rmtree(old_path)

    def _save_index(self) -> None:
        """Save checkpoint index to file."""
        index = {
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "best_checkpoints": [c.to_dict() for c in self.best_checkpoints],
            "config": asdict(self.config),
        }
        index_path = os.path.join(self.config.checkpoint_dir, "index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def _load_index(self) -> None:
        """Load checkpoint index from file."""
        index_path = os.path.join(self.config.checkpoint_dir, "index.json")
        if not os.path.exists(index_path):
            return

        with open(index_path, 'r') as f:
            index = json.load(f)

        self.checkpoints = [
            CheckpointMetadata.from_dict(c)
            for c in index.get("checkpoints", [])
        ]
        self.best_checkpoints = [
            CheckpointMetadata.from_dict(c)
            for c in index.get("best_checkpoints", [])
        ]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
        return os.path.join(self.config.checkpoint_dir, self.checkpoints[-1].checkpoint_id)

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if not self.best_checkpoints:
            return None
        best = max(self.best_checkpoints, key=lambda m: m.win_rate)
        return os.path.join(self.config.checkpoint_dir, "best", best.checkpoint_id)

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        return self.checkpoints.copy()

    def list_best_checkpoints(self) -> List[CheckpointMetadata]:
        """List all best checkpoints."""
        return self.best_checkpoints.copy()

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoint state."""
        return {
            "total_checkpoints": len(self.checkpoints),
            "best_checkpoints": len(self.best_checkpoints),
            "latest_timestep": self.checkpoints[-1].timestep if self.checkpoints else 0,
            "best_win_rate": self.best_checkpoints[-1].win_rate if self.best_checkpoints else 0,
            "checkpoint_dir": self.config.checkpoint_dir,
        }


class CheckpointCallback:
    """SB3 callback for checkpoint management."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_freq: int = 100000,
        curriculum_manager: Optional[Any] = None,
        elo_tracker: Optional[Any] = None,
        opponent_pool: Optional[Any] = None,
        eval_callback: Optional[Any] = None,
        verbose: int = 1,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_manager: CheckpointManager instance.
            checkpoint_freq: Checkpoint frequency in timesteps.
            curriculum_manager: Optional curriculum manager to save.
            elo_tracker: Optional ELO tracker to save.
            opponent_pool: Optional opponent pool to save.
            eval_callback: Evaluation callback to get metrics from.
            verbose: Verbosity level.
        """
        from stable_baselines3.common.callbacks import BaseCallback
        self._base = BaseCallback

        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_freq = checkpoint_freq
        self.curriculum_manager = curriculum_manager
        self.elo_tracker = elo_tracker
        self.opponent_pool = opponent_pool
        self.eval_callback = eval_callback
        self.verbose = verbose


# Create proper callback
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """SB3 callback for checkpoint management."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        checkpoint_freq: int = 100000,
        curriculum_manager: Optional[Any] = None,
        elo_tracker: Optional[Any] = None,
        opponent_pool: Optional[Any] = None,
        get_metrics_fn: Optional[Any] = None,
        verbose: int = 1,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_manager: CheckpointManager instance.
            checkpoint_freq: Checkpoint frequency in timesteps.
            curriculum_manager: Optional curriculum manager to save.
            elo_tracker: Optional ELO tracker to save.
            opponent_pool: Optional opponent pool to save.
            get_metrics_fn: Function that returns (win_rate, avg_vp, episode_count).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_freq = checkpoint_freq
        self.curriculum_manager = curriculum_manager
        self.elo_tracker = elo_tracker
        self.opponent_pool = opponent_pool
        self.get_metrics_fn = get_metrics_fn
        self.last_checkpoint_timestep = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Save checkpoint at specified frequency."""
        # Track episodes
        for info in self.locals.get("infos", []):
            if info.get("is_agent_winner") is not None:
                self.episode_count += 1

        # Check if time to checkpoint
        if self.num_timesteps >= self.last_checkpoint_timestep + self.checkpoint_freq:
            self.last_checkpoint_timestep = self.num_timesteps

            # Get metrics
            win_rate, avg_vp = 0.0, 0.0
            if self.get_metrics_fn:
                win_rate, avg_vp, _ = self.get_metrics_fn()

            # Get curriculum stage
            curriculum_stage = None
            if self.curriculum_manager:
                curriculum_stage = self.curriculum_manager.current_stage_idx

            # Get ELO
            elo_rating = None
            if self.elo_tracker:
                elo_rating = self.elo_tracker.current_elo

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                timestep=self.num_timesteps,
                episode=self.episode_count,
                win_rate=win_rate,
                avg_vp=avg_vp,
                curriculum_stage=curriculum_stage,
                elo_rating=elo_rating,
                curriculum_manager=self.curriculum_manager,
                elo_tracker=self.elo_tracker,
                opponent_pool=self.opponent_pool,
                force=True,
            )

        return True

    def _on_training_end(self) -> None:
        """Save final checkpoint."""
        win_rate, avg_vp = 0.0, 0.0
        if self.get_metrics_fn:
            win_rate, avg_vp, _ = self.get_metrics_fn()

        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            timestep=self.num_timesteps,
            episode=self.episode_count,
            win_rate=win_rate,
            avg_vp=avg_vp,
            curriculum_manager=self.curriculum_manager,
            elo_tracker=self.elo_tracker,
            opponent_pool=self.opponent_pool,
            notes="final",
            force=True,
        )

        if self.verbose > 0:
            summary = self.checkpoint_manager.get_checkpoint_summary()
            print(f"\nCheckpoint Summary:")
            print(f"  Total checkpoints: {summary['total_checkpoints']}")
            print(f"  Best win rate: {summary['best_win_rate']:.2%}")
            print(f"  Directory: {summary['checkpoint_dir']}")


def resume_training(
    checkpoint_path: str,
    model_class,
    device: str = "auto",
) -> Dict[str, Any]:
    """Convenience function to resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or checkpoint ID.
        model_class: Model class (e.g., MaskablePPO).
        device: Device to load model on.

    Returns:
        Dictionary with loaded state for resuming training.
    """
    # Determine if path is a checkpoint dir or ID
    if os.path.isdir(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_id = os.path.basename(checkpoint_path)
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_id = checkpoint_path

    # Create manager and load
    config = CheckpointConfig(checkpoint_dir=checkpoint_dir)
    manager = CheckpointManager(config)

    return manager.load_checkpoint(
        checkpoint_id=checkpoint_id,
        model_class=model_class,
        device=device,
    )
