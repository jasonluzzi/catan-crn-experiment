"""Training module with comprehensive pipeline support.

Includes:
- Curriculum learning (progressive difficulty)
- Self-play training with opponent pool
- Hyperparameter scheduling (LR, entropy, clip range)
- Comprehensive evaluation system
- Checkpoint management and recovery
- ELO rating tracking
- Game logging for replay

Note: Some imports are done lazily to avoid circular dependencies.
"""

from .elo_tracker import ELOTracker, ELORecord, placement_from_vp
from .match_tracker import MatchTracker, MatchRecord, PlayerInfo
from .game_logger import GameLogger, GameLoggerConfig, LogLevel

# Curriculum learning
from .curriculum import (
    CurriculumManager,
    CurriculumConfig,
    CurriculumStage,
    OpponentMix,
    CurriculumCallback,
)

# Schedulers
from .schedulers import (
    Schedule,
    ConstantSchedule,
    LinearSchedule,
    CosineAnnealingSchedule,
    ExponentialSchedule,
    LearningRateScheduler,
    EntropyScheduler,
    ClipRangeScheduler,
    SchedulerConfig,
    SchedulerCallback,
)

# Evaluation
from .evaluation import (
    Evaluator,
    EvaluationConfig,
    EvaluationResult,
    EvaluationCallback,
    QuickEvaluator,
)

# Checkpointing
from .checkpointing import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetadata,
    CheckpointCallback,
    resume_training,
)

# Pipeline
from .pipeline import (
    TrainingPipeline,
    PipelineConfig,
    run_training,
)

# Auxiliary training
from .auxiliary_buffer import (
    AuxiliaryRolloutBuffer,
    AuxiliaryRolloutBufferSamples,
)
from .auxiliary_ppo import AuxiliaryMaskablePPO


# Lazy imports to avoid circular dependencies with models
def get_callbacks():
    """Get callback classes (lazy import)."""
    from .callbacks import (
        WinRateCallback,
        EntropyDecayCallback,
        CheckpointCallback as CheckpointCB,
        GameStatsCallback,
        EvalCallback,
        GameLoggingCallback,
        SelfPlayStatsCallback,
        ELOCallback,
    )
    return {
        'WinRateCallback': WinRateCallback,
        'EntropyDecayCallback': EntropyDecayCallback,
        'CheckpointCallback': CheckpointCB,
        'GameStatsCallback': GameStatsCallback,
        'EvalCallback': EvalCallback,
        'GameLoggingCallback': GameLoggingCallback,
        'SelfPlayStatsCallback': SelfPlayStatsCallback,
        'ELOCallback': ELOCallback,
    }


def get_self_play():
    """Get self-play classes (lazy import)."""
    from .self_play import SelfPlayManager, SelfPlayAgent, SelfPlayCallback
    return {
        'SelfPlayManager': SelfPlayManager,
        'SelfPlayAgent': SelfPlayAgent,
        'SelfPlayCallback': SelfPlayCallback,
    }


__all__ = [
    # ELO
    'ELOTracker',
    'ELORecord',
    'placement_from_vp',
    # Match tracker
    'MatchTracker',
    'MatchRecord',
    'PlayerInfo',
    # Game logger
    'GameLogger',
    'GameLoggerConfig',
    'LogLevel',
    # Curriculum
    'CurriculumManager',
    'CurriculumConfig',
    'CurriculumStage',
    'OpponentMix',
    'CurriculumCallback',
    # Schedulers
    'Schedule',
    'ConstantSchedule',
    'LinearSchedule',
    'CosineAnnealingSchedule',
    'ExponentialSchedule',
    'LearningRateScheduler',
    'EntropyScheduler',
    'ClipRangeScheduler',
    'SchedulerConfig',
    'SchedulerCallback',
    # Evaluation
    'Evaluator',
    'EvaluationConfig',
    'EvaluationResult',
    'EvaluationCallback',
    'QuickEvaluator',
    # Checkpointing
    'CheckpointManager',
    'CheckpointConfig',
    'CheckpointMetadata',
    'CheckpointCallback',
    'resume_training',
    # Pipeline
    'TrainingPipeline',
    'PipelineConfig',
    'run_training',
    # Auxiliary training
    'AuxiliaryRolloutBuffer',
    'AuxiliaryRolloutBufferSamples',
    'AuxiliaryMaskablePPO',
    # Lazy loaders
    'get_callbacks',
    'get_self_play',
]
