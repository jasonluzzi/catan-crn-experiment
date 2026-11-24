"""Learning rate and hyperparameter schedulers for training.

Provides various scheduling strategies for:
- Learning rate (warmup, cosine annealing, linear decay)
- Entropy coefficient (exploration decay)
- Clip range (PPO stability)
- Batch size (optional scaling)
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Union
import math
import numpy as np


class Schedule(ABC):
    """Base class for parameter schedules."""

    @abstractmethod
    def value(self, progress: float) -> float:
        """Get the scheduled value at a given progress point.

        Args:
            progress: Training progress as fraction [0, 1].
                      0 = start of training, 1 = end of training.

        Returns:
            The scheduled parameter value.
        """
        pass

    def __call__(self, progress: float) -> float:
        """Alias for value()."""
        return self.value(progress)


class ConstantSchedule(Schedule):
    """Constant value throughout training."""

    def __init__(self, value: float):
        """Initialize with constant value.

        Args:
            value: The constant value to return.
        """
        self._value = value

    def value(self, progress: float) -> float:
        return self._value


class LinearSchedule(Schedule):
    """Linear interpolation from start to end value."""

    def __init__(
        self,
        start: float,
        end: float,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ):
        """Initialize linear schedule.

        Args:
            start: Starting value.
            end: Ending value.
            start_fraction: Progress at which to start interpolating.
            end_fraction: Progress at which to finish interpolating.
        """
        self.start = start
        self.end = end
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction

    def value(self, progress: float) -> float:
        if progress <= self.start_fraction:
            return self.start
        if progress >= self.end_fraction:
            return self.end

        # Linear interpolation
        local_progress = (progress - self.start_fraction) / (self.end_fraction - self.start_fraction)
        return self.start + local_progress * (self.end - self.start)


class ExponentialSchedule(Schedule):
    """Exponential decay from start to end value."""

    def __init__(
        self,
        start: float,
        end: float,
        decay_rate: float = 5.0,
    ):
        """Initialize exponential schedule.

        Args:
            start: Starting value.
            end: Ending value.
            decay_rate: Controls decay speed (higher = faster decay).
        """
        self.start = start
        self.end = end
        self.decay_rate = decay_rate

    def value(self, progress: float) -> float:
        # Exponential decay: start * exp(-decay_rate * progress) + end
        decay = math.exp(-self.decay_rate * progress)
        return self.end + (self.start - self.end) * decay


class CosineAnnealingSchedule(Schedule):
    """Cosine annealing schedule with optional warmup."""

    def __init__(
        self,
        start: float,
        end: float,
        warmup_fraction: float = 0.0,
        warmup_start: Optional[float] = None,
    ):
        """Initialize cosine annealing schedule.

        Args:
            start: Value after warmup (or initial if no warmup).
            end: Final value at end of training.
            warmup_fraction: Fraction of training for warmup [0, 1].
            warmup_start: Starting value for warmup (defaults to end).
        """
        self.start = start
        self.end = end
        self.warmup_fraction = warmup_fraction
        self.warmup_start = warmup_start if warmup_start is not None else end

    def value(self, progress: float) -> float:
        # Warmup phase
        if progress < self.warmup_fraction:
            warmup_progress = progress / self.warmup_fraction
            return self.warmup_start + warmup_progress * (self.start - self.warmup_start)

        # Cosine annealing phase
        annealing_progress = (progress - self.warmup_fraction) / (1.0 - self.warmup_fraction)
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * annealing_progress))
        return self.end + (self.start - self.end) * cosine_value


class WarmupCosineSchedule(Schedule):
    """Warmup followed by cosine annealing - commonly used for learning rate."""

    def __init__(
        self,
        peak_value: float,
        final_value: float,
        warmup_steps: int,
        total_steps: int,
        warmup_start: float = 0.0,
    ):
        """Initialize warmup + cosine schedule.

        Args:
            peak_value: Value at end of warmup / start of annealing.
            final_value: Value at end of training.
            warmup_steps: Number of warmup steps.
            total_steps: Total training steps.
            warmup_start: Starting value for warmup.
        """
        self.peak_value = peak_value
        self.final_value = final_value
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start = warmup_start

    def value(self, progress: float) -> float:
        current_step = int(progress * self.total_steps)

        # Warmup phase
        if current_step < self.warmup_steps:
            warmup_progress = current_step / self.warmup_steps
            return self.warmup_start + warmup_progress * (self.peak_value - self.warmup_start)

        # Cosine annealing phase
        annealing_steps = self.total_steps - self.warmup_steps
        annealing_progress = (current_step - self.warmup_steps) / annealing_steps
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * annealing_progress))
        return self.final_value + (self.peak_value - self.final_value) * cosine_value

    def value_at_step(self, step: int) -> float:
        """Get value at a specific step number."""
        progress = step / self.total_steps
        return self.value(progress)


class PiecewiseSchedule(Schedule):
    """Piecewise linear schedule with multiple segments."""

    def __init__(
        self,
        milestones: List[float],
        values: List[float],
    ):
        """Initialize piecewise schedule.

        Args:
            milestones: List of progress points [0, m1, m2, ..., 1].
            values: List of values at each milestone.
        """
        if len(milestones) != len(values):
            raise ValueError("milestones and values must have same length")
        if milestones[0] != 0.0:
            milestones = [0.0] + milestones
            values = [values[0]] + values
        if milestones[-1] != 1.0:
            milestones = milestones + [1.0]
            values = values + [values[-1]]

        self.milestones = milestones
        self.values = values

    def value(self, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))

        # Find segment
        for i in range(len(self.milestones) - 1):
            if progress <= self.milestones[i + 1]:
                # Linear interpolation within segment
                local_progress = (progress - self.milestones[i]) / (
                    self.milestones[i + 1] - self.milestones[i]
                )
                return self.values[i] + local_progress * (self.values[i + 1] - self.values[i])

        return self.values[-1]


class LearningRateScheduler:
    """Learning rate scheduler with SB3 compatibility.

    Creates a callable that can be passed to MaskablePPO's learning_rate parameter.
    """

    def __init__(
        self,
        initial_lr: float = 3e-4,
        final_lr: float = 1e-5,
        schedule_type: str = "cosine",
        warmup_fraction: float = 0.05,
        total_timesteps: Optional[int] = None,
    ):
        """Initialize learning rate scheduler.

        Args:
            initial_lr: Initial/peak learning rate.
            final_lr: Final learning rate.
            schedule_type: One of "constant", "linear", "cosine", "exponential".
            warmup_fraction: Fraction of training for warmup (0 = no warmup).
            total_timesteps: Total training timesteps (needed for some schedules).
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.schedule_type = schedule_type
        self.warmup_fraction = warmup_fraction
        self.total_timesteps = total_timesteps

        # Create internal schedule
        if schedule_type == "constant":
            self._schedule = ConstantSchedule(initial_lr)
        elif schedule_type == "linear":
            self._schedule = LinearSchedule(initial_lr, final_lr)
        elif schedule_type == "cosine":
            self._schedule = CosineAnnealingSchedule(
                initial_lr, final_lr,
                warmup_fraction=warmup_fraction,
                warmup_start=final_lr,
            )
        elif schedule_type == "exponential":
            self._schedule = ExponentialSchedule(initial_lr, final_lr)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def __call__(self, progress_remaining: float) -> float:
        """Get learning rate for SB3 (uses progress_remaining = 1 - progress).

        Args:
            progress_remaining: Fraction of training remaining [0, 1].
                               1 at start, 0 at end.

        Returns:
            Current learning rate.
        """
        progress = 1.0 - progress_remaining
        return self._schedule.value(progress)

    def get_schedule_fn(self) -> Callable[[float], float]:
        """Get a callable for SB3's learning_rate parameter."""
        return self


class EntropyScheduler:
    """Entropy coefficient scheduler for controlling exploration."""

    def __init__(
        self,
        initial_ent_coef: float = 0.1,
        final_ent_coef: float = 0.01,
        schedule_type: str = "exponential",
        decay_fraction: float = 0.8,
    ):
        """Initialize entropy scheduler.

        Args:
            initial_ent_coef: Starting entropy coefficient (high exploration).
            final_ent_coef: Final entropy coefficient (low exploration).
            schedule_type: One of "constant", "linear", "exponential", "step".
            decay_fraction: Fraction of training over which to decay.
        """
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.schedule_type = schedule_type
        self.decay_fraction = decay_fraction

        if schedule_type == "constant":
            self._schedule = ConstantSchedule(initial_ent_coef)
        elif schedule_type == "linear":
            self._schedule = LinearSchedule(
                initial_ent_coef, final_ent_coef,
                end_fraction=decay_fraction
            )
        elif schedule_type == "exponential":
            self._schedule = ExponentialSchedule(
                initial_ent_coef, final_ent_coef,
                decay_rate=3.0 / decay_fraction  # Reach ~95% of decay by decay_fraction
            )
        elif schedule_type == "step":
            # Step decay at 25%, 50%, 75% of decay_fraction
            milestones = [0, decay_fraction * 0.25, decay_fraction * 0.5, decay_fraction * 0.75, decay_fraction]
            decay_per_step = (initial_ent_coef - final_ent_coef) / 4
            values = [
                initial_ent_coef,
                initial_ent_coef - decay_per_step,
                initial_ent_coef - 2 * decay_per_step,
                initial_ent_coef - 3 * decay_per_step,
                final_ent_coef,
            ]
            self._schedule = PiecewiseSchedule(milestones, values)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def value(self, progress: float) -> float:
        """Get entropy coefficient at given progress."""
        return self._schedule.value(progress)

    def __call__(self, progress: float) -> float:
        return self.value(progress)


class ClipRangeScheduler:
    """PPO clip range scheduler."""

    def __init__(
        self,
        initial_clip: float = 0.2,
        final_clip: float = 0.1,
        schedule_type: str = "linear",
    ):
        """Initialize clip range scheduler.

        Args:
            initial_clip: Starting clip range.
            final_clip: Final clip range.
            schedule_type: One of "constant", "linear".
        """
        self.initial_clip = initial_clip
        self.final_clip = final_clip

        if schedule_type == "constant":
            self._schedule = ConstantSchedule(initial_clip)
        elif schedule_type == "linear":
            self._schedule = LinearSchedule(initial_clip, final_clip)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def __call__(self, progress_remaining: float) -> float:
        """Get clip range for SB3."""
        progress = 1.0 - progress_remaining
        return self._schedule.value(progress)


class BatchSizeScheduler:
    """Optional batch size scheduler for scaling during training.

    Note: Changing batch size mid-training requires special handling
    and is not directly supported by SB3. This is for reference.
    """

    def __init__(
        self,
        initial_batch_size: int = 256,
        final_batch_size: int = 1024,
        scale_fraction: float = 0.5,
    ):
        """Initialize batch size scheduler.

        Args:
            initial_batch_size: Starting batch size.
            final_batch_size: Final batch size.
            scale_fraction: Fraction of training to scale over.
        """
        self.initial_batch_size = initial_batch_size
        self.final_batch_size = final_batch_size
        self.scale_fraction = scale_fraction

    def value(self, progress: float) -> int:
        """Get batch size at given progress."""
        if progress >= self.scale_fraction:
            return self.final_batch_size

        scale_progress = progress / self.scale_fraction
        # Linear scaling (could also use log scaling for smoother increase)
        raw_size = self.initial_batch_size + scale_progress * (
            self.final_batch_size - self.initial_batch_size
        )
        # Round to nearest power of 2 for efficiency
        return 2 ** round(math.log2(raw_size))


class SchedulerConfig:
    """Configuration container for all schedulers."""

    def __init__(
        self,
        # Learning rate
        lr_initial: float = 3e-4,
        lr_final: float = 1e-5,
        lr_schedule: str = "cosine",
        lr_warmup: float = 0.05,
        # Entropy
        ent_initial: float = 0.05,
        ent_final: float = 0.01,
        ent_schedule: str = "exponential",
        ent_decay_fraction: float = 0.8,
        # Clip range
        clip_initial: float = 0.2,
        clip_final: float = 0.1,
        clip_schedule: str = "constant",  # Usually keep constant
        # Total timesteps (needed for some calculations)
        total_timesteps: Optional[int] = None,
    ):
        """Initialize scheduler configuration.

        Args:
            lr_initial: Initial learning rate.
            lr_final: Final learning rate.
            lr_schedule: LR schedule type.
            lr_warmup: Warmup fraction for LR.
            ent_initial: Initial entropy coefficient.
            ent_final: Final entropy coefficient.
            ent_schedule: Entropy schedule type.
            ent_decay_fraction: Entropy decay fraction.
            clip_initial: Initial clip range.
            clip_final: Final clip range.
            clip_schedule: Clip schedule type.
            total_timesteps: Total training timesteps.
        """
        self.lr_scheduler = LearningRateScheduler(
            initial_lr=lr_initial,
            final_lr=lr_final,
            schedule_type=lr_schedule,
            warmup_fraction=lr_warmup,
            total_timesteps=total_timesteps,
        )
        self.ent_scheduler = EntropyScheduler(
            initial_ent_coef=ent_initial,
            final_ent_coef=ent_final,
            schedule_type=ent_schedule,
            decay_fraction=ent_decay_fraction,
        )
        self.clip_scheduler = ClipRangeScheduler(
            initial_clip=clip_initial,
            final_clip=clip_final,
            schedule_type=clip_schedule,
        )

    def get_lr_schedule(self) -> Callable[[float], float]:
        """Get learning rate schedule function for SB3."""
        return self.lr_scheduler

    def get_clip_schedule(self) -> Callable[[float], float]:
        """Get clip range schedule function for SB3."""
        return self.clip_scheduler

    def get_entropy_at_progress(self, progress: float) -> float:
        """Get entropy coefficient at given training progress."""
        return self.ent_scheduler(progress)

    @classmethod
    def default(cls, total_timesteps: int = 1_000_000) -> 'SchedulerConfig':
        """Create default scheduler configuration."""
        return cls(
            lr_initial=3e-4,
            lr_final=1e-5,
            lr_schedule="cosine",
            lr_warmup=0.05,
            ent_initial=0.05,
            ent_final=0.01,
            ent_schedule="exponential",
            ent_decay_fraction=0.8,
            clip_initial=0.2,
            clip_final=0.1,
            clip_schedule="constant",
            total_timesteps=total_timesteps,
        )

    @classmethod
    def aggressive_exploration(cls, total_timesteps: int = 1_000_000) -> 'SchedulerConfig':
        """Configuration with higher exploration for complex environments."""
        return cls(
            lr_initial=1e-4,
            lr_final=1e-6,
            lr_schedule="cosine",
            lr_warmup=0.1,
            ent_initial=0.1,
            ent_final=0.01,
            ent_schedule="linear",
            ent_decay_fraction=0.9,
            clip_initial=0.3,
            clip_final=0.1,
            clip_schedule="linear",
            total_timesteps=total_timesteps,
        )

    @classmethod
    def stable_training(cls, total_timesteps: int = 1_000_000) -> 'SchedulerConfig':
        """Conservative configuration for stable training."""
        return cls(
            lr_initial=1e-4,
            lr_final=5e-6,
            lr_schedule="linear",
            lr_warmup=0.02,
            ent_initial=0.02,
            ent_final=0.005,
            ent_schedule="exponential",
            ent_decay_fraction=0.7,
            clip_initial=0.2,
            clip_final=0.2,
            clip_schedule="constant",
            total_timesteps=total_timesteps,
        )


# SB3-compatible callback for entropy scheduling
from stable_baselines3.common.callbacks import BaseCallback


class SchedulerCallback(BaseCallback):
    """Callback to update scheduled parameters during training."""

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        total_timesteps: int,
        update_freq: int = 1000,
        verbose: int = 0,
    ):
        """Initialize scheduler callback.

        Args:
            scheduler_config: SchedulerConfig instance.
            total_timesteps: Total training timesteps.
            update_freq: How often to update entropy (in steps).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.scheduler_config = scheduler_config
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        """Update scheduled parameters."""
        if self.n_calls % self.update_freq == 0:
            progress = self.num_timesteps / self.total_timesteps

            # Update entropy coefficient
            new_ent_coef = self.scheduler_config.get_entropy_at_progress(progress)
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = new_ent_coef

            # Log values
            if self.verbose > 0 and self.n_calls % (self.update_freq * 10) == 0:
                lr = self.scheduler_config.lr_scheduler(1.0 - progress)
                self.logger.record("scheduler/learning_rate", lr)
                self.logger.record("scheduler/entropy_coef", new_ent_coef)
                self.logger.record("scheduler/progress", progress)

        return True
