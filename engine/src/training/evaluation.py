"""Comprehensive evaluation system for Catan RL agent.

Provides multi-baseline evaluation, statistical analysis, and detailed metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto
import numpy as np
import json
import os
from datetime import datetime


class BaselineType(Enum):
    """Types of baseline opponents for evaluation."""
    RANDOM = auto()
    GREEDY = auto()
    HEURISTIC = auto()
    POOL_SAMPLE = auto()
    PREVIOUS_BEST = auto()
    FIXED_MODEL = auto()


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    baseline: str
    n_episodes: int
    wins: int
    win_rate: float
    win_rate_ci_lower: float  # 95% confidence interval
    win_rate_ci_upper: float
    avg_vp: float
    avg_turns: float
    avg_reward: float
    placement_distribution: Dict[int, float]  # 1st, 2nd, 3rd, 4th place rates
    vp_distribution: Dict[str, float]  # min, max, mean, std
    timestep: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "n_episodes": self.n_episodes,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "win_rate_ci_lower": self.win_rate_ci_lower,
            "win_rate_ci_upper": self.win_rate_ci_upper,
            "avg_vp": self.avg_vp,
            "avg_turns": self.avg_turns,
            "avg_reward": self.avg_reward,
            "placement_distribution": self.placement_distribution,
            "vp_distribution": self.vp_distribution,
            "timestep": self.timestep,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    n_episodes: int = 100               # Episodes per evaluation
    baselines: List[str] = field(default_factory=lambda: ["random", "greedy"])
    eval_deterministic: bool = True     # Use deterministic actions during eval
    confidence_level: float = 0.95      # For confidence intervals
    save_detailed_results: bool = True  # Save per-episode data
    parallel_envs: int = 4              # Parallel eval envs for speedup


class Evaluator:
    """Comprehensive evaluation system for the Catan agent."""

    def __init__(
        self,
        env_factory: Callable[..., Any],
        config: Optional[EvaluationConfig] = None,
        save_path: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            env_factory: Factory function to create environments.
                        Should accept opponents kwarg.
            config: Evaluation configuration.
            save_path: Path to save evaluation results.
        """
        self.env_factory = env_factory
        self.config = config or EvaluationConfig()
        self.save_path = save_path

        # Results history
        self.results_history: List[EvaluationResult] = []

        # Best metrics tracking
        self.best_win_rates: Dict[str, float] = {}
        self.best_models: Dict[str, str] = {}  # baseline -> model path

        # Load existing results if available
        if save_path:
            self._load_history()

    def evaluate(
        self,
        model,
        timestep: int = 0,
        baselines: Optional[List[str]] = None,
        n_episodes: Optional[int] = None,
        save_model_if_best: bool = False,
        model_save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """Run comprehensive evaluation against all baselines.

        Args:
            model: The model to evaluate.
            timestep: Current training timestep.
            baselines: List of baseline names to evaluate against.
            n_episodes: Override number of episodes.
            save_model_if_best: Save model if it achieves best win rate.
            model_save_path: Path to save best model.
            verbose: Print results.

        Returns:
            Dictionary mapping baseline name to EvaluationResult.
        """
        baselines = baselines or self.config.baselines
        n_episodes = n_episodes or self.config.n_episodes
        results = {}

        for baseline in baselines:
            if verbose:
                print(f"Evaluating against {baseline}...", end=" ", flush=True)

            result = self._evaluate_baseline(
                model, baseline, n_episodes, timestep
            )
            results[baseline] = result
            self.results_history.append(result)

            if verbose:
                print(f"Win rate: {result.win_rate:.2%} "
                      f"[{result.win_rate_ci_lower:.2%}, {result.win_rate_ci_upper:.2%}]")

            # Check for new best
            prev_best = self.best_win_rates.get(baseline, 0)
            if result.win_rate > prev_best:
                self.best_win_rates[baseline] = result.win_rate
                if save_model_if_best and model_save_path:
                    best_path = os.path.join(
                        model_save_path,
                        f"best_vs_{baseline}.zip"
                    )
                    model.save(best_path)
                    self.best_models[baseline] = best_path
                    if verbose:
                        print(f"  New best! Saved to {best_path}")

        # Save results
        if self.save_path:
            self._save_history()

        return results

    def _evaluate_baseline(
        self,
        model,
        baseline: str,
        n_episodes: int,
        timestep: int,
    ) -> EvaluationResult:
        """Evaluate against a specific baseline.

        Args:
            model: Model to evaluate.
            baseline: Baseline name.
            n_episodes: Number of episodes.
            timestep: Current timestep.

        Returns:
            EvaluationResult with metrics.
        """
        n_parallel = self.config.parallel_envs

        # Use parallel evaluation if configured
        if n_parallel > 1:
            return self._evaluate_baseline_parallel(
                model, baseline, n_episodes, timestep, n_parallel
            )

        # Sequential evaluation (original implementation)
        return self._evaluate_baseline_sequential(
            model, baseline, n_episodes, timestep
        )

    def _evaluate_baseline_sequential(
        self,
        model,
        baseline: str,
        n_episodes: int,
        timestep: int,
    ) -> EvaluationResult:
        """Sequential evaluation (one episode at a time)."""
        # Create opponents based on baseline type
        opponents = self._create_opponents(baseline)

        # Create evaluation environment
        env = self.env_factory(opponent_policies=opponents)

        # Run evaluation episodes
        wins = 0
        vps = []
        turns = []
        rewards = []
        placements = {1: 0, 2: 0, 3: 0, 4: 0}

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action_mask = env.action_masks()
                action, _ = model.predict(
                    obs,
                    deterministic=self.config.eval_deterministic,
                    action_masks=action_mask
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            # Record results
            if info.get("is_agent_winner", False):
                wins += 1

            agent_vp = info.get("total_vp", info.get("vp", 0))
            vps.append(agent_vp)
            turns.append(info.get("turn", 0))
            rewards.append(episode_reward)

            # Calculate placement
            opponent_vps = info.get("opponent_total_vps", info.get("opponent_vps", [0, 0, 0]))
            placement = self._calculate_placement(agent_vp, opponent_vps)
            placements[placement] += 1

        env.close()

        # Calculate statistics
        win_rate = wins / n_episodes
        ci_lower, ci_upper = self._wilson_confidence_interval(wins, n_episodes)

        placement_dist = {k: v / n_episodes for k, v in placements.items()}
        vp_dist = {
            "min": float(np.min(vps)),
            "max": float(np.max(vps)),
            "mean": float(np.mean(vps)),
            "std": float(np.std(vps)),
        }

        return EvaluationResult(
            baseline=baseline,
            n_episodes=n_episodes,
            wins=wins,
            win_rate=win_rate,
            win_rate_ci_lower=ci_lower,
            win_rate_ci_upper=ci_upper,
            avg_vp=float(np.mean(vps)),
            avg_turns=float(np.mean(turns)),
            avg_reward=float(np.mean(rewards)),
            placement_distribution=placement_dist,
            vp_distribution=vp_dist,
            timestep=timestep,
        )

    def _evaluate_baseline_parallel(
        self,
        model,
        baseline: str,
        n_episodes: int,
        timestep: int,
        n_parallel: int,
    ) -> EvaluationResult:
        """Parallel evaluation using vectorized environments."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.common.wrappers import ActionMasker

        # Create opponents
        opponents = self._create_opponents(baseline)

        # Create vectorized environment
        def make_env():
            env = self.env_factory(opponent_policies=opponents)
            return env

        vec_env = DummyVecEnv([make_env for _ in range(n_parallel)])

        # Track results
        wins = 0
        vps = []
        turns = []
        rewards = []
        placements = {1: 0, 2: 0, 3: 0, 4: 0}
        episodes_completed = 0
        episode_rewards = np.zeros(n_parallel)

        # Initial reset
        obs = vec_env.reset()

        while episodes_completed < n_episodes:
            # Get action masks from all envs
            action_masks = np.array([
                vec_env.envs[i].action_masks() for i in range(n_parallel)
            ])

            # Predict actions for all envs
            actions, _ = model.predict(
                obs,
                deterministic=self.config.eval_deterministic,
                action_masks=action_masks
            )

            # Step all envs
            obs, reward_batch, dones, infos = vec_env.step(actions)
            episode_rewards += reward_batch

            # Process completed episodes
            for i, done in enumerate(dones):
                if done and episodes_completed < n_episodes:
                    episodes_completed += 1
                    # IMPORTANT: DummyVecEnv auto-resets, so terminal info is in "terminal_info"
                    info = infos[i].get("terminal_info", infos[i])

                    # Record results
                    if info.get("is_agent_winner", False):
                        wins += 1

                    agent_vp = info.get("total_vp", info.get("vp", 0))
                    vps.append(agent_vp)
                    turns.append(info.get("turn", 0))
                    rewards.append(episode_rewards[i])

                    # Calculate placement
                    opponent_vps = info.get("opponent_total_vps", info.get("opponent_vps", [0, 0, 0]))
                    placement = self._calculate_placement(agent_vp, opponent_vps)
                    placements[placement] += 1

                    # Reset episode reward tracker
                    episode_rewards[i] = 0

        vec_env.close()

        # Calculate statistics
        win_rate = wins / n_episodes
        ci_lower, ci_upper = self._wilson_confidence_interval(wins, n_episodes)

        placement_dist = {k: v / n_episodes for k, v in placements.items()}
        vp_dist = {
            "min": float(np.min(vps)),
            "max": float(np.max(vps)),
            "mean": float(np.mean(vps)),
            "std": float(np.std(vps)),
        }

        return EvaluationResult(
            baseline=baseline,
            n_episodes=n_episodes,
            wins=wins,
            win_rate=win_rate,
            win_rate_ci_lower=ci_lower,
            win_rate_ci_upper=ci_upper,
            avg_vp=float(np.mean(vps)),
            avg_turns=float(np.mean(turns)),
            avg_reward=float(np.mean(rewards)),
            placement_distribution=placement_dist,
            vp_distribution=vp_dist,
            timestep=timestep,
        )

    def _create_opponents(self, baseline: str) -> List:
        """Create opponent agents for a baseline.

        Args:
            baseline: Baseline name.

        Returns:
            List of opponent agents.
        """
        from ..models.baseline import RandomAgent, GreedyAgent, HeuristicAgent

        if baseline == "random":
            return [RandomAgent() for _ in range(3)]
        elif baseline == "greedy":
            return [GreedyAgent() for _ in range(3)]
        elif baseline == "heuristic":
            return [HeuristicAgent() for _ in range(3)]
        elif baseline == "mixed":
            return [RandomAgent(), GreedyAgent(), HeuristicAgent()]
        else:
            # Default to random
            return [RandomAgent() for _ in range(3)]

    def _calculate_placement(self, agent_vp: int, opponent_vps: List[int]) -> int:
        """Calculate agent's placement based on VP."""
        all_vps = [agent_vp] + opponent_vps
        sorted_vps = sorted(all_vps, reverse=True)
        return sorted_vps.index(agent_vp) + 1

    def _wilson_confidence_interval(
        self,
        successes: int,
        total: int,
        z: float = 1.96,  # 95% CI
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for proportions.

        More accurate than normal approximation for small samples or
        extreme proportions.

        Args:
            successes: Number of successes (wins).
            total: Total trials (games).
            z: Z-score for confidence level (1.96 = 95%).

        Returns:
            Tuple of (lower, upper) bounds.
        """
        if total == 0:
            return 0.0, 0.0

        p_hat = successes / total
        denominator = 1 + z**2 / total
        center = (p_hat + z**2 / (2 * total)) / denominator
        spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator

        return max(0, center - spread), min(1, center + spread)

    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of evaluation history.

        Args:
            last_n: Only consider last N evaluations per baseline.

        Returns:
            Summary dictionary.
        """
        summary = {
            "total_evaluations": len(self.results_history),
            "baselines": {},
            "best_win_rates": self.best_win_rates.copy(),
        }

        # Group by baseline
        by_baseline: Dict[str, List[EvaluationResult]] = {}
        for result in self.results_history:
            if result.baseline not in by_baseline:
                by_baseline[result.baseline] = []
            by_baseline[result.baseline].append(result)

        for baseline, results in by_baseline.items():
            if last_n:
                results = results[-last_n:]

            win_rates = [r.win_rate for r in results]
            avg_vps = [r.avg_vp for r in results]

            summary["baselines"][baseline] = {
                "n_evaluations": len(results),
                "current_win_rate": results[-1].win_rate if results else 0,
                "avg_win_rate": float(np.mean(win_rates)) if win_rates else 0,
                "best_win_rate": float(np.max(win_rates)) if win_rates else 0,
                "win_rate_trend": self._calculate_trend(win_rates),
                "avg_vp": float(np.mean(avg_vps)) if avg_vps else 0,
            }

        return summary

    def _calculate_trend(self, values: List[float], window: int = 5) -> str:
        """Calculate trend direction from recent values."""
        if len(values) < window:
            return "insufficient_data"

        recent = values[-window:]
        older = values[-2*window:-window] if len(values) >= 2*window else values[:window]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"

    def _save_history(self) -> None:
        """Save evaluation history to file."""
        if not self.save_path:
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        data = {
            "results": [r.to_dict() for r in self.results_history],
            "best_win_rates": self.best_win_rates,
            "best_models": self.best_models,
        }

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_history(self) -> None:
        """Load evaluation history from file."""
        if not self.save_path or not os.path.exists(self.save_path):
            return

        with open(self.save_path, 'r') as f:
            data = json.load(f)

        self.results_history = [
            EvaluationResult(**r) for r in data.get("results", [])
        ]
        self.best_win_rates = data.get("best_win_rates", {})
        self.best_models = data.get("best_models", {})


class EvaluationCallback:
    """SB3-compatible callback for periodic evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        eval_freq: int = 50000,
        n_episodes: int = 50,
        baselines: Optional[List[str]] = None,
        save_best: bool = True,
        save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        """Initialize evaluation callback.

        Args:
            evaluator: Evaluator instance.
            eval_freq: Evaluate every N timesteps.
            n_episodes: Episodes per evaluation.
            baselines: Baselines to evaluate against.
            save_best: Save model when new best achieved.
            save_path: Path to save best models.
            verbose: Verbosity level.
        """
        from stable_baselines3.common.callbacks import BaseCallback
        self._base = BaseCallback

        self.evaluator = evaluator
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.baselines = baselines or ["random"]
        self.save_best = save_best
        self.save_path = save_path
        self.verbose = verbose


# Create proper callback
from stable_baselines3.common.callbacks import BaseCallback


class EvaluationCallback(BaseCallback):
    """SB3-compatible callback for periodic evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        eval_freq: int = 50000,
        n_episodes: int = 50,
        baselines: Optional[List[str]] = None,
        save_best: bool = True,
        save_path: Optional[str] = None,
        early_stopping: bool = False,
        target_win_rate: float = 0.95,
        target_baseline: str = "random",
        verbose: int = 1,
    ):
        """Initialize evaluation callback.

        Args:
            evaluator: Evaluator instance.
            eval_freq: Evaluate every N timesteps.
            n_episodes: Episodes per evaluation.
            baselines: Baselines to evaluate against.
            save_best: Save model when new best achieved.
            save_path: Path to save best models.
            early_stopping: Stop training when target reached.
            target_win_rate: Target win rate for early stopping.
            target_baseline: Baseline for target win rate.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.evaluator = evaluator
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.baselines = baselines or ["random"]
        self.save_best = save_best
        self.save_path = save_path
        self.early_stopping = early_stopping
        self.target_win_rate = target_win_rate
        self.target_baseline = target_baseline
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        """Run evaluation at specified frequency."""
        if self.num_timesteps >= self.last_eval_timestep + self.eval_freq:
            self.last_eval_timestep = self.num_timesteps

            # Run evaluation
            results = self.evaluator.evaluate(
                model=self.model,
                timestep=self.num_timesteps,
                baselines=self.baselines,
                n_episodes=self.n_episodes,
                save_model_if_best=self.save_best,
                model_save_path=self.save_path,
                verbose=self.verbose > 0,
            )

            # Log to tensorboard
            for baseline, result in results.items():
                self.logger.record(f"eval/{baseline}_win_rate", result.win_rate)
                self.logger.record(f"eval/{baseline}_avg_vp", result.avg_vp)
                self.logger.record(f"eval/{baseline}_ci_lower", result.win_rate_ci_lower)
                self.logger.record(f"eval/{baseline}_ci_upper", result.win_rate_ci_upper)

            # Check early stopping
            if self.early_stopping and self.target_baseline in results:
                if results[self.target_baseline].win_rate >= self.target_win_rate:
                    if self.verbose > 0:
                        print(f"\nTarget win rate {self.target_win_rate:.0%} achieved!")
                        print(f"Win rate vs {self.target_baseline}: "
                              f"{results[self.target_baseline].win_rate:.2%}")
                    return False  # Stop training

        return True

    def _on_training_end(self) -> None:
        """Log final summary."""
        if self.verbose > 0:
            summary = self.evaluator.get_summary(last_n=5)
            print("\nEvaluation Summary:")
            for baseline, stats in summary["baselines"].items():
                print(f"  {baseline}:")
                print(f"    Current: {stats['current_win_rate']:.2%}")
                print(f"    Best: {stats['best_win_rate']:.2%}")
                print(f"    Trend: {stats['win_rate_trend']}")


class QuickEvaluator:
    """Lightweight evaluator for fast mid-training checks."""

    def __init__(
        self,
        env_factory: Callable,
        n_episodes: int = 20,
    ):
        """Initialize quick evaluator.

        Args:
            env_factory: Factory to create eval environments.
            n_episodes: Number of episodes for quick eval.
        """
        self.env_factory = env_factory
        self.n_episodes = n_episodes

    def evaluate_vs_random(self, model) -> Tuple[float, float]:
        """Quick evaluation against random opponents.

        Args:
            model: Model to evaluate.

        Returns:
            Tuple of (win_rate, avg_vp).
        """
        from ..models.baseline import RandomAgent

        env = self.env_factory(opponent_policies=[RandomAgent() for _ in range(3)])

        wins = 0
        vps = []

        for _ in range(self.n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action_mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if info.get("is_agent_winner", False):
                wins += 1
            vps.append(info.get("total_vp", info.get("vp", 0)))

        env.close()
        return wins / self.n_episodes, float(np.mean(vps))
