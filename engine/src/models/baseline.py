"""Baseline strategies for Catan."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..game.game_state import GameState
from ..game.rules import Rules
from ..game.actions import (
    TOTAL_ACTIONS, index_to_action,
    ACTION_END_TURN, ACTION_ROLL_DICE,
    ACTION_BUILD_SETTLEMENT_START, ACTION_BUILD_SETTLEMENT_END,
    ACTION_BUILD_CITY_START, ACTION_BUILD_CITY_END,
    ACTION_BUILD_ROAD_START, ACTION_BUILD_ROAD_END,
    ACTION_BUY_DEV_CARD,
    ACTION_PLACE_SETTLEMENT_START, ACTION_PLACE_SETTLEMENT_END,
    ACTION_BANK_TRADE_START, ACTION_BANK_TRADE_END,
)
from ..game.constants import ActionType, GamePhase


class BaseAgent(ABC):
    """Base class for Catan agents."""

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select an action given the current state.

        Args:
            observation: The encoded game observation.
            action_mask: Boolean mask of valid actions.
            deterministic: Whether to select deterministically.

        Returns:
            The selected action index.
        """
        pass

    def reset(self) -> None:
        """Reset agent state (for episodic tracking)."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects uniformly random valid actions."""

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select a random valid action."""
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # Fallback
        return np.random.choice(valid_actions)


class GreedyAgent(BaseAgent):
    """Agent that prioritizes building and VP-generating actions.

    Priority order:
    1. Build city (if possible)
    2. Build settlement (if possible)
    3. Buy dev card (if possible)
    4. Build road (if possible)
    5. End turn / random
    """

    def __init__(self, exploration_prob: float = 0.1):
        """Initialize the greedy agent.

        Args:
            exploration_prob: Probability of taking a random action.
        """
        self.exploration_prob = exploration_prob

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select action based on greedy heuristics."""
        # Exploration
        if not deterministic and np.random.random() < self.exploration_prob:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)

        # Must roll dice first
        if action_mask[ACTION_ROLL_DICE]:
            return ACTION_ROLL_DICE

        # Setup phase - place settlements with smart scoring
        setup_settlements = np.where(
            action_mask[ACTION_PLACE_SETTLEMENT_START:ACTION_PLACE_SETTLEMENT_END]
        )[0]
        if len(setup_settlements) > 0:
            # Score each settlement location based on observation features
            # Corner features are at rows 20-73 in the observation
            # Higher feature values generally indicate better production
            best_score = -float('inf')
            best_corners = []

            for corner_id in setup_settlements:
                obs_row = 20 + corner_id
                if obs_row < observation.shape[0]:
                    # Sum corner features as production proxy, add small random for tie-breaking
                    score = observation[obs_row].sum() + np.random.random() * 0.01
                    if score > best_score:
                        best_score = score
                        best_corners = [corner_id]
                    elif abs(score - best_score) < 0.1:
                        best_corners.append(corner_id)

            # Pick from top-scored corners
            chosen = np.random.choice(best_corners) if best_corners else setup_settlements[0]
            return ACTION_PLACE_SETTLEMENT_START + chosen

        # Setup roads - pick randomly for expansion diversity
        from ..game.actions import ACTION_PLACE_ROAD_START, ACTION_PLACE_ROAD_END
        setup_roads = np.where(
            action_mask[ACTION_PLACE_ROAD_START:ACTION_PLACE_ROAD_END]
        )[0]
        if len(setup_roads) > 0:
            return ACTION_PLACE_ROAD_START + np.random.choice(setup_roads)

        # Priority: City > Settlement > Road (limited) > Dev Card > End Turn
        # Check for cities
        city_actions = np.where(
            action_mask[ACTION_BUILD_CITY_START:ACTION_BUILD_CITY_END]
        )[0]
        if len(city_actions) > 0:
            return ACTION_BUILD_CITY_START + city_actions[0]

        # Check for settlements - HIGHEST PRIORITY for VP
        settlement_actions = np.where(
            action_mask[ACTION_BUILD_SETTLEMENT_START:ACTION_BUILD_SETTLEMENT_END]
        )[0]
        if len(settlement_actions) > 0:
            return ACTION_BUILD_SETTLEMENT_START + settlement_actions[0]

        # Check for roads - but LIMIT road building to leave resources for settlements
        # Only build roads if we have fewer than 6 roads (enough to reach most spots)
        # After 6 roads, prioritize saving resources for settlements
        road_actions = np.where(
            action_mask[ACTION_BUILD_ROAD_START:ACTION_BUILD_ROAD_END]
        )[0]

        # Count how many roads we can build (proxy for how many resources we have)
        # If we can build settlements, we already took that action above
        # So if we're here, we can't build settlements but might be able to build roads

        # Only build roads if we don't have too many already
        # Having too many roads without settlements means wasted resources
        if len(road_actions) > 0 and len(road_actions) <= 3:
            # Only build if we have limited road options (early game expansion)
            return ACTION_BUILD_ROAD_START + np.random.choice(road_actions)

        # Check for dev card (gives VP or utility)
        if action_mask[ACTION_BUY_DEV_CARD]:
            return ACTION_BUY_DEV_CARD

        # Build more roads only if we can't do anything else productive
        if len(road_actions) > 0:
            return ACTION_BUILD_ROAD_START + np.random.choice(road_actions)

        # Build-targeted bank/port trade: convert a surplus resource into what's blocking
        # the next build, instead of hoarding a bad mix. Only reached when we can't build
        # anything directly. Self resources are decodable from the perspective player row
        # (row 146, cols 4-8 = count*0.1, capped at 1.0). The legal BANK_TRADE actions in
        # the mask already encode affordability AND the best port ratio, so we just pick
        # give = a resource the target build doesn't need, receive = the biggest deficit.
        # Resource order (obs cols & costs): LUMBER, BRICK, WOOL, GRAIN, ORE.
        SELF_ROW, RES0 = 146, 4
        hand = [int(round(observation[SELF_ROW, RES0 + r] * 10)) for r in range(5)]
        if sum(hand) >= 8:  # guard: don't trade a thin hand
            # (name, cost) in priority order: city > settlement > dev card > road
            build_costs = [
                ("city", [0, 0, 0, 2, 3]),
                ("settlement", [1, 1, 1, 1, 0]),
                ("dev", [0, 0, 1, 1, 1]),
                ("road", [1, 1, 0, 0, 0]),
            ]
            bank_offsets = np.where(
                action_mask[ACTION_BANK_TRADE_START:ACTION_BANK_TRADE_END]
            )[0]
            # decode legal bank trades -> (give_idx, receive_idx, action)
            legal_trades = [(int(o) // 5, int(o) % 5, ACTION_BANK_TRADE_START + int(o))
                            for o in bank_offsets]
            for _name, cost in build_costs:
                deficit = [max(0, cost[r] - hand[r]) for r in range(5)]
                total_deficit = sum(deficit)
                if total_deficit == 0 or total_deficit > 2:
                    continue  # already affordable (handled above), or too far this turn
                # give a resource the target build does NOT need (pure surplus);
                # receive a deficit resource. Prefer biggest deficit, then biggest surplus.
                cands = [(deficit[recv], hand[give], act)
                         for (give, recv, act) in legal_trades
                         if deficit[recv] > 0 and cost[give] == 0]
                if cands:
                    cands.sort(reverse=True)
                    return cands[0][2]

        # End turn if possible
        if action_mask[ACTION_END_TURN]:
            return ACTION_END_TURN

        # Fallback to any valid action
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return valid_actions[0]

        return 0


class ProductionGreedyAgent(BaseAgent):
    """Agent that prioritizes high-production settlement locations.

    This agent uses the observation to evaluate settlement locations
    based on their production value.
    """

    def __init__(self, exploration_prob: float = 0.1):
        """Initialize the production-greedy agent.

        Args:
            exploration_prob: Probability of taking a random action.
        """
        self.exploration_prob = exploration_prob

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select action prioritizing production value."""
        # Exploration
        if not deterministic and np.random.random() < self.exploration_prob:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)

        # Must roll dice first
        if action_mask[ACTION_ROLL_DICE]:
            return ACTION_ROLL_DICE

        # Setup phase settlements - evaluate production from observation
        setup_settlements = np.where(
            action_mask[ACTION_PLACE_SETTLEMENT_START:ACTION_PLACE_SETTLEMENT_END]
        )[0]
        if len(setup_settlements) > 0:
            # Get production values from observation
            # Corner features start at row 20, production value is feature ~15
            best_idx = 0
            best_prod = -1.0
            for i, corner_id in enumerate(setup_settlements):
                # Row for this corner in observation
                obs_row = 20 + corner_id
                if obs_row < observation.shape[0]:
                    # Production value feature (approximate index)
                    prod = observation[obs_row, 15] if observation.shape[1] > 15 else 0
                    if prod > best_prod:
                        best_prod = prod
                        best_idx = i
            return ACTION_PLACE_SETTLEMENT_START + setup_settlements[best_idx]

        # Main phase settlements
        settlement_actions = np.where(
            action_mask[ACTION_BUILD_SETTLEMENT_START:ACTION_BUILD_SETTLEMENT_END]
        )[0]
        if len(settlement_actions) > 0:
            best_idx = 0
            best_prod = -1.0
            for i, corner_id in enumerate(settlement_actions):
                obs_row = 20 + corner_id
                if obs_row < observation.shape[0]:
                    prod = observation[obs_row, 15] if observation.shape[1] > 15 else 0
                    if prod > best_prod:
                        best_prod = prod
                        best_idx = i
            return ACTION_BUILD_SETTLEMENT_START + settlement_actions[best_idx]

        # City > Dev Card > Road > End Turn
        city_actions = np.where(
            action_mask[ACTION_BUILD_CITY_START:ACTION_BUILD_CITY_END]
        )[0]
        if len(city_actions) > 0:
            return ACTION_BUILD_CITY_START + city_actions[0]

        if action_mask[ACTION_BUY_DEV_CARD]:
            return ACTION_BUY_DEV_CARD

        road_actions = np.where(
            action_mask[ACTION_BUILD_ROAD_START:ACTION_BUILD_ROAD_END]
        )[0]
        if len(road_actions) > 0:
            return ACTION_BUILD_ROAD_START + road_actions[0]

        if action_mask[ACTION_END_TURN]:
            return ACTION_END_TURN

        # Fallback
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return valid_actions[0]

        return 0


class HeuristicAgent(BaseAgent):
    """More sophisticated agent using multiple heuristics.

    Considers:
    - Production value for settlements
    - Resource diversity
    - Port access
    - Road connectivity for expansion
    """

    def __init__(self, exploration_prob: float = 0.05):
        """Initialize the heuristic agent.

        Args:
            exploration_prob: Probability of taking a random action.
        """
        self.exploration_prob = exploration_prob

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select action using heuristics."""
        # Exploration
        if not deterministic and np.random.random() < self.exploration_prob:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)

        # Must roll dice first
        if action_mask[ACTION_ROLL_DICE]:
            return ACTION_ROLL_DICE

        # Setup phase - use scoring function
        setup_settlements = np.where(
            action_mask[ACTION_PLACE_SETTLEMENT_START:ACTION_PLACE_SETTLEMENT_END]
        )[0]
        if len(setup_settlements) > 0:
            best_action = self._score_settlement_locations(
                observation, setup_settlements, is_setup=True
            )
            return ACTION_PLACE_SETTLEMENT_START + best_action

        # Main phase priorities
        # 1. Build city if we have settlements
        city_actions = np.where(
            action_mask[ACTION_BUILD_CITY_START:ACTION_BUILD_CITY_END]
        )[0]
        if len(city_actions) > 0:
            return ACTION_BUILD_CITY_START + city_actions[0]

        # 2. Build settlement at good location
        settlement_actions = np.where(
            action_mask[ACTION_BUILD_SETTLEMENT_START:ACTION_BUILD_SETTLEMENT_END]
        )[0]
        if len(settlement_actions) > 0:
            best_action = self._score_settlement_locations(
                observation, settlement_actions, is_setup=False
            )
            return ACTION_BUILD_SETTLEMENT_START + best_action

        # 3. Buy dev card (if we have resources)
        if action_mask[ACTION_BUY_DEV_CARD]:
            return ACTION_BUY_DEV_CARD

        # 4. Build roads toward expansion
        road_actions = np.where(
            action_mask[ACTION_BUILD_ROAD_START:ACTION_BUILD_ROAD_END]
        )[0]
        if len(road_actions) > 0:
            # Pick road that connects toward good settlement spots
            return ACTION_BUILD_ROAD_START + road_actions[0]

        # 5. End turn
        if action_mask[ACTION_END_TURN]:
            return ACTION_END_TURN

        # Fallback
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return valid_actions[0]

        return 0

    def _score_settlement_locations(
        self,
        observation: np.ndarray,
        valid_corners: np.ndarray,
        is_setup: bool
    ) -> int:
        """Score settlement locations and return best corner index."""
        best_score = -float('inf')
        best_idx = 0

        for i, corner_id in enumerate(valid_corners):
            score = self._evaluate_corner(observation, corner_id, is_setup)
            if score > best_score:
                best_score = score
                best_idx = i

        return valid_corners[best_idx]

    def _evaluate_corner(
        self,
        observation: np.ndarray,
        corner_id: int,
        is_setup: bool
    ) -> float:
        """Evaluate a corner for settlement placement."""
        score = 0.0

        # Get corner features from observation (row 20 + corner_id)
        obs_row = 20 + corner_id
        if obs_row >= observation.shape[0]:
            return score

        corner_features = observation[obs_row]

        # Production value (heavily weighted)
        if len(corner_features) > 15:
            production = corner_features[15]
            score += production * 10.0

        # Port bonus
        if len(corner_features) > 10:
            # Check if port features are set
            has_port = any(corner_features[4:10])
            if has_port:
                score += 2.0

        # Resource diversity (from individual resource production features)
        if len(corner_features) > 22:
            resource_prods = corner_features[17:22]
            diversity = sum(1 for p in resource_prods if p > 0)
            score += diversity * 0.5

        return score


def evaluate_agent(
    agent: BaseAgent,
    num_episodes: int = 100,
    opponent_type: str = "random"
) -> dict:
    """Evaluate an agent against specified opponents.

    Args:
        agent: The agent to evaluate.
        num_episodes: Number of episodes to run.
        opponent_type: Type of opponents ("random", "greedy", or "self").

    Returns:
        Dictionary with evaluation metrics.
    """
    from ..environment.catan_env import CatanEnv

    # Create opponent agents
    if opponent_type == "random":
        opponents = [RandomAgent() for _ in range(3)]
    elif opponent_type == "greedy":
        opponents = [GreedyAgent() for _ in range(3)]
    elif opponent_type == "self":
        opponents = [agent for _ in range(3)]
    else:
        opponents = [RandomAgent() for _ in range(3)]

    env = CatanEnv(opponent_policies=opponents)

    wins = 0
    total_vp = 0
    total_turns = 0
    total_rewards = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action_mask = env.action_masks()
            action = agent.predict(obs, action_mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        if info.get("is_agent_winner", False):
            wins += 1
        total_vp += info.get("vp", 0)
        total_turns += info.get("turn", 0)
        total_rewards += episode_reward

    return {
        "win_rate": wins / num_episodes,
        "avg_vp": total_vp / num_episodes,
        "avg_turns": total_turns / num_episodes,
        "avg_reward": total_rewards / num_episodes,
        "num_episodes": num_episodes,
    }
