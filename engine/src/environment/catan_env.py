"""Gymnasium environment for Settlers of Catan.

OPTIMIZATION NOTES:
- Lazy state copying: Only copy state when needed for reward calculation
- Cached action masks: Avoid redundant computation
- Pre-computed owned edges/corners: Passed to validation functions
"""

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..game.game_state import GameState
from ..game.rules import Rules
from ..game.actions import TOTAL_ACTIONS, index_to_action
from ..game.constants import NUM_PLAYERS, GamePhase, ActionType

from .state_encoder import StateEncoder
from .action_decoder import ActionDecoder
from .reward import RewardCalculator, RewardConfig


class CatanEnv(gym.Env):
    """Gymnasium environment for Settlers of Catan.

    This environment supports MaskablePPO through the action_masks() method.
    The observation is a 2D array encoding the game state from the current
    player's perspective.

    Supports player-to-player trading with:
    - Trade proposals from MAIN phase
    - Opponent responses (accept/reject/counter)
    - Proposer confirmation of accepting players

    Attributes:
        observation_space: Box space of shape (176, 65)
        action_space: Discrete space of size 475
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        opponent_policies: Optional[List[Any]] = None,
        reward_config: Optional[RewardConfig] = None,
        max_turns: int = 500,
        render_mode: Optional[str] = None,
        agent_position: Optional[int] = None,
        opponent_sampler: Optional[callable] = None,
        enable_p2p_trading: bool = True,
    ):
        """Initialize the Catan environment.

        Args:
            opponent_policies: List of 3 opponent policies. If None, uses random.
            reward_config: Configuration for rewards.
            max_turns: Maximum turns before episode ends.
            render_mode: Rendering mode ("human" or "ansi").
            agent_position: Fixed position for the agent (0-3), or None for random.
            opponent_sampler: Optional callable that returns (opponents, opponent_infos)
                on each episode reset. Used for self-play to sample from opponent pool.
            enable_p2p_trading: Whether to allow player-to-player trading (default True).
                Set to False during early training to focus on core mechanics.
        """
        super().__init__()

        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        self.reward_calculator = RewardCalculator(reward_config)
        self.opponent_policies = opponent_policies or [None, None, None]
        self.max_turns = max_turns
        self.enable_p2p_trading = enable_p2p_trading
        self.render_mode = render_mode
        self.opponent_sampler = opponent_sampler

        # Track opponent metadata for match logging
        self.opponent_infos = self._build_opponent_infos(opponent_policies)

        # Agent position: None = random each episode, 0-3 = fixed position
        self._agent_position_config = agent_position
        if agent_position is not None and not (0 <= agent_position < NUM_PLAYERS):
            raise ValueError(f"agent_position must be 0-{NUM_PLAYERS-1} or None, got {agent_position}")

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.state_encoder.observation_shape,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Game state
        self.state: Optional[GameState] = None
        self.agent_player_id: int = 0  # Set in reset()

        # Episode tracking
        self._episode_reward = 0.0
        self._episode_length = 0

    def set_p2p_trading(self, enabled: bool) -> None:
        """Toggle P2P trading at runtime (for curriculum learning)."""
        self.enable_p2p_trading = enabled

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options. Supports:
                - "agent_position": Override agent position for this episode (0-3)

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)

        # Sample new opponents if sampler provided (for self-play)
        if self.opponent_sampler is not None:
            self.opponent_policies, self.opponent_infos = self.opponent_sampler()

        # Initialize new game
        self.state = GameState()

        # Determine agent position for this episode
        if options and "agent_position" in options:
            # Override from options
            self.agent_player_id = options["agent_position"]
        elif self._agent_position_config is not None:
            # Use fixed position from config
            self.agent_player_id = self._agent_position_config
        else:
            # Random position (default)
            self.agent_player_id = np.random.randint(0, NUM_PLAYERS)

        self._episode_reward = 0.0
        self._episode_length = 0

        # If it's not the agent's turn, play opponents until it is
        self._play_opponents_until_agent_turn()

        observation = self._get_observation()
        info = self._get_info()
        info["agent_position"] = self.agent_player_id

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: The action index to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

        OPTIMIZATION: Uses lightweight state snapshot for reward calculation
        instead of full deep copy.
        """
        assert self.state is not None, "Must call reset() before step()"

        # Validate action
        action_mask = Rules.get_legal_actions(self.state)
        if not action_mask[action]:
            # Invalid action - return negative reward and don't change state
            observation = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            return observation, -1.0, False, False, info

        # OPTIMIZATION: Capture only the fields needed for reward calculation
        # instead of full state.copy()
        agent_player = self.state.players[self.agent_player_id]
        prev_agent_vp = agent_player.total_victory_points
        prev_agent_resources = agent_player.total_resources
        prev_settlements = len(agent_player.settlements)
        prev_cities = len(agent_player.cities)
        prev_roads = len(agent_player.roads)
        prev_longest_road = agent_player.has_longest_road
        prev_largest_army = agent_player.has_largest_army
        # Capture IDs for strategic shaping
        prev_settlement_ids = list(agent_player.settlements)
        prev_road_ids = list(agent_player.roads)

        # Execute action
        action_obj = index_to_action(action)
        Rules.execute_action(self.state, action)

        # Calculate reward using lightweight snapshot
        # Robber rewards use VP-based heuristic; strategic context in policy
        # provides richer threat signals via competition features
        reward = self.reward_calculator.calculate_reward_fast(
            prev_vp=prev_agent_vp,
            prev_resources=prev_agent_resources,
            prev_settlements=prev_settlements,
            prev_cities=prev_cities,
            prev_roads=prev_roads,
            prev_longest_road=prev_longest_road,
            prev_largest_army=prev_largest_army,
            current_state=self.state,
            action_type=action_obj.type,
            agent_player_id=self.agent_player_id,
            prev_settlement_ids=prev_settlement_ids,
            prev_road_ids=prev_road_ids,
        )
        self._episode_reward += reward

        # Check for termination
        terminated = self.state.is_game_over
        truncated = self._episode_length >= self.max_turns

        if not terminated and not truncated:
            # Play opponents until it's the agent's turn again
            self._play_opponents_until_agent_turn()

            # Check again after opponents played
            terminated = self.state.is_game_over
            truncated = self._episode_length >= self.max_turns

            # Add terminal reward if game ended during opponent turns
            # (opponent won - agent's reward was calculated before this)
            if terminated:
                terminal_reward = self.reward_calculator.calculate_terminal_reward(
                    self.state, self.agent_player_id
                )
                reward += terminal_reward

        # Add terminal reward for truncated episodes (game timeout)
        # Without this, agent gets no win/loss signal when games don't finish
        if truncated and not terminated:
            terminal_reward = self.reward_calculator.calculate_terminal_reward(
                self.state, self.agent_player_id
            )
            reward += terminal_reward

        self._episode_length += 1

        observation = self._get_observation()
        info = self._get_info()

        # Add game completion status for accurate stats tracking
        # Only count games where someone actually won (not truncated/timeout)
        if terminated or truncated:
            info["game_completed"] = terminated  # True if someone won, False if timeout
            info["game_truncated"] = truncated and not terminated

        return observation, reward, terminated, truncated, info

    def _play_opponents_until_agent_turn(self) -> None:
        """Play opponent turns until it's the agent's turn.

        Handles trade phases where:
        - During TRADE_RESPONSE: If agent is the current responder, return to let agent respond
        - During TRADE_CONFIRM: If agent is the proposer, return to let agent confirm
        - Otherwise, play opponents' trade actions automatically
        """
        max_opponent_actions = 1000  # Safety limit
        actions_taken = 0

        while (
            not self.state.is_game_over and
            self._should_continue_opponent_play() and
            actions_taken < max_opponent_actions
        ):
            opponent_id = self.state.current_player_idx
            action_mask = Rules.get_legal_actions(self.state)

            # Get opponent action
            opponent_policy = self._get_opponent_policy(opponent_id)

            # Filter out P2P trading actions for opponents if disabled
            if not self.enable_p2p_trading:
                from ..game.actions import (
                    ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END,
                    ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END,
                    ACTION_CANCEL_TRADE,
                    ACTION_CONFIRM_TRADE_START, ACTION_CONFIRM_TRADE_END,
                    ACTION_DECLINE_TRADE_START, ACTION_DECLINE_TRADE_END,
                )
                action_mask = action_mask.copy()
                action_mask[ACTION_PROPOSE_TRADE_START:ACTION_PROPOSE_TRADE_END] = False
                action_mask[ACTION_COUNTER_OFFER_START:ACTION_COUNTER_OFFER_END] = False
                action_mask[ACTION_CANCEL_TRADE] = False
                action_mask[ACTION_CONFIRM_TRADE_START:ACTION_CONFIRM_TRADE_END] = False
                action_mask[ACTION_DECLINE_TRADE_START:ACTION_DECLINE_TRADE_END] = False

            if opponent_policy is not None:
                # Use opponent policy
                obs = self.state_encoder.encode(self.state, opponent_id)
                action = opponent_policy.predict(obs, action_mask)
            else:
                # Random action (with trade-aware fallback)
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break
                action = self._select_random_action(valid_actions, action_mask)

            # Execute opponent action
            Rules.execute_action(self.state, action)
            actions_taken += 1

    def _should_continue_opponent_play(self) -> bool:
        """Check if we should continue playing opponents.

        Returns False when the agent should take control:
        - Normal turn: when current_player_idx is the agent
        - TRADE_RESPONSE phase: when agent is the current responder
        - TRADE_CONFIRM phase: when agent is the proposer (confirming trade)
        """
        # If it's the agent's normal turn
        if self.state.current_player_idx == self.agent_player_id:
            return False

        # In TRADE_RESPONSE, check if agent is the current responder
        if self.state.phase == GamePhase.TRADE_RESPONSE:
            if self.state.current_responder_idx == self.agent_player_id:
                return False

        # TRADE_CONFIRM is handled by current_player_idx check above
        # (proposer is set as current_player during confirm phase)

        return True

    def _get_opponent_policy(self, opponent_id: int):
        """Get the policy for an opponent player.

        Args:
            opponent_id: The opponent's player ID.

        Returns:
            Opponent policy or None for random play.
        """
        # opponent_policies is indexed 0-2 for the 3 opponents
        # We need to map the absolute player ID to the opponent index
        if opponent_id == self.agent_player_id:
            return None  # Should not happen

        # Use cached mapping for O(1) lookup instead of rebuilding list each time
        opponent_idx = self._get_opponent_index(opponent_id)
        if opponent_idx is not None and self.opponent_policies and opponent_idx < len(self.opponent_policies):
            return self.opponent_policies[opponent_idx]

        return None

    def _get_opponent_index(self, opponent_id: int) -> Optional[int]:
        """Get the opponent index (0-2) for a player ID using cached mapping.

        Args:
            opponent_id: The opponent's player ID.

        Returns:
            Opponent index 0-2, or None if opponent_id is the agent.
        """
        if opponent_id == self.agent_player_id:
            return None

        # Lazy-build cache on first call
        if not hasattr(self, '_opponent_idx_cache') or self._opponent_idx_cache_agent != self.agent_player_id:
            self._opponent_idx_cache = {}
            self._opponent_idx_cache_agent = self.agent_player_id
            idx = 0
            for i in range(NUM_PLAYERS):
                if i != self.agent_player_id:
                    self._opponent_idx_cache[i] = idx
                    idx += 1

        return self._opponent_idx_cache.get(opponent_id)

    def _select_random_action(self, valid_actions: np.ndarray, action_mask: np.ndarray) -> int:
        """Select a random action, with trade-aware heuristics for opponents.

        For trade phases, opponents have simple heuristics:
        - Accept trades that seem favorable (receiving more than giving)
        - Usually reject unfavorable trades
        - Rarely counter-offer

        Args:
            valid_actions: Array of valid action indices.
            action_mask: Full action mask.

        Returns:
            Selected action index.
        """
        from ..game.actions import (
            ACTION_ACCEPT_TRADE, ACTION_REJECT_TRADE,
            ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END
        )

        # For trade response, use simple heuristics
        if self.state.phase == GamePhase.TRADE_RESPONSE:
            # 70% chance to reject, 25% accept (if valid), 5% counter
            roll = np.random.random()
            if roll < 0.70 and ACTION_REJECT_TRADE in valid_actions:
                return ACTION_REJECT_TRADE
            elif roll < 0.95 and ACTION_ACCEPT_TRADE in valid_actions:
                return ACTION_ACCEPT_TRADE
            # Fall through to random for counter-offers

        return np.random.choice(valid_actions)

    def _get_observation(self) -> np.ndarray:
        """Get the current observation for the agent."""
        return self.state_encoder.encode(self.state, self.agent_player_id)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        from ..game.constants import Resource

        player = self.state.players[self.agent_player_id]

        # Get opponent info (players other than agent)
        opponent_vps = []
        opponent_total_vps = []
        opponent_cards = []
        opponent_dev_cards = []
        for i in range(NUM_PLAYERS):
            if i != self.agent_player_id:
                opponent_vps.append(self.state.players[i].victory_points)
                opponent_total_vps.append(self.state.players[i].total_victory_points)
                opponent_cards.append(self.state.players[i].total_resources)
                opponent_dev_cards.append(sum(self.state.players[i].dev_cards.values()))

        # Get all player card counts (indexed by player ID)
        all_player_cards = [p.total_resources for p in self.state.players]
        all_player_dev_cards = [sum(p.dev_cards.values()) for p in self.state.players]

        # Get bank resources
        bank_resources = {
            r.name: self.state.bank_resources.get(r, 0)
            for r in Resource.tradeable()
        }
        bank_total = sum(bank_resources.values())

        return {
            "turn": self.state.turn_number,
            "phase": self.state.phase.name,
            "vp": player.victory_points,
            "total_vp": player.total_victory_points,
            "resources": player.total_resources,
            "settlements": len(player.settlements),
            "cities": len(player.cities),
            "roads": len(player.roads),
            "dev_cards": sum(player.dev_cards.values()),
            "episode_reward": self._episode_reward,
            "episode_length": self._episode_length,
            "winner": self.state.winner,
            "is_agent_winner": self.state.winner == self.agent_player_id if self.state.is_game_over else None,
            # PHASE 3: Victory path labels for auxiliary head training
            # These are set at game end to indicate who achieved largest army / longest road
            "largest_army_holder": self.state.largest_army_holder,
            "longest_road_holder": self.state.longest_road_holder,
            # Opponent info for match tracking
            "opponent_vps": opponent_vps,  # Visible VP only
            "opponent_total_vps": opponent_total_vps,  # Including hidden VP cards
            "opponent_cards": opponent_cards,  # Resource cards per opponent
            "opponent_dev_cards": opponent_dev_cards,  # Dev cards per opponent
            "opponent_infos": self.opponent_infos,
            # All player card counts (indexed by player ID 0-3)
            "all_player_cards": all_player_cards,  # Resource cards per player
            "all_player_dev_cards": all_player_dev_cards,  # Dev cards per player
            # Bank resource tracking
            "bank_resources": bank_resources,  # Dict of resource name -> count
            "bank_total": bank_total,  # Total cards in bank
            # Placement (1-4) based on VP ranking
            "placement": self._calculate_placement(player.total_victory_points, opponent_total_vps),
        }

    def _calculate_placement(self, agent_vp: int, opponent_vps: List[int]) -> int:
        """Calculate agent's placement (1-4) based on VP ranking."""
        all_vps = [agent_vp] + list(opponent_vps)
        sorted_vps = sorted(all_vps, reverse=True)
        return sorted_vps.index(agent_vp) + 1

    def action_masks(self) -> np.ndarray:
        """Get the action mask for valid actions.

        This method is required for MaskablePPO.

        Returns:
            Boolean array where True indicates a valid action.
        """
        mask = Rules.get_legal_actions(self.state)

        # Ensure mask is proper boolean array
        mask = np.asarray(mask, dtype=bool)

        # Disable P2P trading actions if not enabled
        if not self.enable_p2p_trading:
            from ..game.actions import (
                ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END,
                ACTION_ACCEPT_TRADE, ACTION_REJECT_TRADE,
                ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END,
                ACTION_CANCEL_TRADE,
                ACTION_CONFIRM_TRADE_START, ACTION_CONFIRM_TRADE_END,
                ACTION_DECLINE_TRADE_START, ACTION_DECLINE_TRADE_END,
            )
            # Mask all P2P trading actions
            mask[ACTION_PROPOSE_TRADE_START:ACTION_PROPOSE_TRADE_END] = False
            mask[ACTION_ACCEPT_TRADE] = False
            mask[ACTION_COUNTER_OFFER_START:ACTION_COUNTER_OFFER_END] = False
            mask[ACTION_CANCEL_TRADE] = False
            mask[ACTION_CONFIRM_TRADE_START:ACTION_CONFIRM_TRADE_END] = False
            mask[ACTION_DECLINE_TRADE_START:ACTION_DECLINE_TRADE_END] = False
            # Keep REJECT_TRADE enabled so agent can reject opponent offers
            # (This prevents getting stuck in TRADE_RESPONSE phase)

        # Safety check: ensure at least one action is valid to prevent crashes
        # This is critical for numerical stability in MaskableCategorical
        if not mask.any():
            # This shouldn't happen, but if it does, allow END_TURN as fallback
            from ..game.actions import ACTION_END_TURN
            mask[ACTION_END_TURN] = True
            import warnings
            warnings.warn(f"No valid actions in phase {self.state.phase}, allowing END_TURN as fallback")

        return mask

    def render(self) -> Optional[str]:
        """Render the environment.

        Returns:
            String representation if mode is "ansi", None otherwise.
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render the game state as an ASCII string."""
        from ..game.constants import Resource

        lines = []
        lines.append(f"=== Catan Game - Turn {self.state.turn_number} ===")
        lines.append(f"Phase: {self.state.phase.name}")
        lines.append(f"Current Player: {self.state.current_player_idx}")
        lines.append(f"Last Dice: {self.state.last_dice_roll} = {self.state.get_dice_total()}")
        lines.append("")

        # Player summary with card counts
        lines.append("Players:")
        for i, player in enumerate(self.state.players):
            marker = "*" if i == self.state.current_player_idx else " "
            agent = "(AGENT)" if i == self.agent_player_id else ""
            lines.append(
                f"{marker} P{i}{agent}: VP={player.victory_points}({player.total_victory_points}), "
                f"S={len(player.settlements)}, C={len(player.cities)}, "
                f"R={len(player.roads)}, Cards={player.total_resources}, "
                f"DevCards={sum(player.dev_cards.values())}"
            )

        lines.append("")

        # Bank resources
        bank_parts = []
        for r in Resource.tradeable():
            count = self.state.bank_resources.get(r, 0)
            # Highlight scarce resources
            if count < 5:
                bank_parts.append(f"{r.name[:3]}={count}!")
            else:
                bank_parts.append(f"{r.name[:3]}={count}")
        bank_total = sum(self.state.bank_resources.get(r, 0) for r in Resource.tradeable())
        lines.append(f"Bank ({bank_total}): {', '.join(bank_parts)}")

        # Robber location
        robber_tile = self.state.board.get_robber_tile()
        lines.append(f"Robber on tile {robber_tile}")

        # Trade state (if active)
        if self.state.pending_trade is not None:
            trade = self.state.pending_trade
            offer_str = ", ".join(f"{a} {r.name}" for r, a in trade.offering.items() if a > 0)
            request_str = ", ".join(f"{a} {r.name}" for r, a in trade.requesting.items() if a > 0)
            lines.append(f"Pending Trade: P{trade.proposer_id} offers [{offer_str}] for [{request_str}]")
            if self.state.accepting_players:
                lines.append(f"  Accepting: {self.state.accepting_players}")

        # Valid actions
        mask = self.action_masks()
        valid_count = mask.sum()
        lines.append(f"Valid actions: {valid_count}")

        return "\n".join(lines)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def _build_opponent_infos(self, opponent_policies: Optional[List]) -> List[Dict]:
        """Build metadata about opponent policies.

        Args:
            opponent_policies: List of opponent policy objects.

        Returns:
            List of dicts with agent_type and model_path.
        """
        if opponent_policies is None:
            return [{"agent_type": "random"} for _ in range(3)]

        infos = []
        for policy in opponent_policies:
            if policy is None:
                infos.append({"agent_type": "random"})
            elif hasattr(policy, '__class__'):
                class_name = policy.__class__.__name__
                if class_name == "RandomAgent":
                    infos.append({"agent_type": "random"})
                elif class_name == "GreedyAgent":
                    infos.append({"agent_type": "greedy"})
                elif class_name == "SelfPlayAgent":
                    # Try to get model path from SelfPlayAgent
                    model_path = getattr(policy, 'model_path', None)
                    infos.append({
                        "agent_type": "pool",
                        "model_path": model_path,
                    })
                else:
                    infos.append({"agent_type": "unknown"})
            else:
                infos.append({"agent_type": "unknown"})

        return infos

    def set_opponent_infos(self, opponent_infos: List[Dict]) -> None:
        """Set opponent info metadata (for self-play tracking).

        Args:
            opponent_infos: List of dicts with agent_type, model_path, etc.
        """
        self.opponent_infos = opponent_infos

    def get_state(self) -> GameState:
        """Get the current game state (for testing/debugging)."""
        return self.state

    def set_state(self, state: GameState) -> None:
        """Set the game state (for testing/debugging)."""
        self.state = state


class CatanMultiAgentEnv(CatanEnv):
    """Multi-agent version where all players are controlled externally.

    This is useful for self-play training where all players use learning agents.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        max_turns: int = 500,
        render_mode: Optional[str] = None,
    ):
        """Initialize the multi-agent environment.

        Args:
            reward_config: Configuration for rewards.
            max_turns: Maximum turns before episode ends.
            render_mode: Rendering mode.
        """
        super().__init__(
            opponent_policies=None,
            reward_config=reward_config,
            max_turns=max_turns,
            render_mode=render_mode
        )
        self.num_agents = NUM_PLAYERS

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset for multi-agent play."""
        if seed is not None:
            np.random.seed(seed)

        self.state = GameState()
        self._episode_reward = 0.0
        self._episode_length = 0

        # In multi-agent, we return observation for current player
        observation = self._get_observation_for_player(self.state.current_player_idx)
        info = self._get_info()
        info["current_player"] = self.state.current_player_idx

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step for the current player.

        Returns observation for the NEXT player after the action.
        """
        current_player = self.state.current_player_idx
        prev_state = self.state.copy()

        # Validate and execute action
        action_mask = Rules.get_legal_actions(self.state)
        if not action_mask[action]:
            observation = self._get_observation_for_player(current_player)
            info = self._get_info()
            info["invalid_action"] = True
            info["current_player"] = current_player
            return observation, -1.0, False, False, info

        action_obj = index_to_action(action)
        Rules.execute_action(self.state, action)

        # Calculate reward for the player who acted
        reward = self.reward_calculator.calculate_reward(
            prev_state, self.state, action_obj.type, current_player
        )

        terminated = self.state.is_game_over
        truncated = self._episode_length >= self.max_turns
        self._episode_length += 1

        # Return observation for the next player
        next_player = self.state.current_player_idx
        observation = self._get_observation_for_player(next_player)
        info = self._get_info()
        info["acting_player"] = current_player
        info["current_player"] = next_player
        info["reward_for_player"] = {current_player: reward}
        info["action_type"] = action_obj.type
        info["action_details"] = {"action_index": action}
        info["dense_reward"] = reward

        return observation, reward, terminated, truncated, info

    def _get_observation_for_player(self, player_id: int) -> np.ndarray:
        """Get observation from a specific player's perspective."""
        return self.state_encoder.encode(self.state, player_id)

    def get_rewards_for_all_players(self) -> Dict[int, float]:
        """Get terminal rewards for all players (call at episode end)."""
        rewards = {}
        for player_id in range(NUM_PLAYERS):
            rewards[player_id] = self.reward_calculator.calculate_terminal_reward(
                self.state, player_id
            )
        return rewards
