"""
Multi-agent wrapper for Catan environment.

Enables single-policy self-play where one policy plays all 4 positions,
collecting experiences from all players' perspectives for 4x training data.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from src.environment.catan_env import CatanMultiAgentEnv
from src.environment.state_encoder import StateEncoder
from src.environment.reward import RewardConfig
from src.game.rules import Rules
from src.training.game_logger import GameLogger, GameLoggerConfig


@dataclass
class GameStats:
    """Statistics from a completed game."""
    game_length: int = 0
    winner: Optional[int] = None
    timed_out: bool = False
    final_vps: List[int] = field(default_factory=list)
    settlements_built: List[int] = field(default_factory=list)
    cities_built: List[int] = field(default_factory=list)
    roads_built: List[int] = field(default_factory=list)
    dev_cards_bought: List[int] = field(default_factory=list)
    trades_completed: int = 0
    longest_road_holder: Optional[int] = None
    largest_army_holder: Optional[int] = None
    starting_pips: List[int] = field(default_factory=list)  # Pip count from opening settlements
    learning_players: List[int] = field(default_factory=list)  # Which players were learning agents


@dataclass
class PlayerTransition:
    """A single transition from one player's perspective."""
    player_id: int
    observation: np.ndarray
    action: int
    action_mask: np.ndarray
    reward: float = 0.0
    done: bool = False
    value: float = 0.0
    log_prob: float = 0.0


@dataclass
class GameBuffer:
    """Buffer to store all transitions from a single game."""
    transitions: Dict[int, List[PlayerTransition]] = field(
        default_factory=lambda: {i: [] for i in range(4)}
    )

    def add(self, player_id: int, transition: PlayerTransition):
        self.transitions[player_id].append(transition)

    def get_all_transitions(self) -> List[PlayerTransition]:
        """Get all transitions from all players, flattened."""
        all_trans = []
        for pid in range(4):
            all_trans.extend(self.transitions[pid])
        return all_trans

    def assign_final_rewards(self, rewards: Dict[int, float]):
        """Assign final game rewards to each player's last transition."""
        for pid, reward in rewards.items():
            if self.transitions[pid]:
                self.transitions[pid][-1].reward += reward
                self.transitions[pid][-1].done = True

    def clear(self):
        self.transitions = {i: [] for i in range(4)}


class BCPolicyWrapper:
    """Wrapper to make HierarchicalBCModel compatible with pool opponent interface.

    Note: The BC training data is missing REJECT_TRADE and other trade actions,
    so we completely disable player-to-player trading for BC opponents.
    BC model is only used for building/playing decisions.
    """

    # Action indices for trading (to be masked out)
    ACTION_PROPOSE_TRADE_START = 366
    ACTION_PROPOSE_TRADE_END = 416
    ACTION_ACCEPT_TRADE = 416
    ACTION_REJECT_TRADE = 417
    ACTION_COUNTER_OFFER_START = 418
    ACTION_COUNTER_OFFER_END = 468
    ACTION_CANCEL_TRADE = 468
    ACTION_CONFIRM_TRADE_START = 469
    ACTION_CONFIRM_TRADE_END = 472
    ACTION_DECLINE_TRADE_START = 472
    ACTION_DECLINE_TRADE_END = 475

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def parameters(self):
        """Return model parameters (for device detection)."""
        return self.model.parameters()

    def __call__(self, obs_tensor, action_masks=None, deterministic=False):
        """Select action using BC model, matching CatanPolicy interface."""
        import torch
        import numpy as np

        if action_masks is not None:
            mask_np = action_masks.cpu().numpy()[0] if torch.is_tensor(action_masks) else action_masks[0]
            modified_mask = mask_np.copy()

            # Completely disable player-to-player trading for BC
            # Always reject incoming trades
            if mask_np[self.ACTION_REJECT_TRADE]:
                action = torch.tensor([self.ACTION_REJECT_TRADE])
                return action, torch.zeros(1), torch.zeros(1)

            # Mask out all trade initiation actions
            modified_mask[self.ACTION_PROPOSE_TRADE_START:self.ACTION_PROPOSE_TRADE_END] = False
            modified_mask[self.ACTION_COUNTER_OFFER_START:self.ACTION_COUNTER_OFFER_END] = False
            modified_mask[self.ACTION_CANCEL_TRADE] = False

            # Use modified mask - ensure it's on the same device as the model
            action_masks = torch.from_numpy(modified_mask).bool().unsqueeze(0).to(self.device)

        # Ensure obs_tensor is on the correct device
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            # Extract features
            features = self.model.extract_features(obs_tensor)

            # Use hierarchical actor with stochastic sampling
            flat_actions, _ = self.model.hierarchical_actor.get_action(
                features, action_masks, deterministic=False
            )

            # Return action, value (dummy), log_prob (dummy) to match interface
            return flat_actions.cpu(), torch.zeros(1), torch.zeros(1)


class MultiAgentCatanEnv(gym.Env):
    """
    Wrapper that collects experiences from all 4 players' perspectives.

    A single policy plays all 4 seats. During each step:
    1. Current player takes action using shared policy
    2. Transition is recorded from that player's perspective
    3. Next player's observation is returned

    At game end, all 4 players' transitions are available for training.
    """

    def __init__(
        self,
        dense_reward_scale: float = 0.05,
        win_reward: float = 50.0,
        lose_reward: float = -25.0,
        max_turns: int = 500,
        enable_trading: bool = True,
        max_proposals_per_turn: int = 5,
        turn_penalty: float = 0.0,
        game_log_config: Optional[GameLoggerConfig] = None,
        num_learning_agents: int = 4,
        num_random_opponents: int = 0,
        num_greedy_opponents: int = 0,
        num_pool_opponents: int = 0,
        pool_paths: Optional[List[str]] = None,
    ):
        super().__init__()

        # Validate opponent configuration
        total = num_learning_agents + num_random_opponents + num_greedy_opponents + num_pool_opponents
        if total != 4:
            raise ValueError(f"Must have exactly 4 players, got {total} "
                           f"({num_learning_agents}L + {num_random_opponents}R + {num_greedy_opponents}G + {num_pool_opponents}P)")

        # Opponent configuration
        # Players 0 to num_learning_agents-1 are learning agents
        # Next num_random_opponents are pure random
        # Next num_greedy_opponents are greedy setup (best pips) then random
        # Next num_pool_opponents are from historical policy pool
        self.num_learning_agents = num_learning_agents
        self.num_random_opponents = num_random_opponents
        self.num_greedy_opponents = num_greedy_opponents
        self.num_pool_opponents = num_pool_opponents
        self.pool_paths = pool_paths or []

        # Determine player types by index
        self.learning_players = set(range(num_learning_agents))
        random_start = num_learning_agents
        random_end = random_start + num_random_opponents
        self.random_players = set(range(random_start, random_end))
        greedy_start = random_end
        greedy_end = greedy_start + num_greedy_opponents
        self.greedy_players = set(range(greedy_start, greedy_end))
        pool_start = greedy_end
        self.pool_players = set(range(pool_start, 4))

        # Load pool policies if needed
        self.pool_policies = {}
        if num_pool_opponents > 0 and pool_paths:
            self._load_pool_policies(pool_paths)

        # Create reward config
        reward_config = RewardConfig(
            win_reward=win_reward,
            lose_reward=lose_reward,
            dense_scale=dense_reward_scale,
        )

        # Create underlying multi-agent environment (we control all 4 players)
        self.env = CatanMultiAgentEnv(
            reward_config=reward_config,
            max_turns=max_turns,
        )

        # Store trading flag and limits
        self.enable_trading = enable_trading
        self.max_proposals_per_turn = max_proposals_per_turn
        self._proposals_this_turn = 0  # For learning agents
        self._non_learning_proposals = {i: 0 for i in range(4)}  # Per-player tracking
        self._non_learning_last_turn = {i: -1 for i in range(4)}  # Track turn resets
        self._last_turn_number = 0

        # Turn penalty to encourage faster games
        self.turn_penalty = turn_penalty

        # Store reward values for computing final rewards
        self.win_reward = win_reward
        self.lose_reward = lose_reward

        # Game logging
        self.game_logger: Optional[GameLogger] = None
        if game_log_config is not None:
            self.game_logger = GameLogger(game_log_config)

        # Copy spaces from underlying env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # State encoder for getting observations from any player's perspective
        self.encoder = StateEncoder()

        # Game buffer for collecting multi-agent experiences
        self.game_buffer = GameBuffer()

        # Track current observation and mask for policy
        self._current_obs: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None
        self._prev_player: Optional[int] = None

        # Statistics
        self.games_played = 0
        self.total_transitions = 0

        # Per-game stats tracking
        self._current_game_stats: Optional[GameStats] = None
        self._setup_complete: bool = False
        self._completed_game_stats: List[GameStats] = []

    def _load_pool_policies(self, pool_paths: List[str]) -> None:
        """Load policies from checkpoint paths for pool opponents.

        Supports both CatanPolicy checkpoints and HierarchicalBCModel checkpoints.
        """
        import torch
        from src.models.catan_policy import CatanPolicy
        import gymnasium as gym
        from src.game.actions import TOTAL_ACTIONS

        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

        obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(176, 65), dtype=np.float32)
        action_space = gym.spaces.Discrete(TOTAL_ACTIONS)

        for i, path in enumerate(pool_paths):
            if i >= self.num_pool_opponents:
                break
            player_id = list(self.pool_players)[i]
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Check if this is a BC model (has 'hierarchical' key or 'model_state_dict')
            if checkpoint.get('hierarchical', False) or 'model_state_dict' in checkpoint:
                # Load as BC model with wrapper
                self.pool_policies[player_id] = self._load_bc_policy(checkpoint, device)
                print(f"Loaded BC model for pool opponent {player_id} from {path}")
            else:
                # Load as CatanPolicy
                policy = CatanPolicy(
                    observation_space=obs_space,
                    action_space=action_space,
                    lr_schedule=lambda x: 3e-4,
                )
                if "policy_state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["policy_state_dict"])
                else:
                    policy.load_state_dict(checkpoint)
                policy.to(device)
                policy.eval()
                self.pool_policies[player_id] = policy
                print(f"Loaded CatanPolicy for pool opponent {player_id} from {path}")

    def _load_bc_policy(self, checkpoint: dict, device: torch.device):
        """Load a BC model and wrap it to provide CatanPolicy-like interface."""
        import torch
        from src.models.hierarchical_bc_model import HierarchicalBCModel
        from src.models.action_decomposer import ActionDecomposer

        # Get model config from checkpoint
        args = checkpoint.get('args', {})
        use_gnn = args.get('use_gnn', True)

        # Create and load model
        model = HierarchicalBCModel(use_gnn=use_gnn)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Return a wrapper that provides compatible interface
        return BCPolicyWrapper(model, device)

    def _get_non_learning_action(self, player_id: int) -> int:
        """Get action for a non-learning player (random, greedy, or pool)."""
        import torch
        from src.game.constants import GamePhase, Resource
        from src.game.actions import index_to_action, action_to_index, Action, ActionType

        mask = Rules.get_legal_actions(self.env.state)
        state = self.env.state

        # Reset proposal counter on new turn
        if state.turn_number != self._non_learning_last_turn.get(player_id, -1):
            self._non_learning_proposals[player_id] = 0
            self._non_learning_last_turn[player_id] = state.turn_number

        # Apply trading limits for non-learning players (same as learning)
        proposals_made = self._non_learning_proposals.get(player_id, 0)
        if proposals_made >= self.max_proposals_per_turn:
            mask[366:416] = False  # No proposing
            mask[418:468] = False  # No counter-offers

        legal_actions = np.where(mask)[0]

        # Handle trade responses with heuristics (applies to all non-learning types)
        if state.phase.name == 'TRADE_RESPONSE':
            action = self._get_trade_response_action(player_id, mask)
            if action is not None:
                # Track if it's a counter-offer
                if 418 <= action < 468:
                    self._non_learning_proposals[player_id] = proposals_made + 1
                return action

        if player_id in self.random_players:
            # Maybe propose a trade (10% chance during MAIN phase)
            trade_action = self._maybe_propose_trade(player_id, mask)
            if trade_action is not None:
                return trade_action
            # Pure random action
            return np.random.choice(legal_actions)

        elif player_id in self.greedy_players:
            # Greedy setup: pick highest pip corner during setup, random after
            state = self.env.state
            if state.phase in [GamePhase.SETUP_SETTLEMENT_1, GamePhase.SETUP_SETTLEMENT_2]:
                # Find best settlement by pip count
                best_action = None
                best_pips = -1
                PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

                for action_idx in legal_actions:
                    action = index_to_action(action_idx)
                    if action.type == ActionType.PLACE_SETTLEMENT:
                        corner = state.board.corners[action.corner_id]
                        pips = 0
                        for tile_id in corner.adjacent_tiles:
                            if tile_id < len(state.board.tiles):
                                tile = state.board.tiles[tile_id]
                                if tile.number > 0 and tile.resource != Resource.DESERT:
                                    pips += PIPS.get(tile.number, 0)
                        if pips > best_pips:
                            best_pips = pips
                            best_action = action_idx

                if best_action is not None:
                    return best_action

            # Non-setup: maybe propose trade, otherwise random
            trade_action = self._maybe_propose_trade(player_id, mask)
            if trade_action is not None:
                return trade_action
            return np.random.choice(legal_actions)

        elif player_id in self.pool_players and player_id in self.pool_policies:
            # Pool policy action
            policy = self.pool_policies[player_id]
            obs = self._get_observation(player_id)
            with torch.no_grad():
                device = next(policy.parameters()).device
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
                action, _, _ = policy(obs_tensor, action_masks=mask_tensor)
                return action.item()

        else:
            # Fallback to random
            return np.random.choice(legal_actions)

    def _is_learning_player(self, player_id: int) -> bool:
        """Check if player is a learning agent."""
        return player_id in self.learning_players

    def _get_trade_response_action(self, player_id: int, mask: np.ndarray) -> Optional[int]:
        """Get a random trade response for non-learning players.

        No heuristics - purely random to avoid overfitting the learners.

        Returns:
            Action index or None to fall back to random selection
        """
        from src.game.actions import ACTION_ACCEPT_TRADE, ACTION_REJECT_TRADE
        from src.game.actions import ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END

        # Collect all legal trade response actions
        response_actions = []

        if mask[ACTION_ACCEPT_TRADE]:
            response_actions.append(ACTION_ACCEPT_TRADE)
        if mask[ACTION_REJECT_TRADE]:
            response_actions.append(ACTION_REJECT_TRADE)

        # Include counter-offers (already limited by proposal tracking)
        for i in range(ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END):
            if mask[i]:
                response_actions.append(i)

        if response_actions:
            action = np.random.choice(response_actions)
            # Track counter-offers
            if ACTION_COUNTER_OFFER_START <= action < ACTION_COUNTER_OFFER_END:
                proposals_made = self._non_learning_proposals.get(player_id, 0)
                self._non_learning_proposals[player_id] = proposals_made + 1
            return action

        return None

    def _maybe_propose_trade(self, player_id: int, mask: np.ndarray) -> Optional[int]:
        """Maybe propose a trade for non-learning players (low probability).

        No heuristics - purely random to avoid overfitting the learners.

        Returns:
            Action index or None to continue with normal action selection
        """
        from src.game.actions import ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END

        # 10% chance to propose if any proposals are legal
        if np.random.random() > 0.10:
            return None

        propose_actions = [i for i in range(ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END) if mask[i]]
        if propose_actions:
            action = np.random.choice(propose_actions)
            proposals_made = self._non_learning_proposals.get(player_id, 0)
            self._non_learning_proposals[player_id] = proposals_made + 1
            return action

        return None

    def _play_non_learning_turns(self) -> Tuple[bool, bool]:
        """
        Play turns for non-learning players until we reach a learning player's ACTUAL turn.

        Returns:
            (done, truncated) - whether game ended during non-learning turns
        """
        from src.game.constants import GamePhase
        from src.game.actions import index_to_action

        max_steps = 1000  # Safety limit to prevent infinite loops
        debug = False  # Set True to debug

        for step in range(max_steps):
            current = self.current_player
            phase = self.env.state.phase

            if debug:
                print(f"  _play_non_learning: step={step}, P{current}, {phase.name}")

            # Check if current player is learning
            if self._is_learning_player(current):
                if debug:
                    print(f"    -> returning (learning player)")
                return False, False  # Learning player has control

            # Non-learning player needs to act
            action = self._get_non_learning_action(current)
            action_obj = index_to_action(action)

            if debug:
                print(f"    P{current} takes {action_obj.type.name}")

            # Execute action
            _, _, done, truncated, info = self.env.step(action)

            # Track setup completion
            if not self._setup_complete and not self.env.state.is_setup_phase:
                self._setup_complete = True
                self._current_game_stats.starting_pips = self._compute_starting_pips()

            if done or truncated:
                return done, truncated

        # Safety: if we hit max steps, treat as truncated
        return False, True

    @property
    def current_player(self) -> int:
        """Get the current player index."""
        return self.env.state.current_player_idx

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and game buffer."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear game buffer for new game
        self.game_buffer.clear()

        # Initialize game stats
        self._current_game_stats = GameStats()
        self._setup_complete = False
        self._proposals_this_turn = 0
        self._last_turn_number = 0
        # Reset non-learning proposal tracking
        self._non_learning_proposals = {i: 0 for i in range(4)}
        self._non_learning_last_turn = {i: -1 for i in range(4)}

        # Determine opponent type for logging
        has_opponents = self.num_random_opponents > 0 or self.num_greedy_opponents > 0 or self.num_pool_opponents > 0
        if has_opponents:
            opponent_type = f"mixed ({self.num_learning_agents}L/{self.num_random_opponents}R/{self.num_greedy_opponents}G/{self.num_pool_opponents}P)"
        else:
            opponent_type = "self-play"

        # Start logging new game
        if self.game_logger is not None:
            self.game_logger.start_game(
                game_state=self.env.state,
                agent_player_id=0,
                opponent_type=opponent_type,
                training_step=self.games_played,
            )

        # Play non-learning player turns until we reach a learning player
        self._play_non_learning_turns()

        # Get observation and mask for current player (now guaranteed to be learning)
        self._current_obs = self._get_observation(self.current_player)
        self._current_mask = self._get_action_mask()
        self._prev_player = None

        info['action_mask'] = self._current_mask
        info['current_player'] = self.current_player

        return self._current_obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and collect transition from current player's perspective.

        Returns observation for the NEXT player (may be same player in some phases).
        """
        current_player = self.current_player
        current_obs = self._current_obs
        current_mask = self._current_mask

        # Track trade proposals to prevent loops
        # Check if turn changed - reset counter
        current_turn = self.env.state.turn_number
        is_new_turn = (current_turn != self._last_turn_number)
        if is_new_turn:
            self._proposals_this_turn = 0
            self._last_turn_number = current_turn

        # Execute action in underlying environment
        # Note: CatanEnv.step() returns reward for player 0, but we'll compute our own
        _, _, done, truncated, info = self.env.step(action)

        # Track if this was a trade proposal or counter-offer (both count toward limit)
        action_type = info.get('action_type')
        if action_type is not None and action_type.name in ['PROPOSE_TRADE', 'COUNTER_OFFER']:
            self._proposals_this_turn += 1

        # Log action
        if self.game_logger is not None:
            action_type = info.get('action_type')
            action_details = info.get('action_details', {})
            if action_type is not None:
                self.game_logger.log_action(
                    game_state=self.env.state,
                    action_index=action,
                    action_type=action_type,
                    player_id=current_player,
                    details=action_details,
                )

        # Check if setup phase just completed - compute starting pips
        if not self._setup_complete and not self.env.state.is_setup_phase:
            self._setup_complete = True
            self._current_game_stats.starting_pips = self._compute_starting_pips()

        # Compute reward for the player who just acted
        # Dense rewards are computed per-action in the environment
        reward = info.get('dense_reward', 0.0)

        # Apply turn penalty when a new turn starts (encourages faster games)
        if self.turn_penalty != 0.0 and is_new_turn:
            reward += self.turn_penalty

        # Only create transitions for learning players
        if self._is_learning_player(current_player):
            transition = PlayerTransition(
                player_id=current_player,
                observation=current_obs,
                action=action,
                action_mask=current_mask,
                reward=reward,
                done=False,  # Will be set at game end
            )
            self.game_buffer.add(current_player, transition)
            self.total_transitions += 1

        # Check if game is over
        if done or truncated:
            # Assign final rewards to all players
            final_rewards = self._compute_final_rewards()
            self.game_buffer.assign_final_rewards(final_rewards)
            self.games_played += 1

            # Collect final game stats
            self._finalize_game_stats(done, truncated)
            info['game_stats'] = self._current_game_stats
            self._completed_game_stats.append(self._current_game_stats)

            # Log game end
            if self.game_logger is not None:
                self.game_logger.end_game(
                    game_state=self.env.state,
                    winner=self.env.state.winner,
                )

            # Return dummy observation (game is over)
            info['game_buffer'] = self.game_buffer
            info['final_rewards'] = final_rewards
            info['action_mask'] = np.zeros(self.action_space.n, dtype=bool)
            info['current_player'] = -1

            return self._current_obs, 0.0, done, truncated, info

        # Play non-learning player turns until we reach a learning player or game ends
        nl_done, nl_truncated = self._play_non_learning_turns()

        if nl_done or nl_truncated:
            # Game ended during non-learning turns
            final_rewards = self._compute_final_rewards()
            self.game_buffer.assign_final_rewards(final_rewards)
            self.games_played += 1

            self._finalize_game_stats(nl_done, nl_truncated)
            info['game_stats'] = self._current_game_stats
            self._completed_game_stats.append(self._current_game_stats)

            if self.game_logger is not None:
                self.game_logger.end_game(
                    game_state=self.env.state,
                    winner=self.env.state.winner,
                )

            info['game_buffer'] = self.game_buffer
            info['final_rewards'] = final_rewards
            info['action_mask'] = np.zeros(self.action_space.n, dtype=bool)
            info['current_player'] = -1

            return self._current_obs, 0.0, nl_done, nl_truncated, info

        # Get next learning player's observation and mask
        next_player = self.current_player
        self._current_obs = self._get_observation(next_player)
        self._current_mask = self._get_action_mask()
        self._prev_player = current_player

        info['action_mask'] = self._current_mask
        info['current_player'] = next_player

        # Return 0 reward here - rewards are stored in transitions
        # The training loop will use transitions, not step rewards
        return self._current_obs, 0.0, False, False, info

    def _get_observation(self, player_id: int) -> np.ndarray:
        """Get observation from a specific player's perspective."""
        return self.encoder.encode(self.env.state, player_id)

    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for current player."""
        mask = Rules.get_legal_actions(self.env.state)

        # Limit trade proposals AND counter-offers per turn to prevent trading loops
        if self._proposals_this_turn >= self.max_proposals_per_turn:
            # Disable PROPOSE_TRADE (366-416) and COUNTER_OFFER (418-468)
            mask[366:416] = False
            mask[418:468] = False

        return mask

    def _compute_final_rewards(self) -> Dict[int, float]:
        """Compute final rewards for all players based on game outcome."""
        state = self.env.state
        rewards = {}

        winner = state.winner

        for pid in range(4):
            if winner is not None:
                if pid == winner:
                    rewards[pid] = self.win_reward
                else:
                    rewards[pid] = self.lose_reward
            else:
                # Game truncated without winner - small penalty for all
                rewards[pid] = -5.0

        return rewards

    def _compute_starting_pips(self) -> List[int]:
        """Compute total pip count from starting settlements for each player."""
        # Pip values for each dice number
        PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

        pips_per_player = []
        state = self.env.state

        for player in state.players:
            total_pips = 0
            for corner_id in player.settlements:
                corner = state.board.corners[corner_id]
                for tile_id in corner.adjacent_tiles:
                    if tile_id < len(state.board.tiles):
                        tile = state.board.tiles[tile_id]
                        if tile.number in PIPS:
                            total_pips += PIPS[tile.number]
            pips_per_player.append(total_pips)

        return pips_per_player

    def _finalize_game_stats(self, done: bool, truncated: bool) -> None:
        """Collect final game statistics."""
        state = self.env.state
        stats = self._current_game_stats

        stats.game_length = state.turn_number
        stats.winner = state.winner
        stats.timed_out = truncated and not done

        stats.final_vps = [p.victory_points for p in state.players]
        stats.settlements_built = [len(p.settlements) for p in state.players]
        stats.cities_built = [len(p.cities) for p in state.players]
        stats.roads_built = [len(p.roads) for p in state.players]
        stats.dev_cards_bought = [
            sum(p.dev_cards.values()) + p.knights_played
            for p in state.players
        ]

        stats.longest_road_holder = state.longest_road_holder
        stats.largest_army_holder = state.largest_army_holder
        stats.learning_players = list(self.learning_players)

        # Count trades from game history if available
        # For now, just set to 0 - would need action tracking to count properly
        stats.trades_completed = 0

    def get_completed_game_stats(self) -> List[GameStats]:
        """Get and clear completed game stats."""
        stats = self._completed_game_stats.copy()
        self._completed_game_stats = []
        return stats

    def get_game_transitions(self) -> List[PlayerTransition]:
        """Get all transitions from the completed game."""
        return self.game_buffer.get_all_transitions()

    def action_masks(self) -> np.ndarray:
        """Return current action mask (for compatibility with MaskablePPO)."""
        return self._current_mask


class MultiAgentVecEnv:
    """
    Vectorized multi-agent environment for parallel game collection.

    Runs multiple games in parallel, collecting experiences from all 4 players
    in each game. Designed to work with PPO's rollout collection.
    """

    def __init__(
        self,
        num_envs: int = 4,
        **env_kwargs
    ):
        self.num_envs = num_envs
        self.envs = [MultiAgentCatanEnv(**env_kwargs) for _ in range(num_envs)]

        # Track which envs have pending experiences
        self._pending_transitions: List[List[PlayerTransition]] = [[] for _ in range(num_envs)]

        # Copy spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        return np.array(obs_list), info_list

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments with given actions."""
        obs_list = []
        reward_list = []
        done_list = []
        truncated_list = []
        info_list = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(int(action))

            if done or truncated:
                # Collect transitions from completed game
                self._pending_transitions[i].extend(env.get_game_transitions())

                # Auto-reset
                obs, reset_info = env.reset()
                info.update(reset_info)
                info['terminal_observation'] = obs

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done or truncated)
            truncated_list.append(truncated)
            info_list.append(info)

        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(done_list),
            np.array(truncated_list),
            info_list
        )

    def get_action_masks(self) -> np.ndarray:
        """Get action masks for all environments."""
        return np.array([env.action_masks() for env in self.envs])

    def collect_transitions(self) -> List[PlayerTransition]:
        """Collect all pending transitions from completed games."""
        all_transitions = []
        for i in range(self.num_envs):
            all_transitions.extend(self._pending_transitions[i])
            self._pending_transitions[i] = []
        return all_transitions

    def collect_game_stats(self) -> List[GameStats]:
        """Collect all completed game stats from all environments."""
        all_stats = []
        for env in self.envs:
            all_stats.extend(env.get_completed_game_stats())
        return all_stats

    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()

    @property
    def unwrapped(self):
        return self

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if envs are wrapped with a specific wrapper."""
        return False

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Call a method on environments."""
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in indices]

    def get_attr(self, attr_name, indices=None):
        """Get an attribute from environments."""
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        """Set an attribute on environments."""
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self.envs[i], attr_name, value)
