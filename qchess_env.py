from typing import Dict

from gym import spaces
import numpy as np
import functools

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from game import get_str_observation, Game

PLAYERS = 4
MAX_STEPS = 50


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "qchess",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        super().__init__()

        # Create a list of initial agents
        self.agents = [f"player_{i}" for i in range(PLAYERS)]

        # Make a copy of the agent list as the agents are removed, when they are done.
        self.possible_agents = self.agents[:]

        self.game = Game()

        # Create an agent selector, that loops through all agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        # The board has 5 * 10 * n-Players squares.
        # For 3 Players: 5 * 10 * 3 = 150
        # A Queen can move (at maximum)
        #   - 4 + 4 + 4 + 4 = 16 diagonal
        #   - (10 * PLAYERS - 1) + 8 = (10 * PLAYERS + 7) straight
        # A Knight can move (at maximum)
        #   - 8 jumps
        # Queen + Knight together cover all possible move destinations 16 + (10 * player + 7) + 8 = (10 * PLAYERS + 31)
        # The general action space has a size of (5 * 10 * PLAYERS) * (10 * PLAYERS + 31)
        #                                           = 500 * PLAYERS² + 1550 * PLAYERS
        # For PLAYERS = 3 this would be an action space size of 9150
        self.action_space_size = 500 * PLAYERS * PLAYERS + 1550 * PLAYERS
        self.action_spaces = {name: spaces.Discrete(self.action_space_size) for name in self.agents}

        # The board has 5 x (10 * n-Players) squares. Each square has one integer representing the piece type and player
        # Value -1: No piece
        # Value 0: Cannon
        # Value 1 - 8: Player 0
        #   1: Pawn
        #   2: Rook
        #   3: Knight
        #   4: Bishop
        #   5: Ferz
        #   6: Wazir
        #   7: Queen
        #   8: King
        # Value 9 - 16: Player 1
        # Value 17 - 24: Player 2
        # ---
        # observation: Board representation as above (5, 10 * player, 8 * player)
        # action_mask: Binary mask on actions space to mark all possible actions (size equals action space)
        # observation space size = 5 * 10 * PLAYERS * 8 * PLAYERS
        # ! written as less multiplication operations (5*10*8=400)
        self.state_space_size = 400 * PLAYERS * PLAYERS
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(5, 10 * PLAYERS, PLAYERS * 8 + 1,), dtype=int
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.action_space_size,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }

        self.rewards = {name: 0 for name in self.agents}
        self.dones = None
        self.infos = {name: {} for name in self.agents}

        self.step_count = 0
        self._cumulative_rewards = {name: 0 for name in self.agents}

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _accumulate_rewards(self) -> None:
        """
        Adds .rewards dictionary to ._cumulative_rewards dictionary. Typically
        called near the end of a step() method
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if len(self.agents) == 2:
            string = get_str_observation(self.game.get_observation())
        else:
            string = "Game over"
        print(string)

    def observe(self, agent) -> Dict[str, any]:
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        observation = self.game.get_observation()
        agent_id: int = int(agent.split("_")[1])
        player = list(filter(lambda player: agent_id == player.getId(), self.game.chess_game.getPlayers()))[0]
        # observation = np.dstack((observation[:, :], self.board_history))
        legal_moves = (
            self.game.get_legal_moves(player) if f"player_{player.getId()}" == self.agent_selection else []
        )

        # action_mask example:
        # `action_mask[34] == 1` would mean, that a piece on a specific square is allowed to move in a specific way
        action_mask = np.zeros(self.action_space_size, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]

        self.game = Game()

        self.rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}

        self.step_count = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        executed_move_obj = self.game.action_to_move(action)
        self.game.execute_move(executed_move_obj)

        # reduce rewards for each game step to encourage shorter games
        self.rewards[self.agent_selection] -= 1
        
        # add reward for each own piece still alive
        piece_counter = self.game.count_remaining_pieces(self.game.get_observation())
        self.rewards[self.agent_selection] += piece_counter[self.agent_selection]
        
        if self.step_count > MAX_STEPS:

            pass
        # TODO: Check if game has ended (checkmate or step-count limit exceeded)
        game_over = False

        if game_over:
            # big reward for winning the game
            self.rewards[self.agent_selection] += 1000

        # Add rewards to cumulative rewards
        self._accumulate_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
