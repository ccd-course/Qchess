import numpy as np

from environment import qchess_env
from typing import List, TypedDict

ObservationDict = TypedDict('ObservationDict', {
    'observation': np.ndarray,
    'action_mask': np.ndarray,
})


def dev_policy(observation: ObservationDict, agent: int) -> int:
    print("agent: ", agent)
    return 32


def test_env():
    env = qchess_env.env()
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = dev_policy(observation, agent)
        env.step(action)
        break # TODO: Remove after implementing dev_policy


if __name__ == "__main__":
    test_env()
    # game = Game()
    # player = game.chess_game.getActivePlayer()
    # legal_moves = game.get_legal_moves(player)
    #
    # executed_move_obj = game.action_to_move(legal_moves[0])
    # game.execute_move(executed_move_obj)
    # observation = game.get_observation()
    # game_repository = GameRepositoryMock()
    # cgs = ChessGameService(game_repository)
    # ncgs = NewChessGameService(cgs, game_repository)
    # game_id = ncgs.getNewGameID(["A", "B", "C", "D"])
    # print(f"Created game with game id {game_id}.")
    # game = cgs.getGame(game_id)
    # print(f"Players: {game.getChessboard().toString()}")

    print("hello world")


