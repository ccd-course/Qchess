from jvm import init_jvm
init_jvm()
from com.chess.backend.services import ChessboardService, ChessGameService, PlayerService
from com.chess.backend.restController.service import NewChessGameService
from com.chess.backend.repository import GameRepositoryMock
from ..environment.qchess_utils import get_observation 

if __name__ == "__main__": 
    # game_repository = GameRepositoryMock()
    # cgs = ChessGameService(game_repository)
    # ncgs = NewChessGameService(cgs, game_repository)
    # game_id = ncgs.getNewGameID(["A", "B", "C", "D"])
    # print(f"Created game with game id {game_id}.")
    # game = cgs.getGame(game_id)
    # print(f"Players: {game.getChessboard().toString()}")

    print("hello world")