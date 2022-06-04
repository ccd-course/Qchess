from game.jvm import init_jvm
init_jvm()
from com.chess.backend.services import ChessboardService, ChessGameService, PlayerService
from com.chess.backend.restController.service import NewChessGameService
from com.chess.backend.repository import GameRepositoryMock

class Game:
    def __init__(self):
      self.game_repository = GameRepositoryMock()
      self.chess_game_service = ChessGameService(self.game_repository)
      ncgs = NewChessGameService(self.chess_game_service, self.game_repository)
      self.game_id = ncgs.getNewGameID(["A", "B", "C", "D"])
      print(f"Created game with game id {self.game_id}.")
      self.chess_game = self.chess_game_service.getGame(self.game_id)
      print(f"Players: {self.chess_game.getPlayers().toString()}")

    def get_observation(self):
        """Generate observation from chessboard or game.
        This would be an 2-dimensional array with int values representing the piece standing on the square.
        An empty square is represented as -1.

        :return:
        """

        # board = game.getChessboard()
        # b_sq = board.getSquares()
        # sq2 = [list(square) for square in b_sq]

        pass


    def get_str_observation(self):
        """Converts a board observation to a string that can be interpreted by a human.

        :return:
        """
        pass


    def get_legal_moves(self):
        """Aggregates all legal action-codes in one long integer list.

        :return:
        """
        pass
