import numpy as np

from game.jvm import init_jvm
init_jvm()
from com.chess.backend.services import ChessboardService, ChessGameService, PlayerService
from com.chess.backend.restController.service import NewChessGameService
from com.chess.backend.repository import GameRepositoryMock
from com.chess.backend.restController.controller import ExecutedMoveController


def symbol_layer_mapper(piece_symbol: str, player_id: int) -> int:
    layer_base_position = 0
    if (piece_symbol == "R"): layer_base_position = 1
    elif(piece_symbol == "N"): layer_base_position = 2
    elif(piece_symbol == "B"): layer_base_position = 3
    elif(piece_symbol == "Q"): layer_base_position = 4
    elif(piece_symbol == "K"): layer_base_position = 5
    elif(piece_symbol == "C"): layer_base_position = 6
    elif(piece_symbol == "F"): layer_base_position = 7
    elif(piece_symbol == "W"): layer_base_position = 8
    else: raise ValueError(f"Piece symbol is not defined {piece_symbol}")
    return layer_base_position * player_id


def get_str_observation(observation):
    """Converts a board observation to a string that can be interpreted by a human.

    :return:
    """
    return str(np.argmax(observation, axis=-1))


def count_remaining_pieces(observation):
    """Count each player's pieces remaining in a given observation.

    :return: dictionary containing player numbers as keys and number of that player's remaining pieces as corresponding value.
    """
    # Decode one-hot encoding = find index of 1 on last dimension
    pieces = np.argmax(observation, axis=-1)
    # Ignore all zeros = empty squares
    # Floor divide by 8 to get list of corresponding player for each piece
    pieces = pieces[pieces != 0] // 8

    # Count occurences of player number, then save to dictionary with key = player number (int) and value = number of pieces remaining
    # ! Note that players might be missing in the dictionary if they don't have remaining pieces in the game
    piece_count = dict(zip(*np.unique(pieces, return_counts=True)))

    return piece_count


class Game:
    def __init__(self):
      self.game_repository = GameRepositoryMock()
      self.chess_game_service = ChessGameService(self.game_repository)
      ncgs = NewChessGameService(self.chess_game_service, self.game_repository)
      self.game_id = ncgs.getNewGameID(["A", "B", "C", "D"])
      print(f"Created game with game id {self.game_id}.")
      self.chess_game = self.chess_game_service.getGame(self.game_id)
      print(f"Players: {self.chess_game.getPlayers().toString()}")

    def get_observation(self) -> np.NDArray:
        """Generate observation from chessboard or game.
        This would be an 3-dimensional array with one-hot-encoded values representing the piece standing on the square.
        An empty square is represented as vector containing all zeros.

        :return:
        """
        board = self.chess_game.getChessboard()
        b_sq = list(board.getSquares())
        length_line = len(list(b_sq[0]))
        players_number = 3
        observation = [[[0] * 8 * players_number] * len(b_sq)] * length_line

        for line in b_sq:

            for square in line:

                if square.getPiece():
                    line = square.getPosition().getX()
                    line_square_number = square.getPosition().getY()
                    piece = square.getPiece()
                    piece_symbol = piece.getType().getSymbol()
                    piece_layer = symbol_layer_mapper(piece_symbol, piece.getPlayer().getId())
                    observation[line][line_square_number][piece_layer] = 1

        return np.asarray(observation)

    def get_legal_moves(self, player):
        """Aggregates all legal action-codes in one long integer list.

        :return: ArrayList<Integer>
        """
        return ChessboardService.getAllPossibleActions(self.chess_game.getChessboard(), player)

    def action_to_move(self, action):
        """Returns an Java ExecutedMoveObject from an action code

        :return: com.chess.backend.restController.objects.ExecutedMoveObject
        """
        executed_move_obj = ChessboardService.actionToMove(self.chess_game.getChessboard(), action, self.game_id)
        return executed_move_obj

    def execute_move(self, executed_move_obj):
        """Executes a move in the game

        :param executed_move_obj:
        :return:
        """
        return ExecutedMoveController.executedMove(executed_move_obj)
