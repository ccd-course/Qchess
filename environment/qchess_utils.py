from ast import Return
from distutils.log import error
from logging import exception
from msilib.schema import Error
import numpy as np

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

def get_observation(java_chessboard) -> np.NDArray:
    """Generate observation from chessboard or game.
    This would be an 2-dimensional array with int values representing the piece standing on the square.
    An empty square is represented as -1.

    :return:
    """
    board = java_chessboard.getChessboard()
    b_sq = list(board.getSquares())
    length_line =  len(list(b_sq[0]))
    players_number = 3
    observation = [ [ [0] * 8 * players_number  ] * len(b_sq) ] * length_line

    for line in b_sq:

        for square in line:

            if(square.getPiece()):
                line = square.getPosition().getX()
                line_square_number = square.getPosition().getY()
                piece_layer_number = square.getPiece().getType().geySymbol()
                observation[line][line_square_number][piece_layer_number] = 1
                
    return np.asarray(observation)


def get_str_observation():
    """Converts a board observation to a string that can be interpreted by a human.

    :return:
    """
    pass


def get_legal_moves():
    """Aggregates all legal action-codes in one long integer list.

    :return:
    """
    pass

def count_pieces(observation):
    """Count each player's pieces remaining in a given observation.
    
    :return: dictionary containing player numbers as keys and number of that player's remaining pieces as corresponding value.
    """
    # get number of players by dividing last dimension's length by 8 (number of pieces)
    players_number = observation.shape[-1] / 8
    
    # initialize dictionary with value 0 for each player
    piece_count = {player:0 for player in range(players_number)}
    
    # TODO: iterate over observation, check last dimension's 1 index, get player corresponing to index, increase piece_count
    
    return piece_count