from jvm import init_jvm
init_jvm()
from com.chess.backend.services import ChessGameService
from com.chess.backend.restController.service import NewChessGameService
from com.chess.backend.repository import GameRepositoryMock

game_repository = GameRepositoryMock()
cgs = ChessGameService(game_repository)
ncgs = NewChessGameService(cgs, game_repository)
game_id = ncgs.getNewGameID(["A", "B", "C", "D"])
print(f"Created game with game id {game_id}.")
game = cgs.getGame(game_id)
print(f"Players: {game.getPlayers().toString()}")

# We should maybe directly return all moves for all pieces??
# This would be better as a json or dictionary object
x = 0
y = 0


possible_moves_array = cgs.getPossibleMoves(game, (x, y))
possible_moves = f"Possible moves for piece on {x}, {y}:\n"
for i in possible_moves_array:
    possible_moves += f"- "
    for j in i:
        possible_moves += f"{j} "
    possible_moves += "\n"
print(possible_moves)

next_player = cgs.executedMove((game_id), (0, 0), (0, 39))
print(f"Next player: {next_player}")

print(f"Event stream: {game.getEvents()}")