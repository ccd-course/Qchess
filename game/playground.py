from jvm import init_jvm
init_jvm()

from com.chess.backend.services import ChessboardService, ChessGameService, PlayerService

cgh = ChessGameService()
game = cgh.createNewGame(["A", "B", "C", "D"])
print(f"Players: {game.getPlayers().toString()}")

# We should maybe directly return all moves for all pieces??
# This would be better as a json or dictionary object
x = 0
y = 0
jint = cgh.getPossibleMoves(game, (x, y))
possible_moves = f"Possible moves for piece on {x}, {y}:\n"
for i in jint:
    possible_moves += f"- "
    for j in i:
        possible_moves += f"{j} "
    possible_moves += "\n"
print(possible_moves)
