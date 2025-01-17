import numpy as np
import tensorflow as tf
import chess
import time

loaded_model = tf.keras.models.load_model('model.keras')
loaded_model.summary()

def fenToVector(fen):
  board = chess.Board(fen)
  piece_map = board.piece_map()
  features = np.zeros(64)

  for square, piece in piece_map.items():
    features[square] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
  return features

def winProb(fen):
  featureVector = fenToVector(fen)
  featureVector = featureVector.reshape(1, 64)
  featureVector = np.tile(featureVector,(1, 32, 1))
  prob = loaded_model.predict(featureVector, verbose = 0)[0][0]
  return prob.item()

def moveOrdering(board):
  #order move searches (checks, captures, attacks)
  pieceValue = {
    chess.PAWN: 1,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
  }
  def scoreMove(move):
    board.push(move)
    if board.is_checkmate():
      board.pop()
      return 50
    board.pop()
    if move.promotion is not None: return 20
    if board.is_capture(move):
      captured = board.piece_at(move.to_square)
      capturedValue = pieceValue[captured.piece_type] if captured else 0
      capturing = board.piece_at(move.from_square)
      capturingValue = pieceValue[capturing.piece_type] if capturing else 0
      materialWon = capturedValue - capturingValue
      return 10 + materialWon
    if board.gives_check(move): return 9
    return 0
  moves = list(board.legal_moves)
  return sorted(moves, key = scoreMove, reverse = True)

def minimax(board, depth, alpha, beta, player):
  if depth == 0 or board.is_game_over():
    return winProb(board.fen()), None

  legalMoves = moveOrdering(board)
  bestMove = legalMoves[0]
  if player: #player = True for white
    maxEval = float('-inf')
    for move in moveOrdering(board):
      board.push(move)
      eval, _ = minimax(board, depth - 1, alpha, beta, False)
      board.pop()
      maxEval = max(maxEval, eval)
      alpha = max(alpha, maxEval)
      if eval > maxEval:
        maxEval = eval
        bestMove = move
      if beta <= alpha:
        break
    return maxEval, bestMove
  else:
    minEval = float('inf')
    for move in board.legal_moves:
      board.push(move)
      eval, _ = minimax(board, depth - 1, alpha, beta, True)
      board.pop()
      minEval = min(minEval, eval)
      beta = min(beta, minEval)
      print(f'Evaluating {move}:\n  Best Move: {bestMove} [{minEval}]\n  Eval: {eval}')
      if eval < minEval:
        minEval = eval
        bestMove = move
      if beta <= alpha:
        break
    return minEval, bestMove

def iterativeDeepening(board, maxDepth, timeLimit):
  start = time.time()
  player = board.turn
  for depth in range(1, maxDepth + 1):
    elapsed = time.time() - start
    if elapsed > timeLimit:
      break
    bestScore, bestMove = minimax(board, depth, float('-inf'), float('inf'), player)
    print(f"Move {bestMove} (Score: {bestScore:.2f}) found in {elapsed:.2f}s at depth {depth}")

  return bestMove, bestScore


#example
sampleFen = 'r3k2r/pppqbppp/8/3PP3/3p2n1/8/PPP3PP/RNBQ1RK1 w - - 0 13'
board = chess.Board(sampleFen)
print(board)
maxDepth = 3
timeLimit = 15
player = board.turn
print(winProb(sampleFen))
bestmove, bestEval = iterativeDeepening(board, maxDepth, timeLimit)

#the model recommends Rf7, not the best move but a rather creative one which provokes risky choices
#from the opponent. 
# As is was trained on human games, it has such risk-taking and outlandish/creative tendencies in several
#other positions as well.