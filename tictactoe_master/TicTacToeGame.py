from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TicTacToeLogic import Board
import numpy as np

"""
Game class implementation for the game of MasterTicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
class TicTacToeGame(Game):
    def __init__(self, n=3):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")


class MasterTicTacToe(TicTacToeGame):
    def __init__(self, n=9):
        super().__init__(n)
        self.last_action = None  
        self.master_board = Board(3)

    def getNextState(self, board, player, action):
        b,p = super().getNextState(board, player, action)
        self.last_action = action
        return b,p
    
    def getValidMoves(self, board, player): 
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)

        if self.last_action is None:            
            legalMoves =  b.get_legal_moves(player)
            if len(legalMoves)==0:
                valids[-1]=1
                return np.array(valids)
            for x, y in legalMoves:
                valids[self.n*x+y]=1
            return np.array(valids)
        else:
            #moving form big board to small board in last action location 
            b_cp = Board(3)
            col = int(self.last_action/self.n)
            row = self.last_action%self.n
            _x = col%3*3
            _y = row%3*3
            b_cp.pieces = np.copy(b[_x:_x+3, _y:_y+3])
            # getting the legal moves in the small board
            legalMoves =  b_cp.get_legal_moves(player)
            if len(legalMoves)==0:
                valids[-1]=1
                return np.array(valids)
            #returning the valid coords in the big board
            for x, y in legalMoves:
                x_ = _x + x
                y_= _y + y
                valids[self.n*x_+y_]=1
            return np.array(valids)

    
    def getGameEnded(self, board, player):
        g = TicTacToeGame(3)
        
        for row in range(0, self.n, 3):
            for col in range(0, self.n, 3):
                b = Board(3)
                b.pieces = np.copy(board[col:col+3, row:row+3]) 
                winner = g.getGameEnded(b.pieces, player)
                if winner == 0:
                    continue
                elif winner > 0 and winner < 1:
                    self.master_board[col//3][row//3] = 2
                else:
                    if winner == 1:
                        self.master_board[col//3][row//3] = player
                    elif winner == -1:
                        self.master_board[col//3][row//3] = -player

        winner = g.getGameEnded(self.master_board.pieces, player)
        if winner != 0:
            for col in range(self.master_board.n):
                for row in range(self.master_board.n):
                   self.master_board[col][row] = 0
        return winner


