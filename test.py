import logging

import coloredlogs
from random import shuffle
from MCTS import MCTS
from Arena import Arena
from Coach import Coach
from tictactoe_master.TicTacToeGame import MasterTicTacToe as Game
from tictactoe_master.keras.NNet import NNetWrapper as nn
from utils import *
import numpy as np

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','checkpoint_0.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

log = logging.getLogger(__name__)

g= Game(9)
nnet = nn(g)
nnet.load_checkpoint(args.checkpoint, filename='temp.pth.tar')
c = Coach(g, nnet, args)

# training new network, keeping a copy of the old one
pnet = nn(g)
pnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')

c.loadTrainExamples()

trainExamples = []
for e in c.trainExamplesHistory:
    trainExamples.extend(e)
shuffle(trainExamples)

pmcts = MCTS(g, pnet, args)

nnet.train(trainExamples)
nmcts = MCTS(g, nnet, args)

log.info('PITTING AGAINST PREVIOUS VERSION')
arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), g, g.display)
pwins, nwins, draws = arena.playGames(args.arenaCompare, verbose=True)
