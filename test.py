#!/usr/bin/env python3.7

import sys
sys.path.insert(1, '/Users/Cameron/Desktop/transfer_ggp')

from model import Model

from mcts import MCTSNode, simulation
from propnet.propnet import load_propnet
import time

start = time.time()

# propnet = load_propnet('connect4match1')
# propnet = load_propnet('tictactoe1')
data, propnet = load_propnet('connectFour')

root = MCTSNode(propnet, data)
# exit(0)
for i in range(400):
    simulation(root)
root.print_node()

print('Took', time.time() - start, 'seconds')
