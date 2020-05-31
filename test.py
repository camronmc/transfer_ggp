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
