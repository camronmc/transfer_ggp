import sys
sys.path.insert(1, '/Users/Cameron/Desktop/transfer_ggp')

from b1 import B1Node, do_game
from propnet.propnet import load_propnet
from model import Model
from utils.pauser import set_pauser
import time
import sys

game = sys.argv[1]

data, propnet = load_propnet(game)
model = Model(propnet)
cur = [None]

set_pauser({
    'cur': cur,
    'model': model,
    'propnet': propnet,
})

start = time.time()
for i in range(50000):
    cur[0] = B1Node(propnet, data, model=model)
    print('Game number', i)
    start_game = time.time()
    do_game(cur, propnet, model, z=0.5)
    print("took ", time.time()-start_game, "seconds to play game")
    start_train = time.time()
    model.train(epochs=10)
    print("took ", time.time()-start_train, "seconds to train")
    if i and i % 50 == 0:
        # model.save(game, i)
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
