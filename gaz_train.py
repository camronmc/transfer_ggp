import sys
sys.path.insert(1, '/home/Cameron/transfer_ggp')

from b1 import B1Node, do_game
from propnet.propnet import load_propnet
from model import Model
from utils.pauser import set_pauser
import time
import sys

if len(sys.argv) < 3:
    print("Usage: transfer.py <base_game> <train_to>")

game = sys.argv[1]
trainTo = sys.argv[2]

data, propnet = load_propnet(game)
model = Model(propnet)
# model.load('models/connect4_adrian/step-006000.ckpt')
cur = [None]

set_pauser({
    'cur': cur,
    'model': model,
    'propnet': propnet,
})

model.save(game,0)
start = time.time()
for i in range(int(trainTo)):
    cur[0] = B1Node(propnet, data, model=model)
    print('Game number', i)
    start_game = time.time()
    do_game(cur, propnet, model, z=0.5)
    print("took ", time.time()-start_game, "seconds to play game")
    start_train = time.time()
    model.train(epochs=10)
    print("took ", time.time()-start_train, "seconds to train")
    if i and i % 50 == 0:
        model.save(game, i)
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
