import sys
sys.path.insert(1, '/Users/Cameron/Desktop/transfer_ggp')

from b1 import B1Node, do_game
from propnet.propnet import load_propnet
from model import Model
from utils.pauser import set_pauser
import time
import sys
import tensorflow as tf
import re

if len(sys.argv) < 4:
    print("Usage: transfer.py <base_game> <ckpt> <to_game>")

from_game = sys.argv[1]
ckpt = sys.argv[2]
to_game = sys.argv[3]

## load propnet for new game
data, old_prop = load_propnet(from_game)

## load propnet for old game
data, new_prop = load_propnet(to_game)

## check dimensions of the input and output layers
oldPropNumActions = {role: len(actions) for role, actions in old_prop.legal_for.items()}
newPropNumActions = {role: len(actions) for role, actions in new_prop.legal_for.items()}


if  (len(old_prop.roles) == len(new_prop.roles) and 
    oldPropNumActions == newPropNumActions and
    len(old_prop.propositions) == len(new_prop.propositions)):

    model = Model(new_prop, create=False)

    model.load('./models_ad/' + from_game+ '/step-%06d.ckpt' % int(ckpt))

    model.clear_output_layer()
    
else:

    print("complete complex transfer")
    model = Model(new_prop, transfer=True, base_dims=[685, 342, 171, 85, 50], roles_dim=[])

    model.perform_transfer('./models_ad/' + from_game+ '/step-%06d.ckpt' % int(ckpt), True)

    model.clear_output_layer()

## train
cur = [None]

set_pauser({
    'cur': cur,
    'model': model,
    'propnet': new_prop,
})

model.save(to_game,0,transfer=True,from_game=from_game)
start = time.time()
for i in range(10000):
    cur[0] = B1Node(new_prop, data, model=model)
    print('Game number', i)
    start_game = time.time()
    do_game(cur, new_prop, model, z=0.5)
    print("took ", time.time()-start_game, "seconds to play game")
    start_train = time.time()
    model.train(epochs=10)
    print("took ", time.time()-start_train, "seconds to train")
    if i and i % 50 == 0:
        model.save(to_game, i, transfer=True, from_game=from_game)
        with open(f'models_transfer/times-{to_game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')