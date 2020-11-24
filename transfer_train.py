import sys
sys.path.insert(1, '/home/Cameron/transfer_ggp')

from b1 import B1Node, do_game
from propnet.propnet import load_propnet
from model import Model
from utils.pauser import set_pauser
import time
import sys
import tensorflow as tf
import re

if len(sys.argv) < 6:
    print("Usage: transfer.py <base_game> <ckpt> <to_game> <mode> <train_to> - mode is 0pad mean or clear")

from_game = sys.argv[1]
ckpt = sys.argv[2]
to_game = sys.argv[3]
mode = sys.argv[4]
train_to = sys.argv[5]


zeroPad = False
reuse = False
clear = False
breakthroughMap = False 
if mode == "0pad":
    zeroPad = True
    reuse = True
elif mode == "mean":
    zeroPad = False
    reuse = True
elif mode == "clear":
    clear = True
elif mode == 'map':
    reuse = True
    breakthroughMap = True
elif mode == "none":
    reuse = False
else:
    print("incorrect mode, please use 0pad, mean or clear")
    exit(0)

bd = []
rd = []
if 'breakthrough' in from_game.lower():
    role_a = ('black',)
    role_b = ('white',)
    if 'breakthrough-6x6' in from_game.lower():
        bd = [5650, 2825, 1412, 706, 353, 176, 88, 50]
        rd = []
    elif 'breakthroughSmall' in from_game.lower():
        bd = [2070, 1035, 517, 258, 129, 64, 50]
        rd = []
elif 'connect' in from_game.lower():
    role_a = ('red',)
    role_b = ('white',)
    bd = [685, 342, 171, 85, 50]
    rd = []

## load propnet for new game
oldData, old_prop = load_propnet(from_game)

## load propnet for old game
data, new_prop = load_propnet(to_game)

## check dimensions of the input and output layers
oldPropNumActions = {role: len(actions) for role, actions in old_prop.legal_for.items()}
newPropNumActions = {role: len(actions) for role, actions in new_prop.legal_for.items()}

if  (len(old_prop.roles) == len(new_prop.roles) and 
    oldPropNumActions == newPropNumActions and
    len(old_prop.propositions) == len(new_prop.propositions)):

    model = Model(new_prop, create=False)

    model.load('./models/' + from_game + '/step-%06d.ckpt' % int(ckpt))
    
else:

    print("complete complex transfer")
    model = Model(new_prop, transfer=True, base_dims=bd, roles_dim=rd)

    model.perform_transfer('./models/' + from_game+ '/step-%06d.ckpt' % int(ckpt), reuse_output=reuse, pad_0=zeroPad, breakthroughMap=breakthroughMap)

if clear:
    model.clear_output_layer()

## train
cur = [None]

set_pauser({
    'cur': cur,
    'model': model,
    'propnet': new_prop,
})

model_name = to_game + "from" + from_game + ckpt + mode
model.save(model_name,0, transfer=True)
start = time.time()
for i in range(int(train_to)):
    cur[0] = B1Node(new_prop, data, model=model)
    print('Game number', i)
    start_game = time.time()
    do_game(cur, new_prop, model, z=0.5)
    print("took ", time.time()-start_game, "seconds to play game")
    start_train = time.time()
    model.train(epochs=10)
    print("took ", time.time()-start_train, "seconds to train")
    if i and i % 50 == 0:
        model.save(model_name, i, transfer=True)
        with open(f'models_transfer/times-{to_game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
