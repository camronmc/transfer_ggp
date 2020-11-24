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
import pickle

if len(sys.argv) < 4:
    print("Usage: transfer.py <game> <ckpt> <total_rounds>")

game = sys.argv[1]
expertckpt = sys.argv[2]
totalRounds = int(sys.argv[3])

data, propnet = load_propnet(game)
m = Model(propnet, create=False)
m.load('./models/' + game + '/step-%06d.ckpt' % int(expertckpt))

cur = [None]

start = time.time()
gameIndex = 0
for i in range(totalRounds+1):
    cur[0] = B1Node(propnet, data, model=m)
    print('Game number', i)
    start_game = time.time()
    do_game(cur, propnet, m, z=0.5)
    print("took ", time.time()-start_game, "seconds to play game")

pickle.dump(m.getBuffer(), open( "buffers/" + game + expertckpt + "for" + str(totalRounds) + "rounds.p", "wb" ) )