import sys
sys.path.insert(1, '/Users/Cameron/Desktop/transfer_ggp')

from b1MultiNet import B1Node, do_game
from propnet.propnet import load_propnet
from model import Model
from utils.pauser import set_pauser
import time
import sys
import tensorflow as tf
import re
import pickle

expertGames = ['connect4', 'connect5', 'connect4-miss1', 'connect4-zigzag']
to_game = sys.argv[1]

clear = False

if len(sys.argv) > 2:
    clear = True

propnets = dict()
expertModels = dict()
gameData = dict()
players = dict()

totalRounds = 1601
rotateEvery = 50

replay_buffer = dict()

for game in expertGames:
    replay_buffer[game] = pickle.load(open( "buffers/" + game + "2500for400rounds.p", "rb" ) )

data, propnet = load_propnet(to_game)
mnModel = Model(propnet, multiNet=True, games=expertGames, replay_buffer=replay_buffer)

if clear:
    print("CLEARING")
    mnModel.clear_output_layer()

## train
cur = [None]

mnModel.save(to_game,0,multiNet=True)
start = time.time()
gameIndex = 0
for i in range(totalRounds+1):
    if i % rotateEvery == 0:
        gameIndex = (gameIndex + 1) % len(expertGames)
        currGame = expertGames[gameIndex]
    
    print('training round', i, currGame)
    start_train = time.time()
    mnModel.train(epochs=10, game=currGame)
    print("took ", time.time()-start_train, "seconds to train")
    if i and i % 50 == 0:
        mnModel.save(to_game, i, multiNet=True)
        with open(f'models_transfer/times-{to_game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
