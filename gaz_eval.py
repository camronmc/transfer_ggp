from b1 import B1Node, simulation
from mcts import MCTSNode
from propnet.propnet import load_propnet
from model import Model
import numpy as np
import random
import time
import sys, os, pathlib

if len(sys.argv) < 4:
    print("Usage: transfer.py <base_game> <start> <finish>")

game = sys.argv[1]
models_base = 'models/'
start = sys.argv[2]
fin = sys.argv[3]
models = [game]


TIME_BASED = True
# TIME_BASED = False
random.seed(74)


def choose_move(cur, roles, N, legal, rand=False):
    if not roles:
        return ()
    if rand:
        return tuple(random.choice(legal[role]).id for role in roles)
    if all(len(legal[role]) == 1 for role in roles):
        return tuple(legal[role][0].id for role in roles)
    end = time.time() + N
    if TIME_BASED:
        c = 0
        while time.time() < end:
            simulation(cur)
            c += 1
    else:
        c = N
        for i in range(int(N)):
            simulation(cur)
    print(roles, 'did', c, 'simulations, took', time.time() - end + N, 'seconds')
    return tuple(cur.get_final_move(role) for role in roles)


def run_game(b1_role, mcts_role, b1_N, mcts_N, model, rand=0):
    print(f'b1 ({" ".join(b1_role)}), N={b1_N}')
    print(f'mcts ({" ".join(mcts_role)}), N={mcts_N}')
    curb1 = B1Node(propnet, data, model=model)
    curmcts = MCTSNode(propnet, data)
    for step in range(1000):
        legal = curb1.propnet.legal_moves_dict(curb1.data)
        b1_moves = choose_move(curb1, b1_role, b1_N, legal, step < rand)
        mcts_moves = choose_move(curmcts, mcts_role, mcts_N, legal, step < rand)
        taken_moves = dict(list(zip(b1_role, b1_moves)) + list(zip(mcts_role, mcts_moves)))
        moves = tuple(taken_moves[role] for role in propnet.roles)
        curb1 = curb1.get_or_make_child(moves)
        curmcts = curmcts.get_or_make_child(moves)
        print('Moves were:')
        for move in propnet.legal:
            if move.id in moves and move.move_gdl.strip() != 'noop':
                print(move.move_role, move.move_gdl)
                if 'drop' in move.move_gdl:
                    col = int(move.move_gdl.split()[2]) - 1
        if curb1.terminal:
            break
    print('Results:', curb1.scores)
    return (
        sum(curb1.scores[role] for role in b1_role),
        sum(curb1.scores[role] for role in mcts_role)
    )


def eval_game(b1_N, mcts_N, model, X, rand):
    totalb1g = 0
    totalb1p = 0
    totalmctsg = 0
    totalmctsp = 0
    allb1g = []
    allb1p = []
    allmctsg = []
    allmctsp = []
    for i in range(X):
        print("eval: " + str(i) + " of " + str(X))  
        b1score, mctsscore = run_game(role_a, role_b, b1_N, mcts_N, model, rand)
        allb1g.append(b1score)
        allmctsp.append(mctsscore)
        totalb1g += b1score
        totalmctsp += mctsscore
        b1score, mctsscore = run_game(role_b, role_a, b1_N, mcts_N, model, rand)
        allb1p.append(b1score)
        allmctsg.append(mctsscore)
        totalb1p += b1score
        totalmctsg += mctsscore
        print(totalb1g+totalb1p, 'vs', totalmctsp+totalmctsg)
        print(totalb1g, totalb1p)
        print(totalmctsg, totalmctsp)

    return totalb1g, totalb1p, totalmctsg, totalmctsp, np.std(allb1g), np.std(allb1p), np.std(allmctsg), np.std(allmctsp)



checkpoints = list(range(int(start),int(fin),50))
data, propnet = load_propnet(game)
model = Model(propnet)
print(len(propnet.propositions))
print(len(propnet.nodes))

if 'breakthrough' in game.lower():
    role_a = ('black',)
    role_b = ('white',)
elif 'nim' in game.lower():
    role_a = ('player1',)
    role_b = ('player2',)
elif 'connect' in game.lower():
    role_a = ('red',)
    role_b = ('white',)
elif 'chinesecheckers1' == game.lower():
    role_a = ('red',)
    role_b = ('red',)
elif 'chinesecheckers2' == game.lower():
    role_a = ('red',)
    role_b = ('teal',)
elif 'chinesecheckers3' == game.lower():
    role_a = ('red',)
    role_b = ('green',)
    role_c = ('green',)

path = os.path.join('eval', "")
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

for model_name in models:
    for ckpt in checkpoints:
        print(ckpt)
        model.load(models_base+model_name+'/step-%06d.ckpt'%ckpt)
        res = eval_game(2, 2, model, 25, 2)
        print(*res)
        results = ",".join(map(str, res))
        with open(path + model_name+str(start)+str(fin)+'results.csv', 'a') as f:
            f.write(f'{model_name},{ckpt},{results}\n')
exit(0)

