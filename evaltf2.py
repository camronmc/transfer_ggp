from b1 import B1Node, simulation
from mcts import MCTSNode
from propnet.propnet import load_propnet
from modeltf2 import Model
import numpy as np
import random
import time

# game = 'pacman3psmall'
# game = 'breakthroughSmall'
game = 'connect4'
# game = 'babel'
models_base = 'models/'
# models_base = '/home/adrian/honours-models/'
# models = ['connect4/connect4-z.5/connect4-z.5']
# models = ['pacman3p/pacman3psmall-z.5']
# models = [game+'-z.5']
# models = [game+'-init']
models = [game]


TIME_BASED = True
# TIME_BASED = False

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
    # board = [list('.'*8) for i in range(6)]
    for step in range(1000):
        # print(*(''.join(b) for b in board[::-1]), sep='\n')
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
                # if 'drop' in move.move_gdl:
                    # col = int(move.move_gdl.split()[2]) - 1
                    # for i in range(len(board)):
                        # if board[i][col] == '.':
                            # board[i][col] = move.move_role[0]
                            # break
        if curb1.terminal:
            # print(*(''.join(b) for b in board[::-1]), sep='\n')
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



checkpoints = list(range(0, 9000, 50))[1:]
# checkpoints = [6000]
# checkpoints = list(range(1050, 5001, 50))
data, propnet = load_propnet(game)
model = Model(propnet, game)
print(len(propnet.propositions))
print(len(propnet.nodes))

if game == 'breakthroughSmall':
    role_a = ('black',)
    role_b = ('white',)
elif game == 'connect4':
    role_a = ('red',)
    role_b = ('white',)
elif game.startswith('pacman3p'):
    role_a = ('inky', 'blinky')
    role_b = ('pacman',)
elif game == 'babel':
    role_a = ('builder1', 'builder2', 'builder3')
    role_b = ()

for model_name in models:
    for ckpt in checkpoints:
        print(ckpt)
        model.load(models_base+model_name+'/'+model_name+'-%d.h5'%ckpt, with_training=False)
        res = eval_game(2, 2, model, 25, 2)
        print(*res)
        results = ",".join(map(str, res))
        with open('results'+model_name+'.csv', 'a') as f:
            f.write(f'{model_name},{ckpt},{results}\n')
exit(0)

# role_a = ('builder1', 'builder2')
# role_b = ('builder3',)

# for model_name in models:
#     for ckpt in checkpoints:
#         print(ckpt)
#         model.load(models_base+model_name+'/'+model_name+'-%d.ckpt'%ckpt)
#         res = eval_game(2, 2, model, 25, 2)
#         print(*res)
#         results = ",".join(map(str, res))
#         with open('results'+model_name+'.csv', 'a') as f:
#             f.write(f'{model_name},{ckpt},{results}\n')
