from b1 import B1Node, simulation
from mcts import MCTSNode
from propnet.propnet import load_propnet
from model import Model
import numpy as np
import random
import time
import sys

game = sys.argv[1]
start = int(sys.argv[2])
fin = int(sys.argv[3])
models_base = 'models_ad/'
models = [game]


TIME_BASED = True
# TIME_BASED = False

def choose_move(cur, roles, N, legal, rand=False):
    if not roles:
        return ()
    if rand:
        random.seed(74)
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


def run_game(b1_role, mcts_role, mcts2_role, b1_N, mcts_N, mcts2_N, model, rand=0):
    print(f'b1 ({" ".join(b1_role)}), N={b1_N}')
    print(f'mcts ({" ".join(mcts_role)}), N={mcts_N}')
    print(f'mcts2 ({" ".join(mcts2_role)}), N={mcts2_N}')
    curb1 = B1Node(propnet, data, model=model)
    curmcts = MCTSNode(propnet, data)
    curmcts2 = MCTSNode(propnet, data)
    board = [list('.'*10) for i in range(8)]
    for step in range(1000):
        print(*(''.join(b) for b in board[::-1]), sep='\n')
        legal = curb1.propnet.legal_moves_dict(curb1.data)
        b1_moves = choose_move(curb1, b1_role, b1_N, legal, step < rand)
        mcts_moves = choose_move(curmcts, mcts_role, mcts2_N, legal, step < rand)
        mcts2_moves = choose_move(curmcts2, mcts2_role, mcts_N, legal, step < rand)
        taken_moves = dict(list(zip(b1_role, b1_moves)) + list(zip(mcts_role, mcts_moves)) + list(zip(mcts2_role, mcts2_moves)))
        moves = tuple(taken_moves[role] for role in propnet.roles)
        curb1 = curb1.get_or_make_child(moves)
        curmcts = curmcts.get_or_make_child(moves)
        curmcts2 = curmcts2.get_or_make_child(moves)
        print('Moves were:')
        for move in propnet.legal:
            if move.id in moves and move.move_gdl.strip() != 'noop':
                print(move.move_role, move.move_gdl)
                if 'drop' in move.move_gdl:
                    col = int(move.move_gdl.split()[2]) - 1
                    for i in range(len(board)):
                        if board[i][col] == '.':
                            board[i][col] = move.move_role[0]
                            break
        if curb1.terminal:
            print(*(''.join(b) for b in board[::-1]), sep='\n')
            break
    print('Results:', curb1.scores)
    return (
        sum(curb1.scores[role] for role in b1_role),
        sum(curb1.scores[role] for role in mcts_role),
        sum(curb1.scores[role] for role in mcts2_role),
    )


def eval_game(b1_N, mcts_N, mcts2_N, model, X, rand):
    b1_rolea = []
    b1_roleb = []
    b1_rolec = []
    mcts_rolea = []
    mcts_roleb = []
    mcts_rolec = []
    mcts2_rolea = []
    mcts2_roleb = []
    mcts2_rolec = []
    for i in range(X):
        print("eval: " + str(i) + " of " + str(X))  
        b1score, mctsscore, mcts2score = run_game(role_a, role_b, role_c, b1_N, mcts_N, mcts2_N, model, rand)
        b1_rolea.append(b1score)
        mcts_roleb.append(mctsscore)
        mcts2_rolec.append(mcts2score)
        b1score, mctsscore, mcts2score = run_game(role_b, role_c, role_a,  b1_N, mcts_N, mcts2_N, model, rand)
        b1_roleb.append(b1score)
        mcts_rolec.append(mctsscore)
        mcts2_rolea.append(mcts2score)
        b1score, mctsscore, mcts2score = run_game(role_c, role_a, role_b,  b1_N, mcts_N, mcts2_N, model, rand)
        b1_rolec.append(b1score)
        mcts_rolea.append(mctsscore)
        mcts2_roleb.append(mcts2score)

        print(sum(b1_rolea) + sum(b1_roleb) + sum(b1_rolec), 'vs', sum(mcts_rolea) + sum(mcts_roleb) + sum(mcts_rolec), "vs",  sum(mcts2_rolea) + sum(mcts2_roleb) + sum(mcts2_rolec))
        print(sum(b1_rolea), sum(b1_roleb), sum(b1_rolec))
        print(sum(mcts_rolea),  sum(mcts_roleb) , sum(mcts_rolec))
        print(sum(mcts2_rolea),  sum(mcts2_roleb),  sum(mcts2_rolec))

    return sum(b1_rolea), sum(b1_roleb), sum(b1_rolec), sum(mcts_rolea),  sum(mcts_roleb) , sum(mcts_rolec), sum(mcts2_rolea),  sum(mcts2_roleb),  sum(mcts2_rolec), np.std(b1_rolea), np.std(b1_roleb), np.std(b1_rolec), np.std(mcts_rolea), np.std(mcts_roleb), np.std(mcts_rolec), np.std(mcts2_rolea), np.std(mcts2_roleb), np.std(mcts2_rolec)



# checkpoints = list(range(0, 1501, 50))[1:]
checkpoints = list(range(start,fin,50))
# checkpoints = list(range(1050, 5001, 50))
data, propnet = load_propnet(game)
model = Model(propnet)
print(len(propnet.propositions))
print(len(propnet.nodes))

role_a = ('red',)
role_b = ('white',)
role_c = ('green',)

for model_name in models:
    for ckpt in checkpoints:
        print(ckpt)
        model.load(models_base+model_name+'/step-%06d.ckpt'%ckpt)
        res = eval_game(2, 2, 2, model, 25, 2)
        print(*res)
        results = ",".join(map(str, res))
        with open(model_name+str(start)+str(fin)+'results.csv', 'a') as f:
            f.write(f'{model_name},{ckpt},{results}\n')
exit(0)
