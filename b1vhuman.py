from b1 import B1Node, simulation
from propnet.propnet import load_propnet
from model import Model
import time
import sys

game = sys.argv[1]
my_role = sys.argv[2]

print("We're playing", game)
print('I am', my_role)

data, propnet = load_propnet(game)
model = Model(propnet)
model.load_most_recent(game)


board = [list('.'*8) for i in range(6)]
N = 500

cur = B1Node(propnet, data, model=model)
for step in range(1000):
    print(*(''.join(b) for b in board[::-1]), sep='\n')
    legal = cur.propnet.legal_moves_dict(cur.data)
    taken_moves = {}
    for role in propnet.roles:
        if role != my_role:
            moves = legal[role]
            if len(moves) > 1:
                print('Valid moves for', role)
                print(*(move.move_gdl for move in moves), sep='\n')
                m = input('Enter move: ')
                matching = [move for move in moves if m in move.move_gdl]
                while not matching:
                    print('No moves containing %r' % m)
                    m = input('Enter move: ')
                    matching = [move for move in moves if m in move.move_gdl]
                print('Making move', matching[0].move_gdl)
                taken_moves[role] = matching[0]
            else:
                taken_moves[role] = moves[0]

    if len(legal[my_role]) == 1:
        taken_moves[my_role] = legal[my_role][0]
        start = time.time()
    else:
        model.print_eval(propnet.get_state(cur.data))
        # cur.make_root()
        start = time.time()
        for i in range(N):
            simulation(cur)
        probs = cur.get_probs(1)
        taken_moves[my_role] = None
        best = 0
        print('Counts were:')
        counts = probs[my_role]
        for id, count in counts.items():
            print(propnet.id_to_move[id].move_role, propnet.id_to_move[id].move_gdl, count)
            if count > best:
                best = count
                taken_moves[my_role] = propnet.id_to_move[id]
    moves = [taken_moves[role].id for role in propnet.roles]

    cur = cur.get_or_make_child(tuple(moves))
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
    print('Play took %.4f seconds' % (time.time() - start))
    if cur.terminal:
        print(*(''.join(b) for b in board[::-1]), sep='\n')
        break

for role, score in cur.scores.items():
    print(role, 'got', score)
