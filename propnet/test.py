from .persistent_array import PersistentArray
from .propnet import load_propnet
import random


def run_test_array(N=1000, T=100):
    o = p = PersistentArray(N)
    ol = list(o.values())
    pl = list(p.values())
    for i in range(T):
        i = random.randint(1, N)
        c = p.copy()
        c[i] = 4
        cl = list(c.values())
        assert ol == list(o.values())
        assert pl == list(p.values())
        assert cl != ol
        t = pl[i]
        pl[i] = 4
        assert cl == pl
        pl[i] = t

        pl = cl
        p = c


def run_test_propnet():
    d, p = load_propnet('connectFour')
    datas = [d]

    def step(actions):
        print('STEP')
        d = datas[-1]
        for move in p.legal_moves(d):
            print(move)
        if p.is_terminal(d):
            print('terminal')
            print(p.get_scores(d))
        d = d.copy()
        p.do_step(d, actions)
        datas.append(d)
    
    step(['noop', '(drop 1)'])
    step(['(drop 2)', 'noop'])
    step(['noop', '(drop 1)'])
    step(['(drop 2)', 'noop'])
    step(['noop', '(drop 1)'])
    step(['(drop 2)', 'noop'])
    step(['noop', '(drop 1)'])
    print('Done?')
    d = datas[-1]
    for move in p.legal_moves(d):
        print(move)
    if p.is_terminal(d):
        print('terminal')
        print(p.scores(d))
    return p, datas



def print_state(data):
    board = [[[] for i in range(7)] for j in range(7)]
    for b in propnet.base:
        *_, c, r, p, _, _ = b.gdl.split()
        if 'cell' in b.gdl and data[b.id]:
            board[int(r)][int(c)-1].append(p)
    print(board)
    for row in board[::-1]:
        print(' '.join(x[0][0] if x else '.' for x in row))



print('Testing array')
run_test_array()
print('Testing propnet')
p, datas = run_test_propnet()
