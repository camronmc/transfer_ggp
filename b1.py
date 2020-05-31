from mcts import MCTSNode, simulation
import random
import time
import numpy as np


class B1Node(MCTSNode):
    def __init__(self, propnet, data, actions=None, model=None, root=False):
        super().__init__(propnet, data, actions=actions, sum_start=0.5)
        self.model = model
        self.root = root
        self.children_args = [self.model]
        state = self.propnet.get_state(self.data)
        self.priors, self.pred_scores = model.eval(state)
        self.C = 2

    def make_root(self):
        # if root, generate dirichlet noise to add
        self.root = True
        self.dirichlet = {}
        for role in self.propnet.roles:
            nmoves = len(self.actions[role])
            alpha = 10/(nmoves + 8)
            self.dirichlet[role] = np.random.dirichlet([alpha] * nmoves)

    def choose_action(self, role):
        numerator = self.count ** 0.5
        best = float('-inf')
        ops = [None]
        # print('Choosing action for', role)
        # print('priors', self.priors[role])
        for i, move in enumerate(self.actions[role]):
            # print('Looking at', move.move_gdl)
            denom = self.move_counts[role][move.id] + 1
            # print('move count:', denom)
            if denom == 0:
                return move
            Q = self.win_sums[role][move.id]/denom
            # if denom == 1:
                # Q = self.pred_scores[role]
                # Q = 0.5
            # print('Win sum:', self.win_sums[role][move.id])
            # print('Q:', Q)
            # import pdb; pdb.set_trace()
            prior = self.priors[role][move.id]
            if self.root:
                noise_weight = 0.25
                d = self.dirichlet[role][i]
                prior = (1-noise_weight) * prior + noise_weight * d
            # print('Prior:', prior)
            val = Q + self.C * prior * numerator/denom
            # print('UCB:', prior * numerator/denom)
            # print('Combination val:', val)
            if val > best:
                best = val
                ops = [move]
            elif val == best:
                ops.append(move)
        return random.choice(ops)

    def get_pred_scores(self):
        return self.pred_scores

    def get_probs(self, tau):
        return {
            role: {i: c**tau for i, c in counts.items()}
            for role, counts in self.move_counts.items()
        }


def make_choice(moves, probs, step):
    total = sum(probs.values())
    if step < 1e9:
        choice = random.random() * total
        for i, pr in probs.items():
            choice -= pr
            if choice <= 0:
                return i
    # else:
        # if random.random() < 0.9:
            # return make_choice(moves, probs, -1)
        # return random.choice(moves.values()).move_gdl


def average(scores, q, z):
    return {role: z*sc + (1-z)*q[role]
            for role, sc in scores.items()}


def do_game(curl, propnet, model, z=1, N=300):
    cur = curl[0]
    states = []
    board = [list('.'*8) for i in range(6)]
    for step in range(1000):
        print(*(''.join(b) for b in board[::-1]), sep='\n')
        model.print_eval(propnet.get_state(cur.data))
        cur.make_root()
        start = time.time()
        for i in range(N):
            simulation(cur)
        probs = cur.get_probs(1)
        print('Counts were:')
        for role, counts in probs.items():
            print('For', role)
            for id, count in counts.items():
                print(propnet.id_to_move[id].move_role, propnet.id_to_move[id].move_gdl, count)
            print('New expected return:', cur.q[role]/cur.count)
        # if any(sum(x.values()) < 10 for x in probs.values()):
            # import pdb; pdb.set_trace()
        formatted_probs = {}
        # print('Probs were:')
        for role in propnet.roles:
            # print('For', role)
            formatted_probs[role] = [0] * len(propnet.legal_for[role])
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in probs[role]:
                    formatted_probs[role][i] = probs[role][legal.id]
            total = sum(formatted_probs[role])
            if total == 0:
                total = 1
            for i, prob in enumerate(formatted_probs[role]):
                # print(propnet.legal[i].move_gdl, prob/total)
                formatted_probs[role][i] = prob/total
        state = propnet.get_state(cur.data)
        qs = {role: q/cur.count for role, q in cur.q.items()}
        states.append((state, formatted_probs, qs))
        moves = []
        for role in propnet.roles:
            moves.append(make_choice(cur.actions[role], probs[role], step))
        # cur = cur.children[tuple(moves)]
        cur = cur.get_or_make_child(tuple(moves))
        curl[0] = cur
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
    scores = cur.scores
    for s, p, q in states:
        model.add_sample(s, p, average(scores, q, z))
