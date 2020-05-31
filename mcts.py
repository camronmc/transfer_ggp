import math
import importlib
import random
from collections import defaultdict


class MCTSNode:
    def __init__(self, propnet, data, actions=None, sum_start=0):
        self.data = data.copy()
        self.propnet = propnet
        if actions is not None:
            self.propnet.do_step(self.data, actions)

        self.terminal = False
        if self.propnet.is_terminal(self.data):
            self.terminal = True
            self.scores = self.propnet.scores(self.data)
            if not self.scores:
                import pdb; pdb.set_trace()

        self.C = 2
        self.actions = self.propnet.legal_moves_dict(self.data)
        self.children = defaultdict(dict)
        self.children_args = []
        self.win_sums = {}
        self.move_counts = {}
        self.q = defaultdict(int)
        for role, moves in self.actions.items():
            self.win_sums[role] = {}
            self.move_counts[role] = {}
            for move in moves:
                self.win_sums[role][move.id] = sum_start
                self.move_counts[role][move.id] = 0
        self.count = 1

    def print_node(self):
        print(self.__class__.__name__)
        if self.terminal:
            print('Terminal, scores:', self.scores)
            return
        print('Count:', self.count)
        for role, moves in self.actions.items():
            print(role, end=':\n')
            for move in moves:
                print('%s: count = %d, win = %f' %
                      (move.move_gdl,
                       self.move_counts[role][move.id],
                       self.win_sums[role][move.id]))

    def choose_action(self, role):
        numerator = math.log(self.count) ** 0.5
        best = (-float('inf'), None)
        for move in self.actions[role]:
            denom = self.move_counts[role][move.id] ** 0.5
            if denom == 0:
                return move
            Q = self.win_sums[role][move.id]/self.move_counts[role][move.id]
            val = Q + self.C * numerator/denom
            # print(best, val, move)
            # import pdb; pdb.set_trace()
            if val > best[0]:
                best = (val, move)
        return best[1]

    def choose_actions(self):
        return {role: self.choose_action(role) for role in self.actions}

    def get_or_make_child(self, ids):
        if ids not in self.children:
            self.children[ids] = self.__class__(self.propnet, self.data, set(ids), *self.children_args)
        return self.children[ids]

    def get_child(self):
        actions = self.choose_actions()
        ids = tuple(actions[role].id for role in self.propnet.roles)
        if ids in self.children:
            return False, ids, self.children[ids]
        self.children[ids] = self.__class__(self.propnet, self.data, set(ids), *self.children_args)
        scores = self.children[ids].get_pred_scores()
        return True, ids, scores

    def get_pred_scores(self):
        return self.mc_rollout()

    def get_final_move(self, role):
        best, choice = -1, None
        for i, c in self.move_counts[role].items():
            if c > best:
                best, choice = c, i
        return choice

    def mc_rollout(self):
        copy = list(self.data.values())
        while not self.propnet.is_terminal(copy):
            options = self.propnet.legal_moves_dict(copy)
            chosen = set()
            for role, moves in options.items():
                chosen.add(random.choice(moves).input_id)
            self.propnet.do_step(copy, chosen)
        return self.propnet.scores(copy)

    def update(self, ids, scores):
        self.count += 1
        for role, id in zip(self.propnet.roles, ids):
            self.q[role] += scores[role]
            self.move_counts[role][id] += 1
            self.win_sums[role][id] += scores[role]

    def size(self):
        return 1 + sum(c.size() for c in self.children.values())


def simulation(root):
    stack = []
    stop, ids, child = root.get_child()
    prev = root
    if stop:
        stack.append((ids, prev))
    elif child.terminal:
        stack.append((ids, prev))
        child, stop = child.scores, True
    while not stop:
        stack.append((ids, prev))
        prev = child
        stop, ids, child = child.get_child()
        if not stop and child.terminal:
            child, stop = child.scores, True
    scores = child
    for ids, state in stack:
        state.update(ids, scores)

