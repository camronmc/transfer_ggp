from player import Player, run_player
from mcts import MCTSNode, simulation


class UCTPlayer(Player):
    def start(self, endtime):
        pass

    def play(self, endtime):
        root = MCTSNode(self.propnet)
        for i in range(500):
            simulation(root)
        root.print_node()
        best, choice = -1, None
        for i, c in root.move_counts[self.role].items():
            if c > best:
                best, choice = c, i
        move = self.propnet.id_to_move[choice].move_gdl
        print('Made move', move)
        return move


if __name__ == '__main__':
    run_player(UCTPlayer, 'uct-player')
