from player import Player, run_player
from model import Model
from b1 import B1Node, simulation


class B1Player(Player):
    def start(self, endtime):
        self.model = Model(self.propnet)
        self.model.load_most_recent(self.match_id)

    def play(self, endtime):
        root = B1Node(self.propnet, model=self.model)
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
    run_player(B1Player, 'b1player')
