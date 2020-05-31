from player import Player, run_player

class LegalPlayer(Player):
    def play(self):
        move = next(self.propnet.legal_moves_for(self.role)).move_gdl
        print('Made move', move)
        return move

if __name__ == '__main__':
    run_player(LegalPlayer, 'player')
