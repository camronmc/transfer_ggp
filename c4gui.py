import pygame
from b1 import B1Node, simulation
from propnet.propnet import load_propnet
from model import Model
import time
import sys
import re


def show_current_player(turn):
    if cur.terminal:
        return
    if turn == 0:
        screen.blit(your_turn, (1300, 10))
    else:
        screen.blit(computer_turn, (1300, 10))

    col = [your_role, my_role][turn]
    if col == 'red':
        pygame.draw.circle(screen, BLUE, [1425, 350], 110, 10)
    else:
        pygame.draw.circle(screen, BLUE, [125, 350], 110, 10)


def show_state(state):
    screen.fill(WHITE)
    screen.blit(RESET, (0, 5))
    screen.blit(reset_text, (70, 15))
    pygame.draw.rect(screen, BLUE, [1, 1, 220, 60], 2)
    screen.blit(left_player, (60, 300))
    screen.blit(right_player, (1360, 300))
    pygame.draw.rect(screen, BLUE, [250, 50, 1050, 900])
    pygame.draw.circle(screen, GREEN, [125, 400], 50)
    pygame.draw.circle(screen, RED, [1425, 400], 50)
    X = 130
    for i, row in enumerate(state):
        for j, cell in enumerate(row):
            COL = cols[cell]
            pygame.draw.circle(screen, COL, [250 + j*X + X, 50 + i*X + X], 50)
    if cur.terminal:
        if cur.scores['red'] > 0:
            pygame.draw.rect(screen, YELLOW, [1315, 270, 220, 220], 10)
        if cur.scores['white'] > 0:
            pygame.draw.rect(screen, YELLOW, [15, 270, 220, 200], 10)
        if cur.scores[your_role] == 1:
            screen.blit(human_win, (600, 15))
        elif cur.scores[my_role] == 1:
            screen.blit(computer_win, (600, 15))
        else:
            screen.blit(draw, (600, 15))


def clean(state):
    for row in state:
        for j, cell in enumerate(row):
            if cell == 'g':
                row[j] = '.'


def drop(state, col, player):
    clean(state)
    for row in state[::-1]:
        if row[col] == '.':
            row[col] = player
            return True
    return False


def get_computer_move():
    start = time.time()
    model.print_eval(propnet.get_state(cur.data))
    for i in range(N):
        simulation(cur)
    move_id = cur.get_final_move(my_role)
    move_gdl = propnet.id_to_move[move_id].move_gdl
    move = re.findall(r'\d', move_gdl)[0]
    end = time.time()
    print('Play took', end - start, 'seconds')
    return int(move) - 1


def get_move(legal, role, col):
    if col is None:
        assert len(legal[role]) == 1
        return legal[role][0]
    else:
        moves = legal[role]
        matching = [move for move in moves if str(col+1) in move.move_gdl]
        return matching[0]


def make_moves(my_col, your_col):
    legal = cur.propnet.legal_moves_dict(cur.data)
    my_move = get_move(legal, my_role, my_col)
    your_move = get_move(legal, your_role, your_col)
    taken_moves = {my_role: my_move, your_role: your_move}
    moves = [taken_moves[role].id for role in propnet.roles]
    return tuple(moves)


def check_terminal():
    if cur.terminal:
        print('DONE')
        print(cur.scores)
        print(cur.scores[your_role])
    return cur.terminal


def save_results():
    if cur.terminal:
        with open('c4.txt', 'a') as f:
            print(cur.scores[my_role], file=f)
    else:
        with open('c4-half.txt', 'a') as f:
            print(my_role, state, file=f)


game = 'connect4'
your_role = sys.argv[1]

print("We're playing", game)
print('You are', your_role)

data, propnet = load_propnet(game)
model = Model(propnet)
model.load_most_recent(game)
# model.load('models/connect4/step-004650.ckpt')

# Initialize the game engine
pygame.init()

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
YELLOW =(  255, 255, 0)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
GREY =  (192, 192, 192)

cols = {
    '.': WHITE,
    'r': RED,
    'w': GREEN,
    'g': GREY,
}

RESET = pygame.image.load('reset.png')
RESET = pygame.transform.scale(RESET, (60, 50))
pygame.font.init()
myfont = pygame.font.SysFont('Arial', 30)
reset_text = myfont.render('New game', False, (0, 0, 0))
a0_text = myfont.render('AlphaZero', False, (0, 0, 0))
human_text = myfont.render('    You  ', False, (0, 0, 0))
your_turn = myfont.render('Your turn', False, (0, 0, 0))
computer_turn = myfont.render("Computer's turn", False, (0, 0, 0))
computer_win = myfont.render("You lost!", False, (0, 0, 0))
human_win = myfont.render("You won!", False, (0, 0, 0))
draw = myfont.render("Draw!", False, (0, 0, 0))
size = [1550, 1000]
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Connect-4 against AlphaZero')
clock = pygame.time.Clock()
HUMAN = your_role[0]
my_role = list(set(propnet.roles) - {your_role})[0]
COMPUTER = my_role[0]
left_player, right_player = human_text, a0_text

N = 500

col = None
done = False
cur = B1Node(propnet, data, model=model)
state = [list('.'*7) for i in range(6)]

if my_role == 'white':
    show_state(state)
    move = 3  # get_computer_move()
    drop(state, move, COMPUTER)
    moves = make_moves(move, None)
    cur = cur.get_or_make_child(moves)
    left_player, right_player = right_player, left_player

while not done:

    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True
            break
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[0] <= 220 and event.pos[1] <= 60:
                if not cur.terminal:  # otherwise already saved
                    save_results()
                col = None
                state = [list('.'*7) for i in range(6)]
                show_state(state)
                HUMAN, COMPUTER = COMPUTER, HUMAN
                my_role, your_role = your_role, my_role
                left_player, right_player = right_player, left_player
                cur = B1Node(propnet, data, model=model)
                if my_role == 'white':
                    move = 3  # get_computer_move()
                    drop(state, move, COMPUTER)
                    moves = make_moves(move, None)
                    cur = cur.get_or_make_child(moves)
                continue
            if cur.terminal:
                continue
            if col is not None and drop(state, col, HUMAN):
                show_state(state)
                show_current_player(1)
                pygame.display.flip()
                moves = make_moves(None, col)
                cur = cur.get_or_make_child(moves)
                if check_terminal():
                    save_results()
                    continue
                print('Computers turn')
                move = get_computer_move()
                drop(state, move, COMPUTER)
                moves = make_moves(move, None)
                cur = cur.get_or_make_child(moves)
                if check_terminal():
                    save_results()
                    continue
        elif event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            col = (x-250) / 130 - 0.5
            if 0 < col < 7:
                col = int(col)
                if cur.terminal:
                    clean(state)
                else:
                    drop(state, col, 'g')
            else:
                clean(state)
                col = None

    show_state(state)
    show_current_player(0)
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
