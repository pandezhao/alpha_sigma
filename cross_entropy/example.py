import pygame
import os
import random

# from pprint import pprint
# import numpy as np

WIDTH = 720
HEIGHT = 720
GRID_WIDTH = WIDTH // 20
FPS = 30

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("五子棋")
clock = pygame.time.Clock()

all_sprites = pygame.sprite.Group()

base_folder = os.path.dirname(__file__)

# 加载各种资源
img_folder = os.path.join(base_folder, 'images')
background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()

background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
back_rect = background.get_rect()


# draw background lines
def draw_background(surf):
    # 加载背景图片
    screen.blit(background, back_rect)

    # 画网格线，棋盘为 19行 19列的
    # 1. 画出边框
    rect_lines = [
        ((GRID_WIDTH, GRID_WIDTH), (GRID_WIDTH, HEIGHT - GRID_WIDTH)),
        ((GRID_WIDTH, GRID_WIDTH), (WIDTH - GRID_WIDTH, GRID_WIDTH)),
        ((GRID_WIDTH, HEIGHT - GRID_WIDTH),
         (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
        ((WIDTH - GRID_WIDTH, GRID_WIDTH),
         (WIDTH - GRID_WIDTH, HEIGHT - GRID_WIDTH)),
    ]
    for line in rect_lines:
        pygame.draw.line(surf, BLACK, line[0], line[1], 2)

    for i in range(17):
        pygame.draw.line(surf, BLACK,
                         (GRID_WIDTH * (2 + i), GRID_WIDTH),
                         (GRID_WIDTH * (2 + i), HEIGHT - GRID_WIDTH))
        pygame.draw.line(surf, BLACK,
                         (GRID_WIDTH, GRID_WIDTH * (2 + i)),
                         (HEIGHT - GRID_WIDTH, GRID_WIDTH * (2 + i)))

    circle_center = [
        (GRID_WIDTH * 4, GRID_WIDTH * 4),
        (WIDTH - GRID_WIDTH * 4, GRID_WIDTH * 4),
        (WIDTH - GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
        (GRID_WIDTH * 4, HEIGHT - GRID_WIDTH * 4),
        (GRID_WIDTH * 10, GRID_WIDTH * 10)
    ]
    for cc in circle_center:
        pygame.draw.circle(surf, BLACK, cc, 5)


last_coin = None
movements = []
remain = set(range(1, 19 ** 2 + 1))

player_score_metrix = [[0] * 20 for i in range(20)]
ai_score_metrix = [[0] * 20 for i in range(20)]
color_metrix = [[None] * 20 for i in range(20)]

ai_possible_list = []
ai_optimal_list = []
ai_tabu_list = []
player_optimal_set = set()
player_tabu_list = []

USER, AI = 1, 0

score_level = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 1000000]
# font_name = pygame.font.match_font('华文黑体')
font_name = pygame.font.get_default_font()


def draw_text(surf, text, size, x, y, color=WHITE):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)


def update_score(pos, color, ident):
    hori = 1
    verti = 1
    slash = 1
    backslash = 1
    left = pos[0] - 1

    while left > 0 and color_metrix[left][pos[1]] == color:
        left -= 1
        if hori == 4:
            hori += 1
            break
        if left > 0 and \
                (color_metrix[left][pos[1]] == color or
                 color_metrix[left][pos[1]] is None):
            hori += 1

    right = pos[0] + 1
    while right < 20 and color_metrix[right][pos[1]] == color:
        right += 1
        if hori == 4:
            hori += 1
            break
        if right < 20 and \
                (color_metrix[right][pos[1]] == color or
                 color_metrix[right][pos[1]] is None):
            hori += 1

    hori = score_level[hori]

    up = pos[1] - 1
    while up > 0 and color_metrix[pos[0]][up] == color:
        up -= 1
        if verti == 4:
            verti += 1
            break
        if up > 0 and \
                (color_metrix[pos[0]][up] == color or
                 color_metrix[pos[0]][up] is None):
            verti += 1

    down = pos[1] + 1
    while down < 20 and color_metrix[pos[0]][down] == color:
        down += 1
        if verti == 4:
            verti += 1
            break
        if down < 20 and \
                (color_metrix[pos[0]][down] == color or
                 color_metrix[pos[0]][down] is None):
            verti += 1

    verti = score_level[verti]

    left = pos[0] - 1
    up = pos[1] - 1
    while left > 0 and up > 0 and color_metrix[left][up] == color:
        left -= 1
        up -= 1
        if backslash == 4:
            backslash += 1
            break
        if left > 0 and up > 0 and \
                (color_metrix[left][up] == color or
                 color_metrix[left][up] is None):
            backslash += 1

    right = pos[0] + 1
    down = pos[1] + 1
    while right < 20 and down < 20 and color_metrix[right][down] == color:
        right += 1
        down += 1
        if backslash == 4:
            backslash += 1
            break
        if right < 20 and down < 20 and \
                (color_metrix[right][down] == color or
                 color_metrix[right][down] is None):
            backslash += 1
    backslash = score_level[backslash]

    right = pos[0] + 1
    up = pos[1] - 1
    while right < 20 and up > 0 and color_metrix[right][up] == color:
        right += 1
        up -= 1
        if slash == 4:
            slash += 1
            break
        if right < 20 and up > 0 and (color_metrix[right][up] == color or
                                      color_metrix[right][up] is None):
            slash += 1

    left = pos[0] - 1
    down = pos[1] + 1
    while left > 0 and down < 20 and color_metrix[left][down] == color:
        left -= 1
        down += 1
        if slash == 4:
            slash += 1
            break
        if left > 0 and down < 20 and (color_metrix[left][down] == color or
                                       color_metrix[left][down] is None):
            slash += 1

    slash = score_level[slash]
    # print(pos, color, ident, (hori, verti, slash, backslash))

    if ident == USER:
        player_score_metrix[pos[0]][pos[1]] = \
            int((hori + verti + slash + backslash) * 0.9)
    else:
        ai_score_metrix[pos[0]][pos[1]] = hori + verti + slash + backslash


def game_is_over(pos, color):
    hori = 1
    verti = 1
    slash = 1
    backslash = 1
    left = pos[0] - 1
    while left > 0 and color_metrix[left][pos[1]] == color:
        left -= 1
        hori += 1

    right = pos[0] + 1
    while right < 20 and color_metrix[right][pos[1]] == color:
        right += 1
        hori += 1

    up = pos[1] - 1
    while up > 0 and color_metrix[pos[0]][up] == color:
        up -= 1
        verti += 1

    down = pos[1] + 1
    while down < 20 and color_metrix[pos[0]][down] == color:
        down += 1
        verti += 1

    left = pos[0] - 1
    up = pos[1] - 1
    while left > 0 and up > 0 and color_metrix[left][up] == color:
        left -= 1
        up -= 1
        backslash += 1

    right = pos[0] + 1
    down = pos[1] + 1
    while right < 20 and down < 20 and color_metrix[right][down] == color:
        right += 1
        down += 1
        backslash += 1

    right = pos[0] + 1
    up = pos[1] - 1
    while right < 20 and up > 0 and color_metrix[right][up] == color:
        right += 1
        up -= 1
        slash += 1

    left = pos[0] - 1
    down = pos[1] + 1
    while left > 0 and down < 20 and color_metrix[left][down] == color:
        left -= 1
        down += 1
        slash += 1

    if max([hori, verti, backslash, slash]) >= 5:
        return True


def add_coin(surf, color, pos, ident=USER, radius=16):
    num_pos = gridpos_2_num(pos)
    movements.append(((pos[0] * GRID_WIDTH, pos[1] * GRID_WIDTH), color))
    remain.remove(num_pos)
    if num_pos in player_optimal_set:
        player_optimal_set.remove(num_pos)

    player_score_metrix[pos[0]][pos[1]] = -1 - ident
    ai_score_metrix[pos[0]][pos[1]] = -1 - ident
    color_metrix[pos[0]][pos[1]] = color
    pygame.draw.circle(surf, color,
                       movements[-1][0], radius)
    clock.tick(FPS)

    around = around_grid(pos, 4)
    # print(around)
    for rx in range(around[0], around[1] + 1):
        for ry in range(around[2], around[3] + 1):
            num_pos = gridpos_2_num((rx, ry))
            if num_pos in remain:
                update_score((rx, ry), color, ident)
                if color == BLACK:
                    tpcolor = WHITE
                else:
                    tpcolor = BLACK
                update_score((rx, ry), tpcolor, 1 - ident)


def num_2_gridpos(num):
    return (1 + (num % 19), (num // 19) + 1)


def gridpos_2_num(grid):
    return (grid[1] - 1) * 19 + grid[0] - 1


def around_grid(curr_move_pos, step=2):
    left = curr_move_pos[0] - step if (curr_move_pos[0] - step) > 0 else 1
    right = curr_move_pos[0] + step if (curr_move_pos[0] + step) < 20 else 19
    top = curr_move_pos[1] - step if (curr_move_pos[1] - step) > 0 else 1
    bottom = curr_move_pos[1] + step if (curr_move_pos[1] + step) < 20 else 19

    return (left, right, top, bottom)


def get_next_move(movements, curr_move):
    around = around_grid((curr_move[0][0] // GRID_WIDTH,
                          curr_move[0][1] // GRID_WIDTH))

    for rx in range(around[0], around[1] + 1):
        for ry in range(around[2], around[3] + 1):
            num_pos = gridpos_2_num((rx, ry))
            if num_pos in remain:
                player_optimal_set.add(gridpos_2_num((rx, ry)))

    max_score = -1000000
    next_move = 0
    for i in player_optimal_set:
        grid = num_2_gridpos(i)
        if ai_score_metrix[grid[0]][grid[1]] >= score_level[5]:
            next_move = i
            break
        if player_score_metrix[grid[0]][grid[1]] >= score_level[4]:
            next_move = i
            break
        score = ai_score_metrix[grid[0]][grid[1]] + \
                player_score_metrix[grid[0]][grid[1]]

        if max_score < score:
            max_score = score
            next_move = i
        elif max_score == score:
            if (random.randint(0, 100) % 2) == 0:
                next_move = i

    around = around_grid(num_2_gridpos(next_move))

    for rx in range(around[0], around[1] + 1):
        for ry in range(around[2], around[1] + 1):
            num_pos = gridpos_2_num((rx, ry))
            if num_pos in remain:
                player_optimal_set.add(gridpos_2_num((rx, ry)))
    return next_move


def respond(surf, movements, curr_move):
    next_move = get_next_move(movements, curr_move)
    grid_pos = num_2_gridpos(next_move)

    add_coin(surf, WHITE, grid_pos, AI)

    if game_is_over(grid_pos, WHITE):
        return (False, AI)


def move(surf, pos):
    '''
    Args:
        surf: 我们的屏幕
        pos: 用户落子的位置
    Returns a tuple or None:
        None: if move is invalid else return a
        tuple (bool, player):
            bool: True is game is not over else False
            player: winner (USER or AI)
    '''
    grid = (int(round(pos[0] / (GRID_WIDTH + .0))),
            int(round(pos[1] / (GRID_WIDTH + .0))))

    if grid[0] <= 0 or grid[0] > 19:
        return
    if grid[1] <= 0 or grid[1] > 19:
        return

    pos = (grid[0] * GRID_WIDTH, grid[1] * GRID_WIDTH)

    # num_pos = gridpos_2_num(grid)
    # if num_pos not in remain:
    #     return None
    if color_metrix[grid[0]][grid[1]] is not None:
        return None

    curr_move = (pos, BLACK)
    add_coin(surf, BLACK, grid, USER)

    if game_is_over(grid, BLACK):
        return (False, USER)

    return respond(surf, movements, curr_move)


def draw_movements(surf):
    for move in movements[:-1]:
        pygame.draw.circle(surf, move[1], move[0], 16)
    if movements:
        pygame.draw.circle(surf, GREEN, movements[-1][0], 16)


def show_go_screen(surf, winner=None):
    note_height = 10
    if winner is not None:
        draw_text(surf, 'You {0} !'.format('win!' if winner == USER else 'lose!'),
                  64, WIDTH // 2, note_height, RED)
    else:
        screen.blit(background, back_rect)

    draw_text(surf, 'Five in row', 64, WIDTH // 2, note_height + HEIGHT // 4, BLACK)
    draw_text(surf, 'Press any key to start', 22, WIDTH // 2, note_height + HEIGHT // 2,
              BLUE)
    pygame.display.flip()
    waiting = True

    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYUP:
                waiting = False


game_over = True
running = True
winner = None
while running:
    if game_over:
        show_go_screen(screen, winner)
        game_over = False
        movements = []
        remain = set(range(1, 19 ** 2 + 1))

        player_score_metrix = [[0] * 20 for i in range(20)]
        ai_score_metrix = [[0] * 20 for i in range(20)]
        color_metrix = [[None] * 20 for i in range(20)]

        ai_possible_list = []
        ai_optimal_list = []
        ai_tabu_list = []
        player_optimal_set = set()
        player_tabu_list = []

    # 设置屏幕刷新频率
    clock.tick(FPS)

    # 处理不同事件
    for event in pygame.event.get():

        # 检查是否关闭窗口
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            response = move(screen, event.pos)
            if response is not None and response[0] is False:
                game_over = True
                winner = response[1]
                continue

    # Update
    all_sprites.update()

    # Draw / render
    # screen.fill(BLACK)
    all_sprites.draw(screen)
    draw_background(screen)
    draw_movements(screen)

    # After drawing everything, flip the display
    pygame.display.flip()

pygame.quit()