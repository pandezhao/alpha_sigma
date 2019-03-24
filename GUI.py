import pygame
import os
import utils
import torch
from new_MCTS import MCTS
import argparse

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="display", type=str, help="decide which mode too choose, we can choose:display, game")
    parser.add_argument("--display_file", type=str, help="if we choose display mode, we must select a file to display")
    parser.add_argument("--game_model", type=str, help="if we choose game mode, we must select a trained model to use")
    args = parser.parse_args()
    if args.mode == "display":
        record_file = args.display_file
        oppo = None
    elif args.mode == "game":
        oppo = args.game_model
        record_file = None
    else:
        raise KeyError("we must select a mode between 'display' and 'game'.")
    board_size = 8
    if record_file and not oppo:
        file = utils.read_file(record_file)
        file_record = []
        for i in file:
            file_record.append(i["action"])
        step = 0
        record_file_label = True
    if oppo:
        try:
            Net = torch.load(oppo)
            tree = MCTS(board_size=board_size, neural_network=Net)
        except:
            raise ValueError("The parameter oppo must be a pretrained model")

    GRID_WIDTH = 36
    WIDTH = (board_size+2) * GRID_WIDTH
    HEIGHT = (board_size+2) * GRID_WIDTH
    FPS = 30

    if not oppo:
        record = np.zeros([board_size, board_size])
    else:
        record, game_continue = tree.interact_game_init()

    # define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    # RED = (255, 0, 0)
    # GREEN = (0, 255, 0)
    # BLUE = (0, 0, 255)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("五子棋")
    clock = pygame.time.Clock()

    all_sprites = pygame.sprite.Group()

    base_folder = os.path.dirname(__file__)

    img_folder = os.path.join(base_folder, 'images')
    background_img = pygame.image.load(os.path.join(img_folder, 'back.png')).convert()

    background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
    back_rect = background.get_rect()

    def draw_background(surf):
        screen.blit(background, back_rect)
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

        for i in range(board_size):
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
        ]
        for cc in circle_center:
            pygame.draw.circle(surf, BLACK, cc, 5)


    running = True

    def draw_stone(screen):
        for i in range(board_size):
            for j in range(board_size):
                if record[i,j] == 1:
                    pygame.draw.circle(screen, BLACK, (int((i+1.5)*GRID_WIDTH), int((j+1.5)*GRID_WIDTH)), 16)
                elif record[i,j] == -1:
                    pygame.draw.circle(screen, WHITE, (int((i + 1.5) * GRID_WIDTH), int((j + 1.5) * GRID_WIDTH)), 16)

    def visual_update(matrix, file_record, step):
        if step>= len(file_record):
            return False
        else:
            if step % 2 == 0:
                stone = 1
            else:
                stone = -1
            ss = utils.str_to_move(file_record[step])
            matrix[ss[0], ss[1]] = stone
            return True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not record_file and not oppo:
                    pos = event.pos
                    if pos[0] < GRID_WIDTH or pos[1]<GRID_WIDTH or pos[0]>WIDTH-GRID_WIDTH or pos[1] > HEIGHT - GRID_WIDTH:
                        pass
                    else:
                        grid = (int((pos[0]-GRID_WIDTH)/GRID_WIDTH), int((pos[1]-GRID_WIDTH)/GRID_WIDTH))
                        record[grid[0], grid[1]] = 1
                if oppo:
                    if game_continue:
                        pos = event.pos
                        if pos[0] < GRID_WIDTH or pos[1] < GRID_WIDTH or pos[0] > WIDTH - GRID_WIDTH or pos[
                            1] > HEIGHT - GRID_WIDTH:
                            pass
                        else:
                            grid = (int((pos[0] - GRID_WIDTH) / GRID_WIDTH), int((pos[1] - GRID_WIDTH) / GRID_WIDTH))
                            record, game_continue = tree.interact_game1(grid)
                            draw_background(screen)
                            draw_stone(screen)
                            pygame.display.flip()
                            record, game_continue = tree.interact_game2(grid, game_continue, record)
                    else:
                        pass

                else:
                    if record_file_label:
                        record_file_label=visual_update(record, file_record, step)
                        step += 1
                    else:
                        running = False
        draw_background(screen)
        draw_stone(screen)

        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
