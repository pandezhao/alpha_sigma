from MCTS import MCTS
from five_stone_game import main_process
import utils

 #child node的action我似乎是写错了，每一个node的child之内对应的每一个child node之中都应该有一个action
def main(board_size=11,tree_file=None):
    if tree_file:
        tree = utils.read_file(tree_file)
    else:
        tree = MCTS(board_size=board_size)
    game = main_process(board_size=board_size)
    while True:
        game_continue = True
        next_move = tree.MCTS_step(last_step=None, board=None, game_continue=True)
        while game_continue:
                game_continue, board_state = game.step(next_move)
                next_move=tree.MCTS_step(last_step=next_move, board=board_state, game_continue=game_continue)
        tree.restart()
        game.renew()
        if game.games_counter % 1000 == 0:
            print("We have played {} games, black wins:{}, white wins:{}, heqi:{}".
                  format(game.games_counter, game.games_black_win_counter, game.games_white_win_counter,
                         game.no_win_games_counter))
            print("And in the last 1000 games, black wins:{}, white wins:{}, heqi:{}".
                  format(game.tmp_black_win, game.tmp_white_win, game.tmp_heqi))
            game.clear_tmp()
            print(" ")
        if game.games_counter % 5e4 == 0:
            utils.write_file(tree, 'MCTS_{}.pkl'.format(game.games_counter))
main(tree_file="MCTS_400000.pkl")