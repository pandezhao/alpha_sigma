from new_MCTS import MCTS
import time
import utils
from network import neuralnetwork as nn
import os

import torch
 #child node的action我似乎是写错了，每一个node的child之内对应的每一个child node之中都应该有一个action


def main(board_size=11,tree_file=None, pretrained_model=None, game_file_saved_dict="game_record"):
    if not os.path.exists(game_file_saved_dict):
        os.mkdir(game_file_saved_dict)
    if pretrained_model:
        Net = torch.load(pretrained_model)
    else:
        Net = nn(input_layers=1, board_size=11)
    stack = utils.random_stack()
    if tree_file:
        tree = utils.read_file(tree_file)
    else:
        tree = MCTS(board_size=board_size, neural_network=Net)

    for game_time in range(100):
        game_record = tree.game()
        if len(game_record) % 2 == 1:
            print("game {} completed, black win, this game length is {}".format(game_time, len(game_record)))
        else:
            print("game {} completed, win win, this game length is {}".format(game_time, len(game_record)))
        utils.write_file(game_record, game_file_saved_dict + "/"+time.strftime("%Y%m%d-%H-%M-%S", time.localtime()))
        train_data = utils.generate_training_data(game_record=game_record, board_size=11)
        for i in range(len(train_data)):
            stack.push(train_data[i])
        my_loader = utils.generate_data_loader(stack)
        Net.train(my_loader, game_time)
    torch.save(Net, "model_"+time.strftime("%Y%m%d-%H-%M", time.localtime())+".pkl")

main()

# Net = nn(input_layers=1, board_size=11)
# stack = utils.random_stack()
# game_record = utils.read_file("20190223-17-14-05")
# train_data = utils.generate_training_data(game_record=game_record, board_size=11)
# # test=train_data[0]["distribution"]
# # import numpy as np
# # test = np.array(test)
# # aa = np.sum(test)
# # test = np.exp(test)
# # test = test / np.sum(test)
# # bb = np.sum(test)
#
# for i in range(len(train_data)):
#     stack.push(train_data[i])
# my_loader = utils.generate_data_loader(stack)
# Net.train(my_loader, 1)
print("here we are")