from new_MCTS import MCTS
import time
import utils
from network import neuralnetwork as nn
import os
import matplotlib.pyplot as plt

import torch
 #child node的action我似乎是写错了，每一个node的child之内对应的每一个child node之中都应该有一个action


def main(tree_file=None, pretrained_model="model_100.pkl", game_file_saved_dict="game_record"):
    if not os.path.exists(game_file_saved_dict):
        os.mkdir(game_file_saved_dict)
    if pretrained_model:
        Net = torch.load(pretrained_model)
    else:
        Net = nn(input_layers=1, board_size=utils.board_size, learning_rate=utils.learning_rate)
    Net.adjust_lr(0.01)
    stack = utils.random_stack()
    if tree_file:
        tree = utils.read_file(tree_file)
    else:
        tree = MCTS(board_size=utils.board_size, neural_network=Net)
    record = []
    game_time = 100
    while True:
        game_record, eval, steps = tree.game()
        if len(game_record) % 2 == 1:
            print("game {} completed, black win, this game length is {}".format(game_time, len(game_record)))
        else:
            print("game {} completed, white win, this game length is {}".format(game_time, len(game_record)))
        print("The average eval:{}, the average steps:{}".format(eval, steps))
        utils.write_file(game_record, game_file_saved_dict + "/"+time.strftime("%Y%m%d-%H-%M-%S", time.localtime())+'_game_time:{}.pkl'.format(game_time))
        train_data = utils.generate_training_data(game_record=game_record, board_size=utils.board_size)
        for i in range(len(train_data)):
            stack.push(train_data[i])
        my_loader = utils.generate_data_loader(stack)
        record.extend(Net.train(my_loader, game_time))
        print("train finished")
        print(" ")
        game_time += 1
        if game_time % 50 == 0:
            torch.save(Net, "model_{}.pkl".format(game_time))
            test_game_record, _, _ = tree.game(train=False)
            utils.write_file(test_game_record, game_file_saved_dict + "/"+'test_{}.pkl'.format(game_time))
        plt.figure()
        plt.plot(record)
        plt.title("cross entropy loss")
        plt.xlabel("step passed")
        plt.ylabel("Loss")
        plt.savefig("loss record.svg")
        plt.close()

main()
print("here we are")