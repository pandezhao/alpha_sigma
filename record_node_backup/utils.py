import numpy as np
import pickle
import torch
import torch.utils.data as torch_data

import copy
import random

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

char2num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
            "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20}

temperature = 1
Cpuct = 0.1
batch_size = 20

class distribution_calculater:
    def __init__(self, size):
        self.map = {}
        self.order = []
        for i in range(size):
            for j in range(size):
                name = num2char[i]+num2char[j]
                self.order.append(name)
                self.map[name] = 0

    def push(self, key, value):
        self.map[key] = value

    def get(self):
        result = []
        for key in self.order:
            result.append(np.float_power(self.map[key], 1/temperature))
            self.map[key] = 0

        he = sum(result)
        for i in range(len(result)):
            if result[i]:
                result[i] = result[i] / he
        return self.order[np.argmax(result)], result

def step_child_remove(board_pool, child_pool):
    i = 0
    while i<len(board_pool) and len(child_pool) != 0:
        j = 0
        while j<len(child_pool):
            if np.array_equal(board_pool[i], child_pool[j]):
                board_pool.pop(i)
                child_pool.pop(j)
                i -= 1
                break
            else:
                j += 1
        i+=1
    return board_pool

def write_file(object, file_name):
    filewriter = open(file_name, 'wb')
    pickle.dump(object, filewriter)
    filewriter.close()

def read_file(file_name):
    filereader = open(file_name, 'rb')
    object = pickle.load(filereader)
    filereader.close()
    return object

def move_to_str(action):
    return num2char[action[0]] + num2char[action[1]]

def str_to_move(str):
    return np.array([char2num[str[0]], char2num[str[1]]])

def valid_move(state):
    return list(np.argwhere(state==0))

def generate_new_state(old_name, step, current_player):
    if current_player == 1:
        step = "B" + num2char[step[0]] + num2char[step[1]]
    else:
        step = "W" + num2char[step[0]] + num2char[step[1]]
    for i in range(0, len(old_name), 3):
        if old_name[i+1]>step[1] or (old_name[i+1]==step[1] and old_name[i+2]>step[2]):
            new_name = old_name[:i] + step + old_name[i:]
            return new_name
    new_name = old_name + step
    return new_name

class random_stack:
    def __init__(self, length=1000):
        self.state = []
        self.distrib = []
        self.winner = []
        self.length = length

    def isEmpty(self):
        return len(self.state) == 0

    def push(self, item):
        self.state.append(item["state"])
        self.distrib.append(item["distribution"])
        self.winner.append(item["value"])
        if len(self.state)>= self.length:
            self.state = self.state[1:]
            self.distrib = self.distrib[1:]
            self.winner = self.winner[1:]

    def seq(self):
        return self.state, self.distrib, self.winner

    # def random_seq(self):
    #     tmp = copy.deepcopy(self.data)
    #     random.shuffle(tmp)
    #     return tmp

def generate_training_data(game_record, board_size):
    board = np.zeros([board_size, board_size])
    data = []
    player = 1
    winner = -1 if len(game_record) % 2 == 0 else 1
    for i in range(len(game_record)):
        step = str_to_move(game_record[i]['action'])
        board[step[0], step[1]] = player
        data.append({"state": np.array(board, copy=True), "distribution": game_record[i]['distribution'], "value": winner})
        player, winner = -player, -winner
    return data


def generate_data_loader(stack):
    state, distrib, winner = stack.seq()
    tensor_x = torch.stack([torch.Tensor(s) for s in state])
    tensor_y1 = torch.stack([torch.Tensor(y1) for y1 in distrib])
    tensor_y2 = torch.stack([torch.Tensor(np.array(y2)) for y2 in winner])
    dataset = torch_data.TensorDataset(tensor_x, tensor_y1, tensor_y2)
    my_loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return my_loader